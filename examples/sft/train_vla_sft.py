# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import signal
import uuid
from pathlib import Path

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        raise KeyboardInterrupt("SFT received a stop signal.")

    if cfg.data.get("train_data_source") == "ods":
        signal.signal(signal.SIGTERM, interrupt)
    cfg = validate_cfg(cfg)
    logging.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    if cfg.data.get("train_data_source") == "ods":
        # Close the source before the driver's worker-failure handler exits Ray.
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGUSR1, interrupt)
    if (
        cfg.data.get("train_data_source") == "ods"
        and os.environ.get("RAY_ADDRESS") == "local"
    ):
        import ray

        os.environ["RAY_ADDRESS"] = ray._private.worker.global_worker.gcs_client.address
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")

    if cfg.actor.training_backend not in {"fsdp", "fsdp2"}:
        raise ValueError(f"{cfg.actor.training_backend} backend is not supported yet")
    source_handle = None
    requests = None
    try:
        stream = None
        if cfg.data.get("train_data_source") == "ods":
            from rlinf.data.datasets.openpi.ods import ODSStream, wait_channel
            from rlinf.workers.data.ods_worker import ODSDataWorker

            if (
                cluster.num_nodes != 1
                or component_placement.get_world_size("data") != 1
            ):
                raise ValueError("ODS SFT requires one source owner on a single node.")
            name = f"ods-{uuid.uuid4().hex}"
            world_size = component_placement.get_world_size("actor")
            requests = Channel.create(f"{name}-requests", maxsize=2 * world_size + 1)
            batches = Channel.create(f"{name}-batches", maxsize=1)
            status = Channel.create(f"{name}-status", maxsize=1)
            resume = cfg.runner.get("resume_dir")
            start_step = parse_global_step_from_checkpoint_path(resume) if resume else 0
            source = ODSDataWorker.create_group(
                cfg, world_size=world_size, start_step=start_step
            ).launch(
                cluster,
                name=f"{name}-source",
                placement_strategy=component_placement.get_strategy("data"),
            )
            source_handle = source.run(requests, batches, status)
            ready = wait_channel(
                status.get(key="ready", async_op=True), float(cfg.data.ods.data_timeout)
            )
            if ready["kind"] != "ready":
                raise RuntimeError(
                    f"ODS source initialization failed: {ready.get('error')}."
                )
            stream = ODSStream(requests, batches, ready["metadata"])
        actor_group = FSDPVlaSftWorker.create_group(cfg, ods_stream=stream).launch(
            cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
        )
        stop_condition = None
        if stream is not None and cfg.data.ods.get("calibration", False):
            initial_step = stream.metadata["start_step"]
            updates = (
                Path(cfg.runner.logger.log_path)
                / cfg.runner.logger.experiment_name
                / "ods/updates.jsonl"
            )

            def stop_condition(step: int) -> bool:
                if step - initial_step < 13 or not updates.exists():
                    return False
                rows = [json.loads(line) for line in updates.read_text().splitlines()]
                generations = updates.with_name("generations.jsonl")
                refills = updates.with_name("refills.jsonl")
                if len(rows) < 13 or not generations.exists():
                    return False
                start, end = rows[2]["monotonic_seconds"], rows[-1]["monotonic_seconds"]
                requested = {
                    row["generation"]: row["started"]
                    for row in (
                        json.loads(line) for line in refills.read_text().splitlines()
                    )
                }
                return any(
                    requested[row["generation"]] >= start and row["completed_at"] <= end
                    for row in (
                        json.loads(line)
                        for line in generations.read_text().splitlines()
                    )
                )

        runner = SFTRunner(cfg=cfg, actor=actor_group, stop_condition=stop_condition)
        runner.init_workers()
        runner.run()
    finally:
        if source_handle is not None:
            from rlinf.data.datasets.openpi.ods import wait_channel

            if not source_handle.done():
                wait_channel(requests.put({"kind": "stop"}, async_op=True), 30)
            audit = wait_channel(source_handle, float(cfg.data.ods.shutdown_timeout))[0]
            logging.info("ODS source audit: %s", json.dumps(audit))
            if audit.get("error"):
                raise RuntimeError(f"ODS source failed: {audit['error']}")


if __name__ == "__main__":
    main()
