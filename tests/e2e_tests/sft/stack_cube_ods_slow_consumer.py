# Copyright 2026 The RLinf Authors.
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

"""Exercise one uncommitted ODS batch across real scheduler Channels."""

import os
import time
import uuid
from pathlib import Path

from hydra import compose, initialize_config_dir

from rlinf.data.datasets.openpi.ods import sha256_file, wait_channel
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.data.ods_worker import ODSDataWorker


def main() -> None:
    """Delay delivery consumption and duplicate a request without spending twice."""
    root = Path(os.environ["REPO_PATH"])
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "tests/e2e_tests/sft")
    ):
        cfg = compose(config_name="embodichain_stack_cube_ods_sft_openpi_pi05_ci")
    cfg.runner.logger.experiment_name = "slow_consumer"
    cfg.data.ods.flow.max_generated_episodes = 1
    cluster = Cluster(cluster_cfg=cfg.cluster)
    if os.environ.get("RAY_ADDRESS") == "local":
        import ray

        os.environ["RAY_ADDRESS"] = ray._private.worker.global_worker.gcs_client.address
    placement = HybridComponentPlacement(cfg, cluster)
    name = f"slow-ods-{uuid.uuid4().hex}"
    requests = Channel.create(f"{name}-requests", maxsize=3)
    batches = Channel.create(f"{name}-batches", maxsize=1)
    status = Channel.create(f"{name}-status", maxsize=1)
    group = ODSDataWorker.create_group(cfg, world_size=1).launch(
        cluster,
        name=f"{name}-source",
        placement_strategy=placement.get_strategy("data"),
    )
    serving = group.run(requests, batches, status)
    try:
        ready = wait_channel(status.get(key="ready", async_op=True), 300)
        assert ready["kind"] == "ready", ready
        request = {"kind": "request", "rank": 0, "step": 1}
        wait_channel(requests.put(request, async_op=True), 30)
        wait_channel(requests.put(request, async_op=True), 30)
        # No optimizer commit is sent while the consumer pauses.
        time.sleep(1)
        packet = wait_channel(batches.get(key=0, async_op=True), 30)
        assert packet["kind"] == "batch", packet
        assert packet["batch"]["actions"].shape == (2, 10, 7)
        assert packet["audit"]["delivered_chunks"] == 2
    finally:
        if not serving.done():
            wait_channel(requests.put({"kind": "stop"}, async_op=True), 30)
        audit = wait_channel(serving, 30)[0]
    assert audit["error"] is None, audit
    assert audit["delivered_chunks"] == 2 and audit["committed_chunks"] == 0
    assert audit["quota_final"]["total_delivered_chunks"] == 2
    assert not audit["producer_alive_after_stop"]
    saved = Path(cfg.runner.logger.log_path) / "slow_consumer/ods/supply_audit.json"
    digest = sha256_file(saved)
    repeated = ODSDataWorker.create_group(cfg, world_size=1).launch(
        cluster,
        name=f"{name}-repeated",
        placement_strategy=placement.get_strategy("data"),
    )
    restarted = repeated.run(requests, batches, status)
    rejection = wait_channel(status.get(key="ready", async_op=True), 30)
    assert rejection["kind"] == "error" and "already exists" in rejection["error"]
    assert wait_channel(restarted, 30)[0]["delivered_chunks"] == 0
    assert sha256_file(saved) == digest


if __name__ == "__main__":
    main()
