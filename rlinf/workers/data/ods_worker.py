# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A single simulator-interpreter owner of ODS generation and global quotas."""

from __future__ import annotations

import json
import resource
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.ods import ODSBatchCoordinator
from rlinf.data.datasets.openpi.ods import sha256_file, wait_channel
from rlinf.scheduler import Channel, Worker


class _Stopped(Exception):
    pass


def project_openpi_batch(batch: Any, prompts: list[str]) -> dict[str, Any]:
    """Drop future observations and source-only fields before Channel transport."""
    obs = batch["obs"]
    packet = {
        name: obs["sensor", name, "color"][:, 0, ..., :3].to("cpu").contiguous()
        for name in ("front", "wrist")
    }
    packet.update(
        state=obs["robot", "eef_state"][:, 0].to("cpu").float().contiguous(),
        actions=batch["actions"].to("cpu").float().contiguous(),
        action_mask=batch["loss_mask"].to("cpu").bool().contiguous(),
        prompts=prompts,
    )
    count = packet["state"].shape[0]
    if packet["state"].shape != (count, 8) or packet["actions"].shape != (count, 10, 7):
        raise ValueError("ODS Stack Cube requires state[B,8] and actions[B,10,7].")
    if (
        packet["action_mask"].shape != (count, 10)
        or not packet["action_mask"].any(1).all()
    ):
        raise ValueError("ODS requires nonempty action_mask[B,10].")
    if any(not isinstance(prompt, str) or not prompt for prompt in packet["prompts"]):
        raise ValueError("ODS requires an overall task instruction for every sample.")
    if (
        not torch.isfinite(packet["state"]).all()
        or not torch.isfinite(packet["actions"]).all()
    ):
        raise ValueError("ODS produced nonfinite policy inputs or targets.")
    return packet


class ODSDataWorker(Worker):
    """Own one ODS session and deliver one globally accounted optimizer batch."""

    def __init__(self, cfg: Any, *, world_size: int, start_step: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.world_size, self.start_step = world_size, start_step

    def run(
        self, requests: Channel, batches: Channel, status: Channel
    ) -> dict[str, Any]:
        """Serve until STOP/failure, always closing this worker's simulation child."""
        from embodichain.data_pipeline.configuration import ODSConfiguration
        from embodichain.data_pipeline.datasets import OnlineDataSource, SequenceSpecCfg
        from embodichain.data_pipeline.engine import (
            OnlineDataEngine,
            OnlineDataEngineCfg,
        )
        from embodichain.data_pipeline.runtime import (
            ODSFlowController,
            ODSFlowControllerCfg,
        )

        from rlinf.envs.sim.embodichain.ods import prepare_stack_cube_source

        cfg = self.cfg.data.ods
        output = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "ods"
        )
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            error = (
                f"ODS source directory already exists: {output}. "
                "Choose a fresh runner.logger.experiment_name for each source session."
            )
            wait_channel(
                status.put(
                    {"kind": "error", "error": error}, key="ready", async_op=True
                ),
                30,
            )
            return {"error": error, "delivered_chunks": 0, "committed_chunks": 0}
        pending = None
        messages: deque[dict[str, Any]] = deque()
        last_message = time.monotonic()
        session_started = last_message
        wall_limit = cfg.get("max_wall_seconds")
        deadline = cfg.get("deadline_monotonic")
        observed_generation = 0
        completed_production_seconds = 0.0

        def poll() -> None:
            nonlocal pending, last_message
            nonlocal observed_generation, completed_production_seconds
            if deadline is not None and time.monotonic() >= float(deadline):
                raise TimeoutError("ODS calibration reached its cleanup deadline.")
            if wall_limit is not None and time.monotonic() - session_started > float(
                wall_limit
            ):
                raise TimeoutError("ODS session reached its wall-time limit.")
            if pending is None:
                pending = requests.get(async_op=True)
            if pending.done():
                message, pending = pending.wait(), None
                last_message = time.monotonic()
                if message["kind"] == "stop":
                    raise _Stopped()
                messages.append(message)
            if time.monotonic() - last_message > float(cfg.get("idle_timeout", 1800.0)):
                raise TimeoutError(
                    "ODS actor group stopped responding before committing its batch."
                )
            if engine is not None and engine.is_init:
                counters = engine.get_stats()
                generation = counters["completed_generation"]
                if generation > observed_generation:
                    if generation != observed_generation + 1:
                        raise RuntimeError(
                            "ODS generation audit missed a complete refill."
                        )
                    record = {
                        "generation": generation,
                        "accepted_episodes": engine.buffer_size,
                        "active_seconds": counters["production_seconds"]
                        - completed_production_seconds,
                        "completed_at": time.monotonic(),
                    }
                    with (output / "generations.jsonl").open("a") as file:
                        file.write(json.dumps(record) + "\n")
                    observed_generation = generation
                    completed_production_seconds = counters["production_seconds"]

        engine = None
        coordinator = ODSBatchCoordinator(
            world_size=self.world_size,
            global_batch_size=int(self.cfg.actor.global_batch_size),
            start_step=self.start_step + 1,
        )
        audit: dict[str, Any] = {
            "session_id": uuid.uuid4().hex,
            "start_step": self.start_step,
            "world_size": self.world_size,
            "global_batch_size": coordinator.global_batch_size,
        }
        learner: dict[int, dict[int, dict[str, Any]]] = {}
        error = None
        ready = False
        flow = None
        try:
            norm_sha256 = sha256_file(self.cfg.actor.model.openpi_data.norm_stats_path)
            task, contract = prepare_stack_cube_source(cfg, output)
            plan = None
            if cfg.get("recipe_path"):
                recipe = json.loads(Path(cfg.recipe_path).read_text())
                plan = ODSConfiguration(**recipe["configuration"])
                if plan.producer_envs != int(
                    cfg.producer_envs
                ) or plan.batch_size != int(self.cfg.actor.global_batch_size):
                    raise ValueError(
                        "ODS recipe must match producer count and global batch size."
                    )
            elif not cfg.get("calibration", False):
                raise ValueError(
                    "ODS requires a measured recipe outside explicit calibration runs."
                )
            settings = OmegaConf.to_container(cfg.engine, resolve=True)
            allocation = (
                {
                    "buffer_size": plan.buffer_size,
                    "episode_sample_budget": plan.episode_sample_budget,
                    "episode_sample_hard_limit": plan.episode_sample_hard_limit,
                }
                if plan is not None
                else OmegaConf.to_container(cfg.pilot_cache, resolve=True)
            )
            if set(settings) & set(allocation):
                raise ValueError(
                    "ODS cache and quotas belong to the recipe or pilot_cache, not engine."
                )
            settings.update(allocation)
            settings.setdefault(
                "lifecycle_event_capacity", 4 * int(settings["buffer_size"])
            )
            engine = OnlineDataEngine(
                OnlineDataEngineCfg(
                    **settings,
                    gym_config_path=task,
                    episode_runner="rlinf.envs.sim.embodichain.ods:run_stack_cube_episode",
                    state_dim=9,
                    action_dim=7,
                    action_semantics="policy_request",
                    action_contracts=(
                        "eef_pose.relative_substep_dls@1",
                        "gripper.parallel@1",
                    ),
                    max_episode_steps=600,
                    refill_policy="manual",
                    include_provenance=True,
                )
            )
            audit["bytes_per_episode"] = (
                sum(
                    value.numel() * value.element_size()
                    for value in engine.shared_buffer.values(
                        include_nested=True, leaves_only=True
                    )
                )
                // engine.buffer_size
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                (output / "refills.jsonl").write_text(
                    json.dumps({"generation": 1, "started": time.monotonic()}) + "\n"
                )
                startup = executor.submit(engine.start)
                try:
                    while not startup.done():
                        poll()
                        time.sleep(0.02)
                    startup.result()
                    poll()
                except BaseException:
                    engine.stop()
                    raise
            events_path = output / "lifecycle.jsonl"

            def record_events(events: list[dict[str, Any]]) -> None:
                with events_path.open("a") as file:
                    for event in events:
                        file.write(json.dumps(event) + "\n")

            def on_refill(generation: int, started: float) -> None:
                with (output / "refills.jsonl").open("a") as file:
                    file.write(
                        json.dumps({"generation": generation, "started": started})
                        + "\n"
                    )

            source_wait_seconds = 0.0

            def on_interval(kind: str, started: float, ended: float) -> None:
                nonlocal source_wait_seconds
                source_wait_seconds += ended - started

            flow = ODSFlowController(
                engine,
                ODSFlowControllerCfg(**OmegaConf.to_container(cfg.flow, resolve=True)),
                plan=plan,
                progress=poll,
                on_events=record_events,
                on_refill=on_refill,
                on_interval=on_interval,
            )
            source = OnlineDataSource(engine, schema=contract["schema"])
            sequence = SequenceSpecCfg(history_length=1, prediction_horizon=10)
            generator = torch.Generator().manual_seed(int(self.cfg.actor.seed))
            metadata = {
                **contract,
                **audit,
                "source_worker_sha256": sha256_file(__file__),
                "world_size": self.world_size,
                "norm_stats_sha256": norm_sha256,
                "recipe": asdict(plan) if plan else None,
            }
            (output / "session.json").write_text(json.dumps(metadata, indent=2))
            wait_channel(
                status.put(
                    {"kind": "ready", "metadata": metadata}, key="ready", async_op=True
                ),
                30,
            )
            ready = True
            while True:
                flow.poll()
                if not messages:
                    time.sleep(0.01)
                    continue
                message = messages.popleft()
                rank, step = message["rank"], message["step"]
                if message["kind"] == "request":
                    coordinator.request(rank, step)
                elif message["kind"] == "commit":
                    if step < coordinator.expected_step:
                        continue
                    learner.setdefault(step, {}).setdefault(rank, message)
                    if coordinator.commit(rank, step):
                        row = {
                            "step": step,
                            "completed_chunks": coordinator.global_batch_size,
                            "active_seconds": max(
                                item["active_seconds"]
                                for item in learner.pop(step).values()
                            ),
                            "monotonic_seconds": time.monotonic(),
                            "source": engine.get_stats(),
                        }
                        with (output / "updates.jsonl").open("a") as file:
                            file.write(json.dumps(row) + "\n")
                        refill_step = cfg.get("calibration_refill_step")
                        if (
                            plan is None
                            and refill_step is not None
                            and step - self.start_step == int(refill_step)
                        ):
                            flow.request_refill()
                else:
                    raise ValueError(f"Unexpected ODS message: {message['kind']}.")
                step = coordinator.ready_step
                if step is None:
                    continue

                def sample(size: int) -> dict[str, Any]:
                    if not flow.prepare_batch(size):
                        raise RuntimeError(
                            "ODS accepted-episode budget cannot fund another complete optimizer batch."
                        )
                    batch = flow.sample_context(
                        source, size, sequence, generator=generator
                    )
                    prompts = [
                        row[0] for row in source.resolve_language(batch[:, :1])["task"]
                    ]
                    return project_openpi_batch(batch, prompts)

                source_wait_seconds = 0.0
                packet = coordinator.draw(sample)
                coordinator.mark_delivered(step)
                audit.update(
                    delivered_chunks=coordinator.delivered_chunks,
                    committed_chunks=coordinator.committed_chunks,
                    source=engine.get_stats(),
                )
                local = coordinator.global_batch_size // self.world_size
                for rank in range(self.world_size):
                    wait_channel(
                        batches.put(
                            {
                                "kind": "batch",
                                "step": step,
                                "session_id": audit["session_id"],
                                "source_wait_seconds": source_wait_seconds,
                                "audit": audit.copy(),
                                "batch": {
                                    key: value[rank * local : (rank + 1) * local]
                                    for key, value in packet.items()
                                },
                            },
                            key=rank,
                            async_op=True,
                        ),
                        float(cfg.data_timeout),
                        progress=poll,
                    )
        except _Stopped:
            pass
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.log_error(error)
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except BaseException as exc:
                    error = error or f"ODS cleanup failed: {exc}"
                try:
                    audit["source_final"] = engine.get_stats()
                    audit["quota_final"] = engine.get_sampling_budget_snapshot()
                    if flow is not None:
                        flow.drain_events()
                except BaseException as exc:
                    audit["audit_error"] = str(exc)
            reports = [
                json.loads(file.read_text())
                for file in (output / "attempts").glob("*.json")
            ]
            audit["producer_peak_rss_bytes"] = max(
                (row.get("producer_peak_rss_bytes", 0) for row in reports), default=0
            )
            import psutil

            alive = False
            for row in reports:
                if "producer_pid" not in row:
                    continue
                try:
                    process = psutil.Process(row["producer_pid"])
                    alive |= (
                        process.create_time() == row["producer_create_time"]
                        and process.status() != psutil.STATUS_ZOMBIE
                    )
                except psutil.NoSuchProcess:
                    pass
            audit["producer_alive_after_stop"] = alive
            if alive:
                error = error or "ODS production process remained alive after stop."
            audit.update(
                delivered_chunks=coordinator.delivered_chunks,
                committed_chunks=coordinator.committed_chunks,
                error=error,
                owner_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            )
            (output / "supply_audit.json").write_text(json.dumps(audit, indent=2))
        if error:
            if not ready:
                wait_channel(
                    status.put(
                        {"kind": "error", "error": error}, key="ready", async_op=True
                    ),
                    30,
                )
            else:
                for rank in range(self.world_size):
                    wait_channel(
                        batches.put(
                            {"kind": "error", "error": error}, key=rank, async_op=True
                        ),
                        5,
                    )
        return audit
