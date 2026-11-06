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

"""Native Stack Cube collection and physical admission for online SFT."""

from __future__ import annotations

import json
import os
import resource
import shutil
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import psutil
import torch
from embodichain.lab.gym.envs.managers import SceneEntityCfg

from rlinf.data.datasets.openpi.ods import sha256_file
from rlinf.envs.sim.embodichain.stack_cube import STACK_PROMPT
from rlinf.envs.sim.embodichain.stack_cube_metrics import stack_cube_metrics
from rlinf.envs.sim.embodichain.stack_cube_terms import StackCubeReset


def prepare_stack_cube_source(cfg: Any, output: Path) -> tuple[str, dict[str, Any]]:
    """Materialize immutable SDK input paths and the policy schema for one run."""
    import yaml
    from embodichain.data_pipeline.datasets import data_schema_fingerprint
    from embodichain.data_pipeline.engine import data as sdk_engine

    from rlinf.envs.sim.embodichain.embodichain_env import _resolve_gym_config_path

    source = _resolve_gym_config_path(str(cfg.gym_config_path))
    root = source.parent
    raw = yaml.safe_load(source.read_text())
    if raw["env"]["target_control_frequency"] != 20:
        raise ValueError("ODS Stack Cube requires the qualified 20 Hz control clock.")
    output.mkdir(parents=True, exist_ok=True)
    for key in ("program", "integration", "execution_policy"):
        original = (root / raw["task_program"][key]).resolve(strict=True)
        snapshot = output / f"{key}{original.suffix}"
        shutil.copyfile(original, snapshot)
        raw["task_program"][key] = str(snapshot)
    scene = output / "scene.yaml"
    shutil.copyfile((root / raw["scene"]["component"]).resolve(strict=True), scene)
    raw["scene"]["component"] = str(scene)
    component = (root / raw["embodiment"]["component"]).resolve(strict=True)
    embodiment = yaml.safe_load(component.read_text())
    embodiment["simulation"]["fpath"] = str(Path(cfg.panda_urdf).resolve(strict=True))
    component_path = output / "embodiment.yaml"
    component_path.write_text(yaml.safe_dump(embodiment, sort_keys=False))
    raw["embodiment"]["component"] = str(component_path)
    raw["id"] = f"RLinf-StackCube-ODS-{uuid.uuid4().hex[:10]}-v1"
    raw["num_envs"] = int(cfg.producer_envs)
    raw["max_episode_steps"] = 600
    raw.update(headless=True, device="cuda:0", gpu_id=0, seed=int(cfg.get("seed", 0)))
    raw["env"]["task_instruction"] = STACK_PROMPT
    reset = raw["env"]["events"]["requests"]
    reset["func"] = "OnlineStackCubeReset"
    manifest = output / "request_manifest.json"
    shutil.copyfile(Path(cfg.request_manifest).resolve(strict=True), manifest)
    reset["params"].update(
        request_manifest=str(manifest),
        audit_dir=str(output / "attempts"),
    )
    task = output / "task.yaml"
    task.write_text(yaml.safe_dump(raw, sort_keys=False))
    schema = {
        "version": "rlinf-stack-cube-ods-v1",
        "instruction": STACK_PROMPT,
        "observation": {
            "timing": "pre_action",
            "state": "eef_xyz_euler_xyz_signed_fingers@1",
            "state_transform": raw["env"]["observations"]["eef_state"],
            "state_names": ["x", "y", "z", "rx", "ry", "rz", "finger1", "finger2"],
            "cameras": {item["uid"]: item for item in embodiment["sensor"]},
        },
        "action": {
            "semantics": "policy_request",
            "step_dt": 0.05,
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            "terms": [
                {
                    "term": {"contract": "eef_pose.relative_substep_dls@1"},
                    "params": raw["env"]["actions"]["arm"]["params"],
                },
                {
                    "term": {"contract": "gripper.parallel@1"},
                    "params": raw["env"]["actions"]["gripper"]["params"],
                },
            ],
        },
        "sequence": {"history_length": 1, "prediction_horizon": 10, "action_chunk": 5},
        "panda_sha256": sha256_file(cfg.panda_urdf),
    }
    metadata = {
        "schema": schema,
        "schema_sha256": data_schema_fingerprint(schema),
        "request_manifest_sha256": sha256_file(manifest),
        "scene_sha256": sha256_file(scene),
        "execution_policy_sha256": sha256_file(raw["task_program"]["execution_policy"]),
        "integration_sha256": sha256_file(raw["task_program"]["integration"]),
        "task_program_sha256": sha256_file(raw["task_program"]["program"]),
        "sdk_engine_sha256": sha256_file(sdk_engine.__file__),
        "source_adapter_sha256": sha256_file(__file__),
        "environment_seed": raw["seed"],
    }
    (output / "source_contract.json").write_text(json.dumps(metadata, indent=2))
    return str(task), metadata


class OnlineStackCubeReset(StackCubeReset):
    """Advance a finite training manifest through the formal reset lifecycle."""

    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(cfg, env)
        manifest = json.loads(Path(cfg.params["request_manifest"]).read_text())
        self.requests = manifest["requests"]
        identities = [item["request_id"] for item in self.requests]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError(
                "ODS requires a nonempty manifest with unique request IDs."
            )
        if any(item["split"] != "train" for item in self.requests):
            raise ValueError("ODS production accepts only train-split requests.")
        self.cursor = 0
        self.current: list[dict[str, Any]] = []
        self.audit_dir = Path(cfg.params["audit_dir"])
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        env: Any,
        env_ids: list[int] | torch.Tensor,
        entity_cfgs: list[SceneEntityCfg],
        request_manifest: str,
        audit_dir: str,
        arm_part: str = "arm",
        contact_lift: float = 0.1,
        ground_plane_visible: bool = False,
    ) -> None:
        """Stage fresh requests before applying the existing frozen-pose reset."""
        del request_manifest, audit_dir
        end = self.cursor + env.num_envs
        if end > len(self.requests):
            raise RuntimeError("ODS training request manifest is exhausted.")
        self.current = self.requests[self.cursor : end]
        self.cursor = end
        self.set_requests(self.current)
        super().__call__(
            env, env_ids, entity_cfgs, arm_part, contact_lift, ground_plane_visible
        )


def _metrics(env: Any) -> dict[str, torch.Tensor]:
    objects = [env.sim.get_rigid_object(uid) for uid in ("blue", "red", "green")]
    return stack_cube_metrics(
        torch.stack(
            [obj.get_local_pose(to_matrix=True)[:, :3, 3] for obj in objects], 1
        ),
        env.robot.get_qpos()[:, env.robot.get_joint_ids("hand")],
        torch.stack([obj.body_data.lin_vel.norm(dim=-1) for obj in objects], 1),
        torch.stack([obj.body_data.ang_vel.norm(dim=-1) for obj in objects], 1),
    )


class _MeasuredEpisode:
    def __init__(self, env: Any) -> None:
        self.env = env
        self.stability: list[torch.Tensor] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: Any) -> Any:
        result = self.env.step(action)
        self.stability.append(_metrics(self.env.unwrapped)["stable_frame"].cpu())
        return result


def run_stack_cube_episode(
    env: Any,
    *,
    episode_index: int = 0,
    attempt_id: int = 0,
    should_stop: Any = None,
    progress: Any = None,
) -> Any:
    """Publish only complete programs with twenty stable final frames by step 450.

    Capacity remains 600 so rejected late successes can be diagnosed. Requests
    and settling use the same native reset, controller and physics as collection.
    """
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.lab.gym.envs.types import ControllerAction

    target = env.unwrapped
    resetter = target.event_manager.get_functor_cfg("requests").func
    env.reset(options={"save_data": False})
    measured = _MeasuredEpisode(env)
    previous_guard = getattr(target, "_demo_no_auto_reset", False)
    target._demo_no_auto_reset = True
    try:
        result = execute_demo_episode(
            measured,
            episode_index=episode_index,
            attempt_id=attempt_id,
            instruction=STACK_PROMPT,
            should_stop=should_stop,
            progress=progress,
        )
        program_end = len(measured.stability)
        # Completed SDK rows stay inactive: settling qualifies the physical
        # episode but does not append supervision outside Task Program spans.
        while len(measured.stability) < 600 and not (should_stop and should_stop()):
            measured.step(ControllerAction(target.robot.get_qpos().clone()))
            if len(measured.stability) - program_end >= 20 and bool(
                torch.stack(measured.stability[-20:]).all()
            ):
                break
        length = len(measured.stability)
        stable = (
            torch.stack(measured.stability[-20:]).all(dim=0)
            if length >= 20
            else torch.zeros(target.num_envs, dtype=torch.bool)
        )
        qualified = stable & result.completed & (length <= 450)
        if should_stop and should_stop():
            qualified.zero_()
        accepted = tuple(qualified.tolist())
        audit = {
            "producer_pid": os.getpid(),
            "producer_create_time": psutil.Process().create_time(),
            "producer_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
            "request_ids": [request["request_id"] for request in resetter.current],
            "episode_index": episode_index,
            "attempt_id": attempt_id,
            "program_completed": result.completed,
            "program_steps": program_end,
            "physical_steps": length,
            "stored_steps": list(result.lengths),
            "passed_450": list(accepted),
            "final_stable": stable.tolist(),
        }
        (resetter.audit_dir / f"attempt_{resetter.cursor:08d}.json").write_text(
            json.dumps(audit, indent=2)
        )
        return replace(
            result,
            completed=all(accepted),
            completed_by_env=accepted,
            success=accepted,
            terminal_reason=result.terminal_reason
            if all(accepted)
            else "physical_admission_failed",
            terminal_reasons=tuple(
                "completed" if flag else "physical_admission_failed"
                for flag in accepted
            ),
        )
    finally:
        target._demo_no_auto_reset = previous_guard
