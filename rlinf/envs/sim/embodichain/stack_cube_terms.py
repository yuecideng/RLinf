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

"""Stack Cube terms for EmbodiChain's native managers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from embodichain.lab.gym.envs.managers import Functor, SceneEntityCfg
from embodichain.lab.gym.envs.objectives import ConditionResult
from scipy.spatial.transform import Rotation

from rlinf.envs.sim.embodichain.stack_cube_metrics import stack_cube_metrics

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.envs.managers import ConditionCfg, EventCfg, RewardCfg

__all__ = [
    "StackCubeCompletion",
    "StackCubeReset",
    "StackCubeReward",
    "stack_cube_state",
]


def stack_cube_state(
    env: EmbodiedEnv,
    obs: dict[str, Any],
    link_name: str = "panda_hand",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    gripper_part: str = "hand",
) -> torch.Tensor:
    """Return dataset XYZ, Euler XYZ and two signed finger positions.

    Args:
        env: Native environment owning the robot.
        obs: Native observation payload.
        link_name: Robot link defining the dataset frame.
        offset: Dataset observation point in that link's frame.
        gripper_part: Two independently actuated fingers.

    Returns:
        Float32 state with shape ``(num_envs, 8)``.
    """
    del obs
    poses = env.robot.get_link_pose(link_name, to_matrix=True).cpu().numpy()
    positions = poses[:, :3, 3] + poses[:, :3, :3] @ np.asarray(offset)
    eulers = Rotation.from_matrix(poses[:, :3, :3]).as_euler("xyz")
    fingers = (
        env.robot.get_qpos()[:, env.robot.get_joint_ids(gripper_part)].cpu().numpy()
    )
    if fingers.shape != (env.num_envs, 2):
        raise ValueError(
            f"Expected two fingers per environment, found {fingers.shape}."
        )
    state = np.concatenate((positions, eulers, fingers * [1.0, -1.0]), axis=-1)
    return torch.as_tensor(state, dtype=torch.float32, device=env.device)


class StackCubeCompletion(Functor):
    """Measure stacking once per native condition computation."""

    def __init__(self, cfg: ConditionCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._objects = [
            env.sim.get_rigid_object(cfg.params[key].uid)
            for key in ("blue_entity_cfg", "red_entity_cfg", "green_entity_cfg")
        ]
        if any(obj is None for obj in self._objects):
            raise ValueError(
                "Stack Cube conditions require blue, red and green rigid objects."
            )
        self._hand_ids = env.robot.get_joint_ids(cfg.params.get("gripper_part", "hand"))
        stable_steps = cfg.params.get("stable_steps", 20)
        if type(stable_steps) is not int or stable_steps <= 0:
            raise ValueError("stable_steps must be a positive integer.")
        if len(self._hand_ids) != 2:
            raise ValueError("Stack Cube completion requires two finger joints.")
        self._streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env: EmbodiedEnv,
        blue_entity_cfg: SceneEntityCfg,
        red_entity_cfg: SceneEntityCfg,
        green_entity_cfg: SceneEntityCfg,
        gripper_part: str = "hand",
        stable_steps: int = 20,
        require_stability: bool = False,
    ) -> ConditionResult:
        """Return native task acceptance and the physical stability measurements.

        Args:
            env: Native environment owning the measured state.
            blue_entity_cfg: Bottom object binding.
            red_entity_cfg: Middle object binding.
            green_entity_cfg: Top object binding.
            gripper_part: Configured finger control part.
            stable_steps: Required consecutive stable control steps.
            require_stability: Select stable evaluation acceptance instead of baseline stacking.

        Returns:
            Acceptance with cached per-row physical metrics.
        """
        del blue_entity_cfg, red_entity_cfg, green_entity_cfg, gripper_part
        metrics = stack_cube_metrics(
            torch.stack(
                [obj.get_local_pose(to_matrix=True)[:, :3, 3] for obj in self._objects],
                dim=1,
            ),
            env.robot.get_qpos()[:, self._hand_ids],
            torch.stack(
                [obj.body_data.lin_vel.norm(dim=-1) for obj in self._objects], dim=1
            ),
            torch.stack(
                [obj.body_data.ang_vel.norm(dim=-1) for obj in self._objects], dim=1
            ),
        )
        valid = torch.isfinite(metrics["positions"]).all(dim=(1, 2))
        valid &= torch.isfinite(metrics["linear_speeds"]).all(dim=-1)
        valid &= torch.isfinite(metrics["angular_speeds"]).all(dim=-1)
        self._streak = torch.where(metrics["stable_frame"] & valid, self._streak + 1, 0)
        metrics["stable_success"] = self._streak >= stable_steps
        accepted = (
            metrics["stable_success"]
            if require_stability
            else metrics["baseline_success"]
        )
        return ConditionResult(accepted, valid, metrics)

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Clear only selected rows' stability history."""
        self._streak[slice(None) if env_ids is None else env_ids] = 0


class StackCubeReward(Functor):
    """Score only the first episode while retaining full-horizon evaluation reward."""

    def __init__(self, cfg: RewardCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self, env: EmbodiedEnv, obs: dict, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """Return the native condition's first-episode success bonus."""
        del obs, action
        outcome = info["task_outcome"]
        measured_success = (
            outcome["physical_success"] & outcome["valid"] & ~outcome["failure"]
        )
        success = measured_success & self._active
        if not env.cfg.ignore_terminations:
            self._active &= ~(measured_success | outcome["failure"])
        return success.float()

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        """Enable scoring for selected new episodes."""
        self._active[slice(None) if env_ids is None else env_ids] = True


class StackCubeReset(Functor):
    """Apply a host-selected frozen cohort inside the native reset lifecycle."""

    def __init__(self, cfg: EventCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._pending: tuple[list[dict], list[dict] | None] | None = None

    def set_requests(
        self, requests: list[dict], initial_states: list[dict] | None = None
    ) -> None:
        """Stage one complete batch before calling native ``reset``."""
        if len(requests) != self._env.num_envs or (
            initial_states is not None and len(initial_states) != len(requests)
        ):
            raise ValueError(
                "Frozen reset requires one request and initial state per scene."
            )
        if any(request.get("visual") is not None for request in requests):
            raise ValueError("Visual OOD reset is not implemented.")
        self._pending = (requests, initial_states)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: list[int] | torch.Tensor,
        entity_cfgs: list[SceneEntityCfg],
        arm_part: str = "arm",
        contact_lift: float = 0.1,
        ground_plane_visible: bool = False,
    ) -> None:
        """Clear contacts, place frozen poses and zero dynamics before observations."""
        if self._pending is None:
            return
        if list(torch.as_tensor(env_ids).cpu().tolist()) != list(range(env.num_envs)):
            raise NotImplementedError(
                "Frozen Stack Cube reset requires the full scene batch."
            )
        requests, initial_states = self._pending
        objects = [
            (name, env.sim.get_rigid_object(binding.uid))
            for name, binding in zip(("blue", "red", "green"), entity_cfgs, strict=True)
        ]
        poses = {}
        for name, obj in objects:
            pose = obj.get_local_pose(to_matrix=True).clone()
            if initial_states is not None:
                pose[:] = torch.as_tensor(
                    [state["object_poses"][name] for state in initial_states],
                    device=env.device,
                )
            else:
                pose[:, :3, 3] = torch.as_tensor(
                    [request["objects"][name]["position"] for request in requests],
                    device=env.device,
                )
                angles = np.asarray(
                    [request["objects"][name]["yaw_rad"] for request in requests]
                ).reshape(-1, 1)
                pose[:, :3, :3] = torch.as_tensor(
                    Rotation.from_euler("z", angles).as_matrix(), device=env.device
                )
            poses[name] = pose
        qpos = env.robot.get_qpos().clone()
        if initial_states is not None:
            qpos[:] = torch.as_tensor(
                [state["qpos"] for state in initial_states], device=env.device
            )
        else:
            qpos[:, env.robot.get_joint_ids(arm_part)] += torch.as_tensor(
                [request["arm_joint_offsets"] for request in requests],
                device=env.device,
            )
        limits = env.robot.get_qpos_limits()
        if (
            not bool(torch.isfinite(qpos).all())
            or bool((qpos < limits[..., 0]).any())
            or bool((qpos > limits[..., 1]).any())
        ):
            raise ValueError("Requested robot initial state violates joint limits.")
        env.sim.set_ground_plane_visibility(ground_plane_visible)
        # Preserve the reset-time camera history before the frozen placement.
        with env.sim.render_frame(capture_camera_images=True):
            env.get_obs()
        for name, obj in objects:
            lifted = poses[name].clone()
            lifted[:, 2, 3] += contact_lift
            obj.set_local_pose(lifted)
            obj.clear_dynamics()
        env.sim.update(env.physics_dt, 1, render_final_step=False)
        for name, obj in objects:
            obj.set_local_pose(poses[name])
            obj.clear_dynamics()
        env.robot.set_qpos(qpos, target=False)
        env.robot.set_qpos(qpos, target=True)
        env.robot.clear_dynamics()
        self._pending = None
