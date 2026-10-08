# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

"""Run a closed-loop OpenPI policy in an EmbodiChain task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import OmegaConf

# Executing this file directly puts ``toolkits/standalone_eval_scripts`` at
# ``sys.path[0]``.  That directory contains the legacy OpenPI evaluation
# helpers package, which would shadow the installed ``openpi`` package used by
# the RLinf evaluator below.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:] = [
    entry for entry in sys.path if Path(entry or ".").resolve() != _SCRIPT_DIR
]

from rlinf.envs.sim.embodichain.embodichain_env import EmbodiChainEnv  # noqa: E402
from toolkits.lerobot.calculate_norm_stats import (  # noqa: E402
    _parse_delta_action_mask,
)
from toolkits.lerobot.evaluate_embodichain_openpi import (  # noqa: E402
    _resolve_norm_stats_directory,
)
from toolkits.standalone_eval_scripts.openpi_process_predictor import (  # noqa: E402
    OpenPIProcessPredictor,
)

_DEFAULT_POUR_WATER_RETURN_POSITION = torch.tensor(
    [0.75, -0.10, 0.962], dtype=torch.float32
)
_POUR_WATER_RELATIVE_POSE = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.05],
        [0.0, 1.0, 0.0, -0.10],
        [0.0, 0.0, 1.0, 0.125],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)
_POUR_WATER_ROTATE_ANGLE = -1.0471975511965976  # task_program integration.yaml
_POUR_WATER_TARGET_POSITION_TOLERANCE = 0.03
_POUR_WATER_TARGET_ORIENTATION_TOLERANCE = 0.25
_POUR_WATER_ROTATION_TOLERANCE = 0.25
_POUR_WATER_DWELL_FRAMES = 5
_SIM_MATMUL_PRECISION = "highest"


@contextmanager
def _matmul_precision_scope(precision: str):
    """Temporarily set PyTorch float32 matmul precision and restore it."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _matmul_tf32_flags() -> dict[str, bool]:
    """Return the CUDA matmul and cuDNN TF32 flags for evaluation metadata."""
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _quaternion_relative_angle(initial: torch.Tensor, current: torch.Tensor) -> float:
    """Return the shortest rotation angle between two quaternions."""
    initial = torch.as_tensor(initial, dtype=torch.float32).reshape(-1).cpu()
    current = torch.as_tensor(current, dtype=torch.float32).reshape(-1).cpu()
    if initial.numel() != 4 or current.numel() != 4:
        raise ValueError("Quaternion values must each contain four components.")
    initial = initial / torch.linalg.vector_norm(initial).clamp_min(1.0e-8)
    current = current / torch.linalg.vector_norm(current).clamp_min(1.0e-8)
    dot = torch.abs(torch.dot(initial, current)).clamp(0.0, 1.0)
    return float(2.0 * torch.acos(dot).item())


def _pour_water_physical_success(
    final_position: torch.Tensor | None,
    max_tilt: float,
    *,
    return_position: torch.Tensor = _DEFAULT_POUR_WATER_RETURN_POSITION,
    position_tolerance: float = 0.05,
    tilt_threshold: float = 0.5,
) -> tuple[bool, float | None]:
    """Apply the PourWater physical acceptance rule.

    PourWater's Task Program has no liquid-volume state: its validator checks
    that the bottle returns near ``bottle_return_pose``. A policy rollout does
    not consume the bridge's expert action iterator, so we additionally require
    a meaningful bottle rotation while it was being manipulated. The returned
    distance is useful for diagnosing failed episodes.
    """
    if final_position is None:
        return False, None
    final_position = torch.as_tensor(final_position, dtype=torch.float32).reshape(-1)
    return_position = torch.as_tensor(return_position, dtype=torch.float32).reshape(-1)
    if final_position.numel() < 3 or return_position.numel() != 3:
        raise ValueError("PourWater positions must contain three coordinates.")
    distance = float(
        torch.linalg.vector_norm(final_position[:3] - return_position).item()
    )
    accepted = distance <= position_tolerance and max_tilt >= tilt_threshold
    return accepted, distance


def _sim_rigid_object_pose(
    env: EmbodiChainEnv, uid: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read one rigid-object position/quaternion from the live simulator."""
    try:
        asset = env.env.sim.get_asset(uid)
        pose = asset.get_local_pose(to_matrix=False)
    except (AttributeError, KeyError, RuntimeError):
        return None
    if isinstance(pose, (tuple, list)) and len(pose) == 2:
        position, quaternion = pose
        position = torch.as_tensor(position, dtype=torch.float32).reshape(-1, 3)[0]
        quaternion = torch.as_tensor(quaternion, dtype=torch.float32).reshape(-1, 4)[0]
        return position, quaternion
    pose = torch.as_tensor(pose, dtype=torch.float32)
    if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
        position = pose.reshape(-1, 4, 4)[0, :3, 3]
        rotation = pose.reshape(-1, 4, 4)[0, :3, :3]
        trace = rotation.trace()
        quaternion = torch.tensor(
            [
                1.0 + trace,
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=torch.float32,
        )
        return position, quaternion
    pose = pose.reshape(-1, pose.shape[-1])[0]
    if pose.numel() < 7:
        return None
    return pose[:3], pose[3:7]


def _sim_rigid_object_matrix(env: EmbodiChainEnv, uid: str) -> torch.Tensor | None:
    """Read one rigid-object pose as a world-frame 4x4 matrix."""
    try:
        asset = env.env.sim.get_asset(uid)
        pose = torch.as_tensor(
            asset.get_local_pose(to_matrix=True), dtype=torch.float32
        )
    except (AttributeError, KeyError, RuntimeError):
        return None
    if pose.ndim == 3 and pose.shape[-2:] == (4, 4):
        return pose[0].cpu()
    if pose.ndim == 2 and pose.shape == (4, 4):
        return pose.cpu()
    return None


def _rotation_angle(initial: torch.Tensor, current: torch.Tensor) -> float:
    """Return the angle between two 3x3 rotation matrices."""
    relative = initial.transpose(-1, -2) @ current
    cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.acos(cosine).item())


def _task_program_success(info: Any) -> bool:
    """Read the raw RLinf/EmbodiChain success signal without inference."""
    if not isinstance(info, dict):
        return False
    success = info.get("success")
    if success is not None and _bool_at(success):
        return True
    episode = info.get("episode")
    return isinstance(episode, dict) and _bool_at(episode.get("success_once", False))


class _PourWaterPhysicalTracker:
    """Track the bottle pose needed for a policy-side PourWater verdict."""

    def __init__(
        self,
        return_position: torch.Tensor,
        *,
        position_tolerance: float,
        tilt_threshold: float,
        target_position_tolerance: float = _POUR_WATER_TARGET_POSITION_TOLERANCE,
        target_orientation_tolerance: float = _POUR_WATER_TARGET_ORIENTATION_TOLERANCE,
        rotation_tolerance: float = _POUR_WATER_ROTATION_TOLERANCE,
        dwell_frames: int = _POUR_WATER_DWELL_FRAMES,
    ) -> None:
        self.return_position = torch.as_tensor(return_position, dtype=torch.float32)
        self.position_tolerance = float(position_tolerance)
        self.tilt_threshold = float(tilt_threshold)
        self.target_position_tolerance = float(target_position_tolerance)
        self.target_orientation_tolerance = float(target_orientation_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.dwell_frames = int(dwell_frames)
        self.initial_quaternion: torch.Tensor | None = None
        self.final_position: torch.Tensor | None = None
        self.max_tilt = 0.0
        self.max_rotation = 0.0
        self.initial_matrix: torch.Tensor | None = None
        self.final_matrix: torch.Tensor | None = None
        self.max_target_position_error = float("inf")
        self.max_target_orientation_error = float("inf")
        self.min_pour_rotation_error: float | None = None
        self.min_pour_rotation_error_at_valid_position: float | None = None
        self.position_match_frames = 0
        self.rotation_match_frames = 0
        self.joint_pour_pose_frames = 0
        self.max_pour_dwell = 0
        self._pour_dwell = 0
        self.pour_geometry_seen = False

    def reset(self, env: EmbodiChainEnv) -> None:
        self.initial_quaternion = None
        self.final_position = None
        self.max_tilt = 0.0
        self.max_rotation = 0.0
        self.initial_matrix = None
        self.final_matrix = None
        self.max_target_position_error = float("inf")
        self.max_target_orientation_error = float("inf")
        self.min_pour_rotation_error = None
        self.min_pour_rotation_error_at_valid_position = None
        self.position_match_frames = 0
        self.rotation_match_frames = 0
        self.joint_pour_pose_frames = 0
        self.max_pour_dwell = 0
        self._pour_dwell = 0
        self.pour_geometry_seen = False
        self.update(env)

    def update(self, env: EmbodiChainEnv) -> None:
        pose = _sim_rigid_object_pose(env, "bottle")
        if pose is None:
            return
        position, quaternion = pose
        if self.initial_quaternion is None:
            self.initial_quaternion = quaternion.detach().cpu()
        self.final_position = position.detach().cpu()
        self.max_rotation = max(
            self.max_rotation,
            _quaternion_relative_angle(self.initial_quaternion, quaternion),
        )
        bottle_matrix = _sim_rigid_object_matrix(env, "bottle")
        cup_matrix = _sim_rigid_object_matrix(env, "cup")
        if bottle_matrix is None or cup_matrix is None:
            self._pour_dwell = 0
            return
        if self.initial_matrix is None:
            self.initial_matrix = bottle_matrix.clone()
        self.final_matrix = bottle_matrix.clone()
        initial_z = self.initial_matrix[:3, 2]
        current_z = bottle_matrix[:3, 2]
        z_cosine = (
            torch.dot(initial_z, current_z)
            / (
                torch.linalg.vector_norm(initial_z)
                * torch.linalg.vector_norm(current_z)
            ).clamp_min(1.0e-8)
        ).clamp(-1.0, 1.0)
        self.max_tilt = max(self.max_tilt, float(torch.acos(z_cosine).item()))
        target_matrix = cup_matrix @ _POUR_WATER_RELATIVE_POSE
        target_position_error = float(
            torch.linalg.vector_norm(bottle_matrix[:3, 3] - target_matrix[:3, 3]).item()
        )
        target_orientation_error = _rotation_angle(
            target_matrix[:3, :3], bottle_matrix[:3, :3]
        )
        poured_matrix = target_matrix.clone()
        angle = _POUR_WATER_ROTATE_ANGLE
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        poured_matrix[:3, :3] = poured_matrix[:3, :3] @ torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
        )
        pour_rotation_error = _rotation_angle(
            poured_matrix[:3, :3], bottle_matrix[:3, :3]
        )
        position_match = target_position_error <= self.target_position_tolerance
        rotation_match = pour_rotation_error <= self.rotation_tolerance
        target_match = (
            position_match
            and target_orientation_error <= self.target_orientation_tolerance
        )
        pour_match = position_match and rotation_match
        self.position_match_frames += int(position_match)
        self.rotation_match_frames += int(rotation_match)
        self.joint_pour_pose_frames += int(pour_match)
        self.min_pour_rotation_error = (
            pour_rotation_error
            if self.min_pour_rotation_error is None
            else min(self.min_pour_rotation_error, pour_rotation_error)
        )
        if position_match:
            self.min_pour_rotation_error_at_valid_position = (
                pour_rotation_error
                if self.min_pour_rotation_error_at_valid_position is None
                else min(
                    self.min_pour_rotation_error_at_valid_position,
                    pour_rotation_error,
                )
            )
        self.max_target_position_error = min(
            self.max_target_position_error, target_position_error
        )
        self.max_target_orientation_error = min(
            self.max_target_orientation_error, target_orientation_error
        )
        if pour_match and not target_match:
            self._pour_dwell += 1
            self.max_pour_dwell = max(self.max_pour_dwell, self._pour_dwell)
        else:
            self._pour_dwell = 0
        if self.max_pour_dwell >= self.dwell_frames:
            self.pour_geometry_seen = True

    def result(self) -> dict[str, Any]:
        proxy_success, final_distance = _pour_water_physical_success(
            self.final_position,
            self.max_tilt,
            return_position=self.return_position,
            position_tolerance=self.position_tolerance,
            tilt_threshold=self.tilt_threshold,
        )
        final_orientation_error = (
            _rotation_angle(self.initial_matrix[:3, :3], self.final_matrix[:3, :3])
            if self.initial_matrix is not None and self.final_matrix is not None
            else None
        )
        strict_success = bool(
            proxy_success
            and self.pour_geometry_seen
            and final_orientation_error is not None
            and final_orientation_error <= self.target_orientation_tolerance
        )
        return {
            "physical_success": strict_success,
            "physical_proxy_success": proxy_success,
            "strict_geometry_success": strict_success,
            "bottle_final_distance": final_distance,
            "max_bottle_tilt": self.max_tilt,
            "max_bottle_rotation": self.max_rotation,
            "bottle_pose_available": self.final_position is not None,
            "pour_geometry_seen": self.pour_geometry_seen,
            "max_pour_dwell_frames": self.max_pour_dwell,
            "min_bottle_to_target_position_error": self.max_target_position_error,
            "min_bottle_to_target_orientation_error": self.max_target_orientation_error,
            "min_pour_rotation_error": self.min_pour_rotation_error,
            "min_pour_rotation_error_at_valid_position": self.min_pour_rotation_error_at_valid_position,
            "position_match_frames": self.position_match_frames,
            "rotation_match_frames": self.rotation_match_frames,
            "joint_pour_pose_frames": self.joint_pour_pose_frames,
            "final_bottle_orientation_error": final_orientation_error,
            "strict_target_position_tolerance": self.target_position_tolerance,
            "strict_target_orientation_tolerance": self.target_orientation_tolerance,
            "strict_pour_rotation_tolerance": self.rotation_tolerance,
            "strict_pour_dwell_frames": self.dwell_frames,
        }


def _is_pour_water_task(task_config: str, env: EmbodiChainEnv | None = None) -> bool:
    """Identify the task from its public environment or deployment id."""
    if env is not None:
        native = getattr(env, "env", env)
        spec = getattr(getattr(native, "unwrapped", native), "spec", None)
        identifier = getattr(spec, "id", None)
        if identifier:
            return str(identifier).casefold().split("-v", 1)[0] == "pourwater"
    path = Path(task_config).expanduser()
    if not path.is_file() and str(task_config).startswith("embodichain_tasks/"):
        from embodichain.utils.config_paths import resolve_config_path

        path = Path(resolve_config_path(task_config))
    if not path.is_file():
        from rlinf.envs.sim.embodichain.embodichain_env import (
            _resolve_gym_config_path,
        )

        path = _resolve_gym_config_path(str(task_config))
    identifier = OmegaConf.load(path).get("id", "")
    return str(identifier).casefold().split("-v", 1)[0] == "pourwater"


def _bool_at(value: Any, index: int = 0) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[index].item())
    if isinstance(value, (list, tuple)):
        return bool(value[index])
    return bool(value)


def _snap_gripper(action: torch.Tensor, threshold: float) -> torch.Tensor:
    """Snap Franka's two continuous finger targets to open/closed values."""
    if action.shape[-1] < 2:
        return action
    snapped = action.clone()
    fingers = snapped[..., -2:]
    closed = fingers.mean(dim=-1, keepdim=True) <= threshold
    snapped[..., -2:] = torch.where(closed, 0.0, 0.04)
    return snapped


def evaluate(
    task_config: str,
    checkpoint_dir: str,
    *,
    config_name: str,
    action_dim: int,
    norm_stats_path: str,
    num_episodes: int,
    max_steps: int,
    action_chunk: int,
    num_steps: int,
    include_phase_input: bool,
    phase_scale: float,
    initial_hold_steps: int,
    seed: int,
    noise_seed: int | None,
    zero_noise: bool,
    gripper_threshold: float | None,
    gripper_open_steps: int | None,
    gripper_close_step: int | None,
    qpos_feedforward: float,
    action_application_mode: str,
    max_joint_step: float | None,
    qpos_track_max_step: float | None,
    action_settle_steps: int,
    expert_correction: bool,
    correction_max_steps: int,
    correction_threshold: float,
    correction_force_steps: int,
    delta_action_mask: list[bool] | None,
    position_tolerance: float,
    tilt_threshold: float,
    device: str,
    clip_actions: bool = False,
    eval_sft_image_crop: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint and return episode-level metrics."""
    if action_settle_steps <= 0:
        raise ValueError("action_settle_steps must be positive")
    entry_matmul_precision = torch.get_float32_matmul_precision()
    model = None
    env = None
    report = None
    try:
        torch.set_float32_matmul_precision(_SIM_MATMUL_PRECISION)
        model = OpenPIProcessPredictor(
            checkpoint_dir,
            config_name=config_name,
            output_action_dim=action_dim,
            norm_stats_path=norm_stats_path,
            num_steps=num_steps,
            device=device,
            include_phase_input=include_phase_input,
            phase_scale=phase_scale,
            delta_action_mask=delta_action_mask,
            noise_seed=noise_seed,
            zero_noise=zero_noise,
            eval_sft_image_crop=eval_sft_image_crop,
        )
        model_matmul_precision = model.metadata["matmul_precision"]
        model_tf32_flags = model.metadata["tf32_flags"]
        sim_tf32_flags = _matmul_tf32_flags()
        env_cfg = SimpleNamespace(
            gym_config_path=task_config,
            headless=True,
            sim_device=device,
            seed=seed,
            state_keys=["qpos"],
            main_camera_uid="cam_high",
            wrist_camera_uids=[],
            auto_reset=False,
            ignore_terminations=False,
            max_episode_steps=max_steps,
            video_cfg=None,
            is_eval=True,
            disable_dataset_recording=True,
            action_application_mode=action_application_mode,
            clip_actions=clip_actions,
            correction=SimpleNamespace(
                enabled=expert_correction,
                max_steps=correction_max_steps,
                deviation_threshold=correction_threshold,
                force_steps=correction_force_steps,
            ),
        )
        env = EmbodiChainEnv(
            env_cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=SimpleNamespace(cluster_node_rank=0, rank=0),
        )
        is_pour_water = _is_pour_water_task(task_config, env)
        successes = 0
        physical_successes = 0
        physical_proxy_successes = 0
        strict_geometry_successes = 0
        strict_task_program_successes = 0
        task_program_successes = 0
        intervention_steps = 0
        lengths: list[int] = []
        returns: list[float] = []
        episode_reports: list[dict[str, Any]] = []
        for episode_index in range(num_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            physical_tracker = (
                _PourWaterPhysicalTracker(
                    _DEFAULT_POUR_WATER_RETURN_POSITION,
                    position_tolerance=position_tolerance,
                    tilt_threshold=tilt_threshold,
                )
                if is_pour_water
                else None
            )
            pick_initial_position = None
            pick_final_position = None
            pick_goal_position = None
            pick_max_displacement = 0.0
            if not is_pour_water:
                cube_pose = _sim_rigid_object_pose(env, "cube")
                goal_pose = _sim_rigid_object_pose(env, "goal_marker")
                if cube_pose is not None:
                    pick_initial_position = cube_pose[0].detach().cpu()
                    pick_final_position = pick_initial_position.clone()
                if goal_pose is not None:
                    pick_goal_position = goal_pose[0].detach().cpu()
            if physical_tracker is not None:
                physical_tracker.reset(env)
            previous_action = obs["states"][0, :action_dim].detach().cpu().clone()
            action_queue: list[torch.Tensor] = []
            episode_return = 0.0
            task_program_success = False
            physical_completion_report = None
            length = max_steps
            for step_index in range(max_steps):
                if not action_queue:
                    policy_obs = dict(obs)
                    policy_obs.setdefault(
                        "wrist_images", policy_obs.get("extra_view_images")
                    )
                    actions, _ = model.predict_action_batch(policy_obs)
                    action_queue = []
                    for action in actions[0, :action_chunk]:
                        target = torch.as_tensor(action[..., :action_dim]).to("cpu")
                        action_queue.extend([target.clone()] * action_settle_steps)
                if step_index < initial_hold_steps:
                    action = previous_action.clone()
                else:
                    action = action_queue.pop(0)
                if qpos_track_max_step is not None:
                    measured_qpos = obs["states"][0, :action_dim].detach().cpu()
                    action = measured_qpos + (action - measured_qpos).clamp(
                        -qpos_track_max_step, qpos_track_max_step
                    )
                elif max_joint_step is not None:
                    action = previous_action + (action - previous_action).clamp(
                        -max_joint_step, max_joint_step
                    )
                if gripper_threshold is not None:
                    action = _snap_gripper(action, gripper_threshold)
                # The Franka expert keeps the fingers open while moving to the
                # pre-grasp pose and closes once contact is established.  A
                # policy can predict the arm target a few frames ahead while
                # still being uncertain about the tiny (0.04 m) finger
                # targets.  Allow an explicit deploy-time phase schedule so
                # the safety filter can preserve this contact window without
                # changing the learned arm policy.
                if gripper_open_steps is not None and step_index < gripper_open_steps:
                    action[-2:] = 0.04
                if gripper_close_step is not None and step_index >= gripper_close_step:
                    action[-2:] = 0.0
                if qpos_feedforward:
                    # EmbodiChain's direct tensor action is a target qpos. The
                    # simulator drives the current qpos toward that target
                    # during the four physics substeps in ``env.step``. A
                    # small feed-forward term compensates the measured target
                    # tracking lag without turning the action into a delta.
                    current_qpos = obs["states"][0, :action_dim].detach().cpu()
                    action = action + float(qpos_feedforward) * (action - current_qpos)
                previous_action = action.clone()
                obs, reward, terminated, truncated, info = env.step(action)
                if "applied_action" in info:
                    previous_action = info["applied_action"][0].detach().cpu().clone()
                if pick_initial_position is not None:
                    cube_pose = _sim_rigid_object_pose(env, "cube")
                    if cube_pose is not None:
                        pick_final_position = cube_pose[0].detach().cpu()
                        pick_max_displacement = max(
                            pick_max_displacement,
                            float(
                                torch.linalg.vector_norm(
                                    pick_final_position - pick_initial_position
                                ).item()
                            ),
                        )
                intervene_flag = info.get("intervene_flag")
                if intervene_flag is not None:
                    intervention_steps += int(
                        torch.as_tensor(intervene_flag).sum().item()
                    )
                if physical_tracker is not None:
                    physical_tracker.update(env)
                    current_physical_report = physical_tracker.result()
                    if current_physical_report["strict_geometry_success"]:
                        physical_completion_report = current_physical_report
                episode_return += float(reward.reshape(-1)[0].item())
                task_program_success = task_program_success or _task_program_success(
                    info
                )
                if physical_tracker is None:
                    episode_done = (
                        _bool_at(terminated)
                        or _bool_at(truncated)
                        or task_program_success
                    )
                else:
                    # Raw success may describe demo progress. Pour completion
                    # follows observed geometry; failures and timeouts still end it.
                    terminal_without_raw_success = _bool_at(terminated) and not (
                        _bool_at(info.get("success", False))
                    )
                    episode_done = (
                        physical_completion_report is not None
                        or _bool_at(info.get("fail", False))
                        or _bool_at(truncated)
                        or terminal_without_raw_success
                    )
                if episode_done:
                    length = step_index + 1
                    break
            if physical_completion_report is not None:
                physical_report = physical_completion_report
            elif physical_tracker is not None:
                physical_report = physical_tracker.result()
            else:
                physical_report = {}
            physical_success = bool(physical_report.get("physical_success", False))
            physical_proxy_success = bool(
                physical_report.get("physical_proxy_success", physical_success)
            )
            strict_geometry_success = bool(
                physical_report.get("strict_geometry_success", physical_success)
            )
            strict_task_program_success = bool(
                strict_geometry_success and task_program_success
            )
            success = (
                strict_geometry_success
                if physical_tracker is not None
                else task_program_success
            )
            successes += int(success)
            physical_successes += int(physical_success)
            physical_proxy_successes += int(physical_proxy_success)
            strict_geometry_successes += int(strict_geometry_success)
            strict_task_program_successes += int(strict_task_program_success)
            task_program_successes += int(task_program_success)
            lengths.append(length)
            returns.append(episode_return)
            pick_completion_metrics = {}
            if pick_initial_position is not None:
                pick_completion_metrics["pick_final_hand_qpos"] = (
                    obs["states"][0, -2:].detach().cpu().tolist()
                )
                for key in (
                    "placement_released",
                    "placement_stable_steps",
                    "cube_linear_speed",
                    "cube_angular_speed",
                ):
                    value = info.get("metrics", {}).get(key)
                    pick_completion_metrics[key] = (
                        torch.as_tensor(value).reshape(-1)[0].item()
                        if value is not None
                        else None
                    )
            episode_reports.append(
                {
                    "episode_index": episode_index,
                    "success": success,
                    "task_program_success": task_program_success,
                    "physical_proxy_success": physical_proxy_success,
                    "strict_geometry_success": strict_geometry_success,
                    "strict_task_program_success": strict_task_program_success,
                    **physical_report,
                    **pick_completion_metrics,
                    "pick_initial_position": (
                        pick_initial_position.tolist()
                        if pick_initial_position is not None
                        else None
                    ),
                    "pick_final_position": (
                        pick_final_position.tolist()
                        if pick_final_position is not None
                        else None
                    ),
                    "pick_goal_position": (
                        pick_goal_position.tolist()
                        if pick_goal_position is not None
                        else None
                    ),
                    "pick_max_displacement": pick_max_displacement,
                    "episode_length": length,
                    "return": episode_return,
                }
            )
        report = {
            "episodes": num_episodes,
            "checkpoint": str(Path(checkpoint_dir).expanduser().resolve()),
            "config_name": config_name,
            "task_config": str(Path(task_config).expanduser().resolve()),
            "model_matmul_precision": model_matmul_precision,
            "sim_matmul_precision": _SIM_MATMUL_PRECISION,
            "model_tf32_flags": model_tf32_flags,
            "sim_tf32_flags": sim_tf32_flags,
            "inference_process_mode": "spawn",
            "actor_process_pid": model.process_pid,
            "evaluator_process_pid": os.getpid(),
            "model_provenance": model.metadata["provenance"],
            "seed": seed,
            "action_dim": action_dim,
            "action_horizon": model.action_horizon,
            "action_chunk": action_chunk,
            "num_steps": num_steps,
            "include_phase_input": include_phase_input,
            "phase_scale": phase_scale,
            "eval_sft_image_crop": eval_sft_image_crop,
            "delta_action_mask": delta_action_mask,
            "norm_stats_sha256": hashlib.sha256(
                (
                    _resolve_norm_stats_directory(norm_stats_path) / "norm_stats.json"
                ).read_bytes()
            ).hexdigest(),
            "controller": {
                "action_application_mode": action_application_mode,
                "clip_actions": bool(clip_actions),
                "initial_hold_steps": initial_hold_steps,
                "gripper_open_steps": gripper_open_steps,
                "gripper_close_step": gripper_close_step,
                "gripper_threshold": gripper_threshold,
                "max_joint_step": max_joint_step,
                "qpos_track_max_step": qpos_track_max_step,
                "qpos_feedforward": qpos_feedforward,
                "action_settle_steps": action_settle_steps,
                "zero_noise": zero_noise,
            },
            "successes": successes,
            "success_rate": successes / max(1, num_episodes),
            "mean_episode_length": sum(lengths) / max(1, len(lengths)),
            "mean_return": sum(returns) / max(1, len(returns)),
            "noise_seed": noise_seed,
            "task_success_source": ("pour_water_physical" if is_pour_water else "info"),
            "physical_successes": physical_successes,
            "physical_success_rate": physical_successes / max(1, num_episodes),
            "physical_proxy_successes": physical_proxy_successes,
            "physical_proxy_success_rate": physical_proxy_successes
            / max(1, num_episodes),
            "strict_geometry_successes": strict_geometry_successes,
            "strict_geometry_success_rate": strict_geometry_successes
            / max(1, num_episodes),
            "strict_task_program_successes": strict_task_program_successes,
            "strict_task_program_success_rate": strict_task_program_successes
            / max(1, num_episodes),
            "task_program_successes": task_program_successes,
            "task_program_success_rate": task_program_successes / max(1, num_episodes),
            "evaluation_mode": "assisted" if expert_correction else "unassisted",
            "expert_correction": bool(expert_correction),
            "intervention_steps": intervention_steps,
            "position_tolerance": position_tolerance,
            "tilt_threshold": tilt_threshold,
            "per_episode": episode_reports,
        }
        return report
    finally:
        try:
            if env is not None:
                with _matmul_precision_scope(_SIM_MATMUL_PRECISION):
                    env.close()
        finally:
            try:
                if model is not None:
                    model.close()
                    if report is not None:
                        report["inference_process_cleanup"] = {
                            key: model.metadata.get(key)
                            for key in (
                                "close_response",
                                "process_exit_code",
                                "process_alive_after_close",
                                "close_error",
                            )
                        }
            finally:
                torch.set_float32_matmul_precision(entry_matmul_precision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--config-name", default="pi05_embodichain_joint_state_v2")
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--include-phase-input", action="store_true")
    parser.add_argument("--phase-scale", type=float, default=600.0)
    parser.add_argument(
        "--eval-sft-image-crop",
        action="store_true",
        help="Use the deterministic SFT train=True/rng=None image crop at inference.",
    )
    parser.add_argument(
        "--delta-action-mask",
        default="",
        help="Comma-separated bool mask for action dims converted to deltas.",
    )
    parser.add_argument(
        "--initial-hold-steps",
        type=int,
        default=0,
        help="Hold the reset qpos before requesting the first policy action.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Flow-sampling seed; pass -1 to use nondeterministic noise.",
    )
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="Snap the final two Franka finger targets at this threshold.",
    )
    parser.add_argument(
        "--gripper-open-steps",
        type=int,
        default=None,
        help="Keep Franka fingers open for this many control steps.",
    )
    parser.add_argument(
        "--gripper-close-step",
        type=int,
        default=None,
        help="Close Franka fingers from this control step onward.",
    )
    parser.add_argument("--max-joint-step", type=float, default=None)
    parser.add_argument(
        "--action-settle-steps",
        type=int,
        default=1,
        help="Repeat each predicted qpos target for this many control steps.",
    )
    parser.add_argument(
        "--qpos-feedforward",
        type=float,
        default=0.0,
        help="Add gain*(target-current qpos) before a direct target-qpos step.",
    )
    parser.add_argument(
        "--action-application-mode",
        choices=("target_position", "current_qpos"),
        default="target_position",
        help="Diagnostic EmbodiChain action application path.",
    )
    parser.add_argument(
        "--clip-actions",
        action="store_true",
        help="Clamp finite joint targets to the environment Box bounds; default off.",
    )
    parser.add_argument(
        "--qpos-track-max-step",
        type=float,
        default=None,
        help="Limit each target relative to measured qpos instead of prior target.",
    )
    parser.add_argument("--expert-correction", action="store_true")
    parser.add_argument("--correction-max-steps", type=int, default=50)
    parser.add_argument("--correction-threshold", type=float, default=0.05)
    parser.add_argument("--correction-force-steps", type=int, default=10)
    parser.add_argument(
        "--pour-position-tolerance",
        type=float,
        default=0.05,
        help="PourWater bottle-return position tolerance in meters.",
    )
    parser.add_argument(
        "--pour-tilt-threshold",
        type=float,
        default=0.5,
        help="PourWater maximum relative bottle rotation in radians.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = evaluate(
        args.task_config,
        args.checkpoint_dir,
        config_name=args.config_name,
        action_dim=args.action_dim,
        norm_stats_path=args.norm_stats_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        action_chunk=args.action_chunk,
        num_steps=args.num_steps,
        include_phase_input=args.include_phase_input,
        phase_scale=args.phase_scale,
        eval_sft_image_crop=args.eval_sft_image_crop,
        initial_hold_steps=args.initial_hold_steps,
        seed=args.seed,
        noise_seed=None if args.noise_seed < 0 else args.noise_seed,
        zero_noise=args.zero_noise,
        gripper_threshold=args.gripper_threshold,
        gripper_open_steps=args.gripper_open_steps,
        gripper_close_step=args.gripper_close_step,
        qpos_feedforward=args.qpos_feedforward,
        action_application_mode=args.action_application_mode,
        max_joint_step=args.max_joint_step,
        qpos_track_max_step=args.qpos_track_max_step,
        action_settle_steps=args.action_settle_steps,
        expert_correction=args.expert_correction,
        correction_max_steps=args.correction_max_steps,
        correction_threshold=args.correction_threshold,
        correction_force_steps=args.correction_force_steps,
        delta_action_mask=_parse_delta_action_mask(args.delta_action_mask),
        position_tolerance=args.pour_position_tolerance,
        tilt_threshold=args.pour_tilt_threshold,
        device=args.device,
        clip_actions=args.clip_actions,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
