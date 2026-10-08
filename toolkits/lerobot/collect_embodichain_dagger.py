# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect a small LeRobot DAgger set from EmbodiChain expert correction.

The collector executes the policy in ``EmbodiChainEnv`` and stores the action
actually applied by the environment.  When correction is active this is the
``intervene_action`` returned by the expert planner; otherwise it is the
policy action.  This keeps observations aligned with the action that followed
them and makes the output directly usable by the existing EmbodiChain OpenPI
data config.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from rlinf.data.storage.lerobot import LeRobotDatasetWriter
from rlinf.envs.sim.embodichain.embodichain_env import EmbodiChainEnv
from toolkits.lerobot.audit_embodichain_dataset import audit_dataset
from toolkits.standalone_eval_scripts.embodichain_openpi_eval import (
    _sim_rigid_object_pose,
)
from toolkits.standalone_eval_scripts.openpi_process_predictor import (
    OpenPIProcessPredictor,
)


def _bool_at(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item())
    if isinstance(value, (tuple, list)):
        return bool(value[0])
    return bool(value)


def _numpy(value: Any, *, dtype: np.dtype = np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _episode_success(info: dict[str, Any]) -> bool:
    success = info.get("success")
    if success is not None and _bool_at(success):
        return True
    episode = info.get("episode")
    return isinstance(episode, dict) and _bool_at(episode.get("success_once", False))


def _frame(
    obs: dict[str, Any],
    action: torch.Tensor,
    *,
    task: str,
    intervene: bool,
    done: bool,
    success: bool,
) -> dict[str, Any]:
    image = _numpy(obs["main_images"][0], dtype=np.uint8)
    state = _numpy(obs["states"][0])
    return {
        "observation.images.cam_high": image,
        "observation.state": state,
        "action": _numpy(action),
        "task": task,
        "done": np.asarray([done], dtype=bool),
        "is_success": np.asarray([success], dtype=bool),
        "intervene_flag": np.asarray([intervene], dtype=bool),
    }


def collect(
    *,
    task_config: str,
    checkpoint_dir: str,
    norm_stats_path: str,
    output_root: str,
    num_episodes: int,
    max_steps: int,
    action_dim: int,
    prompt: str,
    action_chunk: int,
    num_steps: int,
    seed: int,
    correction_max_steps: int,
    correction_threshold: float,
    correction_force_steps: int,
    device: str,
    config_name: str = "pi05_embodichain_joint",
) -> dict[str, Any]:
    """Collect corrected trajectories and write audit/norm-stat artifacts."""
    if num_episodes <= 0 or max_steps <= 0 or action_chunk <= 0:
        raise ValueError("num_episodes, max_steps, and action_chunk must be positive")
    output = Path(output_root).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    entry_precision = torch.get_float32_matmul_precision()
    resources = ExitStack()
    resources.callback(torch.set_float32_matmul_precision, entry_precision)
    writer = LeRobotDatasetWriter()
    episode_rows: list[dict[str, Any]] = []
    try:
        torch.set_float32_matmul_precision("highest")
        model = OpenPIProcessPredictor(
            checkpoint_dir,
            config_name=config_name,
            output_action_dim=action_dim,
            norm_stats_path=norm_stats_path,
            num_steps=num_steps,
            device=device,
        )
        resources.callback(model.close)
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
            correction=SimpleNamespace(
                enabled=True,
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
        resources.callback(env.close)
        writer.create(
            repo_id=str(output),
            robot_type="frankapanda",
            fps=25,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": [f"joint_{i}" for i in range(action_dim)],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": [f"joint_{i}" for i in range(action_dim)],
                },
                "observation.images.cam_high": {
                    "dtype": "image",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                },
                "done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
                "is_success": {"dtype": "bool", "shape": (1,), "names": ["is_success"]},
                "intervene_flag": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["intervene_flag"],
                },
            },
            image_writer_threads=2,
            image_writer_processes=0,
        )
        resources.callback(writer.finalize)
        for episode_index in range(num_episodes):
            obs, _ = env.reset(seed=seed + episode_index)
            queue: list[torch.Tensor] = []
            frames: list[dict[str, Any]] = []
            initial_pose = _sim_rigid_object_pose(env, "cube")
            initial = initial_pose[0].tolist() if initial_pose is not None else None
            final = initial
            success = False
            intervention_count = 0
            for step_index in range(max_steps):
                if not queue:
                    policy_obs = dict(obs)
                    policy_obs.setdefault(
                        "wrist_images", policy_obs.get("extra_view_images")
                    )
                    actions, _ = model.predict_action_batch(policy_obs)
                    queue = [
                        torch.as_tensor(a[..., :action_dim]).detach().cpu()
                        for a in actions[0, :action_chunk]
                    ]
                policy_action = queue.pop(0)
                obs_next, _, terminated, truncated, info = env.step(policy_action)
                intervene = bool(_bool_at(info.get("intervene_flag", False)))
                applied = info.get("intervene_action", policy_action)
                applied = (
                    torch.as_tensor(applied).reshape(-1, action_dim)[0].detach().cpu()
                )
                intervention_count += int(intervene)
                success = success or _episode_success(info)
                pose = _sim_rigid_object_pose(env, "cube")
                if pose is not None:
                    final = pose[0].tolist()
                done = _bool_at(terminated) or _bool_at(truncated) or success
                frames.append(
                    _frame(
                        obs,
                        applied,
                        task=prompt,
                        intervene=intervene,
                        done=done,
                        success=success,
                    )
                )
                obs = obs_next
                if done:
                    break
            if frames:
                writer.add_episode(frames)
            distance = None
            if initial is not None and final is not None:
                distance = float(
                    np.linalg.norm(np.asarray(final) - np.asarray(initial))
                )
            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "length": len(frames),
                    "success": success,
                    "intervention_steps": intervention_count,
                    "cube_initial": initial,
                    "cube_final": final,
                    "cube_displacement": distance,
                }
            )
    finally:
        resources.close()

    meta = output / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    sidecar = meta / "embodichain_episodes.jsonl"
    with sidecar.open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(
                json.dumps(
                    {
                        "schema_version": 3,
                        "episode_index": row["episode_index"],
                        "length": row["length"],
                        "completed": row["length"] < max_steps or row["success"],
                        "success": row["success"],
                        "terminated": row["success"],
                        "truncated": not row["success"] and row["length"] >= max_steps,
                        "terminal_reason": "success" if row["success"] else "timeout",
                        "instruction": prompt,
                        "data_type": "dagger_correction",
                        "intervention_steps": row["intervention_steps"],
                    }
                )
                + "\n"
            )
    audit = audit_dataset(
        output,
        expected_action_dim=action_dim,
        expected_state_dim=action_dim,
        expected_episodes=num_episodes,
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (output / "dynamic_replay.json").write_text(
        json.dumps(
            {
                "rows": episode_rows,
                "episodes": num_episodes,
                "successes": sum(bool(r["success"]) for r in episode_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stats_dir = output
    stats_env = os.environ.copy()
    stats_env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("calculate_norm_stats.py")),
            "--config-name",
            config_name,
            "--repo-id",
            str(output),
            "--output-action-dim",
            str(action_dim),
            "--output-dir",
            str(stats_dir),
            "--num-workers",
            "0",
        ],
        check=True,
        env=stats_env,
    )
    return {
        "dataset_root": str(output),
        "audit": audit,
        "episodes": episode_rows,
        "inference_process_mode": "spawn",
        "actor_process_pid": model.process_pid,
        "model_process_cleanup": model.metadata,
        "sim_matmul_precision": "highest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--norm-stats-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--action-dim", type=int, default=9)
    parser.add_argument(
        "--prompt", default="Pick up the cube and place it on the marked target."
    )
    parser.add_argument("--num-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--action-chunk", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4102)
    parser.add_argument("--correction-max-steps", type=int, default=250)
    parser.add_argument("--correction-threshold", type=float, default=0.05)
    parser.add_argument("--correction-force-steps", type=int, default=10)
    parser.add_argument("--config-name", default="pi05_embodichain_joint")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
