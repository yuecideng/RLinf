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

"""Measure decoded Stack Cube commands against fixed ODS expert contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def stack_cube_action_errors(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    translation_scale: float = 0.5,
    rotation_scale: float = 0.5,
) -> dict[str, Any]:
    """Compare valid decoded relative-EEF7 commands in physical target units.

    Position and rotation errors compare command targets at a common state;
    they do not measure achieved robot poses. Each valid control has equal weight.
    Zero gripper command closes, matching the native binary action term.
    """
    if (
        predicted.shape != targets.shape
        or predicted.ndim != 3
        or predicted.shape[-1] != 7
    ):
        raise ValueError("Expected matching predicted/target actions[B,H,7].")
    if mask.dtype != torch.bool or mask.shape != predicted.shape[:2] or not mask.any():
        raise ValueError("Expected a nonempty boolean action mask[B,H].")
    pred, truth = predicted[mask].double(), targets[mask].double()
    if not torch.isfinite(pred).all() or not torch.isfinite(truth).all():
        raise ValueError(
            "Expected finite predicted and target actions at valid controls."
        )
    displacement = (pred[:, :3] - truth[:, :3]).norm(dim=-1) * translation_scale * 1000

    def quaternion(rotvec: torch.Tensor) -> torch.Tensor:
        half_angle = rotvec.norm(dim=-1, keepdim=True) / 2
        xyz = rotvec * 0.5 * torch.sinc(half_angle / torch.pi)
        return torch.cat((xyz, half_angle.cos()), dim=-1)

    pred_q = quaternion(pred[:, 3:6] * rotation_scale)
    true_q = quaternion(truth[:, 3:6] * rotation_scale)
    angle = 2 * (pred_q * true_q).sum(-1).abs().clamp(0, 1).acos() * 180 / torch.pi
    return {
        "valid_controls": int(mask.sum()),
        "translation_target_error_mm_mean": displacement.mean().item(),
        "translation_target_error_mm_p95": torch.quantile(displacement, 0.95).item(),
        "rotation_target_error_deg_mean": angle.mean().item(),
        "rotation_target_error_deg_p95": torch.quantile(angle, 0.95).item(),
        "binary_gripper_accuracy": ((pred[:, -1] > 0) == (truth[:, -1] > 0))
        .double()
        .mean()
        .item(),
        "raw_action_mse_per_dim": (pred - truth).square().mean(0).tolist(),
        "raw_action_mse_real7": (pred - truth).square().mean().item(),
    }


def main() -> None:
    """Evaluate one checkpoint on a frozen CPU packet of expert contexts."""
    from omegaconf import OmegaConf

    from rlinf.data.datasets.openpi.ods import sha256_file
    from rlinf.models import get_model
    from rlinf.utils.logging import get_logger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Resolved Ray evaluation YAML."
    )
    parser.add_argument(
        "--batches",
        required=True,
        type=Path,
        help="Frozen projected ODS CPU packet (.pt).",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    cfg = OmegaConf.load(args.config)
    packet = torch.load(args.batches, map_location="cpu", weights_only=False)
    model = get_model(cfg.rollout.model).to(args.device).eval()
    count = len(packet["prompts"])
    predicted = []
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        obs = {
            "main_images": packet["front"][start:stop],
            "wrist_images": packet["wrist"][start:stop],
            "states": packet["state"][start:stop],
            "task_descriptions": packet["prompts"][start:stop],
        }
        noise = torch.randn(
            stop - start,
            model.action_horizon,
            model.action_dim,
            generator=torch.Generator().manual_seed(args.seed + start),
        ).to(args.device)
        with torch.inference_mode():
            actions, _ = model.predict_action_batch(obs, noise=noise)
        predicted.append(actions.cpu())
    predicted = torch.cat(predicted)
    horizon = predicted.shape[1]
    result = stack_cube_action_errors(
        predicted, packet["actions"][:, :horizon], packet["action_mask"][:, :horizon]
    )
    result.update(
        checkpoint=str(cfg.rollout.model.model_path),
        config_name=str(cfg.rollout.model.openpi.config_name),
        samples=count,
        executed_chunk=horizon,
        seed=args.seed,
        batch_size=args.batch_size,
        normalization_sha256=sha256_file(cfg.rollout.model.openpi_data.norm_stats_path),
        batches_sha256=sha256_file(args.batches),
        metric_scope="Decoded commands on fixed expert states; independent closed-loop evaluation is required.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"), actions=predicted.numpy())
    get_logger().info("Stack Cube decoded action metrics: %s", json.dumps(result))


if __name__ == "__main__":
    main()
