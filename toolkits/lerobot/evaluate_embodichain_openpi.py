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

"""Measure RLinf OpenPI action error on an EmbodiChain LeRobot split.

The evaluator deliberately uses RLinf's ``get_model`` and
``predict_action_batch`` path. That is the same wrapper used by closed-loop
rollouts and therefore handles RLinf ``full_weights.pt`` checkpoints and the
bare ``model.safetensors`` produced by ``sft_to_openpi`` consistently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _resolve_norm_stats_directory(path: str | Path) -> Path:
    """Return the directory accepted by ``openpi.shared.normalize.load``."""
    stats_path = Path(path).expanduser().resolve()
    if stats_path.is_file():
        if stats_path.name != "norm_stats.json":
            raise ValueError(
                "Normalization statistics must be a norm_stats.json file or "
                f"a directory containing one, got {stats_path}."
            )
        return stats_path.parent
    if stats_path.is_dir():
        candidate = stats_path / "norm_stats.json"
        if candidate.is_file():
            return stats_path
        raise FileNotFoundError(f"Normalization statistics not found at {candidate}.")
    raise FileNotFoundError(
        f"Normalization statistics path does not exist: {stats_path}."
    )


def _image_key(sample: dict[str, Any]) -> str:
    """Return the deterministic primary camera key from a LeRobot sample."""
    keys = [key for key in sample if key.startswith("observation.images.")]
    if not keys:
        raise KeyError("LeRobot sample has no observation.images.* field.")
    return sorted(keys)[0]


def _image_tensor(value: Any) -> torch.Tensor:
    """Convert CHW/HWC image data to a batched uint8 HWC tensor."""
    image = np.asarray(value)
    image = np.squeeze(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one RGB image, got shape {image.shape}.")
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB/RGBA image, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)


def _sample_to_env_obs(
    sample: dict[str, Any],
    prompt: str,
    *,
    include_phase_input: bool = False,
) -> dict[str, Any]:
    """Map one LeRobot row to RLinf ``predict_action_batch`` inputs."""
    state = torch.as_tensor(
        np.asarray(sample["observation.state"]), dtype=torch.float32
    )
    if state.ndim == 1:
        state = state.unsqueeze(0)
    observation = {
        "main_images": _image_tensor(sample[_image_key(sample)]),
        "wrist_images": None,
        "states": state,
        "task_descriptions": [prompt],
    }
    if include_phase_input:
        if "annotation.episode_step" not in sample:
            raise KeyError(
                "Phase-conditioned EmbodiChain evaluation requires the "
                "annotation.episode_step feature."
            )
        observation["episode_steps"] = torch.as_tensor(
            np.asarray(sample["annotation.episode_step"]), dtype=torch.int32
        ).reshape(1)
    return observation


def _samples_to_env_obs(
    samples: list[dict[str, Any]],
    prompt: str,
    *,
    include_phase_input: bool = False,
) -> dict[str, Any]:
    """Stack LeRobot samples for one batched RLinf inference call."""
    if not samples:
        raise ValueError("At least one sample is required for batched inference.")
    observations = [
        _sample_to_env_obs(
            sample,
            prompt,
            include_phase_input=include_phase_input,
        )
        for sample in samples
    ]
    result = {
        "main_images": torch.cat(
            [observation["main_images"] for observation in observations], dim=0
        ),
        "wrist_images": None,
        "states": torch.cat(
            [observation["states"] for observation in observations], dim=0
        ),
        "task_descriptions": [prompt] * len(observations),
    }
    if include_phase_input:
        result["episode_steps"] = torch.cat(
            [observation["episode_steps"] for observation in observations], dim=0
        )
    return result


def _select_episode_frames(
    starts: list[int],
    ends: list[int],
    episode_ids: list[int],
    *,
    frames_per_episode: int | None = None,
    max_samples: int | None = None,
) -> list[tuple[int, int]]:
    """Select frame indices, optionally balanced across episode trajectories.

    ``frames_per_episode`` uses integer linspace over each episode's inclusive
    frame range, so a validation sample covers the beginning, middle, and end
    of every selected episode. The returned tuples retain episode identity and
    are suitable for grouping metrics after batched model inference.
    """
    if frames_per_episode is not None and frames_per_episode <= 0:
        raise ValueError("frames_per_episode must be positive when provided.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    if frames_per_episode is not None and max_samples is not None:
        raise ValueError("Pass at most one of frames_per_episode and max_samples.")

    selected: list[tuple[int, int]] = []
    for episode_id in episode_ids:
        start, end = int(starts[episode_id]), int(ends[episode_id])
        if end <= start:
            continue
        if frames_per_episode is None:
            frame_indices = range(start, end)
        else:
            count = min(frames_per_episode, end - start)
            frame_indices = np.linspace(
                start, end - 1, num=count, dtype=np.int64
            ).tolist()
        selected.extend((episode_id, int(index)) for index in frame_indices)

    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def _joint_delta_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Convert absolute joint targets to the state-relative action space."""
    actions = np.asarray(actions, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if actions.ndim < 2 or state.ndim != 1:
        raise ValueError(
            f"Expected actions [H,D] and state [D], got {actions.shape} and {state.shape}."
        )
    if actions.shape[-1] != state.shape[-1]:
        raise ValueError(
            "Action/state dimensions must match for joint deltas: "
            f"{actions.shape[-1]} != {state.shape[-1]}."
        )
    return actions - state[None, ...]


def _checkpoint_source(checkpoint: Path) -> tuple[Path, str]:
    """Find a supported checkpoint file and its relative staging path."""
    if checkpoint.is_file():
        if checkpoint.suffix == ".safetensors":
            return checkpoint, "model.safetensors"
        if checkpoint.suffix == ".pt":
            return checkpoint, "actor/model_state_dict/full_weights.pt"
        raise ValueError(f"Unsupported checkpoint file: {checkpoint}")
    for relative in (
        "model.safetensors",
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
        "full_weights.pt",
    ):
        candidate = checkpoint / relative
        if candidate.is_file():
            return candidate, relative
    raise FileNotFoundError(
        f"No model.safetensors or full_weights.pt found under {checkpoint}."
    )


def _stage_checkpoint(
    checkpoint: str | Path,
    norm_stats_path: str | Path,
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Stage weights and stats in the layout expected by RLinf ``get_model``.

    ``get_model`` resolves stats by the config asset id
    ``RLinf/embodichain_joint/norm_stats.json``. The conversion command also
    emits a convenient top-level JSON file, so we create a temporary symlink
    tree rather than mutating either the source checkpoint or its stats.
    """
    source_checkpoint = Path(checkpoint).expanduser().resolve()
    source_weights, relative_weights = _checkpoint_source(source_checkpoint)
    stats_dir = _resolve_norm_stats_directory(norm_stats_path)
    stats_file = stats_dir / "norm_stats.json"
    staging = tempfile.TemporaryDirectory(prefix="rlinf-embodichain-eval-")
    root = Path(staging.name)
    weight_target = root / relative_weights
    weight_target.parent.mkdir(parents=True, exist_ok=True)
    weight_target.symlink_to(source_weights)
    asset_target = root / "RLinf" / "embodichain_joint"
    asset_target.mkdir(parents=True, exist_ok=True)
    stats_payload = json.loads(stats_file.read_text(encoding="utf-8"))
    stats_inner = stats_payload.get("norm_stats", stats_payload)
    if "actions" not in stats_inner and "action" in stats_inner:
        stats_inner["actions"] = stats_inner["action"]
    if "state" not in stats_inner and "observation.state" in stats_inner:
        stats_inner["state"] = stats_inner["observation.state"]
    stats_payload = {"norm_stats": stats_inner}
    (asset_target / "norm_stats.json").write_text(
        json.dumps(stats_payload), encoding="utf-8"
    )
    return root, staging


def _build_model(
    checkpoint_dir: Path,
    *,
    config_name: str,
    output_action_dim: int,
    norm_stats_dir: Path,
    num_steps: int,
    device: str | None,
    include_phase_input: bool = False,
    phase_scale: float = 600.0,
    delta_action_mask: list[bool] | None = None,
    eval_sft_image_crop: bool = False,
):
    """Build the RLinf model and install transforms for this action dimension."""
    from openpi.shared import normalize

    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.transforms.pipeline import (
        build_openpi_transforms,
    )

    official_model = get_openpi_config(config_name).model
    openpi_data = {
        "norm_stats_path": str(norm_stats_dir / "norm_stats.json"),
        "output_action_dim": output_action_dim,
        "include_phase_input": include_phase_input,
        "phase_scale": phase_scale,
    }
    if delta_action_mask is not None:
        openpi_data["delta_action_mask"] = delta_action_mask
    model_cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(checkpoint_dir),
            "precision": None,
            "is_lora": False,
            "lora_rank": 32,
            "action_dim": output_action_dim,
            # Keep the full ten-step model horizon for H1/H5/H10 metrics. The
            # closed-loop runner may still execute only its first five actions.
            "num_action_chunks": 10,
            "num_steps": num_steps,
            "openpi": {
                "task": "eval",
                "config_name": config_name,
                "action_horizon": 10,
                "action_chunk": 10,
                "action_env_dim": output_action_dim,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "num_images_in_input": 1,
                "num_steps": num_steps,
                "discrete_state_input": official_model.discrete_state_input,
                "eval_sft_image_crop": eval_sft_image_crop,
            },
            "openpi_data": openpi_data,
        }
    )
    model = get_model(model_cfg)
    data_kwargs = {
        "output_action_dim": output_action_dim,
        "norm_stats_path": str(norm_stats_dir / "norm_stats.json"),
        "include_phase_input": include_phase_input,
        "phase_scale": phase_scale,
    }
    if delta_action_mask is not None:
        data_kwargs["delta_action_mask"] = delta_action_mask
    input_transforms, output_transforms = build_openpi_transforms(
        str(checkpoint_dir),
        config_name,
        data_kwargs=data_kwargs,
    )
    loaded_stats = normalize.load(norm_stats_dir)
    # Older LeRobot metadata calls these fields ``action`` and
    # ``observation.state``. RLinf/OpenPI transforms use ``actions`` and
    # ``state``; accept both spellings while keeping one mapping for the model.
    norm_stats = dict(loaded_stats)
    if "actions" not in norm_stats and "action" in norm_stats:
        norm_stats["actions"] = norm_stats["action"]
    if "state" not in norm_stats and "observation.state" in norm_stats:
        norm_stats["state"] = norm_stats["observation.state"]
    if "actions" not in norm_stats:
        raise KeyError("Normalization statistics must contain an 'actions' entry.")
    model.setup_transforms(input_transforms, output_transforms)
    model.to(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model, norm_stats, True


def _normalise_actions(
    actions: np.ndarray,
    stats: Any,
    *,
    use_quantiles: bool,
) -> np.ndarray:
    """Apply the same action normalization used by the OpenPI data wrapper."""
    values = np.asarray(actions, dtype=np.float32)
    action_stats = stats["actions"]
    if use_quantiles:
        if action_stats.q01 is None or action_stats.q99 is None:
            raise ValueError("Quantile action normalization requires q01 and q99.")
        q01 = np.asarray(action_stats.q01)[..., : values.shape[-1]]
        q99 = np.asarray(action_stats.q99)[..., : values.shape[-1]]
        if q01.shape[-1] < values.shape[-1] or q99.shape[-1] < values.shape[-1]:
            raise ValueError(
                "Normalization statistics action dimension is smaller than the "
                f"requested output dimension: stats={q01.shape[-1]}, "
                f"output={values.shape[-1]}."
            )
        return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = np.asarray(action_stats.mean)[..., : values.shape[-1]]
    std = np.asarray(action_stats.std)[..., : values.shape[-1]]
    if mean.shape[-1] < values.shape[-1] or std.shape[-1] < values.shape[-1]:
        raise ValueError(
            "Normalization statistics action dimension is smaller than the "
            f"requested output dimension: stats={mean.shape[-1]}, "
            f"output={values.shape[-1]}."
        )
    return (values - mean) / (std + 1e-6)


def evaluate_action_error(
    dataset_root: str | Path,
    *,
    config_name: str,
    checkpoint_dir: str | Path,
    prompt: str,
    output_action_dim: int,
    num_steps: int = 5,
    max_samples: int | None = None,
    max_episodes: int | None = None,
    frames_per_episode: int | None = None,
    batch_size: int = 1,
    noise_seed: int | None = 0,
    device: str | None = None,
    norm_stats_path: str | Path | None = None,
    include_phase_input: bool = False,
    phase_scale: float = 600.0,
    eval_sft_image_crop: bool = False,
    delta_action_mask: list[bool] | None = None,
) -> dict[str, Any]:
    """Evaluate normalized H1/H5/H10 action error without episode padding.

    Use ``max_episodes=8, frames_per_episode=8`` for an external validation
    checkpoint: this evaluates eight uniformly spaced frames from each of the
    eight episodes, rather than interpreting eight as a global frame budget.
    ``noise_seed`` fixes the flow-sampling noise so checkpoints can be compared
    on exactly the same observations and noise tensors.
    """
    if max_samples is not None and frames_per_episode is not None:
        raise ValueError("Pass at most one of frames_per_episode and max_samples.")
    if output_action_dim <= 0:
        raise ValueError("output_action_dim must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if norm_stats_path is None:
        raise ValueError("norm_stats_path is required for reproducible evaluation.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from rlinf.data.storage.lerobot import episode_boundaries

    dataset_root = Path(dataset_root).expanduser().resolve()
    norm_stats_dir = _resolve_norm_stats_directory(norm_stats_path)
    checkpoint_root, staging = _stage_checkpoint(checkpoint_dir, norm_stats_dir)
    model = None
    try:
        model, norm_stats, use_quantiles = _build_model(
            checkpoint_root,
            config_name=config_name,
            output_action_dim=output_action_dim,
            norm_stats_dir=norm_stats_dir,
            num_steps=num_steps,
            device=device,
            include_phase_input=include_phase_input,
            phase_scale=phase_scale,
            delta_action_mask=delta_action_mask,
            eval_sft_image_crop=eval_sft_image_crop,
        )
        dataset = LeRobotDataset(str(dataset_root))
        starts, ends = episode_boundaries(dataset)
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError("max_episodes must be positive when provided.")
            episode_ids = list(range(min(max_episodes, len(starts))))
        else:
            episode_ids = list(range(len(starts)))
        selected_frames = _select_episode_frames(
            starts,
            ends,
            episode_ids,
            frames_per_episode=frames_per_episode,
            max_samples=max_samples,
        )

        horizon_values = {1: [], 5: [], 10: []}
        episode_values: dict[int, dict[int, list[float]]] = {}
        for episode_id, _ in selected_frames:
            episode_values.setdefault(episode_id, {1: [], 5: [], 10: []})
        finite = True
        if noise_seed is not None:
            noise_generator = torch.Generator(device="cpu")
            noise_generator.manual_seed(noise_seed)
            common_noise = torch.randn(
                len(selected_frames),
                10,
                int(getattr(model, "action_dim", 32)),
                generator=noise_generator,
                dtype=torch.float32,
            )
        else:
            common_noise = None

        for batch_start in range(0, len(selected_frames), batch_size):
            batch_records = selected_frames[batch_start : batch_start + batch_size]
            samples = [dataset[index] for _, index in batch_records]
            model_input = _samples_to_env_obs(
                samples,
                prompt,
                include_phase_input=include_phase_input,
            )
            noise = None
            if common_noise is not None:
                noise = common_noise[batch_start : batch_start + len(batch_records)].to(
                    model.device
                )
            predictions, _ = model.predict_action_batch(
                model_input, mode="eval", noise=noise
            )
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            predictions = np.asarray(predictions, dtype=np.float32)
            if predictions.ndim == 2:
                predictions = predictions[None, ...]
            predictions = predictions[:, :, :output_action_dim]

            for batch_index, (episode_id, index) in enumerate(batch_records):
                prediction = predictions[batch_index]
                sample_state = np.asarray(
                    samples[batch_index]["observation.state"], dtype=np.float32
                )[:output_action_dim]
                for horizon in horizon_values:
                    stop = min(int(ends[episode_id]), index + horizon)
                    target = np.stack(
                        [
                            np.asarray(
                                dataset[target_index]["action"], dtype=np.float32
                            )[:output_action_dim]
                            for target_index in range(index, stop)
                        ],
                        axis=0,
                    )
                    prediction_horizon = prediction[: target.shape[0]]
                    if prediction_horizon.shape != target.shape:
                        raise ValueError(
                            f"Prediction/target shape mismatch at frame {index}: "
                            f"{prediction_horizon.shape} != {target.shape}."
                        )
                    prediction_delta = _joint_delta_actions(
                        prediction_horizon, sample_state
                    )
                    target_delta = _joint_delta_actions(target, sample_state)
                    error = _normalise_actions(
                        prediction_delta, norm_stats, use_quantiles=use_quantiles
                    ) - _normalise_actions(
                        target_delta, norm_stats, use_quantiles=use_quantiles
                    )
                    if not np.isfinite(error).all():
                        finite = False
                        raise ValueError(
                            f"Non-finite normalized error at frame {index}."
                        )
                    value = float(np.abs(error).mean())
                    horizon_values[horizon].append(value)
                    episode_values[episode_id][horizon].append(value)

        if not selected_frames:
            raise ValueError(f"No frames found in {dataset_root}.")
        normalized_mae = {
            f"h{horizon}": float(np.mean(values))
            for horizon, values in horizon_values.items()
        }
        per_episode = [
            {
                "episode_index": episode_id,
                "frames": len(values[1]),
                "source_frames": int(ends[episode_id]) - int(starts[episode_id]),
                "normalized_mae": {
                    f"h{horizon}": float(np.mean(values[horizon]))
                    for horizon in values
                    if values[horizon]
                },
            }
            for episode_id, values in episode_values.items()
        ]
        episode_summary = {
            "count": len(per_episode),
            **{
                f"mean_h{horizon}": float(
                    np.mean(
                        [
                            row["normalized_mae"][f"h{horizon}"]
                            for row in per_episode
                            if f"h{horizon}" in row["normalized_mae"]
                        ]
                    )
                )
                for horizon in horizon_values
            },
        }
        return {
            "dataset_root": str(dataset_root),
            "checkpoint": str(Path(checkpoint_dir).expanduser().resolve()),
            "config_name": config_name,
            "prompt": prompt,
            "norm_stats": str(norm_stats_dir / "norm_stats.json"),
            "norm_stats_sha256": hashlib.sha256(
                (norm_stats_dir / "norm_stats.json").read_bytes()
            ).hexdigest(),
            "action_horizon": model.action_horizon,
            "num_steps": num_steps,
            "delta_action_mask": delta_action_mask,
            "samples": len(selected_frames),
            "episodes": len(per_episode),
            "episode_indices": [row["episode_index"] for row in per_episode],
            "frames_per_episode": frames_per_episode,
            "batch_size": batch_size,
            "noise_seed": noise_seed,
            "action_dim": output_action_dim,
            "include_phase_input": include_phase_input,
            "phase_scale": phase_scale,
            "eval_sft_image_crop": eval_sft_image_crop,
            "finite": finite,
            "normalized_action_mae": normalized_mae,
            # Keep action_mae as the smoke-gate compatibility alias for H1.
            "action_mae": normalized_mae["h1"],
            "episode_summary": episode_summary,
            "per_episode": per_episode,
        }
    finally:
        del model
        staging.cleanup()


def main() -> None:
    """Parse arguments and print a JSON action-error report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--config-name", default="pi05_embodichain_joint_state_v2")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-action-dim", type=int, required=True)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        help="Uniformly spaced frames to sample from each selected episode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of selected frames per model inference call.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=0,
        help="Common flow-sampling noise seed; use -1 for fresh noise.",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--norm-stats",
        "--norm-stats-path",
        dest="norm_stats",
        type=Path,
        required=True,
        help="norm_stats.json or its containing directory.",
    )
    parser.add_argument(
        "--include-phase-input",
        action="store_true",
        help="Use annotation.episode_step as the phase-conditioned state input.",
    )
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from toolkits.lerobot.calculate_norm_stats import _parse_delta_action_mask

    report = evaluate_action_error(
        args.dataset_root,
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        output_action_dim=args.output_action_dim,
        num_steps=args.num_steps,
        max_samples=args.max_samples,
        max_episodes=args.max_episodes,
        frames_per_episode=args.frames_per_episode,
        batch_size=args.batch_size,
        noise_seed=None if args.noise_seed < 0 else args.noise_seed,
        device=args.device,
        norm_stats_path=args.norm_stats,
        include_phase_input=args.include_phase_input,
        phase_scale=args.phase_scale,
        eval_sft_image_crop=args.eval_sft_image_crop,
        delta_action_mask=_parse_delta_action_mask(args.delta_action_mask),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
