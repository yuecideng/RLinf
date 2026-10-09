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

"""Audit an EmbodiChain LeRobot dataset before using it for SFT.

The audit intentionally operates on metadata and Parquet columns instead of
loading the full LeRobot Python dataset. This keeps it useful in data
generation environments where only PyArrow and the generated dataset are
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parquet_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("**/*.parquet"))


def audit_dataset(
    dataset_root: str | Path,
    *,
    expected_action_dim: int,
    expected_state_dim: int,
    expected_episodes: int | None = None,
    min_success_rate: float = 0.0,
) -> dict[str, Any]:
    """Return a JSON-serializable audit report and raise on hard failures."""
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

    info = _read_json(info_path)
    parquet_files = _parquet_files(root)
    if not parquet_files:
        raise ValueError(f"No data/*.parquet files found under {root}.")
    episodes = _read_jsonl(root / "meta" / "embodichain_episodes.jsonl")
    total_episodes = int(info.get("total_episodes", len(episodes)))
    total_frames = int(info.get("total_frames", 0))
    if total_frames <= 0:
        raise ValueError(f"Dataset must contain at least one frame: {root}.")
    if expected_episodes is not None and total_episodes != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} episodes, found {total_episodes} in {root}."
        )
    if len(episodes) not in (0, total_episodes):
        raise ValueError(
            "EmbodiChain episode sidecar length does not match info.json: "
            f"{len(episodes)} != {total_episodes}."
        )

    successes = sum(bool(row.get("success", False)) for row in episodes)
    completed = sum(bool(row.get("completed", False)) for row in episodes)
    truncated = sum(bool(row.get("truncated", False)) for row in episodes)
    success_rate = successes / max(1, len(episodes))
    if success_rate < min_success_rate:
        raise ValueError(
            f"Success rate {success_rate:.3f} is below {min_success_rate:.3f}."
        )

    finite = True
    action_rows = 0
    state_rows = 0
    action_min = np.full(expected_action_dim, np.inf, dtype=np.float64)
    action_max = np.full(expected_action_dim, -np.inf, dtype=np.float64)
    state_min = np.full(expected_state_dim, np.inf, dtype=np.float64)
    state_max = np.full(expected_state_dim, -np.inf, dtype=np.float64)

    columns = ["action", "observation.state"]
    for parquet_path in parquet_files:
        table = parquet.read_table(parquet_path, columns=columns)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != expected_action_dim:
            raise ValueError(
                f"{parquet_path}: action shape {actions.shape}, expected (*, {expected_action_dim})."
            )
        if states.ndim != 2 or states.shape[1] != expected_state_dim:
            raise ValueError(
                f"{parquet_path}: state shape {states.shape}, expected (*, {expected_state_dim})."
            )
        finite = (
            finite
            and bool(np.isfinite(actions).all())
            and bool(np.isfinite(states).all())
        )
        action_rows += actions.shape[0]
        state_rows += states.shape[0]
        action_min = np.minimum(action_min, actions.min(axis=0))
        action_max = np.maximum(action_max, actions.max(axis=0))
        state_min = np.minimum(state_min, states.min(axis=0))
        state_max = np.maximum(state_max, states.max(axis=0))

    if action_rows != total_frames or state_rows != total_frames:
        raise ValueError(
            f"Frame count mismatch: info={total_frames}, actions={action_rows}, states={state_rows}."
        )
    if not finite:
        raise ValueError("Dataset contains non-finite state or action values.")

    spatial_bins: list[int] = []
    spatial_profiles: list[str] = []
    for episode in episodes:
        profile = episode.get("spatial_profile")
        if isinstance(profile, str):
            spatial_profiles.append(profile)
        for segment in episode.get("segments", []):
            metadata = segment.get("metadata", segment)
            value = (
                metadata.get("spatial_bin_id") if isinstance(metadata, dict) else None
            )
            if isinstance(value, int):
                spatial_bins.append(value)
            elif isinstance(value, (list, tuple)):
                spatial_bins.extend(
                    int(item) for item in value if isinstance(item, (int, float))
                )

    report = {
        "dataset_root": str(root),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "completed_episodes": completed,
        "successful_episodes": successes,
        "truncated_episodes": truncated,
        "success_rate": success_rate,
        "schema_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "feature_names": sorted((info.get("features") or {}).keys()),
        "action_dim": expected_action_dim,
        "state_dim": expected_state_dim,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "state_min": state_min.tolist(),
        "state_max": state_max.tolist(),
        "segments": sum(len(row.get("segments", [])) for row in episodes),
        "spatial_bin_ids": sorted(set(spatial_bins)),
        "spatial_profiles": sorted(set(spatial_profiles)),
        "finite": finite,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_dataset(
        args.dataset_root,
        expected_action_dim=args.action_dim,
        expected_state_dim=args.state_dim,
        expected_episodes=args.expected_episodes,
        min_success_rate=args.min_success_rate,
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
