#!/usr/bin/env python3

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

"""Convert an EmbodiChain EEF LeRobot dataset from RPY to Rot6D.

The input action and state layout must be ``[x, y, z, roll, pitch, yaw,
gripper]``. The output layout is ``[x, y, z, rot6d, gripper]`` where Rot6D
contains the first two columns of the rotation matrix.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from rlinf.utils.rot6d import matrix_to_rot6d

_EEF_KEYS = ("observation.state", "action", "observation.eef_pose")
_ROT6D_NAMES = [
    "x",
    "y",
    "z",
    "r11",
    "r21",
    "r31",
    "r12",
    "r22",
    "r32",
    "gripper",
]


def _convert_rows(values: list[list[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != 7:
        raise ValueError(f"Expected an EEF array with shape [N, 7], got {array.shape}.")
    rotation = Rotation.from_euler("xyz", array[:, 3:6]).as_matrix()
    return np.concatenate(
        (array[:, :3], matrix_to_rot6d(rotation), array[:, 6:7]), axis=-1
    ).astype(np.float32)


def _replace_column(table: pa.Table, key: str) -> pa.Table:
    if key not in table.column_names:
        return table
    converted = _convert_rows(table[key].to_pylist())
    column = pa.array(converted.tolist(), type=pa.list_(pa.float32(), 10))
    return table.set_column(table.column_names.index(key), key, column)


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def convert_dataset(input_dir: Path, output_dir: Path) -> None:
    """Convert one dataset directory and write a standard LeRobot copy."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    shutil.copytree(input_dir, output_dir)

    converted_values: dict[str, list[np.ndarray]] = {key: [] for key in _EEF_KEYS}
    for data_path in sorted((output_dir / "data").rglob("*.parquet")):
        table = pq.read_table(data_path)
        for key in _EEF_KEYS:
            if key in table.column_names:
                values = _convert_rows(table[key].to_pylist())
                converted_values[key].append(values)
        for key in _EEF_KEYS:
            table = _replace_column(table, key)
        pq.write_table(table, data_path)

    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key in _EEF_KEYS:
        if key in info.get("features", {}):
            info["features"][key]["shape"] = [10]
            info["features"][key]["names"] = _ROT6D_NAMES
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    for filename in ("stats.json", "norm_stats.json"):
        stats_path = output_dir / "meta" / filename
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        target = stats.setdefault("norm_stats", {})
        for key in ("observation.state", "action"):
            chunks = converted_values[key]
            if chunks:
                target[key] = _stats(np.concatenate(chunks, axis=0))
        stats_path.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    convert_dataset(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
