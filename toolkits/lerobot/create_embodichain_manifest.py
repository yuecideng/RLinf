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

"""Create a deterministic manifest for an EmbodiChain LeRobot experiment.

The manifest records the exact dataset roots, split audits, task-config hashes,
and dataset fingerprints used for one experiment.  It does not generate data
or start training.  Keeping this as a small, dependency-light command makes it
safe to run after each data-generation stage and gives later evaluators a
single source of truth for the split and spatial-randomization profile.

Example::

    python toolkits/lerobot/create_embodichain_manifest.py \
        --task pour_water --profile smoke --seed 0 \
        --action-dim 14 --state-dim 14 \
        --split train=/data/pour_water/smoke/train \
        --split val=/data/pour_water/smoke/val \
        --config train=/path/to/task.cobotmagic_smoke_train.yaml \
        --config val=/path/to/task.cobotmagic_smoke_val.yaml \
        --output /data/pour_water/smoke/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from toolkits.lerobot.audit_embodichain_dataset import audit_dataset


def _parse_mapping(value: str, *, option: str) -> tuple[str, Path]:
    """Parse one ``name=path`` command-line mapping."""
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(f"{option} expects NAME=PATH, got {value!r}.")
    return name, Path(raw_path).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(root: Path, *, exclude: set[Path] | None = None) -> str:
    """Hash all dataset files in stable relative-path order."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    digest = hashlib.sha256()
    excluded = {path.resolve() for path in (exclude or set())}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded
    )
    if not files:
        raise ValueError(f"Dataset root contains no files: {root}")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _git_revision(root: Path | None = None) -> str | None:
    """Return the current revision for a checkout when it is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _config_record(path: Path) -> dict[str, str]:
    """Return a stable path/hash record for one task config."""
    if not path.is_file():
        raise FileNotFoundError(f"Task config does not exist: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def create_manifest(
    *,
    task: str,
    profile: str,
    action_dim: int,
    state_dim: int,
    splits: dict[str, str | Path],
    configs: dict[str, str | Path] | None = None,
    seed: int | None = None,
    min_success_rate: float = 0.95,
    embodichain_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_norm_stats: bool = True,
) -> dict[str, Any]:
    """Audit split roots and return a reproducible experiment manifest.

    Args:
        task: Logical EmbodiChain task name, for example ``pour_water``.
        profile: Data-generation profile, for example ``smoke`` or ``ood``.
        action_dim: Environment action dimension for all supplied splits.
        state_dim: Environment state dimension for all supplied splits.
        splits: Mapping from split names to LeRobot dataset roots.
        configs: Optional mapping from split names to EmbodiChain YAML files.
        seed: Seed used by the data generator, if fixed for this run.
        min_success_rate: Minimum expert success rate required by every split.

    Returns:
        A JSON-serializable manifest.  ``audit_dataset`` raises before a
        manifest is returned if any split fails its schema or quality gate.
    """
    if not splits:
        raise ValueError("At least one dataset split is required.")
    if action_dim <= 0 or state_dim <= 0:
        raise ValueError("action_dim and state_dim must be positive.")
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be in [0, 1].")

    output = Path(output_path).expanduser().resolve() if output_path else None
    split_records: dict[str, Any] = {}
    for split_name, split_root in sorted(splits.items()):
        root = Path(split_root).expanduser().resolve()
        norm_stats_candidates = (
            root / "norm_stats.json",
            root / "meta" / "norm_stats.json",
            root.parent / "norm_stats.json",
            root.parent / "meta" / "norm_stats.json",
        )
        norm_stats_path = next(
            (path for path in norm_stats_candidates if path.is_file()), None
        )
        split_records[split_name] = {
            "root": str(root),
            "fingerprint": _dataset_fingerprint(
                root, exclude={output} if output else None
            ),
            "norm_stats": (
                {
                    "path": str(norm_stats_path),
                    "sha256": _sha256_file(norm_stats_path),
                }
                if norm_stats_path is not None
                else None
            ),
            "audit": audit_dataset(
                root,
                expected_action_dim=action_dim,
                expected_state_dim=state_dim,
                min_success_rate=min_success_rate,
            ),
        }

    norm_stats_records = {
        name: record["norm_stats"]
        for name, record in split_records.items()
        if record["norm_stats"] is not None
    }
    if require_norm_stats and not norm_stats_records:
        raise FileNotFoundError(
            "No norm_stats.json found under the supplied splits. Run "
            "calculate_norm_stats.py before creating the manifest."
        )

    config_records = {
        name: _config_record(Path(path).expanduser().resolve())
        for name, path in sorted((configs or {}).items())
    }
    return {
        "schema_version": "embodichain-sft-manifest-v1",
        "task": task,
        "profile": profile,
        "seed": seed,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "min_success_rate": min_success_rate,
        "rlinf_revision": _git_revision(),
        "embodichain_revision": _git_revision(
            Path(embodichain_path).expanduser().resolve()
            if embodichain_path is not None
            else None
        ),
        "norm_stats": norm_stats_records,
        "splits": split_records,
        "configs": config_records,
    }


def main() -> None:
    """Parse arguments, write a manifest, and print it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--state-dim", type=int, required=True)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset split root; repeat for train/val/test splits.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional task config hash; repeat for each generation config.",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument(
        "--allow-missing-norm-stats",
        action="store_true",
        help="Allow a pre-normalization manifest (not suitable for SFT).",
    )
    parser.add_argument("--embodichain-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits = dict(_parse_mapping(value, option="--split") for value in args.split)
    configs = dict(_parse_mapping(value, option="--config") for value in args.config)
    if len(splits) != len(args.split):
        raise ValueError("Duplicate split names are not allowed.")
    if len(configs) != len(args.config):
        raise ValueError("Duplicate config names are not allowed.")

    manifest = create_manifest(
        task=args.task,
        profile=args.profile,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        splits=splits,
        configs=configs,
        seed=args.seed,
        min_success_rate=args.min_success_rate,
        embodichain_path=args.embodichain_path,
        output_path=args.output,
        require_norm_stats=not args.allow_missing_norm_stats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
