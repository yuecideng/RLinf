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

"""Measure online PI0.5 SFT and qualify SDK-generated ODS recipes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rlinf.utils.logging import get_logger

LOGGER = get_logger()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def measure_run(output: Path, global_batch_size: int) -> dict[str, Any]:
    """Measure a warmup-free window containing a complete cache replacement."""
    from embodichain.data_pipeline.configuration import (
        ODSRateMeasurementCfg,
        measure_ods_throughput,
    )

    rows = _read_rows(output / "ods/updates.jsonl")
    if len(rows) < 13:
        raise ValueError(
            "Calibration requires three warmups and ten optimizer updates."
        )
    baseline, end = rows[2], rows[-1]
    start_time, end_time = baseline["monotonic_seconds"], end["monotonic_seconds"]
    refills = {
        row["generation"]: row["started"]
        for row in _read_rows(output / "ods/refills.jsonl")
    }
    complete = [
        row
        for row in _read_rows(output / "ods/generations.jsonl")
        if refills[row["generation"]] >= start_time and row["completed_at"] <= end_time
    ]
    if not complete:
        raise ValueError(
            "Calibration window must enclose a complete post-warmup refill."
        )
    production_seconds = sum(row["active_seconds"] for row in complete)
    active = sum(row["active_seconds"] for row in rows[3:])
    completed = len(rows[3:]) * global_batch_size
    rates = measure_ods_throughput(
        ODSRateMeasurementCfg(
            unit="accepted_episodes",
            count=sum(row["accepted_episodes"] for row in complete),
            start_monotonic_seconds=start_time,
            end_monotonic_seconds=end_time,
            active_seconds=production_seconds,
        ),
        ODSRateMeasurementCfg(
            unit="completed_chunks",
            count=completed,
            start_monotonic_seconds=start_time,
            end_monotonic_seconds=end_time,
            active_seconds=active,
        ),
    )
    first: dict[int, float] = {}
    publications_in_window = 0
    for event in _read_rows(output / "ods/lifecycle.jsonl"):
        if event["kind"] == "published":
            publications_in_window += int(
                start_time <= event["monotonic_ns"] / 1e9 <= end_time
            )
            generation = event["generation"]
            latency = event["monotonic_ns"] / 1e9 - refills[generation]
            first[generation] = min(first.get(generation, float("inf")), latency)
    if len(first) < 2:
        raise ValueError(
            "Calibration requires initial publication and a refill publication."
        )
    # Nearest-rank P95 is conservative for the small number of complete cycles.
    values = sorted(first.values())
    import math

    p95 = values[math.ceil(0.95 * len(values)) - 1]
    audit = json.loads((output / "ods/supply_audit.json").read_text())
    session_file = output / "ods/session.json"
    contracts = json.loads(session_file.read_text()) if session_file.exists() else {}
    if audit["error"] or audit["committed_chunks"] != len(rows) * global_batch_size:
        raise ValueError("Calibration failed or global committed accounting differs.")
    return {
        "norm_stats_sha256": contracts.get("norm_stats_sha256"),
        "schema_sha256": contracts.get("schema_sha256"),
        "rates": asdict(rates),
        "first_publication_p95_seconds": p95,
        "publication_cycles": len(first),
        "completed_updates": len(rows[3:]),
        "completed_chunks": completed,
        "learner_active_seconds": active,
        "producer_active_seconds": production_seconds,
        "active_overlap_lower_bound_seconds": max(
            0.0, production_seconds + active - (end_time - start_time)
        ),
        "complete_generations": [row["generation"] for row in complete],
        "production_wall_rate_scope": "selected complete generations inside learner window",
        "published_episodes_per_wall_second": publications_in_window
        / (end_time - start_time),
        "window_start_monotonic": start_time,
        "window_end_monotonic": end_time,
        "bytes_per_episode": audit["bytes_per_episode"],
        "owner_peak_rss_bytes": audit["owner_peak_rss_bytes"],
    }


def build_recipe(
    measurement: dict[str, Any], cfg: Any, memory_bytes: int
) -> dict[str, Any]:
    """Call the SDK planner with a PI0.5-specific provisional reuse profile."""
    from embodichain.data_pipeline.configuration import (
        ODSReuseProfileCfg,
        ODSRuntimeCfg,
        configure_ods,
    )

    profile = ODSReuseProfileCfg(
        profile_id="pi05-stack-cube-h1-h10-c5-provisional-v1",
        scope="PI0.5 Stack Cube; two cameras; H1/H10/C5; episode-uniform chunks; <=450 physical steps",
        qualified=False,
        min_chunks_per_episode=32,
        max_chunks_per_episode=128,
        hard_max_chunks_per_episode=128,
    )
    runtime = ODSRuntimeCfg(
        batch_size=int(cfg.actor.global_batch_size),
        producer_envs=int(cfg.data.ods.producer_envs),
        first_publication_p95_seconds=measurement["first_publication_p95_seconds"],
        cache_memory_bytes=memory_bytes,
        bytes_per_episode=measurement["bytes_per_episode"],
        min_cache_episodes=8,
        measurement_scope="concurrent GPU0 producer and GPU4/5 two-rank FSDP run; completed generation and optimizer intervals",
    )
    rates = measurement["rates"]
    plan = configure_ods(
        rates["accepted_episodes_per_active_second"],
        rates["training_chunks_per_active_second"],
        profile=profile,
        runtime=runtime,
    )
    return {
        "configuration": asdict(plan),
        "profile": profile.to_dict(),
        "measurement": measurement,
    }


def select_concurrency(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the smallest qualified producer count within five percent of best."""
    eligible = [row for row in results if row.get("passed")]
    if not eligible:
        raise RuntimeError("No ODS concurrency candidate passed qualification.")
    best = max(
        row["measurement"]["rates"]["training_chunks_per_wall_second"]
        for row in eligible
    )
    return min(
        (
            row
            for row in eligible
            if row["measurement"]["rates"]["training_chunks_per_wall_second"]
            >= 0.95 * best
        ),
        key=lambda row: row["producer_envs"],
    )


def _run(cfg: Any, directory: Path, model_python: str, *, timeout: float) -> None:
    from omegaconf import OmegaConf

    if timeout <= 60:
        raise TimeoutError("ODS candidate has no remaining time for recipe validation.")
    directory.mkdir(parents=True, exist_ok=False)
    cfg.runner.logger.log_path = str(directory)
    cfg.runner.logger.experiment_name = "run"
    cfg.runner.save_interval = -1
    cfg.data.ods.calibration = True
    cfg.data.ods.max_wall_seconds = timeout - 60
    if cfg.data.ods.flow.max_generated_episodes is None:
        cfg.data.ods.flow.max_generated_episodes = 128
    deadline = time.monotonic() + timeout
    cfg.data.ods.deadline_monotonic = deadline - 75
    resolved = OmegaConf.merge(
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
        {
            "hydra": {
                "run": {"dir": "."},
                "output_subdir": None,
                "job": {"chdir": False},
            }
        },
    )
    OmegaConf.save(resolved, directory / "run.yaml", resolve=True)
    env = os.environ.copy()
    ray_tmp = tempfile.mkdtemp(prefix="ods-ray-")
    (directory / "ray_tmp.txt").write_text(ray_tmp)
    env.update(RAY_ADDRESS="local", RAY_TMPDIR=ray_tmp)
    command = [
        model_python,
        str(Path(__file__).parents[2] / "examples/sft/train_vla_sft.py"),
        "--config-path",
        str(directory),
        "--config-name",
        "run",
    ]
    peaks: dict[str, int] = {}
    telemetry_timeouts = 0
    consecutive_timeouts = 0
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline - 60:
                    raise TimeoutError(
                        "ODS candidate exceeded its thirty-minute limit."
                    )
                try:
                    sample = subprocess.run(
                        [
                            "nvidia-smi",
                            "--id=0,4,5",
                            "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=5,
                    )
                except subprocess.TimeoutExpired:
                    telemetry_timeouts += 1
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        raise RuntimeError(
                            "GPU memory telemetry missed three consecutive bounded probes."
                        ) from None
                    continue
                consecutive_timeouts = 0
                for line in sample.stdout.splitlines():
                    device, used = line.split(",")
                    if int(device) in {0, 4, 5}:
                        peaks[device.strip()] = max(
                            peaks.get(device.strip(), 0), int(used) * 1024 * 1024
                        )
                time.sleep(1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (directory / "gpu_memory_peak.json").write_text(json.dumps(peaks, indent=2))
            (directory / "gpu_memory_telemetry.json").write_text(
                json.dumps(
                    {
                        "sampled_peak_bytes": peaks,
                        "timed_out_probes": telemetry_timeouts,
                    },
                    indent=2,
                )
            )
    if process.returncode:
        raise RuntimeError(
            f"ODS candidate exited with status {process.returncode}; see {directory / 'run.log'}."
        )


def main() -> None:
    """Run isolated pilot/planned trials and publish the selected measured recipe."""
    import copy

    from hydra import compose, initialize_config_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--cache-memory-gib", type=float, default=16)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(Path(__file__).parents[2] / "examples/sft/config"),
    ):
        base = compose(
            config_name="embodichain_stack_cube_ods_sft_openpi_pi05",
            overrides=args.override,
        )
    results = []
    for producers in sorted(set(args.candidates)):
        trial = {"producer_envs": producers, "passed": False}
        deadline = time.monotonic() + 1800
        try:
            cfg = copy.deepcopy(base)
            cfg.data.ods.producer_envs = producers
            if producers < 1 or int(cfg.data.ods.pilot_cache.buffer_size) % producers:
                raise ValueError(
                    "Pilot producer count must divide its initial cache size."
                )
            pilot = output / f"p{producers}-pilot"
            _run(cfg, pilot, args.model_python, timeout=deadline - time.monotonic())
            measured = measure_run(pilot / "run", int(cfg.actor.global_batch_size))
            recipe = build_recipe(measured, cfg, int(args.cache_memory_gib * 1024**3))
            recipe_path = output / f"p{producers}-recipe.json"
            recipe_path.write_text(json.dumps(recipe, indent=2))
            cfg.data.ods.recipe_path = str(recipe_path)
            verified = output / f"p{producers}-planned"
            _run(cfg, verified, args.model_python, timeout=deadline - time.monotonic())
            trial.update(
                passed=True,
                recipe_path=str(recipe_path),
                measurement=measure_run(
                    verified / "run", int(cfg.actor.global_batch_size)
                ),
            )
        except Exception as error:
            trial["error"] = f"{type(error).__name__}: {error}"
            LOGGER.error("ODS candidate %s failed: %s", producers, error)
        results.append(trial)
        (output / "candidates.json").write_text(json.dumps(results, indent=2))
    selected = select_concurrency(results)
    (output / "selected.json").write_text(json.dumps(selected, indent=2))
    LOGGER.info(
        "Selected producer_envs=%s, recipe=%s",
        selected["producer_envs"],
        selected["recipe_path"],
    )


if __name__ == "__main__":
    main()
