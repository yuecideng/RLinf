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

"""Apply the EmbodiChain PI 0.5 smoke convergence gate.

The command consumes small JSON artifacts produced by the training/evaluation
steps.  It never loads a checkpoint and never starts training, so it is safe to
run in orchestration code before scheduling the expensive formal data stage.

Accepted training metric formats are either a JSON list/JSONL of records with
one of ``loss``, ``train_loss``, ``train/loss`` or ``actor/loss`` fields, a JSON
object containing ``loss_history``/``metrics``, or a TensorBoard event file or
directory. Action and closed-loop reports use the JSON emitted by the
companion evaluator scripts. Closed-loop reports must explicitly set
``evaluation_mode`` to ``unassisted``; expert-correction reports are diagnostic
artifacts and cannot satisfy this gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_LOSS_KEYS = ("loss", "train_loss", "train/loss", "actor/loss")


def _read_json_or_jsonl(path: str | Path) -> Any:
    """Read one JSON document or a JSONL stream."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Artifact is empty: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
        return records


def _finite_number(value: Any, *, field: str) -> float:
    """Convert a value to a finite float with a useful error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {number!r}.")
    return number


def _metric_records(payload: Any) -> list[dict[str, Any]]:
    """Normalize supported metric artifact layouts to a list of records."""
    if isinstance(payload, dict):
        for key in ("metrics", "loss_history", "history", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
        else:
            payload = [payload]
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError("Training metrics must be a JSON object/list of records.")
    return payload


def _loss_values(path: str | Path) -> list[float]:
    """Extract finite loss values in record order."""
    path = Path(path).expanduser().resolve()
    if path.is_dir() or path.name.startswith("events.out.tfevents"):
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard metrics require the tensorboard package; pass a "
                "JSON/JSONL export instead."
            ) from exc
        accumulator = EventAccumulator(str(path))
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        loss_tag = next(
            (
                tag
                for tag in tags
                if tag in _LOSS_KEYS or tag.rsplit("/", 1)[-1] in _LOSS_KEYS
            ),
            None,
        )
        if loss_tag is None:
            raise ValueError(
                f"No loss scalar found in TensorBoard artifact {path}; "
                f"available tags: {tags}."
            )
        values = [
            _finite_number(event.value, field=loss_tag)
            for event in accumulator.Scalars(loss_tag)
        ]
        if len(values) < 2:
            raise ValueError(
                f"TensorBoard loss must contain at least two records: {path}."
            )
        return values
    records = _metric_records(_read_json_or_jsonl(path))
    values: list[float] = []
    for row in records:
        for key in _LOSS_KEYS:
            if key in row:
                values.append(_finite_number(row[key], field=key))
                break
    if len(values) < 2:
        raise ValueError(
            f"Training metrics must contain at least two loss records: {path}."
        )
    return values


def _report_number(path: str | Path, key: str) -> float:
    """Read one finite scalar from a JSON evaluator report."""
    payload = _read_json_or_jsonl(path)
    if not isinstance(payload, dict) or key not in payload:
        raise ValueError(f"{path} must contain a {key!r} field.")
    return _finite_number(payload[key], field=f"{path}:{key}")


def _read_closed_loop_report(path: str | Path) -> dict[str, Any]:
    """Load and validate an explicitly unassisted closed-loop report."""
    payload = _read_json_or_jsonl(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one closed-loop JSON object.")
    if payload.get("evaluation_mode") != "unassisted":
        raise ValueError(
            f"{path} must declare evaluation_mode='unassisted'; assisted or "
            "legacy unlabeled reports cannot satisfy the smoke gate."
        )
    if bool(payload.get("expert_correction", False)) or bool(
        payload.get("assisted", False)
    ):
        raise ValueError(
            f"{path} is marked as expert-assisted and cannot pass the gate."
        )
    correction = payload.get("correction")
    if isinstance(correction, dict) and bool(correction.get("enabled", False)):
        raise ValueError(f"{path} enables expert correction and cannot pass the gate.")
    return payload


def _validate_episode_counts(
    path: str | Path, payload: dict[str, Any]
) -> tuple[int, int, float]:
    """Require episode counts and success rate to describe the same result."""
    episodes_value = _finite_number(payload.get("episodes"), field=f"{path}:episodes")
    successes_value = _finite_number(
        payload.get("successes"), field=f"{path}:successes"
    )
    if episodes_value <= 0 or episodes_value != int(episodes_value):
        raise ValueError(f"{path}:episodes must be a positive integer.")
    if successes_value < 0 or successes_value != int(successes_value):
        raise ValueError(f"{path}:successes must be a non-negative integer.")
    episodes = int(episodes_value)
    successes = int(successes_value)
    if successes > episodes:
        raise ValueError(f"{path}:successes cannot exceed episodes.")
    success_rate = _finite_number(
        payload.get("success_rate"), field=f"{path}:success_rate"
    )
    expected_rate = successes / episodes
    if not math.isclose(success_rate, expected_rate, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"{path}:success_rate={success_rate} does not match "
            f"successes/episodes={expected_rate}."
        )
    for prefix in ("physical", "task_program"):
        count_key = f"{prefix}_successes"
        rate_key = f"{prefix}_success_rate"
        if count_key not in payload and rate_key not in payload:
            continue
        if count_key not in payload or rate_key not in payload:
            raise ValueError(
                f"{path} must provide both {count_key!r} and {rate_key!r}."
            )
        count = _finite_number(payload[count_key], field=f"{path}:{count_key}")
        rate = _finite_number(payload[rate_key], field=f"{path}:{rate_key}")
        if count < 0 or count != int(count) or count > episodes:
            raise ValueError(f"{path}:{count_key} must be an integer in [0, episodes].")
        expected = int(count) / episodes
        if not math.isclose(rate, expected, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"{path}:{rate_key}={rate} does not match "
                f"{count_key}/episodes={expected}."
            )
    return episodes, successes, success_rate


def evaluate_smoke_gate(
    *,
    train_metrics: str | Path,
    initial_action_report: str | Path,
    trained_action_report: str | Path,
    closed_loop_report: str | Path,
    loss_window: int = 20,
    min_loss_reduction: float = 0.5,
    min_action_mae_reduction: float = 0.3,
    min_closed_loop_success_rate: float = 0.5,
) -> dict[str, Any]:
    """Return the smoke gate report and raise if inputs are malformed.

    The returned ``passed`` field is false for a failed quality gate. The CLI
    exits with status 1 in that case, which lets a data-generation orchestrator
    stop before formal data are produced.
    """
    if not 0.0 <= min_loss_reduction < 1.0:
        raise ValueError("min_loss_reduction must be in [0, 1).")
    if not 0.0 <= min_action_mae_reduction < 1.0:
        raise ValueError("min_action_mae_reduction must be in [0, 1).")
    if not 0.0 <= min_closed_loop_success_rate <= 1.0:
        raise ValueError("min_closed_loop_success_rate must be in [0, 1].")
    if loss_window <= 0:
        raise ValueError("loss_window must be positive.")

    losses = _loss_values(train_metrics)
    effective_window = min(loss_window, max(1, len(losses) // 2))
    initial_loss = float(sum(losses[:effective_window]) / effective_window)
    final_loss = float(sum(losses[-effective_window:]) / effective_window)
    if initial_loss <= 0.0:
        raise ValueError(f"Initial loss must be positive, got {initial_loss}.")
    loss_reduction = (initial_loss - final_loss) / initial_loss

    initial_mae = _report_number(initial_action_report, "action_mae")
    trained_mae = _report_number(trained_action_report, "action_mae")
    if initial_mae < 0.0 or trained_mae < 0.0:
        raise ValueError("Action MAE cannot be negative.")
    if initial_mae == 0.0:
        # A zero baseline has no meaningful relative improvement. Keep the
        # value finite and fail the reduction check instead of emitting inf.
        action_mae_reduction = 0.0
    else:
        action_mae_reduction = (initial_mae - trained_mae) / initial_mae

    checks = {
        "loss_reduction": loss_reduction >= min_loss_reduction,
        "action_mae_reduction": action_mae_reduction >= min_action_mae_reduction,
    }
    closed_loop_payload = _read_closed_loop_report(closed_loop_report)
    (
        closed_loop_episodes,
        closed_loop_successes,
        closed_loop_success_rate,
    ) = _validate_episode_counts(closed_loop_report, closed_loop_payload)
    checks["closed_loop_episodes"] = closed_loop_episodes >= 8
    checks["closed_loop_successes"] = closed_loop_successes >= 4
    checks["closed_loop_success_rate"] = (
        closed_loop_success_rate >= min_closed_loop_success_rate
    )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_loss_reduction": min_loss_reduction,
            "min_action_mae_reduction": min_action_mae_reduction,
            "min_closed_loop_success_rate": min_closed_loop_success_rate,
            "loss_window": effective_window,
        },
        "loss": {
            "initial": initial_loss,
            "final": final_loss,
            "reduction": loss_reduction,
            "records": len(losses),
        },
        "action_mae": {
            "initial": initial_mae,
            "trained": trained_mae,
            "reduction": action_mae_reduction,
        },
        "closed_loop_success_rate": closed_loop_success_rate,
        "closed_loop_episodes": int(closed_loop_episodes),
        "closed_loop_successes": int(closed_loop_successes),
    }


def main() -> None:
    """Run the gate and write a machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--initial-action-report", type=Path, required=True)
    parser.add_argument("--trained-action-report", type=Path, required=True)
    parser.add_argument("--closed-loop-report", type=Path, required=True)
    parser.add_argument("--min-loss-reduction", type=float, default=0.5)
    parser.add_argument("--min-action-mae-reduction", type=float, default=0.3)
    parser.add_argument("--min-closed-loop-success-rate", type=float, default=0.5)
    parser.add_argument("--loss-window", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_smoke_gate(
        train_metrics=args.train_metrics,
        initial_action_report=args.initial_action_report,
        trained_action_report=args.trained_action_report,
        closed_loop_report=args.closed_loop_report,
        loss_window=args.loss_window,
        min_loss_reduction=args.min_loss_reduction,
        min_action_mae_reduction=args.min_action_mae_reduction,
        min_closed_loop_success_rate=args.min_closed_loop_success_rate,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
