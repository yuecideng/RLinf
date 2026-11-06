#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${REPO_PATH:?Set REPO_PATH to the repository checkout.}"
: "${STACK_CI_RUN:?Set STACK_CI_RUN to a new output directory.}"
: "${STACK_CUBE_ENV_PYTHON:?Set the simulator interpreter.}"
export ODS_SIM_PYTHONPATH="${ODS_SIM_PYTHONPATH:-$REPO_PATH}"
export RAY_ADDRESS=local
export RAY_TMPDIR="$(mktemp -d /tmp/rlinf-ods-ray.XXXXXX)"
export STACK_CUBE_ODS_TRAIN_REQUESTS="$STACK_CI_RUN/ods_train.json"
python toolkits/lerobot/prepare_stack_cube_data.py manifest \
    --profile formal --split train --seed 40000 --episodes 8 \
    --output "$STACK_CUBE_ODS_TRAIN_REQUESTS"

config_dir="$REPO_PATH/tests/e2e_tests/sft"
config_name=embodichain_stack_cube_ods_sft_openpi_pi05_ci
python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name"
first="$STACK_CI_RUN/ods_sft/ci-test"
cp "$first/ods/supply_audit.json" "$STACK_CI_RUN/ods_first_audit.json"
checkpoint="$first/checkpoints/global_step_2"

python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name" \
    runner.logger.experiment_name=resume \
    runner.resume_dir="$checkpoint" runner.max_steps=3 runner.save_interval=-1

python - "$STACK_CI_RUN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
first = json.loads((root / "ods_first_audit.json").read_text())
resumed = json.loads((root / "ods_sft/resume/ods/supply_audit.json").read_text())
assert first["error"] is None and resumed["error"] is None
assert first["source_final"]["completed_generation"] == 2
assert first["delivered_chunks"] == first["committed_chunks"] == 4
assert resumed["delivered_chunks"] == resumed["committed_chunks"] == 2
assert first["session_id"] != resumed["session_id"]
assert not first["producer_alive_after_stop"] and not resumed["producer_alive_after_stop"]
checkpoint = root / "ods_sft/ci-test/checkpoints/global_step_2/actor"
metadata = json.loads((checkpoint / "ods_source.json").read_text())
assert metadata["committed_step"] == 2
assert metadata["resume_semantics"] == "training_state_only"
assert (checkpoint / "dcp_checkpoint/.metadata").is_file()
PY

STACK_CUBE_ODS_SFT_PLACEMENT="${STACK_CUBE_ODS_SINGLE_PLACEMENT:-0}" \
python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name" \
    runner.logger.experiment_name=single_rank runner.max_steps=1 \
    runner.save_interval=-1 data.ods.calibration_refill_step=null
python - "$STACK_CI_RUN/ods_sft/single_rank/ods/supply_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
assert audit["error"] is None and audit["world_size"] == 1
assert audit["delivered_chunks"] == audit["committed_chunks"] == 2
assert not audit["producer_alive_after_stop"]
PY

python "$REPO_PATH/tests/e2e_tests/sft/stack_cube_ods_slow_consumer.py"

python - "$STACK_CUBE_ODS_TRAIN_REQUESTS" "$STACK_CI_RUN/ods_one_request.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
manifest["requests"] = manifest["requests"][:1]
Path(sys.argv[2]).write_text(json.dumps(manifest))
PY
export STACK_CUBE_ODS_TRAIN_REQUESTS="$STACK_CI_RUN/ods_one_request.json"
if timeout 300 python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name" \
    runner.logger.experiment_name=source_failure runner.save_interval=-1; then
    echo "Expected the exhausted online source to fail before the second update." >&2
    exit 1
fi
python - "$STACK_CI_RUN/ods_sft/source_failure/ods/supply_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text())
assert "manifest is exhausted" in audit["error"]
assert audit["delivered_chunks"] == audit["committed_chunks"] == 2
assert not audit["producer_alive_after_stop"]
PY
