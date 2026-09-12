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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.sim.embodichain.embodichain_env import (
    _resolve_sim_device_and_gpu_id,
)


def _compose_cartpole_config(name: str) -> Any:
    config_dir = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=name)


def test_embodichain_cartpole_uses_current_task_layout():
    cfg = _compose_cartpole_config("embodichain_ppo_cart_pole")

    assert (
        cfg.env.train.gym_config_path
        == "embodichain_tasks/configs/tasks/classic_control/cart_pole/env.json"
    )
    assert cfg.env.eval.gym_config_path == cfg.env.train.gym_config_path
    assert cfg.cluster.num_nodes == 1
    assert cfg.cluster.component_placement["actor,env,rollout"] == "6-7"


def test_embodichain_uses_worker_physical_gpu_for_dexsim(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = OmegaConf.create({"sim_device": "cuda"})
    monkeypatch.setenv("LOCAL_HARDWARE_RANKS", "7")

    device, gpu_id = _resolve_sim_device_and_gpu_id(
        cfg, OmegaConf.create({"accelerator_rank": 0})
    )

    assert str(device) == "cuda:7"
    assert gpu_id == 7
