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

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.action_utils import prepare_actions_for_embodichain
from rlinf.envs.sim.embodichain.embodichain_env import (
    EmbodiChainEnv,
    _format_joint_position_gripper_state,
)
from rlinf.models.embodiment.openpi.env_io import EnvIO


def _compose_embodiment_config(name: str) -> Any:
    config_dir = Path(__file__).parents[2] / "examples" / "embodiment" / "config"
    return _compose_config(config_dir, name)


def _compose_config(config_dir: Path, name: str) -> Any:
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=name)


def test_embodichain_vla_config_declares_standard_io_mapping():
    cfg = _compose_embodiment_config("embodichain_repeated_pick_place_vla_eval")

    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.env.eval.env_type == "embodichain"
    assert cfg.env.eval.observation_mapping.main_images == "sensor/cam_high/color"
    assert list(cfg.env.eval.observation_mapping.states) == [
        "robot/eef_pose",
    ]
    assert cfg.env.eval.action_adapter.type == "eef_pose_gripper"
    assert cfg.rollout.model.model_type == "openpi"
    assert cfg.rollout.model.action_dim == 7
    assert cfg.rollout.model.openpi.task == "eval"


def test_embodichain_vla_action_adapter_preserves_pose_and_clips_gripper():
    actions = np.array([[[0.2, -0.3, 0.4, 1.5, -1.5, 0.2, 2.0]]], dtype=np.float32)

    converted = prepare_actions_for_embodichain(actions)

    np.testing.assert_allclose(converted[..., :6], actions[..., :6])
    assert converted[..., 6].item() == 1.0


def test_embodichain_sft_config_reuses_vla_contract():
    config_dir = Path(__file__).parents[2] / "examples" / "sft" / "config"
    cfg = _compose_config(config_dir, "embodichain_sft_openpi_pi05")

    assert cfg.runner.task_type == "sft"
    assert cfg.actor.model.openpi.task == "sft"
    assert cfg.actor.model.openpi.config_name == "pi05_franka_state"
    assert cfg.actor.model.action_dim == 7
    assert cfg.data.train_data_paths


def test_embodichain_joint_sft_config_uses_joint_contract():
    config_dir = Path(__file__).parents[2] / "examples" / "sft" / "config"
    cfg = _compose_config(config_dir, "embodichain_sft_openpi_pi05_joint")

    assert cfg.runner.task_type == "sft"
    assert cfg.actor.model.openpi.config_name == "pi05_rlt_maniskill_joint"
    assert cfg.actor.model.action_dim == 8
    assert cfg.actor.model.openpi.action_horizon == 10
    assert cfg.data.train_data_paths


def test_embodichain_joint_vla_config_uses_eight_dim_action():
    cfg = _compose_embodiment_config("embodichain_repeated_pick_place_joint_vla_eval")

    assert cfg.runner.task_type == "embodied_eval"
    assert cfg.env.eval.action_adapter.type == "joint_position_gripper"
    assert list(cfg.env.eval.observation_mapping.states) == ["robot/qpos"]
    assert cfg.rollout.model.action_dim == 8
    assert cfg.rollout.model.openpi.config_name == "pi05_rlt_maniskill_joint"
    assert cfg.rollout.model.openpi.task == "eval"


def test_joint_state_adapter_collapses_franka_mimic_finger():
    state = np.arange(9, dtype=np.float32).reshape(1, 9)
    converted = _format_joint_position_gripper_state(torch.from_numpy(state))

    np.testing.assert_array_equal(converted.numpy(), state[:, [0, 1, 2, 3, 4, 5, 6, 7]])


def test_openpi_output_transform_receives_dataset_state_alias():
    class FakeOpenPIEnvIO(EnvIO):
        device = torch.device("cpu")

    received = {}

    def output_transform(sample):
        received.update(sample)
        return {"actions": sample["actions"]}

    model = FakeOpenPIEnvIO()
    model._output_transform_fn = output_transform
    model._input_transform_fn = lambda sample: sample
    model.action_chunk = 5

    actions = torch.zeros(1, 5, 7)
    state = torch.ones(1, 7)
    decoded = model.decode_actions(actions, state)

    assert "observation.state" in received
    assert "action" in received
    np.testing.assert_allclose(received["observation.state"], np.ones(7))
    torch.testing.assert_close(decoded, actions)


def test_embodichain_step_wraps_flat_action_for_dict_space():
    class FakeEmbodiChainEnv:
        def __init__(self):
            self.received_actions = None

        def step(self, actions):
            self.received_actions = actions
            return (
                {"robot": {"qpos": torch.zeros(1, 2)}},
                torch.zeros(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

    adapter = EmbodiChainEnv.__new__(EmbodiChainEnv)
    adapter._device = torch.device("cpu")
    adapter._action_key = "eef_pose"
    adapter.num_envs = 1
    adapter.env = FakeEmbodiChainEnv()
    adapter._elapsed_steps = torch.zeros(1, dtype=torch.int32)
    adapter.ignore_terminations = False
    adapter.auto_reset = False
    adapter.state_keys = ["qpos"]
    adapter.cfg = {}
    adapter._record_metrics = lambda rewards, infos: infos

    adapter.step(torch.zeros(1, 7))

    assert set(adapter.env.received_actions) == {"eef_pose"}
    torch.testing.assert_close(
        adapter.env.received_actions["eef_pose"], torch.zeros(1, 7)
    )
