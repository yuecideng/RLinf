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

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.action_utils import prepare_actions_for_embodichain
from rlinf.envs.sim.embodichain.embodichain_env import (
    EmbodiChainEnv,
    _convert_pose_matrix_to_rot6d,
    _format_joint_position_gripper_state,
)
from rlinf.models.embodiment.openpi.env_io import EnvIO
from toolkits.convert_embodichain_eef_to_rot6d import convert_dataset


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


def test_embodichain_rot6d_action_adapter_converts_to_quaternion():
    actions = np.array(
        [[[0.2, -0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0]]],
        dtype=np.float32,
    )
    cfg = SimpleNamespace(action_adapter=SimpleNamespace(type="eef_pose_rot6d_gripper"))

    converted = prepare_actions_for_embodichain(actions, env_cfg=cfg)

    assert converted.shape == (1, 1, 8)
    np.testing.assert_allclose(converted[..., :3], actions[..., :3])
    np.testing.assert_allclose(
        converted[..., 3:7], np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)
    )
    assert converted[..., 7].item() == 1.0


def test_embodichain_rot6d_action_adapter_rejects_wrong_width():
    actions = np.zeros((1, 1, 7), dtype=np.float32)
    cfg = SimpleNamespace(action_adapter=SimpleNamespace(type="eef_pose_rot6d_gripper"))

    with pytest.raises(ValueError, match=r"expects \[xyz, rot6d, gripper\]"):
        prepare_actions_for_embodichain(actions, env_cfg=cfg)


def test_embodichain_rot6d_state_adapter_uses_rotation_columns():
    pose = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    pose[0, :3, 3] = torch.tensor([0.1, -0.2, 0.3])

    converted = _convert_pose_matrix_to_rot6d(pose)

    assert converted.shape == (1, 9)
    torch.testing.assert_close(
        converted,
        torch.tensor([[0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    )


def test_embodichain_rot6d_dataset_converter_updates_data_and_metadata(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "meta").mkdir()
    values = [[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.5]]
    table = pa.table(
        {
            "observation.state": values,
            "action": values,
            "observation.eef_pose": values,
        }
    )
    pq.write_table(table, source / "data" / "episode.parquet")
    features = {
        key: {"shape": [7], "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]}
        for key in ("observation.state", "action", "observation.eef_pose")
    }
    (source / "meta" / "info.json").write_text(json.dumps({"features": features}))
    (source / "meta" / "norm_stats.json").write_text(
        json.dumps({"norm_stats": {"observation.state": {}, "action": {}}})
    )

    output = tmp_path / "output"
    convert_dataset(source, output)

    converted = pq.read_table(output / "data" / "episode.parquet")
    np.testing.assert_allclose(
        np.asarray(converted["observation.state"].to_pylist(), dtype=np.float32),
        [[0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]],
    )
    metadata = json.loads((output / "meta" / "info.json").read_text())
    assert metadata["features"]["action"]["shape"] == [10]
    stats = json.loads((output / "meta" / "norm_stats.json").read_text())
    assert set(stats["norm_stats"]) == {"observation.state", "action"}
    assert len(stats["norm_stats"]["action"]["mean"]) == 10


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


def test_embodichain_rot6d_vla_config_uses_official_contract():
    cfg = _compose_embodiment_config("embodichain_repeated_pick_place_rot6d_vla_eval")

    assert cfg.env.eval.action_adapter.type == "eef_pose_rot6d_gripper"
    assert cfg.env.eval.observation_mapping.state_representation == (
        "xyz_rot6d_gripper"
    )
    assert cfg.rollout.model.action_dim == 10
    assert cfg.rollout.model.openpi.config_name == "pi05_franka_rot6d"


def test_embodichain_rot6d_sft_config_uses_ten_dim_contract():
    config_dir = Path(__file__).parents[2] / "examples" / "sft" / "config"
    cfg = _compose_config(config_dir, "embodichain_sft_openpi_pi05_rot6d")

    assert cfg.cluster.component_placement.actor == "0,2"
    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_rank == 4
    assert cfg.actor.micro_batch_size == 1
    assert cfg.actor.model.action_dim == 10
    assert cfg.actor.model.openpi.config_name == "pi05_franka_rot6d"


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
