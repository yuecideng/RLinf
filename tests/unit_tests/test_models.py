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

"""Model registration, embeddings, inference adapters, and reward helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from openpi.shared.normalize import NormStats

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import (
    OpenPIApxInfAdapter,
    _active_token_ids,
)
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


def _openpi_predictor_test_worker(
    connection, checkpoint_dir, norm_stats_path, model_kwargs, noise_seed, zero_noise
):
    """Replace the external model at the spawned process boundary."""
    from toolkits.lerobot import evaluate_embodichain_openpi as offline
    from toolkits.standalone_eval_scripts import openpi_process_predictor as predictor

    class CPUModel:
        device = torch.device("cpu")
        action_dim = 32
        action_horizon = 10

        def predict_action_batch(self, observation, *, mode, rng, noise):
            assert mode == "eval"
            assert "goal_pose" not in observation
            assert isinstance(observation["states"], np.ndarray)
            assert observation.get("wrist_images") is None
            prompt = observation["task_descriptions"][0]
            if prompt == "fail":
                raise RuntimeError("model prediction failed")
            if prompt == "hang":
                time.sleep(2.0)
            observation["states"][...] = 123.0
            observation["main_images"][...] = 255
            sampled = (
                noise if noise is not None else torch.randn((1, 10, 32), generator=rng)
            )
            return sampled[..., : model_kwargs["output_action_dim"]], {}

    def build(*_args, **_kwargs):
        if checkpoint_dir == "fail":
            raise RuntimeError("model initialization failed")
        assert _kwargs["eval_sft_image_crop"] is model_kwargs["eval_sft_image_crop"]
        return CPUModel(), None, None

    offline._build_model = build
    offline._stage_checkpoint = lambda *_args: (
        Path(norm_stats_path).parent,
        SimpleNamespace(cleanup=lambda: None),
    )
    predictor._model_process(
        connection,
        checkpoint_dir,
        norm_stats_path,
        model_kwargs,
        noise_seed,
        zero_noise,
    )


def _openpi_predictor_sleep_worker(*_args):
    """Represent a model process that does not finish initialization."""
    time.sleep(5.0)


@pytest.mark.parametrize(
    ("action_dim", "config_name", "zero_noise", "eval_sft_image_crop"),
    [
        (9, "pi05_embodichain_joint_state_v2", False, False),
        (14, "pi05_embodichain_joint", False, True),
        (9, "pi05_embodichain_joint_state_v2", True, True),
    ],
)
def test_openpi_process_predictor_preserves_observations_and_noise(
    monkeypatch, tmp_path, action_dim, config_name, zero_noise, eval_sft_image_crop
):
    """Spawn isolation preserves CPU inputs and one RNG across episode resets."""
    from toolkits.standalone_eval_scripts import openpi_process_predictor as module

    monkeypatch.setattr(module, "_model_process", _openpi_predictor_test_worker)
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    observation = {
        "states": torch.zeros((1, action_dim)),
        "main_images": torch.zeros((1, 4, 5, 3), dtype=torch.uint8),
        "wrist_images": None,
        "episode_steps": torch.zeros(1, dtype=torch.int64),
        "task_descriptions": ["pick and place"],
        "goal_pose": np.eye(4),
    }
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    model = None
    try:
        model = module.OpenPIProcessPredictor(
            "checkpoint",
            config_name=config_name,
            output_action_dim=action_dim,
            norm_stats_path=str(stats),
            num_steps=5,
            device="cpu",
            include_phase_input=action_dim == 9,
            phase_scale=600.0,
            delta_action_mask=[True] * action_dim,
            noise_seed=41,
            zero_noise=zero_noise,
            eval_sft_image_crop=eval_sft_image_crop,
        )
        assert model.process_pid != os.getpid()
        expected_rng = torch.Generator().manual_seed(41)
        for _ in range(2):
            actions, diagnostics = model.predict_action_batch(observation)
            expected = (
                torch.zeros((1, 10, 32))
                if zero_noise
                else torch.randn((1, 10, 32), generator=expected_rng)
            )
            torch.testing.assert_close(actions, expected[..., :action_dim])
            assert actions.device.type == "cpu"
            assert diagnostics["shape"] == [1, 10, action_dim]
            assert diagnostics["matmul_precision"] == "high"
            assert diagnostics["tf32_flags"]["cuda_matmul_allow_tf32"] is True
            assert torch.get_float32_matmul_precision() == "medium"
            assert torch.count_nonzero(observation["states"]) == 0
            assert torch.count_nonzero(observation["main_images"]) == 0
        provenance = model.metadata["provenance"]
        assert provenance["config_name"] == config_name
        assert provenance["output_action_dim"] == action_dim
        assert provenance["include_phase_input"] is (action_dim == 9)
        assert provenance["delta_action_mask"] == [True] * action_dim
        assert provenance["noise_seed"] == 41
        assert provenance["eval_sft_image_crop"] is eval_sft_image_crop
        model.close()
        model.close()
        assert model.metadata["close_response"]["inference_calls"] == 2
        assert model.metadata["process_exit_code"] == 0
        assert model.metadata["process_alive_after_close"] is False
        with pytest.raises(RuntimeError, match="closed"):
            model.predict_action_batch(observation)
    finally:
        if model is not None:
            model.close()
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("failure", ["initialize", "predict", "timeout"])
def test_openpi_process_predictor_closes_failed_worker(monkeypatch, tmp_path, failure):
    """Model errors and prediction timeouts leave no live owned process."""
    from toolkits.standalone_eval_scripts import openpi_process_predictor as module

    monkeypatch.setattr(module, "_model_process", _openpi_predictor_test_worker)
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    kwargs = {
        "config_name": "pi05_embodichain_joint_state_v2",
        "output_action_dim": 9,
        "norm_stats_path": str(stats),
        "num_steps": 5,
        "device": "cpu",
        "prediction_timeout_s": 0.05 if failure == "timeout" else 120.0,
    }
    children_before = {child.pid for child in mp.active_children()}
    if failure == "initialize":
        with pytest.raises(RuntimeError, match="model initialization failed"):
            module.OpenPIProcessPredictor("fail", **kwargs)
    else:
        model = module.OpenPIProcessPredictor("checkpoint", **kwargs)
        observation = {
            "states": np.zeros((1, 9), dtype=np.float32),
            "main_images": np.zeros((1, 4, 5, 3), dtype=np.uint8),
            "task_descriptions": ["hang" if failure == "timeout" else "fail"],
        }
        exception = TimeoutError if failure == "timeout" else RuntimeError
        with pytest.raises(exception, match="timed out|model prediction failed"):
            model.predict_action_batch(observation)
        model.close()
        assert model.metadata["process_alive_after_close"] is False
    assert {child.pid for child in mp.active_children()} == children_before


def test_openpi_process_predictor_rolls_back_startup_timeout(monkeypatch, tmp_path):
    """A worker that never sends ready is terminated before construction raises."""
    from toolkits.standalone_eval_scripts import openpi_process_predictor as module

    monkeypatch.setattr(module, "_model_process", _openpi_predictor_sleep_worker)
    children_before = {child.pid for child in mp.active_children()}
    with pytest.raises(TimeoutError, match="initialize"):
        module.OpenPIProcessPredictor(
            "checkpoint",
            config_name="pi05_embodichain_joint_state_v2",
            output_action_dim=9,
            norm_stats_path=str(tmp_path / "norm_stats.json"),
            num_steps=5,
            device="cpu",
            startup_timeout_s=0.05,
        )
    assert {child.pid for child in mp.active_children()} == children_before


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


class _ImagePreprocessingPi0(Pi0):
    """Small image-conditioned model exercising the public Euler sampler."""

    def __init__(self, eval_sft_image_crop: bool = False):
        torch.nn.Module.__init__(self)
        self.action_dim = 32
        self.action_horizon = 10
        self.eval_sft_image_crop = eval_sft_image_crop
        self.dropout = torch.nn.Dropout(0.5)

    def build_prefix_cache(self, observation):
        signal = self.dropout(observation.images["base_0_rgb"].mean(dim=(1, 2, 3)))
        return None, None, signal

    def run_suffix(self, _observation, state, _time, cache, _prefix_mask):
        return cache[:, None, None].expand_as(state)

    def velocity_from_suffix(self, suffix):
        return suffix


@pytest.mark.parametrize("eval_sft_image_crop", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA device is unavailable"
            ),
        ),
    ],
)
def test_openpi_eval_sft_image_crop_is_exact_and_preserves_inference_state(
    eval_sft_image_crop,
    device,
):
    """The opt-in sampler uses exact deterministic SFT pixels without dropout."""
    from rlinf.models.embodiment.openpi.modules import model as model_module

    x = torch.linspace(-1.0, 1.0, 224, device=device)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    image = torch.stack((xx, yy, xx * yy), dim=-1).unsqueeze(0)
    observation = model_module.Observation(
        images={key: image.clone() for key in model_module.IMAGE_KEYS},
        image_masks={
            key: torch.tensor(["wrist" not in key], dtype=torch.bool, device=device)
            for key in model_module.IMAGE_KEYS
        },
        state=torch.arange(32, dtype=torch.float32, device=device).reshape(1, 32),
        tokenized_prompt=torch.arange(20, dtype=torch.int64, device=device).reshape(
            1, 20
        ),
        tokenized_prompt_mask=torch.ones((1, 20), dtype=torch.bool, device=device),
    )
    expected = model_module.preprocess_observation(
        observation, train=eval_sft_image_crop, rng=None
    )
    model = _ImagePreprocessingPi0(eval_sft_image_crop).to(device).eval()
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone() if device == "cuda" else None
    prepared = model._preprocess_eval_observation(observation)
    for key in observation.images:
        assert torch.equal(prepared.images[key], expected.images[key])
        assert torch.equal(prepared.image_masks[key], observation.image_masks[key])
    assert torch.equal(prepared.state, observation.state)
    assert torch.equal(prepared.tokenized_prompt, observation.tokenized_prompt)
    assert torch.equal(
        prepared.tokenized_prompt_mask, observation.tokenized_prompt_mask
    )
    assert torch.equal(torch.get_rng_state(), before)
    noise = torch.zeros((1, 10, 32), device=device)
    result = model.sample_actions(observation, noise=noise, num_steps=5)
    expected_signal = expected.images["base_0_rgb"].mean(dim=(1, 2, 3))
    torch.testing.assert_close(result, -expected_signal[:, None, None].expand_as(noise))
    assert model.training is False
    assert model.dropout.training is False
    assert torch.equal(torch.get_rng_state(), before)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
    if not eval_sft_image_crop:
        default = _ImagePreprocessingPi0().to(device).eval()
        assert torch.equal(
            default.sample_actions(observation, noise=noise, num_steps=5), result
        )
    else:
        model.train()
        with pytest.raises(ValueError, match="model.eval"):
            model.sample_actions(observation, noise=noise, num_steps=5)


@pytest.mark.parametrize("enabled", [None, False, True])
def test_openpi_factory_forwards_eval_sft_image_crop(monkeypatch, tmp_path, enabled):
    """Factory configuration reaches the inference model constructor."""
    import rlinf.models.embodiment.openpi as factory
    import rlinf.models.embodiment.openpi.tasks.eval as eval_task

    class StubModel(torch.nn.Module):
        def __init__(self, _config, **kwargs):
            super().__init__()
            self.eval_sft_image_crop = kwargs["eval_sft_image_crop"]

    monkeypatch.setattr(eval_task, "Pi0Eval", StubModel)
    monkeypatch.setattr(factory, "resolve_model_safetensors", lambda _path: None)
    monkeypatch.setattr(
        factory, "resolve_full_weights", lambda _path: tmp_path / "weights.pt"
    )
    monkeypatch.setattr(factory, "load_full_weights", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(factory, "_install_transforms", lambda *_args: None)
    monkeypatch.setattr(factory, "_apply_openpi_param_dtypes", lambda *_args: None)
    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "precision": None,
            "action_dim": 9,
            "num_action_chunks": 5,
            "num_steps": 5,
            "openpi": {
                "task": "eval",
                "config_name": "pi05_embodichain_joint_state_v2",
                "action_horizon": 10,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "eval_sft_image_crop": enabled,
            },
        }
    )
    if enabled is None:
        del cfg.openpi.eval_sft_image_crop
    assert factory.get_model(cfg).eval_sft_image_crop is bool(enabled)


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_embodichain_joint_policy_pads_images_and_actions():
    """EmbodiChain joint samples use one RGB view and a padded Pi05 action."""
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
        EmbodiChainJointOutputs,
    )

    inputs = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=9,
        model_type=openpi_model.ModelType.PI05,
    )
    sample = inputs(
        {
            "observation/image": np.zeros((4, 5, 4), dtype=np.uint8),
            "observation/state": np.arange(9, dtype=np.float32),
            "actions": np.ones((10, 9), dtype=np.float32),
            "prompt": "pick and place",
        }
    )

    assert sample["state"].shape == (32,)
    assert sample["actions"].shape == (10, 32)
    assert sample["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert sample["image"]["base_0_rgb"].dtype == np.uint8
    assert bool(sample["image_mask"]["base_0_rgb"])
    assert not bool(sample["image_mask"]["left_wrist_0_rgb"])
    assert sample["prompt"] == "pick and place"

    outputs = EmbodiChainJointOutputs(output_action_dim=9)(
        {"actions": np.zeros((2, 10, 32), dtype=np.float32)}
    )
    assert outputs["actions"].shape == (2, 10, 9)


@pytest.mark.parametrize("physical_dim", [9, 14])
@pytest.mark.parametrize("include_phase", [False, True])
def test_embodichain_joint_policy_can_defer_model_padding(physical_dim, include_phase):
    """State-token recipes retain physical dimensions until model transforms."""
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
    )

    step = np.asarray(37, dtype=np.int64)
    sample = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=physical_dim,
        model_type=openpi_model.ModelType.PI05,
        include_phase_input=include_phase,
        phase_scale=600.0,
        pad_inputs_to_model_dim=False,
    )(
        {
            "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation/state": np.arange(physical_dim, dtype=np.float32),
            "observation/episode_step": step,
            "actions": np.ones((10, physical_dim), dtype=np.float32),
        }
    )

    assert sample["state"].shape == (physical_dim + int(include_phase),)
    assert sample["actions"].shape == (10, physical_dim)
    if include_phase:
        assert sample["state"][-1] == pytest.approx(37.0 / 600.0)


def test_embodichain_joint_policy_appends_episode_phase():
    """The optional phase input occupies one padded state dimension."""
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
    )

    inputs = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=9,
        model_type=openpi_model.ModelType.PI05,
        include_phase_input=True,
        phase_scale=100.0,
    )
    sample = inputs(
        {
            "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
            "observation/state": np.zeros(9, dtype=np.float32),
            "observation/episode_step": np.asarray(25, dtype=np.int32),
        }
    )

    assert sample["state"].shape == (32,)
    assert sample["state"][9] == pytest.approx(0.25)
    assert np.all(sample["state"][10:] == 0.0)


def test_embodichain_joint_policy_rejects_short_actions():
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
    )

    inputs = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=9,
        model_type=openpi_model.ModelType.PI05,
    )
    with pytest.raises(ValueError, match="do not match"):
        inputs(
            {
                "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
                "observation/state": np.zeros(9, dtype=np.float32),
                "actions": np.zeros((10, 7), dtype=np.float32),
            }
        )


def test_embodichain_joint_policy_scales_normalized_float_images():
    """Float images in the common [0, 1] range become uint8 RGB inputs."""
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
    )

    inputs = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=9,
        model_type=openpi_model.ModelType.PI05,
    )
    sample = inputs(
        {
            "observation/image": np.full((2, 3, 3), 0.5, dtype=np.float32),
            "observation/state": np.zeros(9, dtype=np.float32),
        }
    )

    assert sample["image"]["base_0_rgb"].dtype == np.uint8
    assert np.all(sample["image"]["base_0_rgb"] == 127)


def test_embodichain_joint_policy_absolute_delta_roundtrip_with_norm_stats():
    """Non-zero normalization stats do not alter absolute action decoding."""
    from openpi import transforms
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.openpi.policies.embodichain_policy import (
        EmbodiChainJointInputs,
        EmbodiChainJointOutputs,
    )

    state = np.linspace(-0.8, 0.8, 9, dtype=np.float32)
    absolute_actions = state + np.linspace(0.05, 0.5, 9, dtype=np.float32)
    sample = {
        "observation/image": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation/state": state,
        "actions": np.stack([absolute_actions, absolute_actions + 0.1]),
    }
    data_input = EmbodiChainJointInputs(
        action_dim=32,
        output_action_dim=9,
        model_type=openpi_model.ModelType.PI05,
    )
    converted = data_input(sample)
    delta = transforms.DeltaActions(transforms.make_bool_mask(9))
    restored_delta = transforms.AbsoluteActions(transforms.make_bool_mask(9))
    stats = {
        "state": NormStats(
            mean=np.full(32, 0.25, dtype=np.float32),
            std=np.full(32, 2.0, dtype=np.float32),
        ),
        "actions": NormStats(
            mean=np.full(32, -0.4, dtype=np.float32),
            std=np.full(32, 1.7, dtype=np.float32),
        ),
    }
    delta_sample = delta(
        {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in converted.items()
        }
    )
    normalized = transforms.Normalize(stats)(delta_sample)
    unnormalized = transforms.Unnormalize(stats)(normalized)
    decoded = restored_delta(unnormalized)
    decoded = EmbodiChainJointOutputs(output_action_dim=9)(decoded)

    np.testing.assert_allclose(decoded["actions"], sample["actions"], atol=1e-6)


def test_openpi_sft_action_weights_preserve_mean_and_validate_shape():
    """Optional SFT weights emphasize contact dims without changing scale."""
    from rlinf.models.embodiment.openpi.pi0 import Pi0

    model = object.__new__(Pi0)
    model.action_chunk = 5
    model.action_env_dim = 9
    loss = torch.ones(2, 10, 32)
    reduced = model._reduce_sft_loss(
        loss,
        use_action_chunk_loss=True,
        action_loss_weights=[1.0] * 7 + [4.0, 4.0],
        action_step_weights=[3.0, 2.0, 1.0, 1.0, 1.0],
    )
    assert reduced.item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="action_loss_weights"):
        model._reduce_sft_loss(
            loss,
            use_action_chunk_loss=True,
            action_loss_weights=[1.0] * 8,
        )


def test_pour_water_strict_tracker_does_not_accept_return_pose_alone():
    """Pour success requires configured pour geometry beyond the return pose."""
    from toolkits.standalone_eval_scripts.embodichain_openpi_eval import (
        _PourWaterPhysicalTracker,
    )

    tracker = _PourWaterPhysicalTracker(
        torch.tensor([0.75, -0.10, 0.962]),
        position_tolerance=0.05,
        tilt_threshold=0.5,
    )
    tracker.final_position = torch.tensor([0.75, -0.10, 0.962])
    tracker.max_tilt = 1.0
    tracker.initial_matrix = torch.eye(4)
    tracker.final_matrix = torch.eye(4)
    tracker.pour_geometry_seen = False

    result = tracker.result()
    assert result["physical_proxy_success"] is True
    assert result["strict_geometry_success"] is False

    tracker.pour_geometry_seen = True
    result = tracker.result()
    assert result["strict_geometry_success"] is True


@pytest.fixture
def pour_tracker_pose_fixture():
    """Expose mutable external object poses through their public reader API."""
    from toolkits.standalone_eval_scripts import embodichain_openpi_eval as evaluator

    class PoseAsset:
        def __init__(self, position):
            self.matrix = torch.eye(4)
            self.matrix[:3, 3] = torch.tensor(position)
            self.quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        def get_local_pose(self, *, to_matrix):
            if to_matrix:
                return self.matrix.unsqueeze(0)
            return self.matrix[:3, 3].unsqueeze(0), self.quaternion.unsqueeze(0)

    bottle = PoseAsset([0.75, -0.10, 0.962])
    cup = PoseAsset([0.75, 0.10, 0.90])
    assets = {"bottle": bottle, "cup": cup}
    env = SimpleNamespace(
        env=SimpleNamespace(sim=SimpleNamespace(get_asset=lambda uid: assets[uid]))
    )
    tracker = evaluator._PourWaterPhysicalTracker(
        torch.tensor([0.75, -0.10, 0.962]),
        position_tolerance=0.05,
        tilt_threshold=0.5,
    )
    tracker.reset(env)

    def observe(*, axis, angle, at_target=False):
        radians = torch.tensor(angle)
        c, s = torch.cos(radians), torch.sin(radians)
        if axis == "x":
            rotation = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
            quaternion = torch.tensor(
                [torch.cos(radians / 2), torch.sin(radians / 2), 0.0, 0.0]
            )
        else:
            rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            quaternion = torch.tensor(
                [torch.cos(radians / 2), 0.0, 0.0, torch.sin(radians / 2)]
            )
        bottle.matrix = (
            cup.matrix @ evaluator._POUR_WATER_RELATIVE_POSE
            if at_target
            else torch.eye(4)
        )
        if not at_target:
            bottle.matrix[:3, 3] = torch.tensor([0.75, -0.10, 0.962])
        bottle.matrix[:3, :3] = rotation
        bottle.quaternion = quaternion
        tracker.update(env)
        return tracker.result()

    return tracker, observe


def test_pour_water_proxy_excludes_pure_yaw(pour_tracker_pose_fixture):
    tracker, observe = pour_tracker_pose_fixture
    observe(axis="z", angle=0.708)
    result = observe(axis="z", angle=0.0)
    assert result["max_bottle_rotation"] == pytest.approx(0.708, abs=1e-6)
    assert result["max_bottle_tilt"] == pytest.approx(0.0, abs=1e-6)
    assert result["physical_proxy_success"] is False
    assert result["min_pour_rotation_error_at_valid_position"] is None
    assert result["position_match_frames"] == 0


def test_pour_water_proxy_accepts_genuine_axis_tilt(pour_tracker_pose_fixture):
    _, observe = pour_tracker_pose_fixture
    observe(axis="x", angle=-0.7)
    result = observe(axis="x", angle=0.0)
    assert result["max_bottle_tilt"] == pytest.approx(0.7, abs=1e-6)
    assert result["physical_proxy_success"] is True
    # Tilt and return alone still do not establish the cup-relative pour pose.
    assert result["strict_geometry_success"] is False


def test_pour_water_diagnostics_require_same_frame_position_and_rotation(
    pour_tracker_pose_fixture,
):
    from toolkits.standalone_eval_scripts import embodichain_openpi_eval as evaluator

    _, observe = pour_tracker_pose_fixture
    observe(axis="x", angle=0.0, at_target=True)
    result = observe(axis="x", angle=evaluator._POUR_WATER_ROTATE_ANGLE)
    assert result["min_pour_rotation_error"] == pytest.approx(0.0, abs=1e-6)
    assert result["min_pour_rotation_error_at_valid_position"] == pytest.approx(
        abs(evaluator._POUR_WATER_ROTATE_ANGLE), abs=1e-6
    )
    assert result["position_match_frames"] == result["rotation_match_frames"] == 1
    assert result["joint_pour_pose_frames"] == 0
    assert result["max_pour_dwell_frames"] == 0
    result = observe(axis="x", angle=evaluator._POUR_WATER_ROTATE_ANGLE, at_target=True)
    assert result["min_pour_rotation_error_at_valid_position"] == pytest.approx(
        0.0, abs=1e-6
    )
    assert result["joint_pour_pose_frames"] == 1
    assert result["max_pour_dwell_frames"] == 1


@pytest.fixture
def embodied_evaluator_pose_fixture(monkeypatch, tmp_path):
    """Provide scripted external poses and model actions to the evaluator."""
    from toolkits.standalone_eval_scripts import embodichain_openpi_eval as evaluator

    class PoseAsset:
        def __init__(self, position):
            self.matrix = torch.eye(4)
            self.matrix[:3, 3] = torch.tensor(position)
            self.quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

        def get_local_pose(self, *, to_matrix):
            if to_matrix:
                return self.matrix.unsqueeze(0)
            return self.matrix[:3, 3].unsqueeze(0), self.quaternion.unsqueeze(0)

    class Predictor:
        def __init__(self, checkpoint, *, output_action_dim, **kwargs):
            self.action_horizon = 10
            self.output_action_dim = output_action_dim
            self.process_pid = os.getpid()
            self.metadata = {
                "matmul_precision": "high",
                "tf32_flags": {},
                "provenance": {"checkpoint": checkpoint},
            }

        def predict_action_batch(self, observation):
            return torch.zeros(1, 10, self.output_action_dim), {}

        def close(self):
            self.metadata.update(
                process_exit_code=0,
                process_alive_after_close=False,
                close_response={"type": "closed"},
                close_error=None,
            )

    norm_path = tmp_path / "norm_stats.json"
    norm_path.write_text('{"norm_stats": {}}')
    monkeypatch.setattr(evaluator, "OpenPIProcessPredictor", Predictor)

    def run(
        *,
        pour,
        physical_sequence,
        raw_success,
        terminated,
        fail=False,
        task_config=None,
        native_task_id=None,
    ):
        environments = []
        action_dim = 14 if pour else 9

        class ExternalEnv:
            def __init__(self, cfg, **kwargs):
                self.cfg = cfg
                self.steps = 0
                self.closed = False
                self.assets = {
                    "bottle": PoseAsset([0.75, -0.10, 0.962]),
                    "cup": PoseAsset([0.75, 0.10, 0.90]),
                }
                self.env = SimpleNamespace(
                    sim=SimpleNamespace(get_asset=lambda uid: self.assets[uid])
                )
                if native_task_id is not None:
                    self.env.spec = SimpleNamespace(id=native_task_id)
                environments.append(self)

            def observation(self):
                return {
                    "states": torch.zeros(1, action_dim),
                    "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
                    "task_descriptions": ["Pour water" if pour else "Pick cube"],
                }

            def reset(self, *, seed):
                return self.observation(), {}

            def step(self, action):
                self.steps += 1
                bottle = self.assets["bottle"]
                if physical_sequence and self.steps <= 5:
                    target = (
                        self.assets["cup"].matrix @ evaluator._POUR_WATER_RELATIVE_POSE
                    )
                    angle = torch.tensor(evaluator._POUR_WATER_ROTATE_ANGLE)
                    c, s = torch.cos(angle), torch.sin(angle)
                    bottle.matrix = target.clone()
                    bottle.matrix[:3, :3] = target[:3, :3] @ torch.tensor(
                        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
                    )
                    bottle.quaternion = torch.tensor(
                        [torch.cos(angle / 2), torch.sin(angle / 2), 0.0, 0.0]
                    )
                elif physical_sequence and self.steps == 6:
                    self.assets["bottle"] = PoseAsset([0.75, -0.10, 0.962])
                elif physical_sequence:
                    self.assets["bottle"] = PoseAsset([2.0, 2.0, 2.0])
                return (
                    self.observation(),
                    torch.zeros(1),
                    torch.tensor([terminated]),
                    torch.tensor([False]),
                    {
                        "success": torch.tensor([raw_success]),
                        "fail": torch.tensor([fail]),
                    },
                )

            def close(self):
                self.closed = True

        monkeypatch.setattr(evaluator, "EmbodiChainEnv", ExternalEnv)
        if task_config is None:
            task_config = tmp_path / (
                "pour_water_fixture.yaml" if pour else "pick_place_fixture.yaml"
            )
            task_config.write_text(
                f"id: {'PourWater-v1' if pour else 'PickPlace-v1'}\n"
            )
        report = evaluator.evaluate(
            str(task_config),
            "fixture_checkpoint",
            config_name="pi05_embodichain_joint_state_v2",
            action_dim=action_dim,
            norm_stats_path=str(norm_path),
            num_episodes=1,
            max_steps=8,
            action_chunk=5,
            num_steps=5,
            include_phase_input=False,
            phase_scale=600.0,
            initial_hold_steps=0,
            seed=0,
            noise_seed=0,
            zero_noise=False,
            gripper_threshold=None,
            gripper_open_steps=None,
            gripper_close_step=None,
            qpos_feedforward=0.0,
            action_application_mode="target_position",
            max_joint_step=None,
            qpos_track_max_step=None,
            action_settle_steps=1,
            expert_correction=False,
            correction_max_steps=0,
            correction_threshold=0.05,
            correction_force_steps=0,
            delta_action_mask=[True] * action_dim,
            position_tolerance=0.05,
            tilt_threshold=0.5,
            device="cpu",
        )
        assert environments[0].closed
        assert report["inference_process_cleanup"]["process_exit_code"] == 0
        return report, environments[0]

    return run


def test_pour_water_scoring_follows_same_deployment_at_renamed_path(
    embodied_evaluator_pose_fixture, monkeypatch, tmp_path
):
    """Moving an identical Pour deployment keeps its strict geometry verdict."""
    original = tmp_path / "pour_water" / "task.yaml"
    original.parent.mkdir()
    original.write_text(
        "id: PourWater-v1\n"
        "environment:\n  component: env_smoke_val.yaml\n"
        "task_program:\n  program: task_program/program.yaml\n"
        "  integration: task_program/integration.yaml\n"
        "  execution_policy: components/trajectory_open_loop_dense.yaml\n"
        "embodiment:\n  component: components/cobotmagic_vla.yaml\n"
        "seed: 5101\n"
    )
    relocated = tmp_path / "generic_deployment.yaml"
    relocated.write_bytes(original.read_bytes())
    kwargs = {
        "pour": True,
        "physical_sequence": True,
        "raw_success": False,
        "terminated": False,
    }
    named, _ = embodied_evaluator_pose_fixture(task_config=original, **kwargs)
    monkeypatch.setenv("EMBODICHAIN_PATH", str(tmp_path))
    renamed, _ = embodied_evaluator_pose_fixture(task_config=relocated.name, **kwargs)
    assert named["successes"] == renamed["successes"] == 1
    assert (
        named["task_success_source"]
        == renamed["task_success_source"]
        == "pour_water_physical"
    )
    assert named["per_episode"] == renamed["per_episode"]


def test_pick_scoring_ignores_pour_words_in_deployment_path(
    embodied_evaluator_pose_fixture, tmp_path
):
    """A Pick deployment retains environment scoring in a misleading folder."""
    path = tmp_path / "pour_water" / "task.yaml"
    path.parent.mkdir()
    path.write_text("id: PickPlace-v1\n")
    report, _ = embodied_evaluator_pose_fixture(
        task_config=path,
        pour=False,
        physical_sequence=False,
        raw_success=True,
        terminated=True,
    )
    assert report["successes"] == 1 and report["task_success_source"] == "info"
    assert report["strict_geometry_successes"] == 0


def test_embodichain_scoring_prefers_public_environment_identity(
    embodied_evaluator_pose_fixture, tmp_path
):
    """A registered native environment identifies package-resolved deployments."""
    path = tmp_path / "generic.yaml"
    path.write_text("id: PickPlace-v1\n")
    pour, _ = embodied_evaluator_pose_fixture(
        task_config=path,
        native_task_id="PourWater-v1",
        pour=True,
        physical_sequence=True,
        raw_success=False,
        terminated=False,
    )
    assert pour["successes"] == 1
    assert pour["task_success_source"] == "pour_water_physical"
    path.write_text("id: PourWater-v1\n")
    pick, _ = embodied_evaluator_pose_fixture(
        task_config=path,
        native_task_id="PickPlace-v1",
        pour=False,
        physical_sequence=False,
        raw_success=True,
        terminated=True,
    )
    assert pick["successes"] == 1 and pick["task_success_source"] == "info"


@pytest.mark.parametrize("raw_success", [False, True])
def test_pour_water_physical_completion_stops_without_program_execution(
    embodied_evaluator_pose_fixture, raw_success
):
    report, env = embodied_evaluator_pose_fixture(
        pour=True,
        physical_sequence=True,
        raw_success=raw_success,
        terminated=raw_success,
    )
    assert report["successes"] == report["strict_geometry_successes"] == 1
    assert report["task_program_successes"] == int(raw_success)
    assert report["strict_task_program_successes"] == int(raw_success)
    assert report["per_episode"][0]["max_pour_dwell_frames"] == 5
    # A seventh action would move the completed bottle away from its return pose.
    assert env.steps == report["per_episode"][0]["episode_length"] == 6


@pytest.mark.parametrize("terminated", [False, True])
def test_pour_water_raw_success_does_not_complete_physical_task(
    embodied_evaluator_pose_fixture, terminated
):
    report, env = embodied_evaluator_pose_fixture(
        pour=True, physical_sequence=False, raw_success=True, terminated=terminated
    )
    assert report["successes"] == report["strict_geometry_successes"] == 0
    assert report["task_program_successes"] == 1
    assert report["strict_task_program_successes"] == 0
    assert env.steps == report["per_episode"][0]["episode_length"] == 8


@pytest.mark.parametrize(
    ("raw_success", "terminated", "fail", "expected_success"),
    [(True, False, False, 1), (False, True, False, 0), (False, True, True, 0)],
)
def test_non_pour_evaluator_preserves_raw_success_and_termination(
    embodied_evaluator_pose_fixture, raw_success, terminated, fail, expected_success
):
    report, env = embodied_evaluator_pose_fixture(
        pour=False,
        physical_sequence=False,
        raw_success=raw_success,
        terminated=terminated,
        fail=fail,
    )
    assert report["successes"] == expected_success
    assert report["task_success_source"] == "info"
    assert env.steps == report["per_episode"][0]["episode_length"] == 1


def test_pour_water_native_failure_still_ends_episode(
    embodied_evaluator_pose_fixture,
):
    report, env = embodied_evaluator_pose_fixture(
        pour=True,
        physical_sequence=False,
        raw_success=False,
        terminated=True,
        fail=True,
    )
    assert report["successes"] == 0
    assert env.steps == report["per_episode"][0]["episode_length"] == 1


@pytest.mark.parametrize(
    ("model_type", "use_chunk_loss", "expected"),
    [
        ("openpi", False, False),
        ("openpi", True, True),
        ("mlp_policy", True, False),
    ],
)
def test_openpi_sft_chunk_loss_requires_explicit_opt_in(
    model_type, use_chunk_loss, expected
):
    """Only recipes that explicitly opt in receive chunk-only SFT loss."""
    from rlinf.workers.sft.fsdp_vla_sft_worker import _use_action_chunk_loss

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": model_type,
                    "openpi": {"use_action_chunk_loss": use_chunk_loss},
                }
            }
        }
    )
    assert _use_action_chunk_loss(cfg) is expected


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(*, prefix_seq_len: int = 5) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules and importlib.util.find_spec("gymnasium") is None:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules and importlib.util.find_spec("imageio") is None:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


class _FakeApxInfModel:
    action_horizon = 10
    action_dim = 32
    num_views = 2
    image_size = 224

    def __init__(self, output_shape=(10, 32)):
        self.output_shape = output_shape
        self.calls = []
        self.closed = False

    def infer_rgb(self, rgb_u8, layout, token_ids, *, noise=None):
        self.calls.append((rgb_u8, layout, token_ids, noise))
        offset = len(self.calls) * 1000
        return (
            np.arange(np.prod(self.output_shape), dtype=np.float32).reshape(
                self.output_shape
            )
            + offset
        )

    def close(self):
        self.closed = True


class _FakeApxInfProcessor:
    def __init__(self):
        self.preprocess_calls = []
        self.postprocess_calls = []

    def preprocess_batch(self, env_obs, *, num_views, image_size):
        self.preprocess_calls.append((env_obs, num_views, image_size))
        prepared = []
        for index in range(len(env_obs["task_descriptions"])):
            prepared.append(
                {
                    "rgb_u8": np.full(
                        (num_views, image_size, image_size, 3),
                        index,
                        dtype=np.uint8,
                    ),
                    "token_ids": np.array([index, index + 1], dtype=np.uint32),
                    "state": np.full(32, index, dtype=np.float32),
                }
            )
        return prepared

    def postprocess_batch(self, normalized_actions, prepared):
        self.postprocess_calls.append((normalized_actions.copy(), prepared))
        return torch.from_numpy(normalized_actions[:, :5, :7].copy())


def _apxinf_model_cfg(**apxinf_overrides):
    apxinf = {
        "action_horizon": 10,
        "num_flow_steps": 5,
        "noise_source": "apxinf",
        "seed": 0,
        **apxinf_overrides,
    }
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": "/not/loaded/in/unit/test",
            "num_action_chunks": 5,
            "action_dim": 7,
            "openpi": {
                "config_name": "pi05_libero",
                "num_steps": 5,
                "noise_method": "flow_sde",
                "noise_level": 0.3,
            },
            "apxinf": apxinf,
        }
    )


def _apxinf_env_obs(batch_size=2):
    return {
        "main_images": torch.zeros(batch_size, 8, 8, 3, dtype=torch.uint8),
        "wrist_images": torch.ones(batch_size, 8, 8, 3, dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.zeros(batch_size, 8),
        "task_descriptions": [f"task {index}" for index in range(batch_size)],
    }


def _apxinf_adapter(*, model=None, processor=None, **apxinf_overrides):
    return OpenPIApxInfAdapter(
        _apxinf_model_cfg(**apxinf_overrides),
        "cpu",
        model=model or _FakeApxInfModel(),
        processor=processor or _FakeApxInfProcessor(),
    )


def test_apxinf_strips_openpi_prompt_padding_before_l1_inference():
    transformed = {
        "tokenized_prompt": np.array([2, 42, 108, 0, 0], dtype=np.int32),
        "tokenized_prompt_mask": np.array([True, True, True, False, False]),
    }

    tokens = _active_token_ids(transformed)

    np.testing.assert_array_equal(tokens, np.array([2, 42, 108], dtype=np.uint32))
    assert tokens.flags.c_contiguous


def test_apxinf_calls_l1_infer_rgb_and_delegates_pre_and_postprocessing():
    model = _FakeApxInfModel()
    processor = _FakeApxInfProcessor()
    adapter = _apxinf_adapter(model=model, processor=processor)
    env_obs = _apxinf_env_obs()

    actions, result = adapter.predict_action_batch(env_obs, mode="eval")

    assert actions.shape == (2, 5, 7)
    assert actions.dtype == torch.float32
    assert processor.preprocess_calls == [(env_obs, 2, 224)]
    assert len(model.calls) == 2
    assert model.calls[0][0].shape == (2, 224, 224, 3)
    assert model.calls[0][0].dtype == np.uint8
    assert model.calls[0][1] == "nhwc"
    assert model.calls[0][2].dtype == np.uint32
    assert model.calls[0][3] is None
    normalized = processor.postprocess_calls[0][0]
    assert normalized.shape == (2, 10, 32)
    assert len(result["apxinf_timing"]) == 2


def test_apxinf_explicit_noise_is_split_and_forwarded_exactly():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model, noise_source="observation")
    env_obs = _apxinf_env_obs()
    env_obs["noise"] = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)

    adapter.predict_action_batch(env_obs)

    np.testing.assert_array_equal(model.calls[0][3], env_obs["noise"][0].numpy())
    np.testing.assert_array_equal(model.calls[1][3], env_obs["noise"][1].numpy())


def test_apxinf_observation_noise_is_required():
    adapter = _apxinf_adapter(noise_source="observation")
    with pytest.raises(ValueError, match="requires env_obs"):
        adapter.predict_action_batch(_apxinf_env_obs())


def test_apxinf_observation_noise_does_not_override_other_noise_sources():
    env_obs = _apxinf_env_obs()
    explicit_noise = torch.full((2, 10, 32), 123.0)
    env_obs["noise"] = explicit_noise

    apxinf_model = _FakeApxInfModel()
    _apxinf_adapter(model=apxinf_model, noise_source="apxinf").predict_action_batch(
        env_obs
    )
    assert all(call[3] is None for call in apxinf_model.calls)

    torch_model = _FakeApxInfModel()
    torch_adapter = _apxinf_adapter(model=torch_model, noise_source="torch")
    torch_adapter.predict_action_batch(env_obs)
    assert all(call[3] is not None for call in torch_model.calls)
    for index, call in enumerate(torch_model.calls):
        assert not np.array_equal(call[3], explicit_noise[index].numpy())


def test_apxinf_torch_noise_is_reproducible_and_has_model_shape():
    model_a = _FakeApxInfModel()
    model_b = _FakeApxInfModel()
    adapter_a = _apxinf_adapter(model=model_a, noise_source="torch", seed=7)
    adapter_b = _apxinf_adapter(model=model_b, noise_source="torch", seed=7)

    adapter_a.predict_action_batch(_apxinf_env_obs())
    adapter_b.predict_action_batch(_apxinf_env_obs())

    assert model_a.calls[0][3].shape == (10, 32)
    np.testing.assert_array_equal(model_a.calls[0][3], model_b.calls[0][3])
    np.testing.assert_array_equal(model_a.calls[1][3], model_b.calls[1][3])


def test_apxinf_rejects_bad_normalized_action_shape():
    adapter = _apxinf_adapter(model=_FakeApxInfModel(output_shape=(10, 7)))
    with pytest.raises(ValueError, match="normalized actions have shape"):
        adapter.predict_action_batch(_apxinf_env_obs(batch_size=1))


def test_apxinf_rejects_mismatched_openpi_and_apxinf_flow_steps():
    with pytest.raises(ValueError, match="must match OpenPI num_steps"):
        _apxinf_adapter(num_flow_steps=10)


def test_apxinf_rejects_training_mode():
    adapter = _apxinf_adapter()
    with pytest.raises(ValueError, match="eval-only"):
        adapter.predict_action_batch(_apxinf_env_obs(), mode="train")


def test_apxinf_close_delegates_to_model():
    model = _FakeApxInfModel()
    adapter = _apxinf_adapter(model=model)
    adapter.close()
    assert model.closed


def _stub_apxinf_robo(monkeypatch, resolved_tactics=None):
    """Install a fake ``apxinf_robo`` and record what ``_load_model`` asks it for."""
    seen = {}

    def load_bare_model(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return _FakeApxInfModel()

    def resolve_tactics(device, precision, **kwargs):
        seen["resolve"] = {"device": device, "precision": precision, **kwargs}
        return resolved_tactics

    module = ModuleType("apxinf_robo")
    module.load_bare_model = load_bare_model
    engine = ModuleType("apxinf_robo.engine")
    engine.resolve_tactics = resolve_tactics
    module.engine = engine
    monkeypatch.setitem(sys.modules, "apxinf_robo", module)
    monkeypatch.setitem(sys.modules, "apxinf_robo.engine", engine)
    return seen


def test_apxinf_loads_through_the_apxinf_robo_l1_entry_point(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(_apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor())

    assert seen["path"] == Path("/not/loaded/in/unit/test")
    kwargs = seen["kwargs"]
    assert kwargs["model"] == "pi05"
    assert kwargs["device"] == "cpu"
    assert kwargs["precision"] == "bf16"
    assert kwargs["action_horizon"] == 10
    assert kwargs["num_flow_steps"] == 5
    assert kwargs["sampling_seed"] == 0
    # Left out so load_bare_model selects the tuned tactics.
    assert "tactics" not in kwargs
    assert "resolve" not in seen


def test_apxinf_a_configured_tactics_file_wins_over_the_default_selection(monkeypatch):
    seen = _stub_apxinf_robo(monkeypatch)

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(tactics="/mine.json"), "cpu", processor=_FakeApxInfProcessor()
    )

    assert seen["kwargs"]["tactics"] == "/mine.json"
    assert "resolve" not in seen


def test_apxinf_an_explicit_weights_file_resolves_tactics_from_the_checkpoint_dir(
    monkeypatch,
):
    seen = _stub_apxinf_robo(monkeypatch, resolved_tactics="/ckpt/tactics.json")

    OpenPIApxInfAdapter(
        _apxinf_model_cfg(checkpoint="/ckpt/model-00001-of-00002.safetensors"),
        "cpu",
        processor=_FakeApxInfProcessor(),
    )

    # The weights file goes to the loader, the directory to the tactics lookup:
    # keying the lookup on the file would miss a checkpoint-local tactics.json.
    assert seen["path"] == Path("/ckpt/model-00001-of-00002.safetensors")
    assert seen["resolve"]["model_dir"] == Path("/not/loaded/in/unit/test")
    assert seen["resolve"]["precision"] == "bf16"
    assert seen["kwargs"]["tactics"] == "/ckpt/tactics.json"


def test_apxinf_a_missing_apxinf_robo_names_what_to_install(monkeypatch):
    # ``None`` in sys.modules is how CPython marks an import as unavailable.
    monkeypatch.setitem(sys.modules, "apxinf_robo", None)

    with pytest.raises(ImportError, match="apxinf_robo"):
        OpenPIApxInfAdapter(
            _apxinf_model_cfg(), "cpu", processor=_FakeApxInfProcessor()
        )


def _cosmos3_sglang_cfg(batching_max_size=None):
    server = (
        {} if batching_max_size is None else {"batching_max_size": batching_max_size}
    )
    return OmegaConf.create(
        {
            "rollout": {
                "model": {"num_action_chunks": 16},
                "sglang": {"server": server},
            },
            "env": {"eval": {"env_type": "libero"}},
        }
    )


def test_cosmos3_sglang_requests_group_envs_by_prompt_up_to_the_server_batch():
    # The Cosmos3 server rejects a batch whose prompts tokenize to different
    # lengths, and one larger than its --batching-max-size.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(batching_max_size=2), rank=0)
    tasks = ["pick the bowl", "open the drawer", "pick the bowl", "pick the bowl"]

    groups = adapter.request_groups({"task_descriptions": tasks})

    assert groups == [[0, 2], [3], [1]]


def test_cosmos3_sglang_requests_default_to_one_env_like_sglang():
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    groups = adapter.request_groups({"task_descriptions": ["a", "a", "a"]})

    assert groups == [[0], [1], [2]]


def test_sglang_request_groups_round_trip_to_env_order():
    from rlinf.models.embodiment.sglang_adapter import (
        gather_env_rows,
        select_env_rows,
    )

    env_obs = {
        "main_images": torch.arange(5).view(5, 1),
        "task_descriptions": ["a", "b", "a", "a", "b"],
        "wrist_images": None,
        "_rlinf_stage_id": 1,
    }
    groups = [[0, 2], [3], [1, 4]]

    parts = [select_env_rows(env_obs, group) for group in groups]
    assert parts[2]["task_descriptions"] == ["b", "b"]
    assert parts[0]["_rlinf_stage_id"] == 1 and parts[0]["wrist_images"] is None

    # Each "request" answers with its envs' rows; gathering restores env order.
    order = [index for group in groups for index in group]
    actions = gather_env_rows([p["main_images"] * 10 for p in parts], order)
    info = gather_env_rows([{"x": {"y": p["main_images"]}} for p in parts], order)
    assert torch.equal(actions, torch.arange(5).view(5, 1) * 10)
    assert torch.equal(info["x"]["y"], torch.arange(5).view(5, 1))


def test_cosmos3_sglang_response_keeps_every_env_in_input_order():
    # The action envelope carries one ``data`` entry per env of the request.
    from rlinf.models.embodiment.cosmos3.sglang_adapter import Cosmos3SGLangAdapter

    adapter = Cosmos3SGLangAdapter(_cosmos3_sglang_cfg(), rank=0)

    def entry(input_index, gripper):
        values = np.zeros((16, 12), dtype=np.float32)
        values[:, 3] = 1.0  # rot6d columns (1, 0, 0) and (0, 1, 0): no rotation
        values[:, 7] = 1.0
        values[:, 9] = gripper
        return {"input_index": input_index, "action": {"values": values}}

    actions, _ = adapter.parse_response(
        {"data": [entry(1, 0.5), entry(0, -0.5)]}, state={}
    )

    assert actions.shape == (2, 16, 7)
    assert torch.all(actions[0, :, 6] == -0.5)
    assert torch.all(actions[1, :, 6] == 0.5)
