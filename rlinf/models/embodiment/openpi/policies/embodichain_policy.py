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

"""OpenPI transforms for EmbodiChain joint-position demonstrations."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Convert an image-like value to uint8 HWC RGB."""
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(
            "EmbodiChain camera observations must have shape [H,W,C] or [C,H,W], "
            f"got {tuple(image.shape)}."
        )
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(
            "EmbodiChain camera observations must have RGB/RGBA channels, "
            f"got {tuple(image.shape)}."
        )
    if np.issubdtype(image.dtype, np.floating):
        finite = bool(np.isfinite(image).all()) if image.size else True
        if finite and image.size and image.min() >= 0.0 and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class EmbodiChainJointInputs(transforms.DataTransformFn):
    """Map an EmbodiChain LeRobot sample to a Pi0/Pi05 observation."""

    action_dim: int
    output_action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05
    include_phase_input: bool = False
    phase_scale: float = 600.0
    pad_inputs_to_model_dim: bool = True

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=np.float32)
        if (
            not self.pad_inputs_to_model_dim
            and state.shape[-1] != self.output_action_dim
        ):
            raise ValueError(
                "EmbodiChain physical state must match output_action_dim before "
                "phase is appended: "
                f"state_dim={state.shape[-1]}, "
                f"output_action_dim={self.output_action_dim}."
            )
        if state.shape[-1] > self.action_dim:
            raise ValueError(
                "EmbodiChain state has more dimensions than the OpenPI model: "
                f"shape={state.shape}, action_dim={self.action_dim}."
            )
        if self.include_phase_input:
            phase = np.asarray(
                data.get("observation/episode_step", 0.0), dtype=np.float32
            )
            phase = np.asarray(phase / max(float(self.phase_scale), 1.0))
            phase = np.broadcast_to(phase, state.shape[:-1])[..., None]
            state = np.concatenate((state, phase), axis=-1)
        if self.pad_inputs_to_model_dim:
            state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/image"])
        zero_image = np.zeros_like(base_image)
        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, zero_image, zero_image)
            image_masks = (np.True_, np.False_, np.False_)
        elif self.model_type == _model.ModelType.PI0_FAST:
            image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, zero_image, zero_image)
            image_masks = (np.True_, np.False_, np.False_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        if "actions" in data:
            actions = data["actions"]
            if torch.is_tensor(actions):
                actions = actions.detach().cpu().numpy()
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape[-1] != self.output_action_dim:
                raise ValueError(
                    "EmbodiChain actions do not match the configured environment "
                    f"dimension: shape={actions.shape}, "
                    f"output_action_dim={self.output_action_dim}."
                )
            inputs["actions"] = (
                transforms.pad_to_dim(actions, self.action_dim)
                if self.pad_inputs_to_model_dim
                else actions
            )

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class EmbodiChainJointOutputs(transforms.DataTransformFn):
    """Trim padded OpenPI outputs back to the environment joint dimension."""

    output_action_dim: int

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.output_action_dim:
            raise ValueError(
                "OpenPI output has fewer dimensions than the EmbodiChain action "
                f"space: shape={actions.shape}, output_action_dim={self.output_action_dim}."
            )
        return {"actions": actions[..., : self.output_action_dim]}
