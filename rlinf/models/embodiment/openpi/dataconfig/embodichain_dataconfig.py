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

"""OpenPI data configuration for EmbodiChain joint-position datasets."""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import Any

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import embodichain_policy


def _pad_stats_array(value: Any, target_dim: int, pad_value: float):
    """Pad one normalization-stat vector without changing its dtype."""
    import numpy as np

    value = np.asarray(value)
    if value.shape[-1] >= target_dim:
        return value[..., :target_dim]
    pad_width = [(0, 0)] * value.ndim
    pad_width[-1] = (0, target_dim - value.shape[-1])
    return np.pad(value, pad_width, constant_values=pad_value)


def _adapt_norm_stats(
    norm_stats: Mapping[str, Any] | None,
    target_dim: int,
    *,
    pad_to_target: bool = True,
):
    """Map LeRobot field names and pad stats to the Pi0 action dimension.

    EmbodiChain recordings use ``observation.state`` and ``action`` names and
    store only the physical joint dimensions. Historical configs pad inputs
    before normalization; v2 keeps statistics at physical dimensions so the
    model transform pads only after discrete state tokenization.
    """
    if not norm_stats:
        return norm_stats

    aliases = {
        "state": ("state", "observation.state"),
        "actions": ("actions", "action"),
    }

    def adapt_array(value: Any, pad_value: float):
        import numpy as np

        array = np.asarray(value)
        if pad_to_target:
            return _pad_stats_array(array, target_dim, pad_value)
        return array[..., :target_dim]

    adapted = {}
    for output_key, candidates in aliases.items():
        stats = next((norm_stats[key] for key in candidates if key in norm_stats), None)
        if stats is None:
            continue
        adapted[output_key] = dataclasses.replace(
            stats,
            mean=adapt_array(stats.mean, 0.0),
            std=adapt_array(stats.std, 1.0),
            q01=(adapt_array(stats.q01, -1.0) if stats.q01 is not None else None),
            q99=(adapt_array(stats.q99, 1.0) if stats.q99 is not None else None),
        )
    return adapted


def _set_phase_norm_stats(stats: Any, phase_index: int, target_dim: int) -> Any:
    """Reserve one state-stat slot for a normalized episode phase.

    Elapsed phase is already scaled by :class:`EmbodiChainJointInputs` and must
    retain identity quantile statistics. Historical padded configs reserve the
    remaining model dimensions; state-v2 reserves only its physical state plus
    this phase slot.
    """
    import numpy as np

    if phase_index < 0 or phase_index >= target_dim:
        raise ValueError(
            f"Episode phase index {phase_index} is outside model state dimension "
            f"{target_dim}."
        )

    def _replace_field(value: Any, fill: float) -> Any:
        if value is None:
            return None
        array = np.asarray(value).copy()
        if array.shape[-1] <= phase_index:
            array = _pad_stats_array(array, target_dim, fill)
        array[..., phase_index] = fill
        return array

    return dataclasses.replace(
        stats,
        mean=_replace_field(stats.mean, 0.0),
        std=_replace_field(stats.std, 1.0),
        q01=_replace_field(stats.q01, -1.0),
        q99=_replace_field(stats.q99, 1.0),
    )


@dataclasses.dataclass(frozen=True)
class EmbodiChainJointDataConfig(DataConfigFactory):
    """Transform absolute joint targets from an EmbodiChain LeRobot dataset."""

    default_prompt: str | None = None
    output_action_dim: int = 14
    extra_delta_transform: bool = False
    delta_action_mask: tuple[bool, ...] | list[bool] | None = None
    include_phase_input: bool = False
    phase_scale: float = 600.0
    pad_inputs_to_model_dim: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        if (
            self.include_phase_input
            and model_config.model_type == _model.ModelType.PI05
            and not model_config.discrete_state_input
        ):
            raise ValueError(
                "PI 0.5 elapsed-step input requires discrete_state_input=True; "
                "use pi05_embodichain_joint_state_v2."
            )
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.cam_high",
                        "observation/state": "observation.state",
                        "actions": "action",
                        **(
                            {"observation/episode_step": "annotation.episode_step"}
                            if self.include_phase_input
                            else {}
                        ),
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                embodichain_policy.EmbodiChainJointInputs(
                    action_dim=model_config.action_dim,
                    output_action_dim=self.output_action_dim,
                    model_type=model_config.model_type,
                    include_phase_input=self.include_phase_input,
                    phase_scale=self.phase_scale,
                    pad_inputs_to_model_dim=self.pad_inputs_to_model_dim,
                )
            ],
            outputs=[
                embodichain_policy.EmbodiChainJointOutputs(
                    output_action_dim=self.output_action_dim
                )
            ],
        )

        if not self.extra_delta_transform:
            if self.delta_action_mask is None:
                delta_mask = _transforms.make_bool_mask(self.output_action_dim)
            else:
                delta_mask = tuple(bool(value) for value in self.delta_action_mask)
                if len(delta_mask) != self.output_action_dim:
                    raise ValueError(
                        "delta_action_mask must match output_action_dim: "
                        f"expected {self.output_action_dim}, got {len(delta_mask)}."
                    )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_mask)],
                outputs=[_transforms.AbsoluteActions(delta_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        base_config = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # ``action`` is the raw LeRobot feature. The repack transform
            # renames it to ``actions`` after the dataset loader creates the
            # horizon window.
            action_sequence_keys=("action",),
        )
        norm_target_dim = (
            model_config.action_dim
            if self.pad_inputs_to_model_dim
            else self.output_action_dim
        )
        norm_stats = _adapt_norm_stats(
            base_config.norm_stats,
            norm_target_dim,
            pad_to_target=self.pad_inputs_to_model_dim,
        )
        if self.include_phase_input and norm_stats and "state" in norm_stats:
            norm_stats = dict(norm_stats)
            norm_stats["state"] = _set_phase_norm_stats(
                norm_stats["state"],
                phase_index=self.output_action_dim,
                target_dim=(
                    model_config.action_dim
                    if self.pad_inputs_to_model_dim
                    else self.output_action_dim + 1
                ),
            )
        return dataclasses.replace(base_config, norm_stats=norm_stats)
