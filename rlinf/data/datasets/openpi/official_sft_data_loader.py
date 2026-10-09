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

"""Adapters for the official OpenPI PyTorch SFT data loader."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import OmegaConf

from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    norm_stats_path_from_data_kwargs,
    select_openpi_norm_stats,
)


def build_official_openpi_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[Any, Any]:
    """Build the SFT loader provided by OpenPI for a LeRobot dataset."""
    repo_id = resolve_lerobot_repo_id(data_paths)
    if repo_id is None:
        raise ValueError(
            "OpenPI SFT requires data.train_data_paths to be set to a local "
            "dataset path or LeRobot repo id."
        )
    _validate_local_lerobot_root(repo_id)

    import openpi.training.data_loader as openpi_data_loader

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    batch_size = cfg.actor.micro_batch_size
    if eval_dataset:
        batch_size = cfg.actor.get("eval_batch_size", batch_size)

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    config = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=batch_size * world_size,
        repo_id=repo_id,
        data_kwargs=data_kwargs,
    )
    # Fail early with the configured path when an explicit stats file is
    # missing. OpenPI otherwise reports this several layers deeper in the
    # loader, which obscures the split/path mistake.
    data_config = config.data.create(config.assets_dirs, config.model)
    select_openpi_norm_stats(
        data_config.norm_stats,
        norm_stats_path=norm_stats_path_from_data_kwargs(data_kwargs),
    )
    config = dataclasses.replace(
        config,
        num_workers=int(
            OmegaConf.select(cfg, "data.num_workers", default=config.num_workers)
        ),
        seed=int(OmegaConf.select(cfg, "actor.seed", default=config.seed)),
    )
    _validate_openpi_model_shape(model_cfg, config)

    sampling_plan = OmegaConf.select(cfg, "data.frame_sampling_plan", default=None)
    if sampling_plan is not None and not eval_dataset:
        if data_config.rlds_data_dir is not None:
            raise ValueError("data.frame_sampling_plan requires a LeRobot dataset.")
        dataset = openpi_data_loader.create_torch_dataset(
            data_config, config.model.action_horizon, config.model
        )
        weights = _load_frame_sampling_weights(sampling_plan, repo_id, len(dataset))
        dataset = openpi_data_loader.transform_dataset(dataset, data_config)
        sampler = WeightedFrameSampler(weights, world_size, rank, config.seed)
        torch_loader = openpi_data_loader.TorchDataLoader(
            dataset,
            local_batch_size=batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            seed=config.seed,
            framework="pytorch",
        )
        data_loader = openpi_data_loader.DataLoaderImpl(data_config, torch_loader)
    else:
        data_loader = openpi_data_loader.create_data_loader(
            config, framework="pytorch", shuffle=not eval_dataset
        )
    return data_loader, data_loader.data_config()


class WeightedFrameSampler(torch.utils.data.Sampler[int]):
    """Draw weighted frames from disjoint rank partitions.

    Each iteration partitions a seeded frame permutation across ranks, drops
    its remainder, and samples with replacement within that rank's partition.
    OpenPI's infinite loader starts a new iteration at each rollover, so the
    sampler advances its epoch itself. ``seed`` and the iteration count make
    the stream reproducible; a resumed loader starts a new stream.
    """

    def __init__(
        self,
        weights: list[float],
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        if (
            self.weights.ndim != 1
            or not torch.isfinite(self.weights).all()
            or not (self.weights > 0).all()
        ):
            raise ValueError("Frame sampling weights must be finite and positive.")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Frame sampler rank must be within a positive world_size.")
        self.num_samples = len(self.weights) // world_size
        if self.num_samples == 0:
            raise ValueError("Frame sampling requires at least one frame per rank.")
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        partition_rng = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.weights), generator=partition_rng)
        partition = permutation[
            self.rank : self.num_samples * self.world_size : self.world_size
        ]
        draw_rng = torch.Generator().manual_seed(
            self.seed + self.epoch * self.world_size + self.rank
        )
        selected = torch.multinomial(
            self.weights[partition],
            self.num_samples,
            replacement=True,
            generator=draw_rng,
        )
        self.epoch += 1
        return iter(partition[selected].tolist())


def _load_frame_sampling_weights(
    plan_path: str, repo_id: str, num_frames: int
) -> list[float]:
    plan = json.loads(Path(plan_path).expanduser().read_text())
    planned_repo = plan["dataset_repo_id"]
    actual_path = Path(repo_id).expanduser()
    if actual_path.exists():
        matches = Path(planned_repo).expanduser().resolve() == actual_path.resolve()
    else:
        matches = planned_repo == repo_id
    if not matches:
        raise ValueError(
            "Frame sampling plan dataset_repo_id does not match data paths."
        )
    weights = plan["weights"]
    if plan["num_frames"] != num_frames or len(weights) != num_frames:
        raise ValueError(
            "Frame sampling plan must contain one weight per dataset frame."
        )
    return weights


def get_official_openpi_sft_num_batches(data_loader: Any) -> int:
    """Return the inner PyTorch ``DataLoader`` length used by OpenPI."""
    openpi_loader = getattr(data_loader, "_data_loader", None)
    torch_loader = getattr(openpi_loader, "_data_loader", None) or getattr(
        openpi_loader, "torch_loader", None
    )
    if torch_loader is None:
        raise TypeError(
            "OpenPI dataloader does not expose an inner torch DataLoader; "
            "cannot infer steps per epoch from len()."
        )
    return len(torch_loader)


def is_official_openpi_sft_dataloader(data_loader: Any) -> bool:
    """Return whether ``data_loader`` has OpenPI's loader wrapper layout."""
    return getattr(data_loader, "_data_loader", None) is not None


def _validate_local_lerobot_root(repo_id: str) -> None:
    """Reject a split directory instead of a concrete LeRobot dataset root."""
    path = Path(repo_id).expanduser()
    if not path.exists() or not path.is_dir():
        return
    if not (path / "meta" / "info.json").is_file():
        raise ValueError(
            "OpenPI SFT local data path must contain meta/info.json directly: "
            f"{path}. Pass the concrete LeRobot dataset directory, not its "
            "split parent."
        )


def _validate_openpi_model_shape(model_cfg: Any, openpi_config: Any) -> None:
    """Keep the local Pi0 architecture consistent with the OpenPI config.

    The official loader sizes the SFT action window from
    ``TrainConfig.model.action_horizon``, not ``num_action_chunks`` (that field
    is the env-executed chunk). ``get_model`` already builds the network from
    the same TrainConfig unless YAML overrides ``openpi.action_horizon``.
    """
    yaml_horizon = OmegaConf.select(model_cfg, "openpi.action_horizon", default=None)
    official_horizon = int(openpi_config.model.action_horizon)
    if yaml_horizon is not None and int(yaml_horizon) != official_horizon:
        raise ValueError(
            "openpi SFT data uses TrainConfig.model.action_horizon="
            f"{official_horizon} from {model_cfg.openpi.config_name}; "
            f"openpi.action_horizon={int(yaml_horizon)} would build a different "
            "network. Unset it, or change the TrainConfig."
        )

    local_action_dim = int(model_cfg.openpi.model_action_dim)
    official_action_dim = int(openpi_config.model.action_dim)
    if local_action_dim != official_action_dim:
        raise ValueError(
            "openpi SFT model action dim must match the official OpenPI "
            f"config: actor.model.openpi.model_action_dim={local_action_dim}, "
            f"{model_cfg.openpi.config_name}.model.action_dim="
            f"{official_action_dim}."
        )

    yaml_state_input = OmegaConf.select(
        model_cfg, "openpi.discrete_state_input", default=None
    )
    official_state_input = bool(openpi_config.model.discrete_state_input)
    if yaml_state_input is not None and bool(yaml_state_input) != official_state_input:
        raise ValueError(
            "openpi.discrete_state_input must match the SFT tokenizer config: "
            f"YAML={yaml_state_input}, "
            f"{model_cfg.openpi.config_name}={official_state_input}. "
            "Select a TrainConfig with the intended state conditioning."
        )
