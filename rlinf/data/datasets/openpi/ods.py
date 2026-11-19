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

"""OpenPI encoding and bounded consumption of an external ODS stream."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.scheduler import Channel


@dataclass(frozen=True)
class ODSStream:
    """Channels and frozen producer metadata, without SDK runtime objects."""

    requests: Channel
    batches: Channel
    metadata: dict[str, Any]


def wait_channel(
    work: Any, timeout: float, *, progress: Callable[[], None] | None = None
) -> Any:
    """Wait with a deadline, polling optional owner health and cancellation."""
    deadline = time.monotonic() + timeout
    while not work.done():
        if progress is not None:
            progress()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ODS channel did not respond within {timeout} seconds.")
        time.sleep(0.01)
    return work.wait()


def sha256_file(path: str | Path) -> str:
    """Hash an immutable training asset in bounded blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OpenPIODSEncoder:
    """Apply the selected OpenPI transforms without constructing LeRobot data."""

    version = "openpi-ods-v1"

    def __init__(self, cfg: Any) -> None:
        from openpi import transforms

        from rlinf.data.datasets.openpi.official_sft_data_loader import (
            _validate_openpi_model_shape,
        )
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from rlinf.models.embodiment.openpi.transforms.pipeline import (
            select_openpi_norm_stats,
        )

        kwargs = OmegaConf.to_container(cfg.actor.model.openpi_data, resolve=True)
        path = kwargs.get("norm_stats_path")
        if not path or not Path(path).is_file():
            raise ValueError("ODS SFT requires an existing train-only norm_stats file.")
        config = get_openpi_config(
            cfg.actor.model.openpi.config_name,
            model_path=cfg.actor.model.model_path,
            data_kwargs=kwargs,
        )
        _validate_openpi_model_shape(cfg.actor.model, config)
        self.data_config = config.data.create(config.assets_dirs, config.model)
        stats = select_openpi_norm_stats(
            self.data_config.norm_stats, norm_stats_path=path
        )
        self.norm_sha256 = sha256_file(path)
        action_stats = stats["actions"]
        spread = (
            action_stats.q99 - action_stats.q01
            if self.data_config.use_quantile_norm
            else action_stats.std
        )
        self._constant_actions = np.asarray(spread) == 0
        self._constant_values = np.asarray(
            action_stats.q01
            if self.data_config.use_quantile_norm
            else action_stats.mean
        )
        self._normalization_warned = False
        contract = {
            "version": self.version,
            "model": OmegaConf.to_container(cfg.actor.model.openpi, resolve=True),
            "data": {
                key: value for key, value in kwargs.items() if key != "norm_stats_path"
            },
        }
        self.contract_sha256 = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()
        self._repack = transforms.compose(self.data_config.repack_transforms.inputs)
        self._transform = transforms.compose(
            [
                *self.data_config.data_transforms.inputs,
                transforms.Normalize(
                    stats, use_quantiles=self.data_config.use_quantile_norm
                ),
                *self.data_config.model_transforms.inputs,
            ]
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        import jax

        from rlinf.models.embodiment.openpi.modules.model import Observation

        if not self._normalization_warned and self._constant_actions.any():
            targets = batch["actions"][batch["action_mask"]][:, self._constant_actions]
            expected = torch.as_tensor(self._constant_values[self._constant_actions])
            if bool(((targets - expected).abs() > 1e-6).any()):
                from rlinf.utils.logging import get_logger

                get_logger().warning(
                    "ODS action statistics have zero range in dimensions %s, but source targets vary. Use fixed native train-only statistics.",
                    np.flatnonzero(self._constant_actions).tolist(),
                )
                self._normalization_warned = True
        rows = []
        for index, prompt in enumerate(batch["prompts"]):
            row = self._repack(
                {
                    "observation.images.front": batch["front"][index].numpy(),
                    "observation.images.wrist": batch["wrist"][index].numpy(),
                    "observation.state": batch["state"][index].numpy(),
                    "action": batch["actions"][index].numpy(),
                }
            )
            row["prompt"] = prompt
            rows.append(self._transform(row))
        prepared = jax.tree.map(
            lambda *items: torch.as_tensor(np.stack([np.asarray(x) for x in items])),
            *rows,
        )
        return {
            "observation": Observation.from_dict(prepared),
            "actions": prepared["actions"],
            "action_mask": batch["action_mask"].clone(),
        }


class OpenPIODSDataLoader:
    """Consume one rank's share of exactly one global optimizer batch.

    The worker prepares and collectively admits the batch before iteration.
    This loader has no dataset length and never creates sampling subprocesses.
    """

    def __init__(self, cfg: Any, stream: ODSStream, rank: int, world_size: int) -> None:
        self.stream, self.rank, self.world_size = stream, rank, world_size
        self.encoder = OpenPIODSEncoder(cfg)
        self.micro_batch_size = int(cfg.actor.micro_batch_size)
        self.local_batch_size = int(cfg.actor.global_batch_size) // world_size
        self.timeout = float(cfg.data.ods.get("data_timeout", 1800.0))
        if self.local_batch_size % self.micro_batch_size:
            raise ValueError("ODS local batch must be divisible by micro_batch_size.")
        if stream.metadata["norm_stats_sha256"] != self.encoder.norm_sha256:
            raise ValueError(
                "ODS producer and actor normalization fingerprints differ."
            )
        if int(stream.metadata["world_size"]) != world_size:
            raise ValueError("ODS stream and actor rank topology differ.")
        self._packet: dict[str, Any] | None = None
        self._offset = 0
        self._step = int(stream.metadata["start_step"])
        self._committed_step = self._step
        self.wait_seconds = 0.0
        self.input_active_seconds = 0.0
        self._encoded: list[dict[str, Any]] = []
        self._audit: dict[str, Any] = {}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def prepare_step(self, step: int) -> None:
        """Request one batch; retries cannot silently consume another quota."""
        if self._packet is not None or step != self._committed_step + 1:
            raise RuntimeError(
                f"Expected next ODS step {self._committed_step + 1}, found {step}."
            )
        started = time.monotonic()
        wait_channel(
            self.stream.requests.put(
                {"kind": "request", "rank": self.rank, "step": step}, async_op=True
            ),
            self.timeout,
        )
        packet = wait_channel(
            self.stream.batches.get(key=self.rank, async_op=True), self.timeout
        )
        self.wait_seconds = time.monotonic() - started
        if packet["kind"] != "batch":
            raise RuntimeError(f"ODS source failed: {packet.get('error', packet)}")
        if (
            packet["step"] != step
            or packet["session_id"] != self.stream.metadata["session_id"]
        ):
            raise RuntimeError("ODS received a batch from a different step or session.")
        batch = packet["batch"]
        if batch["actions"].shape[0] != self.local_batch_size:
            raise ValueError("ODS producer returned an incomplete rank batch.")
        if not bool(batch["action_mask"].any(dim=1).all()):
            raise ValueError("ODS source returned a sample without valid supervision.")
        self._packet, self._offset, self._step = packet, 0, step
        self._audit = packet.get("audit", {})
        self._encoded = [
            self.encoder(
                {
                    key: value[start : start + self.micro_batch_size]
                    for key, value in batch.items()
                }
            )
            for start in range(0, self.local_batch_size, self.micro_batch_size)
        ]
        self.wait_seconds = time.monotonic() - started
        self.input_active_seconds = max(
            0.0, self.wait_seconds - float(packet.get("source_wait_seconds", 0.0))
        )

    def __next__(self) -> dict[str, Any]:
        if self._packet is None or self._offset >= self.local_batch_size:
            raise RuntimeError("Prepare an ODS optimizer batch before consuming it.")
        encoded = self._encoded[self._offset // self.micro_batch_size]
        self._offset += self.micro_batch_size
        return encoded

    def commit_step(self, active_seconds: float) -> None:
        """Acknowledge only after all local microbatches and optimization succeed."""
        if self._packet is None or self._offset != self.local_batch_size:
            raise RuntimeError(
                "Cannot commit a partially consumed ODS optimizer batch."
            )
        wait_channel(
            self.stream.requests.put(
                {
                    "kind": "commit",
                    "rank": self.rank,
                    "step": self._step,
                    "active_seconds": active_seconds,
                    "wait_seconds": self.wait_seconds,
                },
                async_op=True,
            ),
            self.timeout,
        )
        self._committed_step, self._packet = self._step, None
        self._encoded.clear()

    def checkpoint_metadata(self) -> dict[str, Any]:
        """Describe training-state-only recovery without claiming source rollback."""
        return {
            **self.stream.metadata,
            "encoder_version": self.encoder.version,
            "encoder_sha256": self.encoder.contract_sha256,
            "resume_semantics": "training_state_only",
            "committed_step": self._committed_step,
            "supply_audit_at_delivery": self._audit,
        }

    def validate_resume(self, path: str) -> None:
        """Validate a prior ODS contract; an offline checkpoint starts a new stream."""
        file = Path(path) / "ods_source.json"
        if not file.exists():
            return
        saved = json.loads(file.read_text())
        current = self.checkpoint_metadata()
        for key in ("schema_sha256", "norm_stats_sha256", "encoder_sha256"):
            if saved[key] != current[key]:
                raise ValueError(f"ODS resume changes the {key} contract.")


def build_ods_openpi_sft_dataloader(
    cfg: Any, stream: ODSStream, world_size: int, rank: int
) -> tuple[OpenPIODSDataLoader, Any]:
    """Build an SDK-free actor-side loader for the selected ODS stream."""
    loader = OpenPIODSDataLoader(cfg, stream, rank, world_size)
    return loader, loader.encoder.data_config
