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

"""Run OpenPI inference in a spawned process with its own CUDA context."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import traceback
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np
import torch

_OBSERVATION_KEYS = (
    "main_images",
    "wrist_images",
    "extra_view_images",
    "states",
    "episode_steps",
    "task_descriptions",
)


def _precision_metadata() -> dict[str, Any]:
    return {
        "model_process_pid": os.getpid(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_flags": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
    }


def _copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Serialize only policy observations, using owned CPU arrays."""
    copied = {}
    for key in _OBSERVATION_KEYS:
        if key not in observation:
            continue
        value = observation[key]
        if value is None:
            copied[key] = None
        elif key == "task_descriptions":
            copied[key] = list(value)
        else:
            if torch.is_tensor(value):
                value = value.detach().cpu()
                if value.dtype in (torch.bfloat16, torch.float16):
                    value = value.to(torch.float32)
                value = value.numpy()
            copied[key] = np.array(value, copy=True)
    for key in ("main_images", "states", "task_descriptions"):
        if key not in copied:
            raise KeyError(f"OpenPI observation is missing {key!r}.")
    return copied


def _model_process(
    connection: Connection,
    checkpoint_dir: str,
    norm_stats_path: str,
    model_kwargs: dict[str, Any],
    noise_seed: int | None,
    zero_noise: bool,
) -> None:
    """Own checkpoint staging, model construction, and the inference RNG."""
    staging = None
    operation = "initialize"
    try:
        from toolkits.lerobot.evaluate_embodichain_openpi import (
            _build_model,
            _resolve_norm_stats_directory,
            _stage_checkpoint,
        )

        checkpoint_root, staging = _stage_checkpoint(checkpoint_dir, norm_stats_path)
        model, _, _ = _build_model(
            checkpoint_root,
            norm_stats_dir=checkpoint_root / "RLinf" / "embodichain_joint",
            **model_kwargs,
        )
        torch.set_float32_matmul_precision("high")
        noise_rng = None
        if noise_seed is not None:
            noise_rng = torch.Generator(device=model.device).manual_seed(noise_seed)
        stats_file = _resolve_norm_stats_directory(norm_stats_path) / "norm_stats.json"
        provenance = {
            "checkpoint_dir": str(Path(checkpoint_dir).expanduser().resolve()),
            "norm_stats_path": str(stats_file),
            "norm_stats_sha256": hashlib.sha256(stats_file.read_bytes()).hexdigest(),
            **model_kwargs,
            "action_horizon": model.action_horizon,
            "model_action_dim": model.action_dim,
            "noise_seed": noise_seed,
            "zero_noise": zero_noise,
        }
        connection.send(
            {"type": "ready", **_precision_metadata(), "provenance": provenance}
        )
        calls = 0
        while True:
            request = connection.recv()
            operation = request["command"]
            if operation == "close":
                connection.send({"type": "closed", "inference_calls": calls})
                break
            if operation != "predict":
                raise ValueError(f"Unknown model command {operation!r}.")
            observation = request["observation"]
            if any(key not in _OBSERVATION_KEYS for key in observation):
                raise ValueError("Model request contains non-observation fields.")
            actions, _ = model.predict_action_batch(
                observation,
                mode="eval",
                rng=noise_rng,
                noise=(
                    torch.zeros(
                        len(observation["states"]),
                        model.action_horizon,
                        model.action_dim,
                        device=model.device,
                    )
                    if zero_noise
                    else None
                ),
            )
            actions = actions.detach().to("cpu", dtype=torch.float32).numpy().copy()
            calls += 1
            connection.send(
                {
                    "type": "prediction",
                    "request_id": request["request_id"],
                    "actions": actions,
                    "shape": list(actions.shape),
                    "finite": bool(np.isfinite(actions).all()),
                    "inference_calls": calls,
                    **_precision_metadata(),
                }
            )
    except BaseException:
        try:
            connection.send(
                {
                    "type": "error",
                    "operation": operation,
                    "traceback": traceback.format_exc(),
                    **_precision_metadata(),
                }
            )
        except (EOFError, OSError):
            pass
    finally:
        try:
            if staging is not None:
                staging.cleanup()
        finally:
            connection.close()


class OpenPIProcessPredictor:
    """Own an OpenPI model process and return CPU action tensors.

    Construction waits for the model's ready response. Predictions send copied
    RGB, qpos, elapsed-step, and prompt observations; other fields are omitted.
    The child retains its inference precision and RNG across episodes without
    changing the caller's precision. The owner must call ``close``; startup and
    prediction failures clean up the child before raising.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        config_name: str,
        output_action_dim: int,
        norm_stats_path: str,
        num_steps: int,
        device: str,
        include_phase_input: bool = False,
        phase_scale: float = 600.0,
        delta_action_mask: list[bool] | None = None,
        noise_seed: int | None = None,
        zero_noise: bool = False,
        eval_sft_image_crop: bool = False,
        startup_timeout_s: float = 300.0,
        prediction_timeout_s: float = 120.0,
    ) -> None:
        if startup_timeout_s <= 0 or prediction_timeout_s <= 0:
            raise ValueError("Model process timeouts must be positive.")
        context = mp.get_context("spawn")
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_model_process,
            args=(
                child_connection,
                checkpoint_dir,
                norm_stats_path,
                {
                    "config_name": config_name,
                    "output_action_dim": output_action_dim,
                    "num_steps": num_steps,
                    "device": device,
                    "include_phase_input": include_phase_input,
                    "phase_scale": phase_scale,
                    "delta_action_mask": delta_action_mask,
                    "eval_sft_image_crop": eval_sft_image_crop,
                },
                noise_seed,
                zero_noise,
            ),
        )
        self._started = False
        self._closed = False
        self._prediction_timeout_s = prediction_timeout_s
        self._request_id = 0
        self._output_action_dim = output_action_dim
        self.metadata: dict[str, Any] = {}
        try:
            self._process.start()
            self._started = True
            child_connection.close()
            response = self._receive("initialize", startup_timeout_s)
            if response["type"] != "ready":
                raise RuntimeError("OpenPI model process did not send ready.")
            self.metadata = {
                key: value for key, value in response.items() if key != "type"
            }
            self.action_horizon: int = response["provenance"]["action_horizon"]
            self.action_dim: int = response["provenance"]["model_action_dim"]
        except BaseException:
            child_connection.close()
            self._shutdown(graceful=False)
            raise

    @property
    def process_pid(self) -> int:
        """Return the owned model process id."""
        return int(self._process.pid)

    def _receive(self, operation: str, timeout_s: float) -> dict[str, Any]:
        if not self._connection.poll(timeout_s):
            raise TimeoutError(
                f"OpenPI model process {self.process_pid} timed out during "
                f"{operation} after {timeout_s:g} seconds."
            )
        try:
            response = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(
                f"OpenPI model process {self.process_pid} disconnected during "
                f"{operation} (exit code {self._process.exitcode})."
            ) from error
        if response["type"] == "error":
            raise RuntimeError(
                f"OpenPI model process {self.process_pid} failed during "
                f"{response['operation']}:\n{response['traceback']}"
            )
        return response

    def predict_action_batch(
        self, observation: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return decoded CPU actions and model-process prediction diagnostics."""
        if self._closed:
            raise RuntimeError("OpenPI model process is closed.")
        copied = _copy_observation(observation)
        try:
            self._connection.send(
                {
                    "command": "predict",
                    "request_id": self._request_id,
                    "observation": copied,
                }
            )
            response = self._receive("predict", self._prediction_timeout_s)
            if (
                response["type"] != "prediction"
                or response["request_id"] != self._request_id
            ):
                raise RuntimeError(
                    "OpenPI model process returned an unexpected response."
                )
            self._request_id += 1
            actions = np.array(response.pop("actions"), dtype=np.float32, copy=True)
            expected_shape = (
                len(copied["states"]),
                self.action_horizon,
                self._output_action_dim,
            )
            if actions.shape != expected_shape:
                raise RuntimeError(
                    f"OpenPI actions have shape {actions.shape}; expected {expected_shape}."
                )
            if not response["finite"] or not np.isfinite(actions).all():
                raise RuntimeError("OpenPI model process returned non-finite actions.")
            return torch.from_numpy(actions), response
        except BaseException:
            self._shutdown(graceful=False)
            raise

    def _shutdown(self, *, graceful: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if graceful and self._started and self._process.is_alive():
                try:
                    self._connection.send({"command": "close"})
                    self.metadata["close_response"] = self._receive("close", 5.0)
                except (RuntimeError, TimeoutError, OSError) as error:
                    self.metadata["close_error"] = str(error)
        finally:
            self._connection.close()
            if self._started:
                self._process.join(timeout=5.0 if graceful else 0.1)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5.0)
                self.metadata["process_exit_code"] = self._process.exitcode
                self.metadata["process_alive_after_close"] = self._process.is_alive()

    def close(self) -> None:
        """Close the protocol and join the child; repeated calls are harmless."""
        self._shutdown(graceful=True)
