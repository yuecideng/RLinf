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

"""Tests for the FSDP device mesh, the process groups derived from it, and the
units FSDP2 shards a model into.

Two properties of that mesh are easy to get wrong and silent when they are, so
they are pinned here: the timeout its collectives run under, and which of its
dimensions a gradient norm reduces over.

The timeout. ``init_device_mesh`` creates the default process group itself when
none exists, using whatever watchdog timeout the backend ships with — 30 minutes
for NCCL/Gloo, about 60 for HCCL, in every case below the 180 minutes RLinf
gives its own groups. A mesh dimension that spans the whole world reuses that
group, so every FSDP collective inherits that timeout, and no environment
variable can raise it. ``create_device_mesh`` therefore creates the group first,
with the same ``RLINF_TIMEOUT`` that RLinf applies to its own inter-worker
groups.

The reduction group. FSDP leaves each rank only its slice of every gradient, so
a norm over one of them is not the gradient's norm. ``gradient_reduction_group``
picks the dimension the shards are spread over, which stays ``fsdp`` even once a
replicated ``ddp`` dimension exists beside it.

The units. A model that reads embedding weights directly sets
``_fsdp_wrap_embeddings = False`` so they are gathered with the enclosing unit.
"""

import json
import logging
import os
import socket
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy.checkpoint import (
    checkpoint_communication_group,
    should_save_training_state,
    validate_checkpoint_communication_backend,
    validate_checkpoint_format,
)
from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model, create_device_mesh
from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Worker
from rlinf.scheduler.cluster import Cluster

# The timeout a bare init_process_group() installs is backend-specific -- 30
# minutes for NCCL and Gloo, 3636 seconds for HCCL on Ascend -- so no test here
# may hardcode it. What every backend has in common is that the value is not the
# one RLINF_TIMEOUT asked for, and that it is below RLinf's own 180-minute
# default. CONFIGURED_TIMEOUT is an arbitrary value distinguishable from all of
# them.
CONFIGURED_TIMEOUT = timedelta(minutes=97)
RLINF_DEFAULT_TIMEOUT = timedelta(minutes=180)


def free_port() -> str:
    """Reserve an ephemeral port for the rendezvous.

    A fixed port would collide with anything else on a shared CI runner and turn
    every test in this file into an unrelated bind error.

    Returns:
        str: A port that was free a moment ago.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


@pytest.fixture
def single_rank_env(monkeypatch):
    """Give a lone pytest process enough of a rendezvous to build a 1-D mesh.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to scope the environment
            variables and the device type to this test.

    Yields:
        None: Control returns to the test with the environment in place.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", free_port())
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Worker.torch_device_type is only populated inside a live Worker; the mesh
    # itself does not care which device type it is built over.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu", raising=False)
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def group_timeout(group: dist.ProcessGroup) -> timedelta:
    """Read the watchdog timeout a process group was created with.

    Args:
        group (dist.ProcessGroup): The group to inspect.

    Returns:
        timedelta: The timeout carried by the group's backend options. This is
            the same ``_timeout`` attribute ``DeviceMesh`` reads when it forwards
            a timeout to its sub-groups.

    Raises:
        AssertionError: If no backend exposes a timeout. This fails rather than
            skipping, because a skip would leave every assertion in this file
            green on a platform where the timeout cannot be read at all, which
            is exactly when they stop covering anything.
    """
    for device_type in group._device_types:
        options = getattr(group._get_backend(device_type), "options", None)
        timeout = getattr(options, "_timeout", None)
        if timeout is not None:
            return timeout
    raise AssertionError(
        f"no backend of {group} over {list(group._device_types)} exposes a timeout"
    )


def test_mesh_group_uses_the_configured_timeout(single_rank_env, monkeypatch):
    """The mesh's process group carries RLINF_TIMEOUT, not the backend default."""
    monkeypatch.setenv(
        "RLINF_TIMEOUT", str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    )

    mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == CONFIGURED_TIMEOUT


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("dcp", "dcp"),
        (" DCP ", "dcp"),
        ("local_shard", "local_shard"),
        ("LOCAL_SHARD", "local_shard"),
    ],
)
def test_checkpoint_format_accepts_supported_values(value, expected):
    """Checkpoint format aliases normalize to the two supported layouts."""
    assert validate_checkpoint_format(value) == expected


def test_checkpoint_format_rejects_unknown_values():
    """A typo must not silently select the DCP save path."""
    with pytest.raises(ValueError, match="Unsupported FSDP checkpoint format"):
        validate_checkpoint_format("local-shards")


@pytest.mark.parametrize(
    ("interval", "step", "force", "expected"),
    [
        (None, 100, False, True),
        (1000, 100, False, False),
        (1000, 1000, False, True),
        (1000, 750, True, True),
        (1000, 0, False, True),
    ],
)
def test_training_state_save_interval(interval, step, force, expected):
    """Exports can be more frequent than resumable training states."""
    assert should_save_training_state(step, interval, force=force) is expected


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "1000"])
def test_training_state_save_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="null or a positive integer"):
        should_save_training_state(100, interval)


@pytest.fixture
def cpu_checkpoint_platform(single_rank_env, monkeypatch):
    """Model the accelerator API edge while checkpoint tensors stay on CPU."""
    platform = SimpleNamespace(
        set_device=lambda _: None,
        current_device=lambda: torch.device("cpu"),
        synchronize=lambda: None,
        ipc_collect=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: False,
    )
    monkeypatch.setattr(Worker, "torch_platform", platform)


def _checkpoint_manager(
    interval=1000,
    checkpoint_format="dcp",
    full_weights=True,
    communication_backend=None,
):
    cfg = OmegaConf.create(
        {
            "model": {"precision": "fp32"},
            "fsdp_config": {
                "strategy": "fsdp",
                "amp_autocast": {"enabled": False},
                "training_state_save_interval": interval,
                "checkpoint_format": checkpoint_format,
                "save_full_model_weights": full_weights,
                "checkpoint_communication_backend": communication_backend,
            },
        }
    )
    manager = FSDPModelManager(cfg, world_size=1, rank=0)
    manager.model = torch.nn.Linear(4, 3)
    manager.optimizer = torch.optim.AdamW(manager.model.parameters(), lr=0.01)
    manager.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        manager.optimizer, step_size=2, gamma=0.5
    )
    _checkpoint_update(manager)
    return manager


def _checkpoint_update(manager):
    manager.optimizer.zero_grad()
    manager.model(
        torch.arange(8, dtype=torch.float32).reshape(2, 4)
    ).square().mean().backward()
    manager.optimizer.step()
    manager.lr_scheduler.step()


@pytest.mark.parametrize(
    ("checkpoint_format", "communication_backend"),
    [("dcp", None), ("dcp", "gloo"), ("local_shard", None)],
)
def test_model_only_export_and_full_training_resume(
    cpu_checkpoint_platform, tmp_path, checkpoint_format, communication_backend
):
    """Inference exports reload; sparse full saves retain the next update."""
    manager = _checkpoint_manager(
        checkpoint_format=checkpoint_format, communication_backend=communication_backend
    )
    export = tmp_path / "global_step_100" / "actor"
    manager.save_checkpoint(str(export), step=100)
    weights = resolve_full_weights(export)
    assert weights is not None
    reloaded = torch.nn.Linear(4, 3)
    reloaded.load_state_dict(torch.load(weights, weights_only=True))
    inputs = torch.ones((2, 4))
    torch.testing.assert_close(reloaded(inputs), manager.model(inputs), rtol=0, atol=0)
    assert not (export / "dcp_checkpoint").exists()
    assert not (export / "local_shard_checkpoint").exists()
    with pytest.raises(
        FileNotFoundError, match="Inference weights alone cannot resume"
    ):
        manager.load_checkpoint(str(export))

    full = tmp_path / "global_step_1000" / "actor"
    manager.save_checkpoint(str(full), step=1000)
    _checkpoint_update(manager)
    expected = {key: value.clone() for key, value in manager.model.state_dict().items()}
    expected_lr = manager.lr_scheduler.get_last_lr()
    manager.load_checkpoint(str(full))
    _checkpoint_update(manager)
    for key, value in manager.model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert manager.lr_scheduler.get_last_lr() == expected_lr


@pytest.mark.parametrize("backend", ["nccl", "mpi", False, 1])
def test_checkpoint_communication_backend_rejects_unsupported_values(backend):
    with pytest.raises(ValueError, match="null or 'gloo'"):
        validate_checkpoint_communication_backend(backend)


def test_checkpoint_communication_backend_requires_dcp(cpu_checkpoint_platform):
    with pytest.raises(ValueError, match="requires DCP"):
        _checkpoint_manager(
            checkpoint_format="local_shard", communication_backend="gloo"
        )


def test_checkpoint_group_is_scoped_and_keeps_default_group(cpu_checkpoint_platform):
    _checkpoint_manager()
    default = dist.group.WORLD
    with checkpoint_communication_group(None) as group:
        assert group is None and dist.group.WORLD is default
    with pytest.raises(RuntimeError, match="serialization failed"):
        with checkpoint_communication_group("gloo") as group:
            assert group is not default and dist.get_backend(group) == "gloo"
            assert dist.group.WORLD is default
            raise RuntimeError("serialization failed")
    assert dist.group.WORLD is default
    with pytest.raises((ValueError, TypeError)):
        dist.get_backend(group)


def test_failed_dcp_save_releases_checkpoint_group(
    cpu_checkpoint_platform, tmp_path, monkeypatch
):
    import torch.distributed.checkpoint as dcp

    manager = _checkpoint_manager(communication_backend="gloo")
    groups = []

    def fail_save(*args, process_group, **kwargs):
        groups.append(process_group)
        raise OSError("storage unavailable")

    monkeypatch.setattr(dcp, "save", fail_save)
    with pytest.raises(OSError, match="storage unavailable"):
        manager.save_checkpoint(str(tmp_path / "actor"), step=1000)
    with pytest.raises((ValueError, TypeError)):
        dist.get_backend(groups[0])


@pytest.mark.parametrize("interval", [None, 1000])
def test_checkpoint_final_override_and_disabled_export(
    cpu_checkpoint_platform, tmp_path, interval
):
    manager = _checkpoint_manager(interval=interval, full_weights=False)
    path = tmp_path / "actor"
    if interval is not None:
        with pytest.raises(ValueError, match="save_full_model_weights=True"):
            manager.save_checkpoint(str(path), step=100)
    manager.save_checkpoint(str(path), step=750, force_training_state=True)
    assert (path / "dcp_checkpoint" / ".metadata").is_file()
    assert not (path / "model_state_dict").exists()
    assert not (path / "checkpoint_metadata.json").exists()
    manager.load_checkpoint(str(path))


class _CheckpointCall:
    def __init__(self, call):
        self.call = call

    def wait(self):
        return self.call()

    def consume_duration(self):
        return 0.0


class _CheckpointActor:
    """A synchronous fake at the worker-group RPC edge with real serialization."""

    def __init__(self, manager, fail=False):
        self.manager, self.fail = manager, fail

    def get_max_steps_per_epoch(self):
        return _CheckpointCall(lambda: [2])

    def set_global_step(self, step):
        pass

    def run_training(self):
        _checkpoint_update(self.manager)
        return _CheckpointCall(lambda: [{"loss": 0.0}])

    def run_eval(self):
        return _CheckpointCall(lambda: [{"val_loss": 1.0}])

    def save_checkpoint(self, path, step, **kwargs):
        def save():
            metadata_path = Path(path) / "checkpoint_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                assert metadata["complete"] is False
                assert metadata["step"] == step
            self.manager.save_checkpoint(path, step, **kwargs)
            if self.fail:
                raise RuntimeError("worker data save failed after model export")

        return _CheckpointCall(save)


@pytest.mark.parametrize("interval", [None, 1000])
def test_sft_checkpoint_completion_and_final_state(
    cpu_checkpoint_platform, tmp_path, interval
):
    manager = _checkpoint_manager(interval=interval)
    cfg = OmegaConf.create(
        {
            "actor": {"fsdp_config": {"training_state_save_interval": interval}},
            "runner": {
                "max_steps": 2,
                "max_epochs": -1,
                "save_interval": 1,
                "val_check_interval": -1,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "test",
                    "project_name": "test",
                    "logger_backends": [],
                },
            },
        }
    )
    SFTRunner(cfg, _CheckpointActor(manager)).run()
    for step in (1, 2):
        path = tmp_path / "test" / "checkpoints" / f"global_step_{step}" / "actor"
        metadata = path / "checkpoint_metadata.json"
        if interval is None:
            assert not metadata.exists()
        else:
            assert json.loads(metadata.read_text()) == {
                "step": step,
                "save_training_state": step == 2,
                "checkpoint_format": "dcp",
                "complete": True,
            }
            if step == 1:
                with pytest.raises(ValueError, match="inference weights only"):
                    manager.load_checkpoint(str(path))
        if interval is None or step == 2:
            manager.load_checkpoint(str(path))


def test_failed_sft_save_keeps_incomplete_marker(cpu_checkpoint_platform, tmp_path):
    manager = _checkpoint_manager()
    cfg = OmegaConf.create(
        {
            "actor": {"fsdp_config": {"training_state_save_interval": 1000}},
            "runner": {
                "max_steps": 1,
                "max_epochs": -1,
                "save_interval": 1,
                "val_check_interval": -1,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "test",
                    "project_name": "test",
                    "logger_backends": [],
                },
            },
        }
    )
    with pytest.raises(RuntimeError, match="worker data save failed"):
        SFTRunner(cfg, _CheckpointActor(manager, fail=True)).run()
    path = tmp_path / "test" / "checkpoints" / "global_step_1" / "actor"
    assert (
        json.loads((path / "checkpoint_metadata.json").read_text())["complete"] is False
    )
    with pytest.raises(ValueError, match="incomplete"):
        manager.load_checkpoint(str(path))


def test_early_stop_forces_training_state(cpu_checkpoint_platform, tmp_path):
    manager = _checkpoint_manager()
    cfg = OmegaConf.create(
        {
            "actor": {"fsdp_config": {"training_state_save_interval": 1000}},
            "runner": {
                "max_steps": 10,
                "max_epochs": -1,
                "save_interval": 1,
                "val_check_interval": 1,
                "early_stop": {"enabled": True, "patience": 1},
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "test",
                    "project_name": "test",
                    "logger_backends": [],
                },
            },
        }
    )
    SFTRunner(cfg, _CheckpointActor(manager)).run()
    path = tmp_path / "test" / "checkpoints" / "global_step_2" / "actor"
    metadata = json.loads((path / "checkpoint_metadata.json").read_text())
    assert metadata["complete"] and metadata["save_training_state"]
    manager.load_checkpoint(str(path))
    assert not (tmp_path / "test" / "checkpoints" / "global_step_3").exists()


def test_mesh_group_defaults_above_the_torch_watchdog(single_rank_env, monkeypatch):
    """With RLINF_TIMEOUT unset the mesh still gets RLinf's 180-minute default."""
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)

    mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == RLINF_DEFAULT_TIMEOUT


def test_existing_process_group_is_left_alone(single_rank_env, monkeypatch, caplog):
    """A default group built by another component keeps its own timeout.

    RLINF_TIMEOUT cannot be applied retroactively, so the only thing left to do
    is say so — otherwise someone who followed the FAQ raises the variable and
    still dies on the backend watchdog with no clue why.
    """
    monkeypatch.setenv(
        "RLINF_TIMEOUT", str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    )
    dist.init_process_group(timeout=timedelta(minutes=11))

    with caplog.at_level(logging.WARNING):
        mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == timedelta(minutes=11)
    assert "RLINF_TIMEOUT" in caplog.text


def test_collective_timeout_matches_the_scheduler_default(monkeypatch):
    """``RLINF_TIMEOUT`` is read with the default the scheduler ships."""
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)
    assert Cluster.get_collective_timeout() == timedelta(minutes=180)

    monkeypatch.setenv("RLINF_TIMEOUT", "5")
    assert Cluster.get_collective_timeout() == timedelta(minutes=5)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("30m", "integer representing minutes"),
        ("0", "positive number of minutes"),
        ("-5", "positive number of minutes"),
    ],
)
def test_collective_timeout_rejects_unusable_values(monkeypatch, value, message):
    """Bad values fail loudly rather than installing a watchdog nobody wants.

    ``0`` and negatives parse as integers but abort the very first collective,
    so they have to be rejected alongside outright malformed input.
    """
    monkeypatch.setenv("RLINF_TIMEOUT", value)
    with pytest.raises(ValueError, match=message):
        Cluster.get_collective_timeout()


def test_torch_still_installs_the_short_timeout_on_its_own(single_rank_env):
    """Pin the upstream behaviour that makes ``create_device_mesh`` necessary.

    Letting ``init_device_mesh`` build the group leaves it on the backend's own
    watchdog, whatever that happens to be, and ``RLINF_TIMEOUT`` is ignored. The
    assertion is deliberately about what the timeout is *not*: the concrete value
    differs per backend (1800s on NCCL/Gloo, 3636s on Ascend HCCL), so pinning a
    number here would fail on some accelerator without anything being wrong.

    If PyTorch ever starts honouring a longer timeout for the implicitly created
    default group, this test fails and the workaround can be reconsidered.
    """
    assert not dist.is_initialized()
    os.environ["RLINF_TIMEOUT"] = str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=["fsdp"])
    finally:
        os.environ.pop("RLINF_TIMEOUT", None)

    group = mesh["fsdp"].get_group()
    assert group is dist.distributed_c10d._get_default_group()
    timeout = group_timeout(group)
    assert timeout != CONFIGURED_TIMEOUT
    assert timeout < RLINF_DEFAULT_TIMEOUT


def test_mesh_dimension_reuses_the_default_group(single_rank_env):
    """The ``fsdp`` dimension is the default group, which is why the fix works.

    ``DeviceMesh`` only hands a mesh dimension its own sub-group when the
    dimension is narrower than the world. For the 1-D mesh RLinf builds, the
    dimension *is* the default group, so setting that group's timeout is enough.
    """
    mesh = create_device_mesh(1)
    assert mesh["fsdp"].get_group() is dist.distributed_c10d._get_default_group()


def test_gradients_reduce_over_the_sharding_dimension(single_rank_env):
    """The norm's process group is the one gradients are sharded over.

    FSDP leaves each rank only its slice of every gradient, so the group has to
    span the sharding dimension. Reducing over nothing reports one rank's shard
    norm rather than the gradient's, and gradient clipping then loosens as the
    job grows; reducing over a replicated dimension counts each shard once per
    replica. Both are silent, so the lookup fails loudly instead of falling back
    when no sharding dimension is present.
    """
    from torch.distributed.device_mesh import init_device_mesh

    from rlinf.hybrid_engines.fsdp.utils import gradient_reduction_group

    sharded = create_device_mesh(1)
    assert gradient_reduction_group(sharded) is sharded["fsdp"].get_group()

    hybrid = init_device_mesh("cpu", (1, 1), mesh_dim_names=["ddp", "fsdp"])
    assert gradient_reduction_group(hybrid) is hybrid["fsdp"].get_group()

    replicated_only = init_device_mesh("cpu", (1,), mesh_dim_names=["ddp"])
    with pytest.raises(KeyError):
        gradient_reduction_group(replicated_only)


class _DomainTable(torch.nn.Module):
    """Reads its embedding weights directly, like cosmos ``DomainAwareLinear``."""

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(4, 3)

    def forward(self, x):
        return x @ self.table.weight.T


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = _DomainTable()

    def forward(self, x):
        return self.head(x)


class _Policy(torch.nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, x):
        return self.block(x)


def _shard(policy: torch.nn.Module) -> torch.nn.Module:
    return apply_fsdp2_to_model(
        policy,
        {},
        create_device_mesh(1),
        MixedPrecisionPolicy(),
        OffloadPolicy(),
        reshard_after_forward=True,
    )


def test_embeddings_are_sharded_as_their_own_units_by_default(single_rank_env):
    policy = _shard(_Policy())

    assert isinstance(policy.block.head.table, FSDPModule)


def test_a_model_can_keep_its_embeddings_in_the_enclosing_unit(single_rank_env):
    policy = _Policy()
    policy._fsdp_wrap_embeddings = False
    policy = _shard(policy)

    assert not isinstance(policy.block.head.table, FSDPModule)
