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

"""Optimizer-batch admission and accounting for a single ODS source owner."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_Batch = TypeVar("_Batch")


class ODSBatchCoordinator:
    """Admit one global draw and commit it after every actor rank succeeds.

    The source owner calls these methods serially and owns the sampled payload.
    Counters cover this coordinator's lifetime; ``start_step`` identifies the
    first optimizer batch, without counting earlier batches again.
    """

    def __init__(
        self, *, world_size: int, global_batch_size: int, start_step: int = 0
    ) -> None:
        """Set the actor topology and the first optimizer-step identifier."""
        for name, value in (
            ("world_size", world_size),
            ("global_batch_size", global_batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer; found {value!r}.")
        if global_batch_size % world_size:
            raise ValueError("global_batch_size must divide evenly across world_size.")
        self._validate_step(start_step)
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.expected_step = start_step
        self.delivered_chunks = 0
        self.committed_chunks = 0
        self._requests: set[int] = set()
        self._next_requests: set[int] = set()
        self._commits: set[int] = set()
        self._phase = "waiting"

    @staticmethod
    def _validate_step(step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError(
                f"Optimizer step must be a nonnegative integer; found {step!r}."
            )

    def _validate_message(self, rank: int, step: int) -> None:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(
                f"Actor rank must be in [0, {self.world_size}); found {rank!r}."
            )
        self._validate_step(step)

    @property
    def ready_step(self) -> int | None:
        """Return the step all ranks requested, until its one draw is attempted."""
        if self._phase == "waiting" and len(self._requests) == self.world_size:
            return self.expected_step
        return None

    @property
    def failed(self) -> bool:
        """Whether this source draw failed and cannot be retried."""
        return self._phase == "failed"

    def request(self, rank: int, step: int) -> None:
        """Request a batch, accepting duplicates and one committed-rank lookahead.

        Args:
            rank: Actor rank requesting its share of the global batch.
            step: Optimizer-step identifier. Earlier committed steps are no-ops.
        """
        self._validate_message(rank, step)
        if step < self.expected_step:
            return
        if step == self.expected_step:
            self._requests.add(rank)
            return
        if step != self.expected_step + 1:
            raise ValueError(
                f"Expected step {self.expected_step} or {self.expected_step + 1}; found {step}."
            )
        if rank not in self._commits:
            raise RuntimeError(
                "An actor must commit its current batch before requesting the next."
            )
        self._next_requests.add(rank)

    def draw(self, sampler: Callable[[int], _Batch]) -> _Batch:
        """Call the source sampler once after all current requests arrive.

        Args:
            sampler: Callback receiving the global batch size. Its returned
                payload remains owned by the caller, not the coordinator.

        Returns:
            The sampled global batch, before projection or delivery.

        Raises:
            RuntimeError: If no step is ready, including after a prior attempt.
        """
        if self.ready_step is None:
            raise RuntimeError("No optimizer batch is ready for a new ODS draw.")
        self._phase = "sampling"
        try:
            batch = sampler(self.global_batch_size)
        except BaseException:
            self._phase = "failed"
            raise
        self._phase = "sampled"
        return batch

    def mark_delivered(self, step: int) -> bool:
        """Count a successfully projected global packet once, before rank delivery.

        Args:
            step: Step whose packet is ready for all actor-rank channels.

        Returns:
            True for a new delivery; false for a duplicate committed/delivered step.
        """
        self._validate_step(step)
        if step < self.expected_step:
            return False
        if step != self.expected_step:
            raise ValueError(
                f"Expected delivered step {self.expected_step}; found {step}."
            )
        if self._phase == "delivered":
            return False
        if self._phase != "sampled":
            raise RuntimeError(
                "A successful ODS draw must precede delivery accounting."
            )
        self._phase = "delivered"
        self.delivered_chunks += self.global_batch_size
        return True

    def commit(self, rank: int, step: int) -> bool:
        """Commit an actor's completed optimizer batch and advance after all commits.

        Args:
            rank: Actor whose microbatches and optimizer step all succeeded.
            step: Delivered optimizer-step identifier.

        Returns:
            True only when this commit advances the global optimizer step.
        """
        self._validate_message(rank, step)
        if step < self.expected_step:
            return False
        if step != self.expected_step:
            raise ValueError(
                f"Expected committed step {self.expected_step}; found {step}."
            )
        if self._phase != "delivered":
            raise RuntimeError("An actor cannot commit an undelivered ODS batch.")
        self._commits.add(rank)
        if len(self._commits) != self.world_size:
            return False
        self.committed_chunks += self.global_batch_size
        self.expected_step += 1
        self._requests = self._next_requests
        self._next_requests = set()
        self._commits.clear()
        self._phase = "waiting"
        return True
