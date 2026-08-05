# SPDX-License-Identifier: Apache-2.0
"""Track request builds that may still enqueue a pre-LM batch item."""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)


class OutstandingBuildTracker:
    """Counts request builds that may still enqueue a pre-LM batch item.

    The batch drain loop keeps its wait window open while the count is
    positive. ``on_idle`` fires on every transition to zero (outside the
    tracker lock, so the callback may take other locks) so the drain loop
    can re-check its flush condition immediately instead of sleeping until
    the window deadline.

    Counting sides: submit-side counting is enabled iff builds run on an
    executor. The scheduler then increments at submit — a build can wait in
    the executor queue before its builder starts — and refunds builds
    cancelled before they start via ``note_cancelled``. Without an executor
    each build increments itself in ``note_started``. Either way the build
    settles exactly once: ``note_started`` arms a thread-local flag and
    ``settle`` only decrements while it is armed, so repeated settles are
    no-ops.

    Failure modes: a leaked positive count only degrades batching back to
    the full wait window and is visible as ``count`` in service stats. An
    underflow would flush batches early, so ``note_cancelled``/``settle``
    raise instead of going negative.

    Boundary: submit-side counting also counts builds queued behind busy
    executor workers. When every worker blocks on the current encode (for
    example merged single-flight followers), those queued builds cannot
    join the current batch but still hold its window open until the
    deadline. That is a bounded latency cost, not a deadlock.
    """

    def __init__(self, *, on_idle: Callable[[], None] | None = None) -> None:
        self._count = 0
        self._lock = threading.Lock()
        self._local = threading.local()
        self._submit_side = False
        self._on_idle = on_idle

    def enable_submit_side_counting(self) -> None:
        """Count builds at scheduler submission instead of at build start."""
        self._submit_side = True

    def note_submitted(self) -> None:
        """Increment before a build enters the executor queue."""
        with self._lock:
            self._count += 1

    def note_cancelled(self) -> None:
        """Refund a submitted build cancelled before it starts."""
        self._decrement()

    def note_started(self) -> None:
        """Arm this thread's build so it settles exactly once."""
        self._local.unsettled = True
        if self._submit_side:
            return
        with self._lock:
            self._count += 1

    def settle(self) -> None:
        """Decrement this build's count unless an earlier path already did."""
        if getattr(self._local, "unsettled", False):
            self._local.unsettled = False
            self._decrement()

    @contextlib.contextmanager
    def tracked_build(self) -> Iterator[None]:
        """Count one build for the duration of the block."""
        self.note_started()
        try:
            yield
        finally:
            self.settle()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def _decrement(self) -> None:
        with self._lock:
            if self._count <= 0:
                raise RuntimeError(
                    "outstanding-build count would go negative; an "
                    "increment/decrement pair is mismatched"
                )
            self._count -= 1
            reached_zero = self._count == 0
        if reached_zero and self._on_idle is not None:
            try:
                self._on_idle()
            except Exception:
                logger.exception("outstanding-build idle callback failed")


__all__ = ["OutstandingBuildTracker"]
