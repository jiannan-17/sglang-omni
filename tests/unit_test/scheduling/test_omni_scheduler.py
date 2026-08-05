# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import threading

from sglang_omni.scheduling.omni_scheduler import OmniScheduler


class _Counter:
    def __init__(self) -> None:
        self.cancelled = 0

    def note_cancelled(self) -> None:
        self.cancelled += 1


class _FakeExecutor:
    def __init__(self) -> None:
        self.shutdown_calls: list[dict] = []

    def shutdown(self, wait=True, cancel_futures=False):  # noqa: ANN001
        self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})


def _bare_scheduler(counter, executor, pending):
    """Construct only the scheduler state exercised by shutdown tests."""
    scheduler = object.__new__(OmniScheduler)
    scheduler._request_build_executor = executor
    scheduler._request_build_tracker = counter
    scheduler._pending_request_builds = pending
    scheduler._request_admission_lock = threading.RLock()
    return scheduler


def test_executor_shutdown_refunds_cancelled_pending_builds() -> None:
    """Shutdown decrements counts only for builds cancelled before they start."""
    cancelled_future: concurrent.futures.Future = concurrent.futures.Future()
    assert cancelled_future.cancel()
    running_future: concurrent.futures.Future = concurrent.futures.Future()
    assert running_future.set_running_or_notify_cancel()
    done_future: concurrent.futures.Future = concurrent.futures.Future()
    done_future.set_running_or_notify_cancel()
    done_future.set_result(object())

    counter = _Counter()
    executor = _FakeExecutor()
    pending = {
        "req-cancelled": (object(), False, cancelled_future),
        "req-running": (object(), False, running_future),
        "req-done": (object(), False, done_future),
    }
    scheduler = _bare_scheduler(counter, executor, pending)

    scheduler._shutdown_request_build_executor()

    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert scheduler._request_build_executor is None
    assert counter.cancelled == 1
    # Removing the cancelled build prevents abort from decrementing it again.
    assert set(pending) == {"req-running", "req-done"}

    # A second shutdown must not decrement the count again.
    scheduler._shutdown_request_build_executor()
    assert counter.cancelled == 1


def test_executor_shutdown_without_counter_is_safe() -> None:
    """Shutdown works without an outstanding-build counter."""
    future: concurrent.futures.Future = concurrent.futures.Future()
    assert future.cancel()
    scheduler = _bare_scheduler(
        None, _FakeExecutor(), {"req-1": (object(), False, future)}
    )

    scheduler._shutdown_request_build_executor()

    assert scheduler._request_build_executor is None
