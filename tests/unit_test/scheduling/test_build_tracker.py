# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest

from sglang_omni.scheduling.build_tracker import OutstandingBuildTracker


def test_idle_fires_on_every_transition_to_zero() -> None:
    idle_calls: list[int] = []
    tracker = OutstandingBuildTracker(on_idle=lambda: idle_calls.append(1))

    tracker.note_started()
    tracker.settle()
    assert tracker.count == 0
    assert len(idle_calls) == 1

    tracker.note_submitted()
    tracker.note_submitted()
    tracker.note_cancelled()
    assert len(idle_calls) == 1, "2 -> 1 must not fire on_idle"
    tracker.note_cancelled()
    assert len(idle_calls) == 2, "1 -> 0 via cancel must fire on_idle"


def test_settle_is_idempotent_per_build() -> None:
    idle_calls: list[int] = []
    tracker = OutstandingBuildTracker(on_idle=lambda: idle_calls.append(1))

    tracker.note_started()
    tracker.settle()
    tracker.settle()
    tracker.settle()

    assert tracker.count == 0
    assert len(idle_calls) == 1


def test_cancel_underflow_raises() -> None:
    tracker = OutstandingBuildTracker()
    with pytest.raises(RuntimeError, match="negative"):
        tracker.note_cancelled()
    assert tracker.count == 0


def test_submit_side_counting_pairs_submit_with_settle() -> None:
    idle_calls: list[int] = []
    tracker = OutstandingBuildTracker(on_idle=lambda: idle_calls.append(1))
    tracker.enable_submit_side_counting()

    tracker.note_submitted()
    tracker.note_started()  # must not double-count in submit-side mode
    assert tracker.count == 1
    tracker.settle()
    assert tracker.count == 0
    assert len(idle_calls) == 1


def test_tracked_build_settles_on_exception() -> None:
    tracker = OutstandingBuildTracker()

    with pytest.raises(RuntimeError, match="boom"):
        with tracker.tracked_build():
            assert tracker.count == 1
            raise RuntimeError("boom")

    assert tracker.count == 0


def test_settle_on_thread_without_started_build_is_noop() -> None:
    tracker = OutstandingBuildTracker()
    tracker.note_started()  # main thread's build stays outstanding

    def foreign_settle() -> None:
        tracker.settle()

    thread = threading.Thread(target=foreign_settle)
    thread.start()
    thread.join(timeout=5)

    assert tracker.count == 1, "another thread must not settle this build"
    tracker.settle()
    assert tracker.count == 0


def test_idle_callback_errors_are_swallowed() -> None:
    def bad_callback() -> None:
        raise RuntimeError("callback boom")

    tracker = OutstandingBuildTracker(on_idle=bad_callback)
    tracker.note_started()
    tracker.settle()  # must not raise

    assert tracker.count == 0
