# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_asr.encoder_service import (
    FunASRPreLMEncoderService,
    _expected_audio_tokens,
    build_cache_namespace,
)

_HIDDEN_SIZE = 4
_NAMESPACE = "testns"
_SERVICES: list[FunASRPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def _close_services() -> Iterator[None]:
    yield
    for service in _SERVICES:
        service.close()
    _SERVICES.clear()


class _StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.audio_tower = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        )
        self.dtype = dtype
        self.encode_calls = 0
        self.fail = False
        self.fail_multi_item = False
        self.encode_gate: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = _expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, _HIDDEN_SIZE), fill, dtype=self.dtype))
        return torch.cat(parts, dim=0)


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 4,
) -> FunASRPreLMEncoderService:
    service = FunASRPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    _SERVICES.append(service)
    return service


def _run_build(
    service: FunASRPreLMEncoderService,
    item: SimpleNamespace,
    after_start: Callable[[], object] | None = None,
) -> None:
    """Run an encode and always decrement its outstanding-build count."""
    service.note_build_started()
    try:
        if after_start is not None:
            after_start()
        service.encode_item(item)
    finally:
        service.note_build_finished()


def _await(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _item(
    audio_hash: int | None,
    num_audio_tokens: int,
    *,
    with_feature: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        hash=audio_hash,
        audio_fingerprint=str(audio_hash) if audio_hash is not None else None,
        num_audio_tokens=num_audio_tokens,
        feature=torch.zeros(1, 560, 8) if with_feature else None,
        precomputed_embeddings=None,
    )


def test_encode_attaches_lm_ready_embedding_and_clears_feature() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(7, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.dtype
    assert (
        item.precomputed_embeddings.device
        == next(model.audio_tower.parameters()).device
    )
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1
    assert model.grad_enabled_during_encode is False
    assert service.stats()["misses"] == 1


def test_close_stops_worker() -> None:
    service = _make_service()

    service.close()

    assert not service._thread.is_alive()


def test_batch_context_unwinds_inference_mode_when_stream_context_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(FunASRPreLMEncoderService)
    service._stream = object()

    def fail_stream(_stream):  # noqa: ANN001, ANN202
        raise RuntimeError("stream context failed")

    monkeypatch.setattr(torch.cuda, "stream", fail_stream)

    assert not torch.is_inference_mode_enabled()
    with pytest.raises(RuntimeError, match="stream context failed"):
        with service._batch_context():
            pass
    assert not torch.is_inference_mode_enabled()


def test_cache_hit_skips_reencode() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(11, 3)
    second = _item(11, 3)
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)
    assert second.feature is None
    assert service.stats()["hits"] == 1


def test_extended_audio_never_reuses_prefix_embedding() -> None:
    model = _StubModel()
    service = _make_service(model)

    short = _item(111, 3)
    extended = _item(222, 5)
    service.encode_item(short)
    service.encode_item(extended)

    assert model.encode_calls == 2
    assert extended.precomputed_embeddings.shape == (5, _HIDDEN_SIZE)
    assert len(service._cache) == 2
    assert not torch.equal(
        short.precomputed_embeddings[0], extended.precomputed_embeddings[0]
    )


def test_cache_key_prefers_full_waveform_fingerprint() -> None:
    model = _StubModel()
    service = _make_service(model)
    first = _item(7, 3)
    second = _item(7, 3)
    first.audio_fingerprint = "full-hash-a"
    second.audio_fingerprint = "full-hash-b"

    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert len(service._cache) == 2


def test_concurrent_identical_requests_encode_once() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    items = [_item(123, 3) for _ in range(n_threads)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    for item in items:
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, items[0].precomputed_embeddings)
    stats = service.stats()
    assert stats["merged"] + stats["hits"] == n_threads - 1


def test_stale_cache_miss_rechecks_before_starting_duplicate_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    service = _make_service(model)
    stale_miss = threading.Event()
    release_stale_reader = threading.Event()
    original_get = service._cache.get

    def controlled_get(key: str | None):  # noqa: ANN202
        cached = original_get(key)
        if (
            threading.current_thread().name == "stale-cache-reader"
            and not stale_miss.is_set()
        ):
            assert cached is None
            stale_miss.set()
            assert release_stale_reader.wait(timeout=10)
        return cached

    monkeypatch.setattr(service._cache, "get", controlled_get)
    follower_item = _item(123, 3)
    errors: list[BaseException] = []

    def follower() -> None:
        try:
            service.encode_item(follower_item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=follower, name="stale-cache-reader")
    thread.start()
    assert stale_miss.wait(timeout=10)

    leader_item = _item(123, 3)
    service.encode_item(leader_item)
    release_stale_reader.set()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 1
    assert torch.equal(
        leader_item.precomputed_embeddings,
        follower_item.precomputed_embeddings,
    )
    assert service.stats()["hits"] == 1


def test_concurrent_identical_requests_deduplicate_without_cache() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model, cache_max_entries=0)
    barrier = threading.Barrier(2)
    items = [_item(123, 3) for _ in range(2)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    assert len(service._cache) == 0
    assert torch.equal(items[0].precomputed_embeddings, items[1].precomputed_embeddings)


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = _StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = _make_service(model)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(_item(55, 3))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) and "boom" in str(exc) for exc in errors)
    assert len(service._cache) == 0
    assert service.stats()["failed"] == 2

    model.fail = False
    item = _item(55, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_merged_follower_token_mismatch_raises_and_counts_failed() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.2
    service = _make_service(model)
    leader_item = _item(321, 3)
    follower_item = _item(321, 5)
    errors: list[BaseException] = []

    def leader() -> None:
        try:
            service.encode_item(leader_item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=leader)
    thread.start()
    deadline = time.monotonic() + 5
    while not service._inflight and time.monotonic() < deadline:
        time.sleep(0.005)
    assert service._inflight, "leader never registered in-flight"

    with pytest.raises(RuntimeError, match="returned an invalid"):
        service.encode_item(follower_item)
    thread.join(timeout=30)

    assert not errors, errors
    assert leader_item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert follower_item.precomputed_embeddings is None
    stats = service.stats()
    assert stats["merged"] == 1
    assert stats["failed"] == 1


def test_multi_item_batch_failure_retries_per_item_and_counts_stats() -> None:
    model = _StubModel()
    model.fail_multi_item = True
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(31, 3), _item(32, 3), _item(33, 4)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # Note (Akazaakane): Queue every leader before releasing the gate so the
    # next drain exercises the multi-item retry path.
    deadline = time.monotonic() + 5
    while len(service._inflight) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(service._inflight) == 3, "items never queued"
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    for item in items:
        assert item.precomputed_embeddings.shape == (
            item.num_audio_tokens,
            _HIDDEN_SIZE,
        )
    stats = service.stats()
    assert stats["failed"] == 0
    assert stats["items"] == 3
    assert stats["batches"] == 3
    assert model.encode_calls == 4
    assert len(service._cache) == 3


def test_eviction_under_byte_budget_triggers_reencode() -> None:
    model = _StubModel()
    service = _make_service(model, cache_max_bytes=100)

    for audio_hash in (1, 2, 3):
        service.encode_item(_item(audio_hash, 3))
    assert model.encode_calls == 3
    assert service._cache.eviction_count >= 1
    assert len(service._cache) == 2

    service.encode_item(_item(1, 3))
    assert model.encode_calls == 4


def test_invalid_cache_entry_is_evicted_and_reencoded() -> None:
    model = _StubModel()
    service = _make_service(model)
    probe = _item(42, 3)
    service.encode_item(probe)
    assert model.encode_calls == 1
    key = service._cache_key(probe)

    for poison in (
        torch.zeros(5, _HIDDEN_SIZE),
        torch.zeros(3, _HIDDEN_SIZE + 1),
        torch.zeros(3, _HIDDEN_SIZE, dtype=torch.float64),
    ):
        service._cache.put(key, poison)
        item = _item(42, 3)
        service.encode_item(item)
        assert model.encode_calls == 2
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, probe.precomputed_embeddings)
        model.encode_calls = 1


def test_invalid_cache_reader_preserves_a_valid_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(42, 3)
    key = service._cache_key(item)
    service._cache.put(key, torch.zeros(2, _HIDDEN_SIZE))
    stale_reader = threading.Event()
    release_reader = threading.Event()
    original_remove = service._cache.remove_if_same

    def controlled_remove(key, expected):  # noqa: ANN001, ANN202
        stale_reader.set()
        assert release_reader.wait(timeout=10)
        return original_remove(key, expected)

    monkeypatch.setattr(service._cache, "remove_if_same", controlled_remove)
    errors: list[BaseException] = []

    def encode() -> None:
        try:
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=encode)
    thread.start()
    assert stale_reader.wait(timeout=10)
    replacement = torch.full((3, _HIDDEN_SIZE), 7.0)
    service._cache.put(key, replacement)
    release_reader.set()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not errors, errors
    assert model.encode_calls == 0
    assert torch.equal(item.precomputed_embeddings, replacement)
    assert torch.equal(service._cache.get(key), replacement)


def test_token_count_mismatch_fails_loudly() -> None:
    model = _StubModel()
    model.row_offset = 1
    service = _make_service(model)
    item = _item(9, 3)

    with pytest.raises(RuntimeError, match="!= expected rows"):
        service.encode_item(item)

    assert item.precomputed_embeddings is None
    assert len(service._cache) == 0


def test_missing_token_count_raises() -> None:
    service = _make_service()
    item = SimpleNamespace(hash=1, feature=None, precomputed_embeddings=None)

    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        service.encode_item(item)


def test_item_without_fingerprint_encodes_without_caching() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(1, 2)
    second = _item(1, 2)
    first.audio_fingerprint = None
    second.audio_fingerprint = None
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert first.feature is None
    assert first.precomputed_embeddings.shape == (2, _HIDDEN_SIZE)
    assert len(service._cache) == 0


def test_expected_audio_tokens_uses_request_metadata() -> None:
    explicit = SimpleNamespace(num_audio_tokens=5, feature=torch.zeros(1, 560, 17))
    assert _expected_audio_tokens(explicit) == 5
    assert _expected_audio_tokens(SimpleNamespace()) is None


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = _StubModel()
    frontend = SimpleNamespace(
        feature_size=80,
        sampling_rate=16000,
        frame_length=25,
        frame_shift=10,
        lfr_m=7,
        lfr_n=6,
        window="hamming",
    )
    base = dict(
        model_path="FunAudioLLM/Fun-ASR-Nano-2512-hf",
        feature_extractor=frontend,
        mm_attention_backend=None,
    )

    namespace = build_cache_namespace(model, **base)
    assert namespace == build_cache_namespace(model, **base)
    assert namespace != build_cache_namespace(
        model, **{**base, "model_path": "other/revision"}
    )
    assert namespace != build_cache_namespace(
        model, **{**base, "mm_attention_backend": "triton_attn"}
    )
    assert namespace != build_cache_namespace(_StubModel(dtype=torch.bfloat16), **base)
    changed_frontend = SimpleNamespace(**{**vars(frontend), "lfr_m": 5})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = _StubModel()
    changed_config.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=_HIDDEN_SIZE), marker="other"
    )
    assert namespace != build_cache_namespace(changed_config, **base)


# --- Outstanding-build counting and batching -------------------------------
# Use generous windows and assert batch composition instead of wall-clock time.


def test_cold_single_request_flushes_immediately() -> None:
    """A lone request flushes when no counted build can add a batch item."""
    service = _make_service(max_batch_wait_ms=500)

    item = _item(9001, 4)
    _run_build(service, item)

    assert item.precomputed_embeddings is not None
    stats = service.stats()
    assert stats["guard_flushes"] == 1, stats
    assert stats["batch_size_hist"] == {1: 1}, stats
    assert stats["outstanding_builds"] == 0, stats


def test_inflight_build_holds_window_open_and_batches_pair() -> None:
    """B's batch window stays open while C may still add an item."""
    model = _StubModel()
    service = _make_service(model, max_batch_wait_ms=500)
    gate = threading.Event()
    model.encode_gate = gate

    thread_a = threading.Thread(target=_run_build, args=(service, _item(9101, 4)))
    thread_a.start()
    assert _await(lambda: model.encode_calls == 1), "A never reached the encoder"

    thread_b = threading.Thread(target=_run_build, args=(service, _item(9102, 4)))
    thread_b.start()
    assert _await(lambda: service._queue.qsize() == 1), "B never queued"

    service.note_build_started()  # C may still add an item to the batch.
    gate.set()  # Let the drain pick up B while C remains counted.
    try:
        service.encode_item(_item(9103, 4))
    finally:
        service.note_build_finished()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    stats = service.stats()
    assert stats["batch_size_hist"] == {1: 1, 2: 1}, stats
    assert stats["guard_flushes"] == 1, stats
    assert stats["early_flushes"] == 1, stats
    assert stats["outstanding_builds"] == 0, stats


def test_closed_loop_c2_pairs_every_round() -> None:
    """Each c=2 round forms a two-item batch while one build is outstanding."""
    service = _make_service(max_batch_wait_ms=500)
    rounds = 30
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def producer(base: int) -> None:
        try:
            for round_no in range(rounds):
                _run_build(
                    service,
                    _item(base + round_no, 4),
                    after_start=lambda: barrier.wait(timeout=10),
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=producer, args=(base,)) for base in (10_000, 20_000)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    stats = service.stats()
    assert stats["batch_size_hist"] == {2: rounds}, stats
    assert stats["guard_flushes"] == 0, stats
    assert stats["early_flushes"] == rounds, stats
    assert stats["outstanding_builds"] == 0, stats


def test_concurrency_migration_keeps_batching_correct_per_phase() -> None:
    """Changing concurrency does not affect batch sizes in later phases."""
    service = _make_service(max_batch_wait_ms=500)

    for round_no in range(3):
        _run_build(service, _item(1_000 + round_no, 4))

    def paired_rounds(bases: tuple[int, int], rounds: int) -> None:
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def producer(base: int) -> None:
            try:
                for round_no in range(rounds):
                    _run_build(
                        service,
                        _item(base + round_no, 4),
                        after_start=lambda: barrier.wait(timeout=10),
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=producer, args=(base,)) for base in bases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not errors, errors

    paired_rounds((2_000, 3_000), 3)

    burst_barrier = threading.Barrier(32)
    burst_errors: list[BaseException] = []

    def burst_producer(base: int) -> None:
        try:
            _run_build(
                service,
                _item(base, 4),
                after_start=lambda: burst_barrier.wait(timeout=10),
            )
        except BaseException as exc:  # noqa: BLE001
            burst_errors.append(exc)

    burst_threads = [
        threading.Thread(target=burst_producer, args=(4_000 + idx,))
        for idx in range(32)
    ]
    for thread in burst_threads:
        thread.start()
    for thread in burst_threads:
        thread.join(timeout=60)
    assert not burst_errors, burst_errors

    paired_rounds((5_000, 6_000), 3)

    stats = service.stats()
    assert stats["batch_size_hist"] == {1: 3, 2: 6, 8: 4}, stats
    assert stats["guard_flushes"] == 3, stats
    assert stats["outstanding_builds"] == 0, stats


def test_long_idle_then_burst_batches_together() -> None:
    """A new burst batches normally after the worker has been idle."""
    service = _make_service(max_batch_wait_ms=500)
    _run_build(service, _item(7_000, 4))

    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def producer(base: int) -> None:
        try:
            _run_build(
                service,
                _item(base, 4),
                after_start=lambda: barrier.wait(timeout=10),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=producer, args=(7_100 + idx,)) for idx in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    stats = service.stats()
    assert stats["batch_size_hist"] == {1: 1, 4: 1}, stats
    assert stats["early_flushes"] == 1, stats
    assert stats["outstanding_builds"] == 0, stats


def test_failed_and_finished_builds_settle_the_counter() -> None:
    """A failed build decrements its count so later batches can flush."""
    service = _make_service(max_batch_wait_ms=500)

    service.note_build_started()
    assert service.stats()["outstanding_builds"] == 1
    service.note_build_finished()
    assert service.stats()["outstanding_builds"] == 0
    service.note_build_finished()
    assert service.stats()["outstanding_builds"] == 0

    bad_item = SimpleNamespace(hash=1, feature=None, precomputed_embeddings=None)
    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        _run_build(service, bad_item)
    assert service.stats()["outstanding_builds"] == 0

    item = _item(8_000, 4)
    _run_build(service, item)
    assert item.precomputed_embeddings is not None
    stats = service.stats()
    assert stats["guard_flushes"] == 1, stats
    assert stats["batch_size_hist"] == {1: 1}, stats


def test_dangling_build_does_not_deadlock_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Shutdown terminates the worker despite an outstanding-build count."""
    caplog.set_level(logging.INFO, logger="sglang_omni.models.fun_asr.encoder_service")
    service = _make_service(max_batch_wait_ms=200)

    service.note_build_started()
    item = _item(8_100, 4)
    thread = threading.Thread(target=service.encode_item, args=(item,))
    thread.start()
    assert _await(
        lambda: service._queue.qsize() == 1 or item.precomputed_embeddings is not None
    )

    service.close()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert not service._thread.is_alive()
    assert item.precomputed_embeddings is not None
    assert "final stats" in caplog.text
    service.note_build_finished()


def test_wait_zero_and_batch_size_one_edges() -> None:
    """Zero wait and batch size one still produce immediate single-item batches."""
    instant = _make_service(max_batch_wait_ms=0)
    for idx in range(2):
        _run_build(instant, _item(8_200 + idx, 4))
    stats = instant.stats()
    assert stats["batch_size_hist"] == {1: 2}, stats

    singles = _make_service(max_batch_size=1, max_batch_wait_ms=500)
    barrier = threading.Barrier(3)
    errors: list[BaseException] = []

    def producer(base: int) -> None:
        try:
            _run_build(
                singles,
                _item(base, 4),
                after_start=lambda: barrier.wait(timeout=10),
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=producer, args=(8_300 + idx,)) for idx in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    stats = singles.stats()
    assert stats["batch_size_hist"] == {1: 3}, stats
    assert stats["outstanding_builds"] == 0, stats


def test_merged_follower_settles_before_leader_finishes() -> None:
    """A merged request decrements its count because it adds no batch item."""
    model = _StubModel()
    service = _make_service(model, max_batch_wait_ms=500)
    gate = threading.Event()
    model.encode_gate = gate

    leader_thread = threading.Thread(
        target=service.encode_item, args=(_item(9_500, 4),)
    )
    leader_thread.start()
    assert _await(lambda: model.encode_calls == 1), "leader never reached the encoder"

    follower_thread = threading.Thread(
        target=_run_build, args=(service, _item(9_500, 4))
    )
    follower_thread.start()
    assert _await(lambda: service.stats()["merged"] == 1), "follower never merged"

    assert _await(lambda: service.stats()["outstanding_builds"] == 0), (
        "merged follower left its build outstanding"
    )
    assert follower_thread.is_alive(), "follower should still be blocked on the leader"

    gate.set()
    leader_thread.join(timeout=10)
    follower_thread.join(timeout=10)
    assert service.stats()["outstanding_builds"] == 0


def test_submit_side_counting_pairs_scheduler_count_with_builder_settle() -> None:
    """The counter increments and decrements once for a submitted build."""
    service = _make_service(max_batch_wait_ms=500)
    service.enable_submit_side_counting()

    service.note_build_submitted()
    assert service.stats()["outstanding_builds"] == 1
    service.note_build_started()
    assert service.stats()["outstanding_builds"] == 1
    item = _item(9_600, 4)
    try:
        service.encode_item(item)
    finally:
        service.note_build_finished()
    assert item.precomputed_embeddings is not None
    assert service.stats()["outstanding_builds"] == 0

    service.note_build_submitted()
    service.note_build_cancelled()
    assert service.stats()["outstanding_builds"] == 0


def test_submitted_but_unstarted_build_holds_window_open() -> None:
    """A build waiting in the executor keeps the batch window open."""
    model = _StubModel()
    service = _make_service(model, max_batch_wait_ms=500)
    service.enable_submit_side_counting()
    gate = threading.Event()
    model.encode_gate = gate

    service.note_build_submitted()
    thread_a = threading.Thread(target=_run_build, args=(service, _item(9_700, 4)))
    thread_a.start()
    assert _await(lambda: model.encode_calls == 1), "A never reached the encoder"

    service.note_build_submitted()
    thread_b = threading.Thread(target=_run_build, args=(service, _item(9_701, 4)))
    thread_b.start()
    assert _await(lambda: service._queue.qsize() == 1), "B never queued"

    service.note_build_submitted()
    gate.set()  # The drain picks up B while C waits in the executor queue.
    thread_c = threading.Thread(target=_run_build, args=(service, _item(9_702, 4)))
    thread_c.start()
    for thread in (thread_a, thread_b, thread_c):
        thread.join(timeout=10)

    stats = service.stats()
    assert stats["batch_size_hist"] == {1: 1, 2: 1}, stats
    assert stats["outstanding_builds"] == 0, stats
