# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
from vllm.multimodal.image_decoders.nvimagecodec import NvImageCodecInput

pytestmark = pytest.mark.cpu_test
_TIMEOUT = 5


def _input(value: int) -> NvImageCodecInput:
    return NvImageCodecInput(object(), b"jpeg", value, 1)


class _FakeDecoder:
    instance: "_FakeDecoder"
    failure: str | None = None
    item_failure: int | None = None
    gate = [threading.Event(), threading.Event()]

    def __init__(self, _batch_cap: int, _device_index: int) -> None:
        self.widths: list[int] = []
        self.closed = False
        type(self).instance = self

    def submit(self, items, permits):
        self.widths.append(len(items))
        if len(self.widths) == 1:
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_TIMEOUT)
        return items, permits

    def collect(self, token):
        if self.failure == "collect":
            raise RuntimeError("collect failed")
        items, permits = token
        for permit in permits:
            permit.release()
        return [
            ValueError("bad JPEG") if item.width == self.item_failure else item.width
            for item in items
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.item_failure = None
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(
        nvimagecodec, "_pinned_output_budget", nvimagecodec._PinnedOutputBudget()
    )
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)


def _service(values=(), *, batch_cap=5):
    service = nvimagecodec._NvImageCodecService(batch_cap)
    service.wait_until_ready()
    decoder = _FakeDecoder.instance
    futures = [service.submit(_input(value)) for value in values]
    if futures:
        assert decoder.gate[0].wait(timeout=_TIMEOUT)
    return service, decoder, futures


def _assert_fails(futures, message: str) -> None:
    errors = [future.exception(timeout=_TIMEOUT) for future in futures]
    assert all(
        isinstance(error, RuntimeError) and message in str(error) for error in errors
    )


def test_batches_fifo_jobs_and_skips_cancellation() -> None:
    service, decoder, _ = _service()
    first = service.submit(_input(1))
    assert decoder.gate[0].wait(timeout=_TIMEOUT)
    assert service.submit(_input(9)).cancel()
    queued = [service.submit(_input(value)) for value in range(2, 9)]
    decoder.gate[1].set()

    assert first.result(timeout=_TIMEOUT) == 1
    assert [future.result(timeout=_TIMEOUT) for future in queued] == list(range(2, 9))
    service.close()
    assert decoder.widths == [1, 5, 2]


def test_close_finishes_active_work_and_rejects_backlog() -> None:
    service, decoder, futures = _service((1, 3, 4), batch_cap=1)
    thread = threading.Thread(target=service.close)
    thread.start()
    _assert_fails(futures[1:], "closed")
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(_input(5))
    decoder.gate[1].set()
    thread.join(timeout=_TIMEOUT)

    assert not thread.is_alive() and decoder.closed
    assert futures[0].result(timeout=_TIMEOUT) == 1


def test_system_failure_is_sticky() -> None:
    _FakeDecoder.failure = "collect"
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(3)]
    assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
    _FakeDecoder.gate[1].set()

    _assert_fails(futures, "collect failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        service.submit(_input(2))
    service.close()


def test_item_failure_does_not_stop_service() -> None:
    service, decoder, _ = _service()
    _FakeDecoder.item_failure = 2
    first = service.submit(_input(1))
    assert decoder.gate[0].wait(timeout=_TIMEOUT)
    failed = service.submit(_input(2))
    decoder.gate[1].set()

    assert first.result(timeout=_TIMEOUT) == 1
    with pytest.raises(ValueError, match="bad JPEG"):
        failed.result(timeout=_TIMEOUT)
    assert service.submit(_input(3)).result(timeout=_TIMEOUT) == 3
    service.close()


def test_pinned_budget_tracks_lifetime_and_has_a_bounded_wait() -> None:
    class Owner:
        pass

    budget = nvimagecodec._PinnedOutputBudget(1, wait_timeout=0)
    (permit,) = budget.try_acquire(1)
    owner = Owner()
    permit.attach(owner)
    with pytest.raises(TimeoutError, match="retained beyond its request"):
        budget.wait_for(1, threading.Event())

    del owner
    assert len(budget.try_acquire(1)) == 1
