# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import weakref

import pytest

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
from vllm.multimodal.image_decoders.nvimagecodec import NvImageCodecInput

pytestmark = pytest.mark.cpu_test
_TIMEOUT = 5


def _input(value: int) -> NvImageCodecInput:
    return NvImageCodecInput(object(), value, 1)


class _FakeDecoder:
    instance: "_FakeDecoder"
    failure: str | None = None
    gate = [threading.Event(), threading.Event()]

    def __init__(self, _batch_cap: int, device_index: int) -> None:
        if self.failure == "init":
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_TIMEOUT)
            raise RuntimeError("constructor failed")
        self.widths: list[int] = []
        self.device_index = device_index
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
        return [item.width for item in items]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(
        nvimagecodec, "_pinned_output_budget", nvimagecodec._PinnedOutputBudget()
    )
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)


def _service(values=(), *, batch_cap=5, failure=None, device_index=0):
    _FakeDecoder.failure = failure
    service = nvimagecodec._NvImageCodecService(batch_cap, device_index)
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


@pytest.mark.parametrize("failure", ["init", "collect"])
def test_system_failure_is_sticky(failure) -> None:
    _FakeDecoder.failure = failure
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(3)]
    assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
    _FakeDecoder.gate[1].set()

    message = "constructor failed" if failure == "init" else "collect failed"
    with pytest.raises(RuntimeError, match=message):
        service.wait_until_ready() if failure == "init" else futures[0].result(_TIMEOUT)
    if failure == "init":
        _assert_fails(futures, "constructor failed")
    else:
        _assert_fails(futures[1:], "collect failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        service.submit(_input(2))
    service.close()


def test_pinned_budget_follows_owner_lifetime() -> None:
    class Owner:
        pass

    budget = nvimagecodec._PinnedOutputBudget(1)
    (permit,) = budget.try_acquire(1)
    owner = Owner()
    permit.attach(owner)
    permit.release()
    budget.configure(1)
    assert not budget.try_acquire(1)

    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    assert len(budget.try_acquire(1)) == 1

    budget = nvimagecodec._PinnedOutputBudget(4)
    permits = budget.try_acquire(4)
    permits[0].release()
    assert not budget.try_acquire(2)
    permits[1].release()
    assert len(budget.try_acquire(2)) == 2


def test_jobs_arriving_during_budget_wait_join_the_wave(monkeypatch) -> None:
    widths = []

    class Decoder:
        def __init__(self, *_args) -> None:
            pass

        def submit(self, items, permits):
            widths.append(len(items))
            return items, permits

        def collect(self, token):
            items, permits = token
            for permit in permits:
                permit.release()
            return [item.width for item in items]

        def close(self) -> None:
            pass

    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", Decoder)
    service = nvimagecodec._NvImageCodecService(2)
    service.wait_until_ready()
    occupied = service._budget.try_acquire(4)
    entered_wait = threading.Event()
    wait_for = service._budget.wait_for

    def observed_wait(*args):
        entered_wait.set()
        return wait_for(*args)

    service._budget.wait_for = observed_wait
    first = service.submit(_input(1))
    assert entered_wait.wait(timeout=_TIMEOUT)
    second = service.submit(_input(2))
    for permit in occupied:
        permit.release()

    assert [first.result(_TIMEOUT), second.result(_TIMEOUT)] == [1, 2]
    assert widths == [2]
    service.close()


def test_service_waits_until_a_complete_wave_fits(monkeypatch) -> None:
    construct = threading.Event()
    widths = []

    class Owner:
        pass

    class Result:
        def __init__(self, owner) -> None:
            self.owner = owner

        def release(self) -> None:
            self.owner = None

    class BorrowingDecoder:
        def __init__(self, _batch_cap: int, _device_index: int) -> None:
            assert construct.wait(timeout=_TIMEOUT)

        def submit(self, items, permits):
            widths.append(len(items))
            owners = [Owner() for _ in items]
            for permit, owner in zip(permits, owners, strict=True):
                permit.attach(owner)
            return owners

        def collect(self, owners):
            return [Result(owner) for owner in owners]

        def close(self) -> None:
            pass

    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", BorrowingDecoder)
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(6)]
    construct.set()
    service.wait_until_ready()

    first = [future.result(timeout=_TIMEOUT) for future in futures[:4]]
    first[0].release()
    first[1].release()
    last = [future.result(timeout=_TIMEOUT) for future in futures[4:]]

    assert widths == [2, 2, 2]
    for result in first[2:] + last:
        result.release()
    service.close()
