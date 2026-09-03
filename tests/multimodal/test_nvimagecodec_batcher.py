# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import weakref
from concurrent.futures import Future

import pytest

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
from vllm.multimodal.image_decoders.nvimagecodec import NvImageCodecInput

pytestmark = pytest.mark.cpu_test
_T = 5


def _input(value: int) -> NvImageCodecInput:
    return NvImageCodecInput(b"", object(), value, 1, 1)


class _FakeDecoder:
    instance: "_FakeDecoder"
    failure: str | None = None
    gate = [threading.Event(), threading.Event()]

    def __init__(self, _batch_cap: int, device_index: int) -> None:
        if self.failure == "init":
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_T)
            raise RuntimeError("constructor failed")
        self.widths: list[int] = []
        self.layouts: list[tuple[str, ...]] = []
        self.device_index = device_index
        self.closed = False
        type(self).instance = self

    def submit(self, items: tuple[NvImageCodecInput, ...], permits) -> object:
        self.widths.append(len(items))
        self.layouts.append(tuple(item.output_layout for item in items))
        if len(self.widths) == 1:
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_T)
        return items, permits

    def collect(self, token) -> list[int | Exception]:
        if self.failure == "collect":
            raise RuntimeError("collect failed")
        items, permits = token
        results = [
            ValueError("invalid image") if item.width == 2 else item.width
            for item in items
        ]
        for permit in permits:
            permit.release()
        return results

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)
    monkeypatch.setattr(
        nvimagecodec, "_pinned_output_budget", nvimagecodec._PinnedOutputBudget()
    )


def _service(values=(), *, batch_cap=5, failure=None, device_index=0):
    _FakeDecoder.failure = failure
    nvimagecodec._pinned_output_budget.configure(2 * batch_cap)
    service = nvimagecodec._NvImageCodecService(batch_cap, device_index)
    service.wait_until_ready()
    decoder = _FakeDecoder.instance
    futures = [service.submit(_input(value)) for value in values]
    if futures:
        assert decoder.gate[0].wait(timeout=_T)
    return service, decoder, futures


def test_service_passes_visible_device_index_to_decoder() -> None:
    service, decoder, _ = _service(device_index=3)
    assert decoder.device_index == 3
    service.close()


def _assert_fails(futures, message: str) -> None:
    errors = [future.exception(timeout=_T) for future in futures]
    assert all(
        isinstance(error, RuntimeError) and message in str(error) for error in errors
    )


def test_batches_fifo_jobs_skips_cancellation_and_isolates_item_errors() -> None:
    service, decoder, _ = _service()
    first = service.submit(_input(1))
    assert decoder.gate[0].wait(timeout=_T)
    assert service.submit(_input(9)).cancel()
    queued = [service.submit(_input(value)) for value in range(2, 9)]
    decoder.gate[1].set()
    assert first.result(timeout=_T) == 1
    assert isinstance(queued[0].exception(timeout=_T), ValueError)
    assert [future.result(timeout=_T) for future in queued[1:]] == list(range(3, 9))
    service.close()
    assert decoder.widths == [1, 5, 2]


def test_service_does_not_mix_output_layouts_in_one_wave() -> None:
    service, decoder, _ = _service(batch_cap=3)
    first = service.submit(_input(1))
    assert decoder.gate[0].wait(timeout=_T)
    chw = [
        service.submit(
            NvImageCodecInput(b"", object(), value, 1, 1, output_layout="chw_rgb")
        )
        for value in (3, 5)
    ]
    hwc = [service.submit(_input(value)) for value in (4, 6)]
    decoder.gate[1].set()

    assert first.result(timeout=_T) == 1
    assert [future.result(timeout=_T) for future in chw] == [3, 5]
    assert [future.result(timeout=_T) for future in hwc] == [4, 6]
    service.close()
    assert decoder.layouts == [
        ("hwc_rgb",),
        ("chw_rgb", "chw_rgb"),
        ("hwc_rgb", "hwc_rgb"),
    ]


def test_close_finishes_active_work_and_rejects_backlog() -> None:
    service, decoder, futures = _service((1, 3, 4), batch_cap=1)
    thread = threading.Thread(target=service.close)
    thread.start()
    _assert_fails(futures[1:], "closed")
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(_input(5))
    decoder.gate[1].set()
    thread.join(timeout=_T)
    assert not thread.is_alive() and decoder.closed
    assert futures[0].result(timeout=_T) == 1


def test_system_failure_is_sticky_and_fails_two_waves_and_fifo_tail() -> None:
    service, decoder, futures = _service(range(6), batch_cap=2, failure="collect")
    decoder.gate[1].set()
    _assert_fails(futures, "collect failed")
    assert decoder.widths == [2, 2]
    with pytest.raises(RuntimeError, match="nvImageCodec decoder failed"):
        service.submit(_input(8))
    service.close()


def test_constructor_failure_fails_early_and_future_work() -> None:
    _FakeDecoder.failure = "init"
    service = nvimagecodec._NvImageCodecService(2)
    assert _FakeDecoder.gate[0].wait(timeout=_T)
    queued = service.submit(_input(1))
    _FakeDecoder.gate[1].set()
    with pytest.raises(RuntimeError, match="constructor failed"):
        service.wait_until_ready()
    _assert_fails([queued], "constructor failed")
    with pytest.raises(RuntimeError, match="nvImageCodec decoder failed"):
        service.submit(_input(2))
    service.close()


def test_system_failure_tolerates_concurrent_future_cancellation() -> None:
    class _CancelBeforeSetException(Future):
        def set_exception(self, exception: BaseException | None) -> None:
            self.cancel()
            super().set_exception(exception)

    service, _, _ = _service()
    future = _CancelBeforeSetException()
    with service._condition:
        service._outstanding.add(future)
    service._fail(RuntimeError("decoder failed"))
    assert future.cancelled()
    service.close()


def test_pinned_budget_follows_attached_owner_lifetime() -> None:
    class Owner:
        pass

    budget = nvimagecodec._PinnedOutputBudget()
    budget.configure(1)
    [permit] = budget.try_acquire(1)
    owner = Owner()
    permit.attach(owner)
    permit.release()

    assert not budget.try_acquire(1)
    owner_ref = weakref.ref(owner)
    del owner
    assert owner_ref() is None
    assert len(budget.try_acquire(1)) == 1


def test_pinned_budget_waits_for_the_whole_next_wave() -> None:
    budget = nvimagecodec._PinnedOutputBudget()
    budget.configure(4)
    permits = budget.try_acquire(4)
    stop = threading.Event()
    ready = threading.Event()

    def wait_for_two() -> None:
        ready.set()
        assert budget.wait_for(2, stop)
        ready.set()

    ready.clear()
    thread = threading.Thread(target=wait_for_two)
    thread.start()
    assert ready.wait(timeout=_T)
    ready.clear()
    permits[0].release()
    assert not ready.wait(timeout=0.05)
    permits[1].release()
    assert ready.wait(timeout=_T)
    thread.join(timeout=_T)
    assert not thread.is_alive()


def test_service_parks_until_a_complete_claimed_wave_fits(monkeypatch) -> None:
    construct = threading.Event()
    third_submit = threading.Event()
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
            assert construct.wait(timeout=_T)

        def submit(self, items, permits):
            widths.append(len(items))
            if len(widths) == 3:
                third_submit.set()
            owners = [Owner() for _ in items]
            for permit, owner in zip(permits, owners, strict=True):
                permit.attach(owner)
            return owners

        def collect(self, owners):
            return [Result(owner) for owner in owners]

        def close(self) -> None:
            pass

    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", BorrowingDecoder)
    nvimagecodec._pinned_output_budget.configure(4)
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(6)]
    construct.set()
    service.wait_until_ready()

    first = [future.result(timeout=_T) for future in futures[:4]]
    first[0].release()
    assert not third_submit.wait(timeout=0.05)
    first[1].release()
    assert third_submit.wait(timeout=_T)
    last = [future.result(timeout=_T) for future in futures[4:]]

    assert widths == [2, 2, 2]
    assert service._thread.is_alive()
    for result in first[2:] + last:
        result.release()
    service.close()


def test_service_tops_up_a_narrow_claim_while_parked(monkeypatch) -> None:
    nvimagecodec._pinned_output_budget.configure(4)
    occupied = nvimagecodec._pinned_output_budget.try_acquire(4)
    submitted = threading.Event()
    widths = []

    class Decoder:
        def __init__(self, _batch_cap: int, _device_index: int) -> None:
            pass

        def submit(self, items, permits):
            widths.append(len(items))
            submitted.set()
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
    first = service.submit(_input(1))
    service.wait_until_ready()
    second = service.submit(_input(3))
    for permit in occupied:
        permit.release()

    assert first.result(timeout=_T) == 1
    assert second.result(timeout=_T) == 3
    assert submitted.wait(timeout=_T)
    assert widths == [2]
    service.close()
