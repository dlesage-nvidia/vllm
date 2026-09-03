# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
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
        self.device_index = device_index
        self.closed = False
        type(self).instance = self

    def submit(self, items: tuple[NvImageCodecInput, ...]) -> object:
        self.widths.append(len(items))
        if len(self.widths) == 1:
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_T)
        return items

    def collect(self, token: tuple[NvImageCodecInput, ...]) -> list[int | Exception]:
        if self.failure == "collect":
            raise RuntimeError("collect failed")
        return [
            ValueError("invalid image") if item.width == 2 else item.width
            for item in token
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)


def _service(values=(), *, batch_cap=5, failure=None, device_index=0):
    _FakeDecoder.failure = failure
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
