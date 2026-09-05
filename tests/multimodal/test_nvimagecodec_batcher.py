# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import pytest

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
from vllm.multimodal.image_decoders.nvimagecodec import NvImageCodecInput

pytestmark = pytest.mark.cpu_test
_TIMEOUT = 5


def _input(value: int) -> NvImageCodecInput:
    return NvImageCodecInput(b"jpeg", object(), value, 1)


class _FakeDecoder:
    instance: "_FakeDecoder"
    failure: str | None = None
    gate = [threading.Event(), threading.Event()]

    def __init__(self, _batch_cap: int, device_index: int) -> None:
        self.widths: list[int] = []
        self.device_index = device_index
        self.closed = False
        type(self).instance = self

    def submit(self, items):
        self.widths.append(len(items))
        if len(self.widths) == 1:
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_TIMEOUT)
        return items

    def collect(self, token):
        if self.failure == "collect":
            raise RuntimeError("collect failed")
        return [
            None if self.failure == "item" and item.width == 2 else _Result(item.width)
            for item in token
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(
        nvimagecodec, "_owned_output_budget", nvimagecodec._OwnedOutputBudget(0)
    )
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)


class _Result:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return self.value == other


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
    assert not first.cancel()
    assert service.submit(_input(9)).cancel()
    queued = [service.submit(_input(value)) for value in range(2, 9)]
    decoder.gate[1].set()

    assert first.result(timeout=_TIMEOUT) == 1
    assert [future.result(timeout=_TIMEOUT) for future in queued] == list(range(2, 9))
    service.close()
    assert decoder.widths == [1, 5, 2]


def test_one_decode_failure_does_not_poison_service() -> None:
    service, _, futures = _service((1, 2), failure="item")
    _FakeDecoder.gate[1].set()
    assert futures[0].result(timeout=_TIMEOUT) == 1
    with pytest.raises(ValueError, match="failed to decode"):
        futures[1].result(timeout=_TIMEOUT)
    assert service.submit(_input(3)).result(timeout=_TIMEOUT) == 3
    service.close()


def test_system_failure_is_sticky_and_releases_budget() -> None:
    _FakeDecoder.failure = "collect"
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(3)]
    assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
    _FakeDecoder.gate[1].set()

    with pytest.raises(RuntimeError, match="collect failed"):
        futures[0].result(_TIMEOUT)
    _assert_fails(futures[1:], "collect failed")
    with pytest.raises(RuntimeError, match="decoder failed"):
        service.submit(_input(2))
    service.close()
    assert service._budget._in_use == 0


def test_service_waits_until_a_complete_wave_fits(monkeypatch) -> None:
    construct = threading.Event()
    widths = []

    class Owner:
        pass

    class OwnedDecoder:
        def __init__(self, _batch_cap: int, _device_index: int) -> None:
            assert construct.wait(timeout=_TIMEOUT)

        def submit(self, items):
            widths.append(len(items))
            return items

        def collect(self, items):
            return [Owner() for _ in items]

        def close(self) -> None:
            pass

    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", OwnedDecoder)
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(6)]
    first_futures, last_futures = futures[:4], futures[4:]
    del futures
    construct.set()
    service.wait_until_ready()

    first = [future.result(timeout=_TIMEOUT) for future in first_futures]
    del first[:2]
    del first_futures[:2]
    last = [future.result(timeout=_TIMEOUT) for future in last_futures]

    assert widths == [2, 2, 2]
    del first, last
    service.close()
