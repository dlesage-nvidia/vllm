# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
import socket
import sys
import threading
import weakref
from dataclasses import replace
from enum import IntEnum
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
import vllm.multimodal.media.image as image_module
from vllm.entrypoints.serve.instrumentator.health import router as health_router
from vllm.exceptions import RetryableRequestError
from vllm.multimodal.image_decoders.nvimagecodec import NvImageCodecInput
from vllm.multimodal.media.image import (
    ImageMediaIO,
    _get_nvimagecodec_decode_service,
    initialize_nvimagecodec_decode_service,
)

pytestmark = pytest.mark.cpu_test
_TIMEOUT = 5


def test_startup_batch_cap_preserves_global_default():
    assert nvimagecodec.get_nvimagecodec_batch_cap() == 20
    assert nvimagecodec.get_nvimagecodec_batch_cap(batch_cap=20) == 20
    with pytest.raises(ValueError, match="batch_cap"):
        nvimagecodec.get_nvimagecodec_batch_cap(batch_cap=21)


def test_shared_owner_uses_one_decoder_batch_and_reservation():
    assert nvimagecodec.get_nvimagecodec_batch_cap(batch_cap=10) == 10
    assert nvimagecodec.get_nvimagecodec_non_context_bytes() == (
        20 * nvimagecodec._NVIMAGECODEC_DEVICE_BYTES_PER_BATCH_SLOT
        + nvimagecodec.NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES
    )


def test_close_reports_native_cleanup_failure():
    service, decoder, _ = _service()
    decoder.failure = "cleanup"
    with pytest.raises(RuntimeError, match="cleanup failed"):
        service.close()


def test_service_rejects_missing_final_destination_without_poisoning_decoder():
    service, decoder, _ = _service()
    try:
        with pytest.raises(ValueError, match="final pinned RGB destination"):
            service.submit(replace(_input(6), destination=None))
        assert service.is_healthy()
    finally:
        decoder.gate[1].set()
        service.close()


@pytest.mark.parametrize("interleaved_output", [False, True])
def test_shared_output_slot_lives_until_the_last_tensor_view(interleaved_output):
    import torch

    from vllm.multimodal.image_decoders.nvimagecodec import _MappedOutputs

    memory = SharedMemory(create=True, size=12)
    buffer = memory.buf
    assert buffer is not None
    buffer[:] = bytes(range(12))
    mapping = _MappedOutputs(memory, interleaved_output)
    released = threading.Event()
    try:
        tensor = mapping.tensor(0, 2, 2, released.set)
        expected = torch.arange(12, dtype=torch.uint8)
        if interleaved_output:
            expected = expected.reshape(2, 2, 3).permute(2, 0, 1)
            alias = tensor.permute(1, 2, 0).view(-1)
        else:
            expected = expected.reshape(3, 2, 2)
            alias = tensor.view(-1)
        assert torch.equal(tensor, expected)
        assert alias.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        alias[0] = 99
        assert buffer[0] == 99
        del tensor
        assert not released.is_set()
        assert mapping.views == 1
        del alias
        assert released.wait(_TIMEOUT)
        assert mapping.views == 0
        mapping.close()
    finally:
        memory.unlink()


def _input(value: int) -> NvImageCodecInput:
    import torch

    return NvImageCodecInput(
        b"jpeg", object(), value, 1, torch.empty(3, 1, value, dtype=torch.uint8)
    )


class _FakeDecoder:
    instance: "_FakeDecoder"
    failure: str | None = None
    behavior: str | None = None
    transport = False
    gate = [threading.Event(), threading.Event()]

    def __init__(
        self, _batch_cap: int, _device_index: int, _interleaved_output: bool = False
    ) -> None:
        self.widths: list[int] = []
        type(self).instance = self
        if self.behavior == "block_init":
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_TIMEOUT)

    def submit(self, items):
        self.widths.append(len(items))
        if len(self.widths) == 1:
            self.gate[0].set()
            assert self.gate[1].wait(timeout=_TIMEOUT)
        return list(items)

    def collect(self, token):
        if self.failure == "collect":
            token.clear()
            raise RuntimeError("collect failed")
        results = [
            None
            if self.failure == "reject" or self.failure == "item" and item.width == 2
            else item.destination.fill_(item.width)
            if self.transport
            else _Result(item.width)
            for item in token
        ]
        token.clear()
        return results

    def close(self) -> None:
        if self.failure == "cleanup":
            raise RuntimeError("cleanup failed")


@pytest.fixture(autouse=True)
def _install_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDecoder.failure = None
    _FakeDecoder.behavior = None
    _FakeDecoder.transport = False
    _FakeDecoder.gate = [threading.Event(), threading.Event()]
    monkeypatch.setattr(nvimagecodec, "_NvImageCodecDecoder", _FakeDecoder)


@pytest.fixture
def cpu_owner(monkeypatch, tmp_path, request):
    import torch

    class CudaError(IntEnum):
        cudaSuccess = 0

    runtime = ModuleType("cuda.bindings.runtime")
    monkeypatch.setattr(runtime, "cudaError_t", CudaError, raising=False)
    bindings = ModuleType("cuda.bindings")
    monkeypatch.setattr(bindings, "runtime", runtime, raising=False)
    cuda = ModuleType("cuda")
    monkeypatch.setattr(cuda, "bindings", bindings, raising=False)
    for module in (cuda, bindings, runtime):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class CodeStream:
        def __init__(self, data):
            with Image.open(BytesIO(data)) as image:
                self.width, self.height = image.size
                self.num_channels = len(image.getbands())
                self.codec_name = image.format.lower()
                self.precision = 8

    native = ModuleType("nvidia.nvimgcodec")
    monkeypatch.setattr(native, "CodeStream", CodeStream, raising=False)
    nvidia = sys.modules.get("nvidia", ModuleType("nvidia"))
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setattr(nvidia, "nvimgcodec", native, raising=False)

    _FakeDecoder.transport = True
    monkeypatch.setattr(torch.accelerator, "set_device_index", lambda index: None)
    unregistered = threading.Event()
    monkeypatch.setattr(
        runtime,
        "cudaHostRegister",
        lambda *args: (CudaError.cudaSuccess,),
        raising=False,
    )

    def unregister(pointer):
        unregistered.set()
        return (CudaError.cudaSuccess,)

    monkeypatch.setattr(runtime, "cudaHostUnregister", unregister, raising=False)
    monkeypatch.setattr(nvimagecodec, "_CLOSE_TIMEOUT", 1)
    parent, child = mp.Pipe()
    api_count, batch_cap, *layout = getattr(request, "param", (1, 1))
    interleaved_output = layout[0] if layout else False
    slots = max(1, 2 * batch_cap // api_count)
    endpoint = nvimagecodec.SharedDecoderEndpoint(
        str(tmp_path / "owner.sock"),
        os.urandom(32),
        batch_cap,
        os.getpid(),
        interleaved_output,
    )
    errors = []

    def run():
        try:
            nvimagecodec._owner(endpoint, child, slots, 0, api_count)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert parent.poll(_TIMEOUT)
    assert parent.recv()[0] == "ready"
    image_module.set_shared_nvimagecodec_endpoint(endpoint)
    try:
        yield endpoint, unregistered, errors
    finally:
        _FakeDecoder.gate[1].set()
        if thread.is_alive():
            parent.send("stop")
        thread.join(_TIMEOUT)
        parent.close()
        image_module.set_shared_nvimagecodec_endpoint(None)
        assert not thread.is_alive()
        if _FakeDecoder.failure not in ("collect", "cleanup"):
            assert not errors


def _transport_input():
    with BytesIO() as buffer:
        Image.new("RGB", (6, 3), (31, 63, 127)).save(buffer, "JPEG")
        return nvimagecodec.preflight_image_nvimagecodec(buffer.getvalue())


@pytest.mark.parametrize("cpu_owner", [(1, 1, False), (1, 1, True)], indirect=True)
def test_disconnect_during_running_copy_releases_owner_views(cpu_owner):
    endpoint, unregistered, errors = cpu_owner
    client = nvimagecodec.SharedNvImageCodecClient(endpoint)
    try:
        client.submit(_transport_input())
        assert _FakeDecoder.gate[0].wait(_TIMEOUT)
        client.close()
        _FakeDecoder.gate[1].set()
        assert unregistered.wait(_TIMEOUT), (
            "Disconnect retained its own completed future"
        )
        assert not errors
    finally:
        _FakeDecoder.gate[1].set()
        client.close()


def test_shared_admission_credit_survives_cancellation_and_storage_aliases(cpu_owner):
    endpoint, _, errors = cpu_owner
    client = nvimagecodec.SharedNvImageCodecClient(endpoint)
    try:
        item = _transport_input()
        first = client.submit(item)
        assert _FakeDecoder.gate[0].wait(_TIMEOUT)
        canceled = client.submit(item)
        assert canceled.cancel()
        with pytest.raises(RetryableRequestError):
            client.submit(item)
        _FakeDecoder.gate[1].set()
        alias = first.result(_TIMEOUT).view(-1)
        del first
        with client.condition:
            assert client.condition.wait_for(
                lambda: len(client.admitted) == 1, _TIMEOUT
            )
        second = client.submit(item)
        second.result(_TIMEOUT)
        with pytest.raises(RetryableRequestError):
            client.submit(item)
        del alias
        with client.condition:
            assert client.condition.wait_for(
                lambda: len(client.admitted) == 1, _TIMEOUT
            )
        assert client.submit(item).result(_TIMEOUT).shape == (3, 3, 6)
        assert not errors
    finally:
        _FakeDecoder.gate[1].set()
        client.close()


@pytest.mark.parametrize(
    "failure,error_type,healthy",
    [("reject", ValueError, True), ("collect", RuntimeError, False)],
)
def test_shared_transport_preserves_image_vs_systemic_failure(
    cpu_owner, failure, error_type, healthy
):
    endpoint, unregistered, _ = cpu_owner
    _FakeDecoder.failure = failure
    client = nvimagecodec.SharedNvImageCodecClient(endpoint)
    try:
        future = client.submit(_transport_input())
        assert _FakeDecoder.gate[0].wait(_TIMEOUT)
        _FakeDecoder.gate[1].set()
        with pytest.raises(error_type):
            future.result(_TIMEOUT)
        assert client.is_healthy() is healthy
    finally:
        _FakeDecoder.gate[1].set()
        client.close()
        assert unregistered.wait(_TIMEOUT)


def test_shared_owner_rejects_extra_active_connections(cpu_owner):
    from multiprocessing.connection import Client

    endpoint, _, _ = cpu_owner
    client = nvimagecodec.SharedNvImageCodecClient(endpoint)
    extra = None
    try:
        with pytest.raises((EOFError, OSError, mp.AuthenticationError)):
            extra = Client(endpoint.address, family="AF_UNIX", authkey=endpoint.authkey)
    finally:
        if extra is not None:
            extra.close()
        client.close()


def test_listener_cleanup_failure_still_stops_shared_owner(cpu_owner):
    from vllm.utils.network_utils import ZmqListener
    from vllm.v1.utils import _shutdown_processes_and_cleanup_listeners

    endpoint, _, _ = cpu_owner
    client = nvimagecodec.SharedNvImageCodecClient(endpoint)

    class FailingSocket(socket.socket):
        def close(self):
            raise OSError("listener close failed")

    transport = FailingSocket()
    listener = ZmqListener("tcp://127.0.0.1:0", transport)
    try:
        with pytest.raises(OSError, match="listener close failed"):
            _shutdown_processes_and_cleanup_listeners(
                [], [listener], timeout=0, image_decoder_manager=client
            )
        assert not client.is_healthy()
    finally:
        client.close()
        socket.socket.close(transport)


class _Result:
    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return self.value == other


def _service(values=(), *, batch_cap=5, failure=None):
    _FakeDecoder.failure = failure
    service = nvimagecodec._NvImageCodecService(batch_cap)
    service.wait_until_ready()
    decoder = _FakeDecoder.instance
    futures = [service.submit(_input(value)) for value in values]
    if futures:
        assert decoder.gate[0].wait(timeout=_TIMEOUT)
    return service, decoder, futures


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


def test_completed_output_is_released_while_service_is_idle() -> None:
    service, decoder, _ = _service()
    released = threading.Event()
    try:
        future = service.submit(_input(1))
        assert decoder.gate[0].wait(timeout=_TIMEOUT)
        future.add_done_callback(
            lambda done: weakref.finalize(done.result(), released.set)
        )
        del future
        decoder.gate[1].set()
        assert released.wait(timeout=_TIMEOUT)
    finally:
        decoder.gate[1].set()
        service.close()


def test_system_failure_is_sticky_and_rejects_outstanding_jobs() -> None:
    _FakeDecoder.failure = "collect"
    service = nvimagecodec._NvImageCodecService(2)
    futures = [service.submit(_input(value)) for value in range(3)]
    assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
    _FakeDecoder.gate[1].set()

    with pytest.raises(RuntimeError, match="collect failed"):
        futures[0].result(_TIMEOUT)
    errors = [future.exception(timeout=_TIMEOUT) for future in futures[1:]]
    assert all(
        isinstance(error, RuntimeError) and "collect failed" in str(error)
        for error in errors
    )
    with pytest.raises(RuntimeError, match="decoder failed"):
        service.submit(_input(2))
    service.close()


@pytest.mark.parametrize(
    "failure,expected_status,error_type",
    [("reject", 200, ValueError), ("collect", 503, RuntimeError)],
)
def test_health_endpoint_reports_systemic_decoder_failure_only(
    cpu_owner, failure: str, expected_status: int, error_type: type[Exception]
) -> None:
    app = FastAPI()
    app.state.engine_client = None
    app.include_router(health_router)
    _FakeDecoder.failure = failure
    release = initialize_nvimagecodec_decode_service(batch_cap=cpu_owner[0].batch_cap)
    try:
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            service = _get_nvimagecodec_decode_service()
            future = service.submit(_transport_input())
            assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
            _FakeDecoder.gate[1].set()
            with pytest.raises(error_type):
                future.result(timeout=_TIMEOUT)
            assert client.get("/health").status_code == expected_status
            assert client.get("/health").status_code == expected_status
    finally:
        _FakeDecoder.gate[1].set()
        release()

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_service_reuses_two_arenas_while_completed_outputs_are_retained() -> None:
    _FakeDecoder.behavior = "block_init"
    service = nvimagecodec._NvImageCodecService(2)
    assert _FakeDecoder.gate[0].wait(timeout=_TIMEOUT)
    futures = [service.submit(_input(value)) for value in range(6)]
    first_futures, last_futures = futures[:4], futures[4:]
    del futures
    _FakeDecoder.gate[1].set()
    service.wait_until_ready()

    first = [future.result(timeout=_TIMEOUT) for future in first_futures]
    del first[:2]
    del first_futures[:2]
    last = [future.result(timeout=_TIMEOUT) for future in last_futures]

    assert _FakeDecoder.instance.widths == [2, 2, 2]
    del first, last
    service.close()


@pytest.mark.parametrize("cpu_owner", [(4, 1)], indirect=True)
def test_startup_batch_cap_bounds_queue_and_retained_outputs(cpu_owner) -> None:
    release = initialize_nvimagecodec_decode_service(4, batch_cap=1)
    service = _get_nvimagecodec_decode_service()
    decoder = _FakeDecoder.instance
    try:
        item = _transport_input()
        first = service.submit(item)
        assert decoder.gate[0].wait(timeout=_TIMEOUT)
        with pytest.raises(RetryableRequestError):
            service.submit(item)
        decoder.gate[1].set()
        assert first.result(timeout=_TIMEOUT).shape == (3, 3, 6)
        with pytest.raises(RetryableRequestError):
            service.submit(item)
        assert decoder.widths == [1]
        del first
        with service.condition:
            assert service.condition.wait_for(lambda: not service.admitted, _TIMEOUT)
        assert service.submit(item).result(timeout=_TIMEOUT).shape == (3, 3, 6)
        assert decoder.widths == [1, 1]
    finally:
        decoder.gate[1].set()
        release()


@pytest.mark.parametrize("cpu_owner", [(4, 20)], indirect=True)
def test_startup_batch_cap_reinitialization_and_runtime_overrides(cpu_owner) -> None:
    release = initialize_nvimagecodec_decode_service(4)
    release_equivalent = None
    try:
        release_equivalent = initialize_nvimagecodec_decode_service(4, batch_cap=20)
        with pytest.raises(RuntimeError, match="batch cap"):
            initialize_nvimagecodec_decode_service(4, batch_cap=1)
        with pytest.raises(RuntimeError, match="RGB output layout"):
            initialize_nvimagecodec_decode_service(4, interleaved_output=True)
        configured = {"backend": "nvimagecodec"}
        assert (
            ImageMediaIO.merge_kwargs(configured, {"batch_cap": 20})["batch_cap"] == 20
        )
        assert (
            ImageMediaIO.merge_kwargs(
                {**configured, "batch_cap": 20}, {"batch_cap": None}
            )["batch_cap"]
            is None
        )
        with pytest.raises(ValueError, match="batch cap is fixed"):
            ImageMediaIO.merge_kwargs(configured, {"batch_cap": 1})
        with pytest.raises(ValueError, match="batch_cap"):
            ImageMediaIO.merge_kwargs(configured, {"batch_cap": True})
    finally:
        if release_equivalent is not None:
            release_equivalent()
        release()


def test_startup_diagnostics_do_not_claim_hardware_selection(caplog) -> None:
    with caplog.at_level("INFO", logger=nvimagecodec.__name__):
        service, _, _ = _service(batch_cap=20)
    try:
        assert "batch cap 20" in caplog.text
        assert "HW JPEG availability is unverified" in caplog.text
        assert "actual backend selection is unobservable" in caplog.text
        assert "fallback is permitted" in caplog.text
    finally:
        service.close()


@pytest.mark.parametrize(
    "configured,requested", [(None, 20), (20, None), (None, None), (10, 10)]
)
def test_runtime_batch_cap_equivalence_is_independent_of_initialization(
    configured, requested
):
    configured_kwargs = {"backend": "nvimagecodec", "batch_cap": configured}
    assert (
        ImageMediaIO.merge_kwargs(configured_kwargs, {"batch_cap": requested})[
            "batch_cap"
        ]
        == requested
    )
