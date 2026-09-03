# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

import vllm.multimodal.media.image as image_module
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    NvImageCodecResult,
    _PinnedImageLease,
)
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_service_state(monkeypatch):
    monkeypatch.setattr(image_module, "_nvimagecodec_state", None)


def _install_native(monkeypatch, result):
    data = b"native JPEG"
    admitted = NvImageCodecInput(object(), 3, 2)
    future: Future[NvImageCodecResult] = Future()
    if isinstance(result, BaseException):
        future.set_exception(result)
    else:
        future.set_result(result)
    service = Mock()
    service.submit.return_value = future
    monkeypatch.setattr(
        image_module, "preflight_image_nvimagecodec", lambda *_a, **_k: admitted
    )
    monkeypatch.setattr(
        image_module, "_get_nvimagecodec_decode_service", lambda: service
    )
    return data, admitted, service


@pytest.mark.asyncio
async def test_async_borrowed_jpeg_uses_native_service(monkeypatch) -> None:
    lease = _PinnedImageLease(object(), width=3, height=2)
    data, admitted, service = _install_native(monkeypatch, lease)

    result = await ImageMediaIO(
        backend="nvimagecodec", _borrow_output=True
    ).load_bytes_async(data)

    assert result.media is lease
    assert result.original_bytes is data
    assert result.io_config == {"backend": "nvimagecodec"}
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
async def test_native_decline_falls_back_to_pillow(monkeypatch) -> None:
    fallback = MediaWithBytes(object(), b"native JPEG")
    data, admitted, service = _install_native(monkeypatch, None)
    image_io = ImageMediaIO(backend="nvimagecodec", _borrow_output=True)
    monkeypatch.setattr(image_io, "_load_bytes_pillow", Mock(return_value=fallback))

    assert await image_io.load_bytes_async(data) is fallback
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
async def test_cancelled_load_releases_late_native_result(monkeypatch) -> None:
    data = b"native JPEG"
    admitted = NvImageCodecInput(object(), 3, 2)
    future: Future[NvImageCodecResult] = Future()
    assert future.set_running_or_notify_cancel()
    service = Mock()
    service.submit.return_value = future
    monkeypatch.setattr(
        image_module, "preflight_image_nvimagecodec", lambda *_a, **_k: admitted
    )
    monkeypatch.setattr(
        image_module, "_get_nvimagecodec_decode_service", lambda: service
    )

    task = asyncio.create_task(
        ImageMediaIO(backend="nvimagecodec", _borrow_output=True).load_bytes_async(data)
    )
    while not service.submit.called:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    lease = _PinnedImageLease(object(), width=3, height=2)
    future.set_result(lease)
    with pytest.raises(RuntimeError, match="expired"):
        lease.borrow_tensor()


@pytest.mark.asyncio
async def test_noneligible_and_sync_loads_use_pillow(monkeypatch) -> None:
    data = b"fallback"
    fallback = Mock(return_value=MediaWithBytes(object(), data))
    monkeypatch.setattr(ImageMediaIO, "_load_bytes_pillow", fallback)
    monkeypatch.setattr(
        image_module,
        "_get_nvimagecodec_decode_service",
        lambda: pytest.fail("service used"),
    )

    image_io = ImageMediaIO(backend="nvimagecodec")
    assert image_io.load_bytes(data) is fallback.return_value
    assert await image_io.load_bytes_async(data) is fallback.return_value
    monkeypatch.setattr(
        image_module, "preflight_image_nvimagecodec", lambda *_a, **_k: None
    )
    borrowed = ImageMediaIO(backend="nvimagecodec", _borrow_output=True)
    assert await borrowed.load_bytes_async(data) is fallback.return_value
    assert fallback.call_count == 3


def test_process_local_service_is_reference_counted(monkeypatch) -> None:
    service = Mock()
    create = Mock(return_value=service)
    monkeypatch.setattr(image_module, "create_nvimagecodec_decode_service", create)
    release_first = image_module.initialize_nvimagecodec_decode_service(4, 2)
    release_second = image_module.initialize_nvimagecodec_decode_service(4, 2)

    create.assert_called_once_with(4, device_index=2)
    service.wait_until_ready.assert_called_once_with()
    assert image_module._get_nvimagecodec_decode_service() is service
    with pytest.raises(RuntimeError, match="different process topology"):
        image_module.initialize_nvimagecodec_decode_service(4, 1)

    release_first()
    release_first()
    service.close.assert_not_called()
    release_second()
    service.close.assert_called_once_with()


def test_failed_service_initialization_is_cleaned_up(monkeypatch) -> None:
    service = Mock()
    service.wait_until_ready.side_effect = RuntimeError("startup failed")
    monkeypatch.setattr(
        image_module, "create_nvimagecodec_decode_service", lambda *_a, **_k: service
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        image_module.initialize_nvimagecodec_decode_service()
    service.close.assert_called_once_with()
    assert image_module._nvimagecodec_state is None


def test_assigned_gpu_is_mapped_to_visible_decoder_ordinal(monkeypatch) -> None:
    from vllm.platforms import current_platform

    variable = current_platform.device_control_env_var
    monkeypatch.delenv(variable, raising=False)
    assert image_module._nvimagecodec_device_index(None) == 0
    assert image_module._nvimagecodec_device_index([2]) == 2

    monkeypatch.setenv(variable, "4,2")
    assert image_module._nvimagecodec_device_index([2]) == 1
    with pytest.raises(RuntimeError, match="physical GPU 3 is not visible"):
        image_module._nvimagecodec_device_index([3])


def test_fork_reset_drops_inherited_service(monkeypatch) -> None:
    state = image_module._NvImageCodecState(1, 0, Mock())
    monkeypatch.setattr(image_module, "_nvimagecodec_state", state)
    old_lock = threading.Lock()
    monkeypatch.setattr(image_module, "_nvimagecodec_service_lock", old_lock)
    image_module._reset_nvimagecodec_decode_service_after_fork()
    assert image_module._nvimagecodec_state is None
    assert image_module._nvimagecodec_service_lock is not old_lock
