# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from concurrent.futures import Future
from unittest.mock import Mock

import pytest

import vllm.multimodal.media.image as image_module
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    NvImageCodecResult,
    _PinnedImageLease,
)
from vllm.multimodal.media import ImageMediaIO

pytestmark = pytest.mark.cpu_test


def _install_native(monkeypatch, future):
    data = b"native JPEG"
    admitted = NvImageCodecInput(object(), data, 3, 2)
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
    future: Future[NvImageCodecResult] = Future()
    future.set_result(lease)
    data, admitted, service = _install_native(monkeypatch, future)

    result = await ImageMediaIO(backend="nvimagecodec").load_bytes_async(data)

    assert result.media is lease
    assert result.original_bytes is data
    assert result.io_config == {"backend": "nvimagecodec"}
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
async def test_cancelled_load_releases_late_native_result(monkeypatch) -> None:
    future: Future[NvImageCodecResult] = Future()
    assert future.set_running_or_notify_cancel()
    data, _, service = _install_native(monkeypatch, future)

    task = asyncio.create_task(
        ImageMediaIO(backend="nvimagecodec").load_bytes_async(data)
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
async def test_native_backend_never_uses_pillow(monkeypatch) -> None:
    data = b"unsupported"
    fallback = Mock(side_effect=AssertionError("Pillow decoded pixels"))
    monkeypatch.setattr(ImageMediaIO, "_load_bytes_pillow", fallback)

    image_io = ImageMediaIO(backend="nvimagecodec")
    with pytest.raises(ValueError, match="asynchronous loading only"):
        image_io.load_bytes(data)

    monkeypatch.setattr(
        image_module,
        "preflight_image_nvimagecodec",
        Mock(side_effect=ValueError("unsupported native input")),
    )
    with pytest.raises(ValueError, match="unsupported native input"):
        await image_io.load_bytes_async(data)
    fallback.assert_not_called()


def test_native_backend_cannot_change_per_request() -> None:
    with pytest.raises(ValueError, match="fixed at startup"):
        ImageMediaIO.merge_kwargs({"backend": "nvimagecodec"}, {"backend": "pillow"})


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
