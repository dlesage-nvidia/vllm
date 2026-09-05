# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import Future
from unittest.mock import Mock

import pytest
import torch

import vllm.multimodal.media.image as image_module
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    NvImageCodecResult,
)
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_service_state(monkeypatch):
    monkeypatch.setattr(image_module, "_nvimagecodec_state", None)


def _install_native(monkeypatch, result):
    data = b"native JPEG"
    admitted = NvImageCodecInput(data, object(), 3, 2)
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
async def test_async_jpeg_uses_native_service(monkeypatch) -> None:
    tensor = torch.zeros((3, 2, 3), dtype=torch.uint8)
    data, admitted, service = _install_native(monkeypatch, tensor)

    image_io = ImageMediaIO(backend="nvimagecodec")
    image_io._load_bytes_pillow = Mock(side_effect=AssertionError("used Pillow"))
    result = await image_io.load_bytes_async(data)

    assert result.media is tensor
    assert result.original_bytes is data
    assert result.io_config == {"backend": "nvimagecodec"}
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
async def test_native_decode_failure_does_not_fall_back(monkeypatch) -> None:
    data, admitted, service = _install_native(monkeypatch, None)
    image_io = ImageMediaIO(backend="nvimagecodec")
    image_io._load_bytes_pillow = Mock(side_effect=AssertionError("used Pillow"))

    with pytest.raises(RuntimeError, match="failed to decode"):
        await image_io.load_bytes_async(data)
    service.submit.assert_called_once_with(admitted)


def test_sync_native_load_fails_instead_of_using_pillow(monkeypatch) -> None:
    tensor = torch.zeros((3, 2, 3), dtype=torch.uint8)
    data, _, service = _install_native(monkeypatch, tensor)
    image_io = ImageMediaIO(backend="nvimagecodec")
    image_io._load_bytes_pillow = Mock(side_effect=AssertionError("used Pillow"))
    with pytest.raises(RuntimeError, match="asynchronous"):
        image_io.load_bytes(data)
    service.submit.assert_not_called()


@pytest.mark.asyncio
async def test_pillow_backend_stays_on_pillow(monkeypatch) -> None:
    data = b"pillow"
    fallback = Mock(return_value=MediaWithBytes(object(), data))
    monkeypatch.setattr(ImageMediaIO, "_load_bytes_pillow", fallback)
    monkeypatch.setattr(
        image_module,
        "_get_nvimagecodec_decode_service",
        lambda: pytest.fail("service used"),
    )

    image_io = ImageMediaIO()
    assert image_io.load_bytes(data) is fallback.return_value
    assert await image_io.load_bytes_async(data) is fallback.return_value
    assert fallback.call_count == 2


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
