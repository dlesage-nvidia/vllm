# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import Mock

import pybase64
import pytest
import torch

import vllm.multimodal.media.connector as connector_module
import vllm.multimodal.media.image as image_module
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    NvImageCodecResult,
    SharedDecoderEndpoint,
)
from vllm.multimodal.media import (
    ImageMediaIO,
    MediaConnector,
    MediaDecodedBudget,
    MediaWithBytes,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("ownership", [None, "local", "shared"])
def test_image_decoder_ownership_is_not_a_configuration_option(ownership):
    with pytest.raises(ValueError, match="ownership is not configurable"):
        MultiModalConfig(
            media_io_kwargs={
                "image": {"backend": "nvimagecodec", "ownership": ownership}
            }
        )
    with pytest.raises(ValueError, match="ownership is not configurable"):
        ImageMediaIO(backend="nvimagecodec", ownership=ownership)
    with pytest.raises(ValueError, match="ownership is not configurable"):
        ImageMediaIO.merge_kwargs({"backend": "nvimagecodec"}, {"ownership": ownership})


def test_workspace_reservation_is_fixed_at_startup():
    configured = {"backend": "nvimagecodec", "workspace_reservation_mib": 640}
    with pytest.raises(ValueError, match="workspace reservation is fixed"):
        ImageMediaIO.merge_kwargs(configured, {"workspace_reservation_mib": 1280})
    assert (
        ImageMediaIO.merge_kwargs(configured, {"workspace_reservation_mib": 640})
        == configured
    )


@pytest.mark.parametrize(
    "configured_value,requested_value", [(None, 1664), (1664, None)]
)
def test_workspace_reservation_accepts_equivalent_default(
    configured_value, requested_value
):
    configured = {
        "backend": "nvimagecodec",
        "workspace_reservation_mib": configured_value,
    }
    merged = ImageMediaIO.merge_kwargs(
        configured, {"workspace_reservation_mib": requested_value}
    )
    assert merged["workspace_reservation_mib"] == requested_value


def test_batch_cap_is_fixed_before_service_initialization():
    configured = {"backend": "nvimagecodec", "batch_cap": 1}
    assert ImageMediaIO.merge_kwargs(configured, {"batch_cap": 1}) == configured
    with pytest.raises(ValueError, match="batch cap is fixed"):
        ImageMediaIO.merge_kwargs(configured, {"batch_cap": 2})
    with pytest.raises(ValueError, match="batch cap is fixed"):
        ImageMediaIO.merge_kwargs(configured, {"batch_cap": None})


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

    monkeypatch.setenv("VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", "24")
    budget = MediaDecodedBudget()
    image_io = ImageMediaIO(backend="nvimagecodec", decoded_budget=budget)
    monkeypatch.setattr(
        image_io, "_load_bytes_pillow", Mock(side_effect=AssertionError("used Pillow"))
    )
    result = await image_io.load_bytes_async(data)

    assert result.media is tensor
    assert result.original_bytes is data
    assert result.io_config == {"backend": "nvimagecodec"}
    assert budget.used == 24
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
async def test_native_decode_failure_does_not_fall_back(monkeypatch) -> None:
    error = ValueError("nvImageCodec failed to decode the JPEG.")
    data, admitted, service = _install_native(monkeypatch, error)
    image_io = ImageMediaIO(backend="nvimagecodec")
    monkeypatch.setattr(
        image_io, "_load_bytes_pillow", Mock(side_effect=AssertionError("used Pillow"))
    )

    with pytest.raises(ValueError, match="failed to decode") as caught:
        await image_io.load_bytes_async(data)
    assert caught.value is error
    service.submit.assert_called_once_with(admitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["bytes", "data", "file"])
async def test_pending_native_decode_does_not_occupy_media_worker(
    monkeypatch, tmp_path, source
) -> None:
    tensor = torch.zeros((3, 2, 3), dtype=torch.uint8)
    data, _, service = _install_native(monkeypatch, tensor)
    future: Future[NvImageCodecResult] = Future()
    submitted = asyncio.Event()

    def submit(item):
        submitted.set()
        return future

    service.submit.side_effect = submit
    path = tmp_path / "image.jpg"
    path.write_bytes(data)
    connector = MediaConnector(
        allowed_local_media_path=str(tmp_path),
        media_io_kwargs={"image": {"backend": "nvimagecodec"}},
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(connector_module, "global_thread_pool", executor)
        if source == "bytes":
            load = ImageMediaIO(backend="nvimagecodec").load_bytes_async(
                data, executor=executor
            )
        else:
            url = (
                "data:image/jpeg;base64," + pybase64.b64encode(data).decode()
                if source == "data"
                else path.as_uri()
            )
            load = connector.fetch_image_async(url)
        task = asyncio.create_task(load)
        try:
            await asyncio.wait_for(submitted.wait(), timeout=5)
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(executor, lambda: None),
                timeout=5,
            )
        finally:
            future.set_result(tensor)
        result = await task
    assert result.media is tensor and result.original_bytes == data


def test_sync_native_load_fails_instead_of_using_pillow() -> None:
    with pytest.raises(RuntimeError, match="asynchronous"):
        ImageMediaIO(backend="nvimagecodec").load_bytes(b"native JPEG")


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

    assert await ImageMediaIO().load_bytes_async(data) is fallback.return_value


def test_shared_owner_client_is_reference_counted(monkeypatch) -> None:
    monkeypatch.setattr(image_module, "_nvimagecodec_state", None)
    service = Mock()
    create = Mock(return_value=service)
    endpoint = SharedDecoderEndpoint("cpu-test-owner", b"test", 20, 0)
    image_module.set_shared_nvimagecodec_endpoint(endpoint)
    monkeypatch.setattr(image_module, "SharedNvImageCodecClient", create)
    release_first = image_module.initialize_nvimagecodec_decode_service(4, 2)
    release_second = image_module.initialize_nvimagecodec_decode_service(4, 2)

    create.assert_called_once_with(endpoint)
    service.wait_until_ready.assert_called_once_with()
    assert image_module._get_nvimagecodec_decode_service() is service
    with pytest.raises(RuntimeError, match="different process topology"):
        image_module.initialize_nvimagecodec_decode_service(4, 1)

    release_first()
    release_first()
    service.close.assert_not_called()
    release_second()
    service.close.assert_called_once_with()
    image_module.set_shared_nvimagecodec_endpoint(None)


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
