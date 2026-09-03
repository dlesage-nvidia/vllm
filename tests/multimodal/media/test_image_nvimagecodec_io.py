# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
import threading
from concurrent.futures import Future
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pybase64
import pytest
from PIL import Image

import vllm.multimodal.media.image as im
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    NvImageCodecResult,
    _PinnedImageLease,
)
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes, VideoMediaIO

pytestmark = pytest.mark.cpu_test


def _service(result=None):
    service = Mock()
    future: Future[np.ndarray] = Future()
    if isinstance(result, BaseException):
        future.set_exception(result)
    else:
        future.set_result(result)
    service.submit.return_value = future
    return service


@pytest.fixture(autouse=True)
def _reset_service_state(monkeypatch):
    monkeypatch.setattr(im, "_nvimagecodec_state", None)


def _install_native(monkeypatch, layout, failure=None):
    data = b"native-image"
    layout = layout or "hwc_rgb"
    admitted = NvImageCodecInput(data, object(), 3, 2, 1, output_layout=layout)
    shape = (3, 2, 3) if layout == "chw_rgb" else (2, 3, 3)
    array = np.full(shape, 7, dtype=np.uint8)
    service = _service(failure or array)
    monkeypatch.setattr(im, "preflight_image_nvimagecodec", lambda *_a, **_k: admitted)
    monkeypatch.setattr(im, "_get_nvimagecodec_decode_service", lambda: service)
    return data, admitted, array, service


@pytest.mark.parametrize(
    ("layout", "async_load", "failure"),
    [
        (None, False, None),
        ("hwc_rgb", False, None),
        ("chw_rgb", True, None),
        ("chw_rgb", True, RuntimeError("systemic decode failure")),
    ],
)
@pytest.mark.asyncio
async def test_native_output_ownership_and_async_failure(
    monkeypatch, layout, async_load, failure
):
    data, admitted, array, service = _install_native(monkeypatch, layout, failure)
    monkeypatch.setattr(
        ImageMediaIO, "_load_bytes_pillow", lambda *_a, **_k: pytest.fail("spill")
    )
    io = ImageMediaIO(backend="nvimagecodec", output_layout=layout)
    if failure:
        with pytest.raises(RuntimeError, match="systemic decode failure"):
            await io.load_bytes_async(data)
        service.submit.assert_called_once_with(admitted)
        return

    result = await io.load_bytes_async(data) if async_load else io.load_bytes(data)
    assert result.original_bytes is data
    assert result.io_config == {
        "backend": "nvimagecodec",
        "output_layout": "CHW" if layout == "chw_rgb" else "HWC",
    }
    service.submit.assert_called_once_with(admitted)
    if layout is None:
        assert isinstance(result.media, Image.Image)
        array.fill(9)
        np.testing.assert_array_equal(result.media, np.full_like(array, 7))
    else:
        assert result.media is array


@pytest.mark.asyncio
async def test_async_qwen_path_selects_borrowed_jpeg_delivery(monkeypatch) -> None:
    data = b"\xff\xd8native-jpeg"
    admitted = NvImageCodecInput(data, object(), 3, 2, 1, output_layout="chw_rgb")
    lease = _PinnedImageLease(object(), width=3, height=2)
    service = _service(lease)
    monkeypatch.setattr(im, "preflight_image_nvimagecodec", lambda *_a, **_k: admitted)
    monkeypatch.setattr(im, "_get_nvimagecodec_decode_service", lambda: service)

    result = await ImageMediaIO(
        backend="nvimagecodec",
        output_layout="chw_rgb",
        _borrow_output=True,
    ).load_bytes_async(data)

    assert result.media is lease
    [submitted] = service.submit.call_args.args
    assert submitted.delivery == "borrowed"


@pytest.mark.asyncio
async def test_cancelled_async_load_releases_late_borrowed_result(monkeypatch) -> None:
    data = b"\xff\xd8native-jpeg"
    admitted = NvImageCodecInput(data, object(), 3, 2, 1, output_layout="chw_rgb")
    future: Future[NvImageCodecResult] = Future()
    assert future.set_running_or_notify_cancel()
    service = Mock()
    service.submit.return_value = future
    monkeypatch.setattr(im, "preflight_image_nvimagecodec", lambda *_a, **_k: admitted)
    monkeypatch.setattr(im, "_get_nvimagecodec_decode_service", lambda: service)
    image_io = ImageMediaIO(
        backend="nvimagecodec",
        output_layout="chw_rgb",
        _borrow_output=True,
    )

    task = asyncio.create_task(image_io.load_bytes_async(data))
    while not service.submit.called:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    lease = _PinnedImageLease(object(), width=3, height=2)
    future.set_result(lease)

    with pytest.raises(RuntimeError, match="expired"):
        lease.borrow_tensor()


def test_semantic_decline_uses_pillow_without_service(monkeypatch):
    data = b"semantic-decline"
    fallback = Mock(return_value=MediaWithBytes(object(), data))
    monkeypatch.setattr(im, "preflight_image_nvimagecodec", lambda *_a, **_k: None)
    monkeypatch.setattr(
        im, "_get_nvimagecodec_decode_service", lambda: pytest.fail("service used")
    )
    monkeypatch.setattr(ImageMediaIO, "_load_bytes_pillow", fallback)
    result = ImageMediaIO(backend="nvimagecodec").load_bytes(data)
    assert result is fallback.return_value
    fallback.assert_called_once_with(data)


def test_video_jpeg_frames_use_native_service(monkeypatch):
    data, admitted, _, service = _install_native(monkeypatch, None)
    frame = pybase64.b64encode(data).decode()
    video_io = VideoMediaIO(ImageMediaIO(backend="nvimagecodec"), num_frames=2)
    frames, _ = video_io.load_base64("video/jpeg", f"{frame},{frame}")

    assert frames.shape == (2, 2, 3, 3)
    assert [call.args for call in service.submit.call_args_list] == [(admitted,)] * 2


def test_uninitialized_and_failed_initialization_cleanup(monkeypatch):
    with pytest.raises(RuntimeError, match="not initialized in this process"):
        im._get_nvimagecodec_decode_service()
    service = _service()
    service.wait_until_ready.side_effect = RuntimeError("startup failed")
    service.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(
        im,
        "create_nvimagecodec_decode_service",
        lambda _n, *, device_index: service,
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        im.initialize_nvimagecodec_decode_service()
    service.close.assert_called_once_with()
    assert im._nvimagecodec_state is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded, use of fork.*:DeprecationWarning"
)
def test_fork_resets_inherited_service_and_locked_mutex(monkeypatch):
    parent_state = im._NvImageCodecState(os.getpid(), 3, 0, _service())
    inherited_lock = threading.Lock()
    monkeypatch.setattr(im, "_nvimagecodec_state", parent_state)
    monkeypatch.setattr(im, "_nvimagecodec_service_lock", inherited_lock)
    inherited_lock.acquire()
    try:
        child_pid = os.fork()
        if child_pid == 0:
            child_lock = im._nvimagecodec_service_lock
            reset = im._nvimagecodec_state is None
            fresh = child_lock is not inherited_lock and child_lock.acquire(False)
            os._exit(0 if reset and fresh else 1)
        _, status = os.waitpid(child_pid, 0)
    finally:
        inherited_lock.release()
    assert os.waitstatus_to_exitcode(status) == 0
    assert im._nvimagecodec_state == parent_state


def test_singleton_fixed_topology_and_pid_owned_close(monkeypatch):
    service = _service()
    create = Mock(return_value=service)
    monkeypatch.setattr(im, "create_nvimagecodec_decode_service", create)
    release_first = im.initialize_nvimagecodec_decode_service(
        api_process_count=4, device_index=2
    )
    release_second = im.initialize_nvimagecodec_decode_service(
        api_process_count=4, device_index=2
    )
    service.wait_until_ready.assert_called_once_with()
    assert im._get_nvimagecodec_decode_service() is service
    assert im._nvimagecodec_state == im._NvImageCodecState(
        os.getpid(), 4, 2, service, 2
    )
    with pytest.raises(RuntimeError, match="different process topology"):
        im.initialize_nvimagecodec_decode_service(api_process_count=4, device_index=1)
    create.assert_called_once_with(4, device_index=2)

    release_first()
    release_first()
    service.close.assert_not_called()
    assert im._nvimagecodec_state == im._NvImageCodecState(os.getpid(), 4, 2, service)
    release_second()
    service.close.assert_called_once_with()
    assert im._nvimagecodec_state is None

    replacement = _service()
    create.return_value = replacement
    release_replacement = im.initialize_nvimagecodec_decode_service(
        api_process_count=2, device_index=1
    )
    assert im._get_nvimagecodec_decode_service() is replacement
    release_replacement()
    replacement.close.assert_called_once_with()

    stale = _service()
    create.return_value = stale
    release_stale = im.initialize_nvimagecodec_decode_service()
    im._close_nvimagecodec_decode_service()
    successor = _service()
    create.return_value = successor
    release_successor = im.initialize_nvimagecodec_decode_service()
    release_stale()
    successor.close.assert_not_called()
    release_successor()
    successor.close.assert_called_once_with()

    im._close_nvimagecodec_decode_service()
    im._close_nvimagecodec_decode_service()
    service.close.assert_called_once_with()
    assert im._nvimagecodec_state is None

    foreign = _service()
    im._nvimagecodec_state = im._NvImageCodecState(os.getpid() + 1, 4, 2, foreign)
    im._close_nvimagecodec_decode_service()
    foreign.close.assert_not_called()
    assert im._nvimagecodec_state is None


def test_assigned_gpu_is_mapped_to_visible_decoder_ordinal(monkeypatch):
    from vllm.platforms import current_platform

    variable = current_platform.device_control_env_var
    monkeypatch.delenv(variable, raising=False)
    assert im._nvimagecodec_device_index(None) == 0
    assert im._nvimagecodec_device_index([2]) == 2

    monkeypatch.setenv(variable, "4,2")
    assert im._nvimagecodec_device_index([2]) == 1
    with pytest.raises(RuntimeError, match="physical GPU 3 is not visible"):
        im._nvimagecodec_device_index([3])


def test_renderer_shutdown_releases_image_backend_once(monkeypatch):
    from vllm.renderers.base import BaseRenderer

    release = Mock()
    monkeypatch.setattr(
        im, "initialize_image_decode_backend", Mock(return_value=release)
    )
    renderer = SimpleNamespace(config=object(), _resources=ExitStack())

    BaseRenderer.initialize_image_decode_backend(renderer)
    BaseRenderer.shutdown(renderer)
    BaseRenderer.shutdown(renderer)

    release.assert_called_once_with()
