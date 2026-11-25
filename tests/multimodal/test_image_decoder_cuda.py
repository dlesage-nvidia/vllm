# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing as mp
import os
import signal
import socket
from dataclasses import replace
from io import BytesIO
from multiprocessing.connection import Client as IpcClient
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.image_decoders.nvimagecodec import (
    SharedNvImageCodecClient,
    SharedNvImageCodecManager,
    _NvImageCodecDecoder,
    preflight_image_nvimagecodec,
)
from vllm.multimodal.media import ImageMediaIO
from vllm.multimodal.media.image import initialize_nvimagecodec_decode_service
from vllm.v1.utils import _shutdown_processes_and_cleanup_listeners

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _jpeg(width: int = 193, height: int = 97, offset: int = 0) -> bytes:
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x * 3, y * 5, x + y * 2), axis=-1).astype(np.uint8)
    pixels[..., 0] += offset
    with BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
        return buffer.getvalue()


def _decode_input(data):
    item = preflight_image_nvimagecodec(data)
    destination = torch.empty(
        (3, item.height, item.width), dtype=torch.uint8, pin_memory=True
    )
    return replace(item, destination=destination)


@pytest.mark.asyncio
async def test_selected_backend_starts_one_owner_and_returns_shared_chw() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    release = initialize_nvimagecodec_decode_service()
    try:
        data = _jpeg()
        result = await ImageMediaIO(backend="nvimagecodec").load_bytes_async(data)
        host = result.media
        expected = np.moveaxis(np.asarray(ImageMediaIO().load_bytes(data).media), -1, 0)
        assert host.device.type == "cpu" and tuple(host.shape) == expected.shape
        assert host.is_contiguous()
        np.testing.assert_allclose(host.numpy(), expected, rtol=0, atol=6)
        snapshot = host.clone()
        for offset in (1, 2):
            await ImageMediaIO(backend="nvimagecodec").load_bytes_async(
                _jpeg(offset=offset)
            )
        assert torch.equal(host, snapshot)
    finally:
        release()


def test_device_images_resize_without_invalidating_owned_outputs() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    decoder = _NvImageCodecDecoder(batch_cap=1)
    images = []
    retained = []
    try:
        for index, (width, height) in enumerate(
            ((193, 97), (385, 193), (97, 49), (513, 257))
        ):
            item = _decode_input(_jpeg(width, height, index))
            arena = decoder.submit((item,))
            if index < 2:
                images.append(arena.images[0])
            else:
                assert arena.images[0] is images[index % 2]
            (host,) = decoder.collect(arena)
            assert host is not None
            assert host.is_pinned() and tuple(host.shape) == (3, height, width)
            retained.append((host, host.clone()))
        for host, snapshot in retained:
            assert torch.equal(host, snapshot)
    finally:
        images.clear()
        decoder.close()


def test_repeated_shutdown_preserves_completed_pinned_outputs() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    decoder = _NvImageCodecDecoder(batch_cap=4)
    data = _jpeg(3840, 2160)
    pending = [
        decoder.submit(tuple(_decode_input(data) for _ in range(width)))
        for width in (4, 3)
    ]
    retained = []
    for arena in pending:
        for host in decoder.collect(arena):
            assert host is not None
            retained.append(host)
    snapshots = [host.clone() for host in retained]
    decoder.close()
    decoder.close()
    assert all(host.is_pinned() for host in retained)
    assert all(
        torch.equal(host, snapshot)
        for host, snapshot in zip(retained, snapshots, strict=True)
    )


def _shared_client_worker(endpoint, data, receipt):
    client = SharedNvImageCodecClient(endpoint)
    try:
        futures = [
            client.submit(preflight_image_nvimagecodec(encoded)) for encoded in data
        ]
        outputs = [future.result(timeout=30) for future in futures]
        for encoded, output in zip(data, outputs, strict=True):
            reference = np.moveaxis(
                np.asarray(ImageMediaIO().load_bytes(encoded).media), -1, 0
            )
            np.testing.assert_allclose(output.numpy(), reference, rtol=0, atol=6)
        expected = outputs[0].clone()
        alias = outputs[0].view(-1)
        client.close()
        assert torch.equal(alias.reshape_as(expected), expected)
        receipt.send(True)
    finally:
        client.close()
        receipt.close()


def test_shared_decoder_four_clients_exit_cleanly_with_retained_views():
    pytest.importorskip("nvidia.nvimgcodec")
    context = mp.get_context("spawn")
    manager = SharedNvImageCodecManager(4, 20)
    workers = []
    try:
        for requester in range(4):
            data = []
            for image_index in range(5):
                with BytesIO() as buffer:
                    Image.new(
                        "RGB",
                        (3840, 2160),
                        (31 + requester * 40, 64 + image_index * 20, 127),
                    ).save(buffer, "JPEG")
                    data.append(buffer.getvalue())
            parent, child = context.Pipe(duplex=False)
            process = context.Process(
                target=_shared_client_worker, args=(manager.endpoint, data, child)
            )
            process.start()
            child.close()
            workers.append((process, parent))
        for process, receipt in workers:
            assert receipt.poll(60)
            assert receipt.recv() is True
            process.join(30)
            assert process.exitcode == 0
        manager.close()
        assert manager.process.exitcode == 0
    finally:
        for process, receipt in workers:
            if process.is_alive():
                process.terminate()
                process.join(5)
            receipt.close()
        manager.close()


def test_shared_slot_cancellation_and_disconnect_preserve_storage_aliases():
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 3)
    with pytest.raises(mp.AuthenticationError):
        IpcClient(manager.endpoint.address, family="AF_UNIX", authkey=b"incorrect")
    client = SharedNvImageCodecClient(manager.endpoint)
    item = preflight_image_nvimagecodec(_jpeg(3840, 2160))
    try:
        first = [client.submit(item) for _ in range(2)]
        outputs = [future.result(timeout=30) for future in first]
        expected = outputs[0].clone()
        alias = outputs[0].view(-1)
        canceled = client.submit(item)
        assert canceled.cancel()
        del first, outputs
        with client.condition:
            assert client.condition.wait_for(
                lambda: len(client.admitted) == 1, timeout=30
            )
        next_output = client.submit(item).result(timeout=30)
        assert torch.equal(next_output, expected)
        assert torch.equal(alias.reshape_as(expected), expected)
        client.close()
        replacement = SharedNvImageCodecClient(manager.endpoint)
        try:
            assert torch.equal(replacement.submit(item).result(timeout=30), expected)
            assert torch.equal(alias.reshape_as(expected), expected)
        finally:
            replacement.close()
    finally:
        client.close()
        manager.close()


@pytest.mark.parametrize("authenticated", [False, True])
def test_shared_owner_shutdown_during_client_registration(authenticated):
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 2)
    connection: Connection | socket.socket
    if authenticated:
        connection = IpcClient(
            manager.endpoint.address, family="AF_UNIX", authkey=manager.endpoint.authkey
        )
    else:
        transport = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        transport.connect(manager.endpoint.address)
        connection = transport
    try:
        manager.close()
        assert manager.process.exitcode == 0
    finally:
        connection.close()
        manager.close()


def test_shared_owner_failure_preserves_outputs_and_rejects_new_requests():
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 2)
    client = SharedNvImageCodecClient(manager.endpoint)
    item = preflight_image_nvimagecodec(_jpeg())
    outputs = [client.submit(item).result(timeout=30) for _ in range(2)]
    try:
        manager.process.kill()
        manager.process.join(10)
        with client.condition:
            assert client.condition.wait_for(
                lambda: client.failure is not None, timeout=10
            )
        with pytest.raises(RuntimeError, match="closed"):
            client.submit(item)
        assert not client.is_healthy()
        client.close()
        with pytest.raises(RuntimeError, match="exited with"):
            manager.close()
    finally:
        client.close()
        if not manager.closed:
            with pytest.raises(RuntimeError, match="exited with"):
                manager.close()
    assert len(outputs) == 2


@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
def test_shared_owner_signal_preserves_parent_driven_cleanup(shutdown_signal):
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 2)
    client = SharedNvImageCodecClient(manager.endpoint)
    output = client.submit(preflight_image_nvimagecodec(_jpeg())).result(timeout=30)
    snapshot = output.clone()
    region_name = client.mapping.memory.name
    owner_pid = manager.process.pid
    assert owner_pid is not None
    try:
        os.kill(owner_pid, shutdown_signal)
        manager.close()
        assert manager.process.exitcode == 0
        assert torch.equal(output, snapshot)
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=region_name)
    finally:
        client.close()
        manager.close()


def test_shared_owner_shutdown_kills_unresponsive_process():
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 2)
    owner_pid = manager.process.pid
    assert owner_pid is not None
    try:
        os.kill(owner_pid, signal.SIGSTOP)
        _, status = os.waitpid(owner_pid, os.WUNTRACED)
        assert os.WIFSTOPPED(status)
        with pytest.raises(RuntimeError, match="exited with -9"):
            manager.close(timeout=0)
        assert manager.process.exitcode == -signal.SIGKILL
    finally:
        if manager.process.is_alive():
            manager.process.kill()
            manager.process.join(10)
        manager.close()


def test_shared_owner_cleanup_outlives_zero_request_shutdown_timeout():
    pytest.importorskip("nvidia.nvimgcodec")
    manager = SharedNvImageCodecManager(2, 2)
    try:
        _shutdown_processes_and_cleanup_listeners(
            [], [], timeout=0, image_decoder_manager=manager
        )
        assert manager.process.exitcode == 0
    finally:
        manager.close()


@pytest.mark.parametrize("batch_cap", [4, 20])
def test_mixed_shape_waves_preserve_retained_outputs(batch_cap: int) -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    shapes = ((193, 97), (385, 193), (97, 49), (513, 257), (1920, 1080), (3840, 2160))
    encoded = [
        _jpeg(width, height, index) for index, (width, height) in enumerate(shapes)
    ]
    expected = [
        np.moveaxis(np.asarray(ImageMediaIO().load_bytes(data).media), -1, 0)
        for data in encoded
    ]
    widths = (batch_cap, batch_cap, batch_cap // 2, batch_cap - 1, batch_cap, batch_cap)
    decoder = _NvImageCodecDecoder(batch_cap=batch_cap)
    retained = []
    try:
        for first_wave in range(0, len(widths), 2):
            pending = []
            for wave in (first_wave, first_wave + 1):
                indices = [
                    (wave + index) % len(shapes) for index in range(widths[wave])
                ]
                items = tuple(_decode_input(encoded[index]) for index in indices)
                pending.append((decoder.submit(items), indices))
            for arena, indices in pending:
                for host, index in zip(decoder.collect(arena), indices, strict=True):
                    assert host is not None and host.is_pinned()
                    assert tuple(host.shape) == expected[index].shape
                    np.testing.assert_allclose(
                        host.numpy(), expected[index], rtol=0, atol=6
                    )
                    if first_wave == 0:
                        retained.append((host, host.clone()))
                for host, snapshot in retained:
                    assert torch.equal(host, snapshot)
    finally:
        decoder.close()
    for host, snapshot in retained:
        assert torch.equal(host, snapshot)
