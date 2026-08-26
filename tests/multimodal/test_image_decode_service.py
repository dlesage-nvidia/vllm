# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import threading
from io import BytesIO
from typing import Any, cast

import pytest
from PIL import Image

import vllm.multimodal.image_decoders.nvimagecodec as nvimagecodec
import vllm.multimodal.media.image_decode_service as image_decode_service
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.image_decode_service import (
    ImageDecodeServiceConfig,
    NvImageCodecDecodeService,
    NvImageCodecQueueFullError,
    acquire_nvimagecodec_decode_service_lease,
    get_nvimagecodec_decode_service,
    get_nvimagecodec_decode_service_stats,
    load_images_with_service_async,
    release_nvimagecodec_decode_service_lease,
    shutdown_nvimagecodec_decode_service,
)

pytestmark = pytest.mark.cpu_test


class _FakeImageIO:
    backend = "nvimagecodec"
    image_mode = "RGB"
    rgba_background_color = (255, 255, 255)
    kwargs: dict[str, object] = {}

    def __init__(
        self,
        load,
        *,
        decoders: int = 1,
        batch_size: int = 5,
        pipeline_depth: int = 2,
        coalesce_timeout_ms: float = 1000,
    ) -> None:
        self.decoders = decoders
        self.batch_size = batch_size
        self.pipeline_depth = pipeline_depth
        self.coalesce_timeout_ms = coalesce_timeout_ms
        self._load = load

    def load_bytes_many(self, encoded_images):
        return self._load(list(encoded_images))


class _IndexedError(ValueError):
    def __init__(self, index: int, error: Exception) -> None:
        super().__init__(str(error))
        self.index = index
        self.error = error


def _result(data: bytes) -> MediaWithBytes[Image.Image]:
    return MediaWithBytes(Image.new("RGB", (1, 1)), data)


@pytest.fixture(autouse=True)
def _reset_service():
    shutdown_nvimagecodec_decode_service()
    yield
    shutdown_nvimagecodec_decode_service()


def test_five_singleton_jpegs_form_one_native_batch():
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load)
    service = get_nvimagecodec_decode_service(image_io)
    encoded = [b"\xff\xd8" + bytes([index]) for index in range(5)]
    futures = [service.submit(image_io, [item]) for item in encoded]
    results = [future.result(timeout=2)[0] for future in futures]

    assert calls == [encoded]
    assert [result.original_bytes for result in results] == encoded
    assert get_nvimagecodec_decode_service_stats().batch_widths == {5: 1}


def test_service_claims_no_more_than_pipeline_capacity_once_ready():
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(
        load,
        batch_size=2,
        pipeline_depth=3,
        coalesce_timeout_ms=0,
    )
    service = get_nvimagecodec_decode_service(image_io)
    encoded = [b"\xff\xd8" + bytes([index]) for index in range(7)]
    with service._cond:
        futures = [service.submit(image_io, [item]) for item in encoded]
    results = [future.result(timeout=2)[0] for future in futures]

    assert calls == [encoded[:6], encoded[6:]]
    assert [result.original_bytes for result in results] == encoded
    assert service.snapshot_stats().batch_widths == {6: 1, 1: 1}
    for result in results:
        result.media.close()


def test_pipeline_does_not_delay_a_ready_native_batch():
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(
        load,
        batch_size=2,
        pipeline_depth=8,
        coalesce_timeout_ms=1000,
    )
    service = get_nvimagecodec_decode_service(image_io)
    encoded = [b"\xff\xd8one", b"\xff\xd8two"]
    futures = [service.submit(image_io, [item]) for item in encoded]
    results = [future.result(timeout=2)[0] for future in futures]

    assert calls == [encoded]
    assert service.snapshot_stats().batch_widths == {2: 1}
    for result in results:
        result.media.close()


def test_partial_batch_flushes_after_timeout():
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load, coalesce_timeout_ms=1)
    service = get_nvimagecodec_decode_service(image_io)
    encoded = b"\xff\xd8partial"

    result = service.submit(image_io, [encoded]).result(timeout=2)

    assert result[0].original_bytes == encoded
    assert calls == [[encoded]]
    assert get_nvimagecodec_decode_service_stats().batch_widths == {1: 1}


def test_indexed_failure_does_not_poison_other_requests():
    calls: list[list[bytes]] = []
    original = ValueError("bad image")
    bad = b"\xff\xd8bad"

    def load(items: list[bytes]):
        calls.append(items)
        if bad in items:
            raise _IndexedError(items.index(bad), original)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load)
    encoded = [b"\xff\xd8" + bytes([index]) for index in range(4)]
    encoded.insert(2, bad)
    service = get_nvimagecodec_decode_service(image_io)
    futures = [service.submit(image_io, [item]) for item in encoded]

    with pytest.raises(ValueError) as exc_info:
        futures[2].result(timeout=2)
    assert exc_info.value is original
    assert [future.result(timeout=2)[0].original_bytes for future in futures[:2]] == (
        encoded[:2]
    )
    assert [future.result(timeout=2)[0].original_bytes for future in futures[3:]] == (
        encoded[3:]
    )
    assert calls == [encoded, encoded[:2] + encoded[3:]]


def test_process_configuration_is_immutable_until_shutdown():
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    get_nvimagecodec_decode_service(image_io)

    incompatible = _FakeImageIO(
        lambda items: [_result(item) for item in items], batch_size=4
    )
    with pytest.raises(RuntimeError, match="already configured"):
        get_nvimagecodec_decode_service(incompatible)

    incompatible_depth = _FakeImageIO(
        lambda items: [_result(item) for item in items], pipeline_depth=3
    )
    with pytest.raises(RuntimeError, match="already configured"):
        get_nvimagecodec_decode_service(incompatible_depth)


@pytest.mark.parametrize(("pipeline_depth", "queue_depth"), [(1, 4), (4, 4), (8, 8)])
def test_pipeline_depth_scales_admission_to_one_full_claim(
    pipeline_depth: int, queue_depth: int
):
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        decoders=2,
        batch_size=3,
        pipeline_depth=pipeline_depth,
    )

    config = ImageDecodeServiceConfig.from_image_io(image_io)  # type: ignore[arg-type]

    assert config.max_pending_items == queue_depth * 2 * 3
    assert config.max_pending_encoded_bytes == (
        queue_depth * 2 * nvimagecodec.NVIMAGECODEC_MAX_ENCODED_BYTES
    )


@pytest.mark.asyncio
async def test_async_cancellation_closes_claimed_result():
    claimed = threading.Event()
    release = threading.Event()
    created: list[Image.Image] = []

    def load(items: list[bytes]):
        claimed.set()
        assert release.wait(timeout=2)
        image = Image.new("RGB", (1, 1))
        created.append(image)
        return [MediaWithBytes(image, items[0])]

    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    task = asyncio.create_task(
        load_images_with_service_async(image_io, [b"\xff\xd8cancel"])
    )
    await asyncio.to_thread(claimed.wait, 2)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        if created and getattr(created[0], "fp", None) is None:
            break
        await asyncio.sleep(0.001)
    assert created
    # close() is idempotent; accessing the core after close raises ValueError.
    with pytest.raises(ValueError):
        created[0].getpixel((0, 0))


def test_non_jpeg_and_plural_inputs_are_direct_jobs():
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load)
    service = get_nvimagecodec_decode_service(image_io)
    png = BytesIO()
    Image.new("RGB", (1, 1)).save(png, "PNG")
    plural = [b"\xff\xd8one", b"\xff\xd8two"]

    png_result = service.submit(image_io, [png.getvalue()]).result(timeout=2)
    plural_result = service.submit(image_io, plural).result(timeout=2)

    assert [item.original_bytes for item in png_result] == [png.getvalue()]
    assert [item.original_bytes for item in plural_result] == plural
    assert calls == [[png.getvalue()], plural]


def test_direct_and_coalesced_jobs_share_decoder_concurrency_limit():
    active = 0
    max_active = 0
    calls: list[list[bytes]] = []
    lock = threading.Lock()
    two_started = threading.Event()
    release = threading.Event()

    def load(items: list[bytes]):
        nonlocal active, max_active
        with lock:
            calls.append(items)
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                two_started.set()
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load, decoders=2, batch_size=2)
    service = get_nvimagecodec_decode_service(image_io)
    futures = [
        service.submit(image_io, [b"\xff\xd8jpeg-1"]),
        service.submit(image_io, [b"\xff\xd8jpeg-2"]),
        service.submit(image_io, [b"direct-1"]),
    ]
    assert two_started.wait(timeout=2)

    last = service.submit(image_io, [b"direct-2"])
    assert not last.running()
    with lock:
        assert len(calls) == 2
        assert max_active == 2

    release.set()
    results = [future.result(timeout=2) for future in [*futures, last]]
    assert sorted(map(len, calls)) == [1, 1, 2]
    for result in results:
        for item in result:
            item.media.close()


def test_admission_backlog_is_bounded_and_recovers_after_cancellation():
    started = threading.Event()
    release = threading.Event()

    def load(items: list[bytes]):
        started.set()
        assert release.wait(timeout=2)
        return [_result(item) for item in items]

    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    first = service.submit(image_io, [b"a"])
    assert started.wait(timeout=2)
    waiting = service.submit(image_io, [b"b"])

    with pytest.raises(NvImageCodecQueueFullError):
        service.submit(image_io, [b"c"])

    # Keep the dispatcher out so cancel() must release admission synchronously.
    with service._cond:
        assert waiting.cancel()
        replacement = service.submit(image_io, [b"d"])

    release.set()
    try:
        assert first.result(timeout=2)[0].original_bytes == b"a"
        assert replacement.result(timeout=2)[0].original_bytes == b"d"
        stats = service.snapshot_stats()
        assert stats.submitted_images == 3
        assert stats.direct_jobs == 3
    finally:
        service.shutdown()


@pytest.mark.asyncio
async def test_async_admission_parks_beyond_bounded_decode_tiers_in_fifo_order():
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        if items == [b"a"]:
            first_started.set()
            assert release_first.wait(timeout=2)
        return [_result(item) for item in items]

    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    tasks = [
        asyncio.create_task(service.submit_async(image_io, [item]))
        for item in (b"a", b"b", b"c")
    ]
    assert await asyncio.to_thread(first_started.wait, 2)

    for _ in range(2000):
        with service._cond:
            parked = len(service._async_admission_waiters) == 1
            bounded = service._owned_items == service._waiting_items == 1
        if parked and bounded:
            break
        await asyncio.sleep(0.001)
    assert parked and bounded
    assert not tasks[2].done()

    release_first.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert calls == [[b"a"], [b"b"], [b"c"]]
    assert [result[0].original_bytes for result in results] == [b"a", b"b", b"c"]
    with service._cond:
        assert service._owned_items == service._waiting_items == 0
        assert not service._async_admission_waiters
        assert service._async_admission_handoff is None
    for result in results:
        result[0].media.close()
    service.shutdown()


@pytest.mark.asyncio
async def test_async_queued_cancellation_releases_fifo_position():
    first_started = threading.Event()
    release_first = threading.Event()
    calls: list[list[bytes]] = []

    def load(items: list[bytes]):
        calls.append(items)
        if items == [b"a"]:
            first_started.set()
            assert release_first.wait(timeout=2)
        return [_result(item) for item in items]

    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    tasks = [
        asyncio.create_task(service.submit_async(image_io, [item]))
        for item in (b"a", b"b", b"c", b"d")
    ]
    assert await asyncio.to_thread(first_started.wait, 2)

    for _ in range(2000):
        with service._cond:
            parked = len(service._async_admission_waiters) == 2
        if parked:
            break
        await asyncio.sleep(0.001)
    assert parked

    tasks[2].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[2]
    release_first.set()
    results = await asyncio.wait_for(
        asyncio.gather(tasks[0], tasks[1], tasks[3]), timeout=2
    )

    assert calls == [[b"a"], [b"b"], [b"d"]]
    with service._cond:
        assert service._owned_items == service._waiting_items == 0
        assert not service._async_admission_waiters
        assert service._async_admission_handoff is None
    for result in results:
        result[0].media.close()
    service.shutdown()


def test_cancelling_async_handoff_atomically_wakes_next_ticket():
    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        batch_size=1,
        coalesce_timeout_ms=0,
    )
    first = service._begin_async_admission(image_io, (b"a",))
    second = service._begin_async_admission(image_io, (b"b",))

    assert first.state == "handoff"
    assert second.state == "queued"
    service._cancel_async_admission(first)

    assert first.state == "released"
    assert second.state == "handoff"
    assert second.ready.result(timeout=0) is None
    with service._cond:
        assert service._waiting_items == service._waiting_bytes == 1
    service._cancel_async_admission(second)
    service.shutdown()


def test_cancelling_parked_request_wakes_next_before_fetch():
    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    first = service._begin_async_request(1)
    second = service._begin_async_request(1)
    cancelled = service._begin_async_request(1)
    following = service._begin_async_request(1)

    assert first.state == second.state == "held"
    assert cancelled.state == following.state == "queued"
    service._cancel_async_request(cancelled)
    service._release_async_request(first)

    assert cancelled.state == "released"
    assert following.state == "held"
    assert following.ready.result(timeout=0) is None
    with service._cond:
        assert service._async_request_items == 2
        assert service._async_request_holders == {second, following}
    service._release_async_request(second)
    service._release_async_request(following)
    service.shutdown()


def test_shutdown_releases_queued_and_handoff_async_admission():
    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        batch_size=1,
        coalesce_timeout_ms=0,
    )
    handoff = service._begin_async_admission(image_io, (b"a",))
    queued = service._begin_async_admission(image_io, (b"b",))
    request_holders = [service._begin_async_request(1) for _ in range(2)]
    request_waiter = service._begin_async_request(1)

    service.shutdown()

    assert handoff.state == queued.state == "released"
    assert all(ticket.state == "released" for ticket in request_holders)
    assert request_waiter.state == "released"
    with pytest.raises(RuntimeError, match="shut down"):
        queued.ready.result(timeout=0)
    with pytest.raises(RuntimeError, match="shut down"):
        request_waiter.ready.result(timeout=0)
    with pytest.raises(RuntimeError, match="shut down"):
        service._validate_async_request(request_holders[0])
    with pytest.raises(RuntimeError, match="shut down"):
        service._consume_async_admission(handoff, image_io, (b"a",))
    with service._cond:
        assert service._waiting_items == service._waiting_bytes == 0
        assert not service._async_admission_waiters
        assert service._async_admission_handoff is None
        assert service._async_request_items == 0
        assert not service._async_request_waiters
        assert not service._async_request_holders


def test_async_request_gate_rejects_nonpositive_item_count():
    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)

    with pytest.raises(ValueError, match="must be positive"):
        service._begin_async_request(0)
    with pytest.raises(ValueError, match="must be positive"):
        service._begin_async_request(-1)

    service.shutdown()


@pytest.mark.asyncio
async def test_async_request_larger_than_one_admission_tier_is_rejected():
    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        batch_size=1,
        coalesce_timeout_ms=0,
    )

    with pytest.raises(NvImageCodecQueueFullError, match="exceeds one"):
        await service.submit_async(image_io, [b"too large"])

    assert not service.started
    service.shutdown()
    with pytest.raises(RuntimeError, match="shutting down"):
        await service.submit_async(image_io, [b"too large"])


@pytest.mark.asyncio
@pytest.mark.parametrize("burst_size", [128, 256])
async def test_async_burst_larger_than_both_decode_tiers_completes(burst_size: int):
    release = threading.Event()
    two_claims_started = threading.Event()
    lock = threading.Lock()
    active = 0

    def load(items: list[bytes]):
        nonlocal active
        with lock:
            active += 1
            if active == 2:
                two_claims_started.set()
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return [_result(item) for item in items]

    image_io = _FakeImageIO(load, decoders=2, pipeline_depth=4)
    service = get_nvimagecodec_decode_service(image_io)
    bounded_items = 2 * service.config.max_pending_items
    encoded = [b"\xff\xd8" + index.to_bytes(2, "little") for index in range(burst_size)]
    tasks = [
        asyncio.create_task(load_images_with_service_async(image_io, [item]))
        for item in encoded
    ]
    assert await asyncio.to_thread(two_claims_started.wait, 2)

    for _ in range(2000):
        with service._cond:
            parked = len(service._async_admission_waiters)
        if parked == burst_size - bounded_items:
            break
        await asyncio.sleep(0.001)
    assert parked == burst_size - bounded_items

    release.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)

    assert [result[0].original_bytes for result in results] == encoded
    assert service.snapshot_stats().submitted_images == burst_size
    with service._cond:
        assert service._owned_items == service._waiting_items == 0
        assert not service._async_admission_waiters
        assert service._async_admission_handoff is None
    for result in results:
        result[0].media.close()


def test_jobs_larger_than_either_admission_tier_are_rejected():
    started = threading.Event()
    release = threading.Event()

    def load(items: list[bytes]):
        started.set()
        assert release.wait(timeout=2)
        return [_result(item) for item in items]

    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)

    with pytest.raises(NvImageCodecQueueFullError):
        service.submit(image_io, [b"too large"])
    first = service.submit(image_io, [b"a"])
    assert started.wait(timeout=2)
    with pytest.raises(NvImageCodecQueueFullError):
        service.submit(image_io, [b"b", b""])

    with service._cond:
        assert service._owned_items == 1
        assert service._owned_bytes == 1
        assert service._waiting_items == 0
        assert service._waiting_bytes == 0

    release.set()
    first.result(timeout=2)[0].media.close()
    service.shutdown()


def test_completion_releases_admission_before_waking_caller():
    started = threading.Event()
    release = threading.Event()

    def load(items: list[bytes]):
        started.set()
        assert release.wait(timeout=2)
        return [_result(item) for item in items]

    config = ImageDecodeServiceConfig(
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
        max_pending_items=1,
        max_pending_encoded_bytes=1,
    )
    service = NvImageCodecDecodeService(config)
    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    first = service.submit(image_io, [b"a"])
    assert started.wait(timeout=2)
    second = service.submit(image_io, [b"b"])
    replacement = []

    def resubmit(_future):
        try:
            replacement.append(service.submit(image_io, [b"c"]))
        except BaseException as error:
            replacement.append(error)

    first.add_done_callback(resubmit)
    release.set()
    first_result = first.result(timeout=2)
    assert replacement and not isinstance(replacement[0], BaseException)
    remaining = [second.result(timeout=2), replacement[0].result(timeout=2)]

    for result in [first_result, *remaining]:
        result[0].media.close()
    service.shutdown()


def test_unhashable_decode_semantics_use_direct_queue_without_leaking_accounting():
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    image_io.image_mode = cast(Any, [])
    service = get_nvimagecodec_decode_service(image_io)

    result = service.submit(image_io, [b"\xff\xd8jpeg"]).result(timeout=2)

    assert result[0].original_bytes == b"\xff\xd8jpeg"
    assert service.snapshot_stats().direct_jobs == 1
    with service._cond:
        assert service._owned_items == 0
        assert service._owned_bytes == 0
    result[0].media.close()


def test_successful_shutdown_logs_machine_readable_stats(monkeypatch):
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    service = get_nvimagecodec_decode_service(image_io)
    futures = [
        service.submit(image_io, [b"\xff\xd8" + bytes([index])]) for index in range(5)
    ]
    for future in futures:
        future.result(timeout=2)[0].media.close()
    calls = []
    monkeypatch.setattr(
        image_decode_service.logger,
        "info",
        lambda *args: calls.append(args),
    )

    shutdown_nvimagecodec_decode_service()

    assert len(calls) == 1
    assert calls[0][0] == "nvImageCodec decode service stats: %s"
    assert json.loads(calls[0][1]) == {
        "submitted_images": 5,
        "direct_jobs": 0,
        "batch_widths": {"5": 1},
    }


def test_atexit_shutdown_suppresses_stats_log(monkeypatch):
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    service = get_nvimagecodec_decode_service(image_io)
    result = service.submit(image_io, [b"direct"]).result(timeout=2)
    result[0].media.close()
    calls = []
    monkeypatch.setattr(
        image_decode_service.logger,
        "info",
        lambda *args: calls.append(args),
    )

    image_decode_service._shutdown_nvimagecodec_decode_service_at_exit()

    assert calls == []


def test_direct_result_count_mismatch_closes_returned_images():
    created = Image.new("RGB", (1, 1))

    def load(items: list[bytes]):
        return [MediaWithBytes(created, items[0])]

    image_io = _FakeImageIO(load, batch_size=1, coalesce_timeout_ms=0)
    service = get_nvimagecodec_decode_service(image_io)
    future = service.submit(image_io, [b"first", b"second"])

    with pytest.raises(RuntimeError, match="expected 2, got 1"):
        future.result(timeout=2)
    with pytest.raises(ValueError):
        created.getpixel((0, 0))


@pytest.mark.asyncio
async def test_coalesced_cancellation_closes_only_cancelled_result():
    claimed = threading.Event()
    release = threading.Event()
    created: dict[bytes, Image.Image] = {}

    def load(items: list[bytes]):
        claimed.set()
        assert release.wait(timeout=2)
        results = []
        for item in items:
            image = Image.new("RGB", (1, 1))
            created[item] = image
            results.append(MediaWithBytes(image, item))
        return results

    image_io = _FakeImageIO(load)
    encoded = [b"\xff\xd8" + bytes([index]) for index in range(5)]
    tasks = [
        asyncio.create_task(load_images_with_service_async(image_io, [item]))
        for item in encoded
    ]
    assert await asyncio.to_thread(claimed.wait, 2)

    tasks[2].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[2]
    release.set()
    results = await asyncio.gather(
        *(task for index, task in enumerate(tasks) if index != 2)
    )

    for _ in range(100):
        try:
            created[encoded[2]].getpixel((0, 0))
        except ValueError:
            break
        await asyncio.sleep(0.001)
    with pytest.raises(ValueError):
        created[encoded[2]].getpixel((0, 0))
    assert [result[0].original_bytes for result in results] == [
        encoded[0],
        encoded[1],
        encoded[3],
        encoded[4],
    ]
    for result in results:
        assert result[0].media.getpixel((0, 0)) == (0, 0, 0)
        result[0].media.close()


def test_pristine_service_is_reinitialized_after_fork(monkeypatch):
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    original = get_nvimagecodec_decode_service(image_io)
    child_pid = original.owner_pid + 1
    monkeypatch.setattr(image_decode_service.os, "getpid", lambda: child_pid)

    child_service = get_nvimagecodec_decode_service(image_io)

    assert child_service is not original
    assert child_service.owner_pid == child_pid


def test_started_service_fails_closed_after_fork(monkeypatch):
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        batch_size=1,
        coalesce_timeout_ms=0,
    )
    service = get_nvimagecodec_decode_service(image_io)
    result = service.submit(image_io, [b"direct"]).result(timeout=2)
    result[0].media.close()
    child_pid = service.owner_pid + 1
    monkeypatch.setattr(image_decode_service.os, "getpid", lambda: child_pid)

    with pytest.raises(RuntimeError, match="cannot be reused after fork"):
        get_nvimagecodec_decode_service(image_io)
    with pytest.raises(RuntimeError, match="cannot be reused after fork"):
        service.submit(image_io, [b"another"])


def test_atexit_ignores_parent_service_after_fork(monkeypatch):
    image_io = _FakeImageIO(
        lambda items: [_result(item) for item in items],
        batch_size=1,
        coalesce_timeout_ms=0,
    )
    service = get_nvimagecodec_decode_service(image_io)
    result = service.submit(image_io, [b"direct"]).result(timeout=2)
    result[0].media.close()
    monkeypatch.setattr(
        image_decode_service.os,
        "getpid",
        lambda: service.owner_pid + 1,
    )

    image_decode_service._shutdown_nvimagecodec_decode_service_at_exit()

    assert image_decode_service._decode_service is service


def test_service_lifetime_tracks_renderer_style_leases():
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    acquire_nvimagecodec_decode_service_lease()
    acquire_nvimagecodec_decode_service_lease()
    service = get_nvimagecodec_decode_service(image_io)

    release_nvimagecodec_decode_service_lease()
    assert get_nvimagecodec_decode_service(image_io) is service

    release_nvimagecodec_decode_service_lease()
    replacement = get_nvimagecodec_decode_service(image_io)
    assert replacement is not service
    assert service._closing


def test_last_lease_resets_decoder_pool_without_service():
    acquire_nvimagecodec_decode_service_lease()
    nvimagecodec.NvImageCodecBackend._configure_decoder_slots(2, 5)

    release_nvimagecodec_decode_service_lease()

    assert nvimagecodec._nvimagecodec_decoder_pool.max_slots is None
    assert nvimagecodec._nvimagecodec_decoder_pool.batch_size is None
    assert nvimagecodec._nvimagecodec_decoder_pool.pipeline_depth is None


def test_explicit_shutdown_resets_decoder_pool_without_service():
    nvimagecodec.NvImageCodecBackend._configure_decoder_slots(2, 5)

    shutdown_nvimagecodec_decode_service()

    assert nvimagecodec._nvimagecodec_decoder_pool.max_slots is None
    assert nvimagecodec._nvimagecodec_decoder_pool.batch_size is None
    assert nvimagecodec._nvimagecodec_decoder_pool.pipeline_depth is None


def test_last_lease_shutdown_serializes_new_service_creation(monkeypatch):
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    acquire_nvimagecodec_decode_service_lease()
    service = get_nvimagecodec_decode_service(image_io)
    service.submit(image_io, [b"direct"]).result(timeout=2)[0].media.close()
    shutdown_entered = threading.Event()
    allow_shutdown = threading.Event()
    release_done = threading.Event()
    get_started = threading.Event()
    get_done = threading.Event()
    replacement: list[NvImageCodecDecodeService] = []
    original_shutdown = service.shutdown

    def blocking_shutdown():
        shutdown_entered.set()
        assert allow_shutdown.wait(timeout=2)
        original_shutdown()

    monkeypatch.setattr(service, "shutdown", blocking_shutdown)

    def release_last_lease():
        release_nvimagecodec_decode_service_lease()
        release_done.set()

    def get_replacement():
        get_started.set()
        replacement.append(get_nvimagecodec_decode_service(image_io))
        get_done.set()

    release_thread = threading.Thread(target=release_last_lease)
    release_thread.start()
    assert shutdown_entered.wait(timeout=2)
    get_thread = threading.Thread(target=get_replacement)
    get_thread.start()
    assert get_started.wait(timeout=2)
    replacement_was_blocked = not get_done.wait(timeout=0.1)

    allow_shutdown.set()
    release_thread.join(timeout=2)
    get_thread.join(timeout=2)
    assert replacement_was_blocked
    assert release_done.is_set()
    assert get_done.is_set()
    assert replacement[0] is not service


def test_failed_shutdown_keeps_service_generation_published(monkeypatch):
    image_io = _FakeImageIO(lambda items: [_result(item) for item in items])
    acquire_nvimagecodec_decode_service_lease()
    service = get_nvimagecodec_decode_service(image_io)

    def fail_pool_shutdown():
        raise RuntimeError("injected pool shutdown failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            nvimagecodec,
            "shutdown_nvimagecodec_decoder_pool",
            fail_pool_shutdown,
        )
        with pytest.raises(RuntimeError, match="injected pool shutdown failure"):
            release_nvimagecodec_decode_service_lease()

    assert image_decode_service._decode_service is service
    assert get_nvimagecodec_decode_service(image_io) is service
    with pytest.raises(RuntimeError, match="shutting down"):
        service.submit(image_io, [b"another"])
    with pytest.raises(RuntimeError, match="did not complete"):
        shutdown_nvimagecodec_decode_service()
    assert image_decode_service._decode_service is service

    # A failed generation is deliberately not restartable; reset only the test
    # double's failure state so fixture teardown can release its resources.
    with service._cond:
        service._closing = False
    shutdown_nvimagecodec_decode_service()
