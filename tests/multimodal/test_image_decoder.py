# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.gpu_ipc_memory import (
    MultiModalGPUMemoryPool,
    set_mm_gpu_ipc_pool,
)
from vllm.multimodal.image_decoders.nvimagecodec import (
    NVIMAGECODEC_MAX_BATCH_SIZE,
    NVIMAGECODEC_MAX_PIPELINE_DEPTH,
    NVIMAGECODEC_MAX_PIXELS,
    NvImageCodecBackend,
    NvImageCodecBatchItemError,
    NvImageCodecDecoderSlot,
    _nvimagecodec_decoder_pool,
    decode_image_nvimagecodec,
    shutdown_nvimagecodec_decoder_pool,
    validate_nvimagecodec_batch_size,
    validate_nvimagecodec_decoders,
    validate_nvimagecodec_pipeline_depth,
)

pytestmark = pytest.mark.cpu_test


@contextmanager
def _fresh_decoder_pool():
    pool = _nvimagecodec_decoder_pool
    old_state = (
        pool.slots,
        pool.active,
        pool.cond,
        pool.max_slots,
        pool.batch_size,
        pool.pipeline_depth,
        pool.owner_pid,
        pool.closing,
        pool.generation,
    )
    pool.slots = []
    pool.active = 0
    pool.cond = threading.Condition()
    pool.max_slots = None
    pool.batch_size = None
    pool.pipeline_depth = None
    pool.owner_pid = os.getpid()
    pool.closing = False
    pool.generation = 0
    try:
        yield pool
    finally:
        (
            pool.slots,
            pool.active,
            pool.cond,
            pool.max_slots,
            pool.batch_size,
            pool.pipeline_depth,
            pool.owner_pid,
            pool.closing,
            pool.generation,
        ) = old_state


@pytest.fixture(autouse=True)
def _reset_decoder_state():
    set_mm_gpu_ipc_pool(None)
    with _fresh_decoder_pool():
        yield
    set_mm_gpu_ipc_pool(None)


def _metadata(
    codec: str,
    *,
    width: int = 8,
    height: int = 4,
    precision: int = 8,
    channels: int = 3,
) -> dict[str, object]:
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "precision": precision,
        "channels": channels,
    }


def _fake_nvimgcodec(
    metadata: dict[bytes, dict[str, object]],
    *,
    outcomes: dict[tuple[str, bytes], bool] | None = None,
    events: list[tuple[Any, ...]] | None = None,
    on_decode=None,
):
    outcome_map: dict[tuple[str, bytes], bool] = outcomes or {}
    event_log: list[tuple[Any, ...]] = [] if events is None else events

    class FakeCodeStream:
        def __init__(self, data: bytes, *, is_substream: bool = False):
            record = metadata[data]
            metadata_error = record.get("metadata_error")
            if isinstance(metadata_error, Exception):
                raise metadata_error
            if metadata_error:
                raise RuntimeError("parser failure")
            self.data = data
            self.codec_name = record["codec"]
            self.width = record["width"]
            self.height = record["height"]
            self.precision = record["precision"]
            self.num_channels = record["channels"]
            self.is_substream = is_substream

        def get_sub_code_stream(self, index: int):
            event_log.append(("substream", self.data, index))
            if index != 0:
                raise RuntimeError("bad substream")
            return FakeCodeStream(self.data, is_substream=True)

    class FakeBackend:
        def __init__(self, kind):
            self.kind = kind

    class FakeDecodeParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            event_log.append(("params", kwargs))

    class FakeOutput:
        def __init__(self, data: bytes, kind: str):
            self.data = data
            self.kind = kind

    class FakeDecoder:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            first_kind = kwargs["backends"][0].kind
            self.kind = "cpu" if first_kind == "cpu" else "gpu"
            event_log.append(("decoder", self.kind, kwargs))

        def decode(self, sources, *, params, cuda_stream):
            source_data = [source.data for source in sources]
            event_log.append(
                (
                    "decode",
                    self.kind,
                    source_data,
                    params.sample_format,
                    cuda_stream,
                )
            )
            if on_decode is not None:
                custom = on_decode(self.kind, sources)
                if custom is not None:
                    return custom
            return [
                FakeOutput(source.data, self.kind)
                if outcome_map.get((self.kind, source.data), True)
                else None
                for source in sources
            ]

    return SimpleNamespace(
        Backend=FakeBackend,
        BackendKind=SimpleNamespace(
            HW_GPU_ONLY="hw",
            GPU_ONLY="gpu",
            HYBRID_CPU_GPU="hybrid",
            CPU_ONLY="cpu",
        ),
        CodeStream=FakeCodeStream,
        ColorSpec=SimpleNamespace(SRGB="srgb"),
        DecodeParams=FakeDecodeParams,
        Decoder=FakeDecoder,
        SampleFormat=SimpleNamespace(I_RGB="rgb", I_RGBA="rgba"),
    )


def _install_fake_backend(
    monkeypatch,
    nvimgcodec,
    *,
    with_gpu_stream: bool = True,
):
    monkeypatch.setattr(
        "vllm.multimodal.image_decoders.nvimagecodec._load_nvimgcodec",
        lambda: nvimgcodec,
    )

    class FakeEvent:
        def synchronize(self):
            pass

    class FakeStream:
        def __init__(self, index: int):
            self.cuda_stream = 123 + index
            self.device = SimpleNamespace(index=0)

        def record_event(self):
            return FakeEvent()

        def synchronize(self):
            pass

    stream = FakeStream(0)

    def create_slot(cls):
        slot = NvImageCodecDecoderSlot(stream if with_gpu_stream else None)
        if with_gpu_stream:
            slot.extra_streams = [
                FakeStream(index) for index in range(1, NVIMAGECODEC_MAX_PIPELINE_DEPTH)
            ]
        return slot

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_create_decoder_slot",
        classmethod(create_slot),
    )

    @contextmanager
    def fake_stream_context(stream):
        yield

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_torch_stream_context",
        staticmethod(fake_stream_context),
    )
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pillow",
        staticmethod(
            lambda output, item: Image.new(
                item.output_mode,
                (item.width, item.height),
            )
        ),
    )
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pinned",
        staticmethod(
            lambda output, item: (
                output,
                np.zeros(
                    (item.height, item.width, len(item.output_mode)), dtype=np.uint8
                ),
            )
        ),
    )


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_validate_nvimagecodec_decoders_rejects_invalid_values(value: object):
    with pytest.raises(ValueError, match="positive integer"):
        validate_nvimagecodec_decoders(value)


@pytest.mark.parametrize(
    "value", [True, 0, -1, 1.5, "5", NVIMAGECODEC_MAX_BATCH_SIZE + 1]
)
def test_validate_nvimagecodec_batch_size_rejects_invalid_values(value: object):
    with pytest.raises(ValueError, match="batch_size"):
        validate_nvimagecodec_batch_size(value)


@pytest.mark.parametrize(
    "value", [True, 0, -1, 1.5, "2", NVIMAGECODEC_MAX_PIPELINE_DEPTH + 1]
)
def test_validate_nvimagecodec_pipeline_depth_rejects_invalid_values(value: object):
    with pytest.raises(ValueError, match="pipeline_depth"):
        validate_nvimagecodec_pipeline_depth(value)


def test_empty_batch_does_not_import_nvimagecodec(monkeypatch):
    monkeypatch.setattr(
        "vllm.multimodal.image_decoders.nvimagecodec._load_nvimgcodec",
        lambda: pytest.fail("empty batch imported nvImageCodec"),
    )

    assert NvImageCodecBackend.decode_many([]) == []


def test_output_modes_must_match_data(monkeypatch):
    with pytest.raises(ValueError, match="same length"):
        NvImageCodecBackend.decode_many([b"a"], output_modes=[])
    with pytest.raises(ValueError, match="RGB"):
        NvImageCodecBackend.decode_many([b"a"], output_modes=["L"])  # type: ignore[list-item]


def test_all_supported_codecs_use_expected_decoder_and_keep_positions(monkeypatch):
    codecs = ("jpeg", "jpeg2k", "tiff", "bmp", "png", "pnm", "webp", "gif")
    encoded = {name: name.encode() for name in codecs}
    metadata = {data: _metadata(codec) for codec, data in encoded.items()}
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(metadata, events=events)
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    results = NvImageCodecBackend.decode_many(
        list(encoded.values()),
        batch_size=16,
    )

    assert [result is not None for result in results] == [True] * 7 + [False]
    decode_events = [event for event in events if event[0] == "decode"]
    assert decode_events[0][1:3] == (
        "gpu",
        [encoded["jpeg"], encoded["jpeg2k"], encoded["tiff"]],
    )
    assert decode_events[1][1:3] == (
        "cpu",
        [encoded["bmp"], encoded["png"], encoded["pnm"], encoded["webp"]],
    )
    assert ("substream", encoded["tiff"], 0) in events


def test_gpu_and_cpu_decoders_are_separate_and_retained(monkeypatch):
    jpeg = b"jpeg"
    png = b"png"
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {jpeg: _metadata("jpeg"), png: _metadata("png")},
        outcomes={("gpu", jpeg): False},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(1024))

    assert all(NvImageCodecBackend.decode_many([jpeg, png], decoders=1, batch_size=5))
    decoder_events = [event for event in events if event[0] == "decoder"]
    assert [event[1] for event in decoder_events] == ["gpu", "cpu"]
    gpu_kwargs = decoder_events[0][2]
    cpu_kwargs = decoder_events[1][2]
    assert [backend.kind for backend in gpu_kwargs["backends"]] == [
        "hw",
        "gpu",
        "hybrid",
    ]
    assert [backend.kind for backend in cpu_kwargs["backends"]] == ["cpu"]
    assert cpu_kwargs["device_id"] == -1


def test_gpu_partial_miss_falls_back_only_missed_positions(monkeypatch):
    data = [b"a", b"b", b"c"]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        outcomes={("gpu", b"b"): False},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(1024))

    results = NvImageCodecBackend.decode_many(data)

    assert all(result is not None for result in results)
    decode_events = [event for event in events if event[0] == "decode"]
    assert decode_events[0][1:3] == ("gpu", data)
    assert decode_events[1][1:3] == ("cpu", [b"b"])


def test_gpu_and_cpu_miss_remains_positional_none(monkeypatch):
    data = [b"a", b"b"]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        outcomes={("gpu", b"b"): False, ("cpu", b"b"): False},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(1024))

    results = NvImageCodecBackend.decode_many(data)

    assert results[0] is not None
    assert results[1] is None


def test_native_batches_are_grouped_by_output_mode(monkeypatch):
    data = [b"rgb-a", b"rgba", b"rgb-b"]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg", channels=4) for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    results = NvImageCodecBackend.decode_many(
        data,
        output_modes=["RGB", "RGBA", "RGB"],
    )

    assert [result.mode for result in results if result is not None] == [
        "RGB",
        "RGBA",
        "RGB",
    ]
    decode_events = [event for event in events if event[0] == "decode"]
    assert decode_events[0][2:4] == ([b"rgb-a", b"rgb-b"], "rgb")
    assert decode_events[1][2:4] == ([b"rgba"], "rgba")


def test_batch_size_chunks_native_calls(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    NvImageCodecBackend.decode_many(data, batch_size=5)

    sizes = [len(event[2]) for event in events if event[0:2] == ("decode", "gpu")]
    assert sizes == [5, 5, 2]


def test_pipeline_submits_refill_before_materializing_ready_chunk(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    class FakeEvent:
        def __init__(self, stream_id: int):
            self.stream_id = stream_id

        def synchronize(self):
            events.append(("sync", self.stream_id))

    def record_event(stream):
        events.append(("record", stream.cuda_stream))
        return FakeEvent(stream.cuda_stream)

    def materialize(host_buffer, item):
        events.append(("materialize", item.index))
        return Image.new(item.output_mode, (item.width, item.height))

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_record_stream_event",
        staticmethod(record_event),
    )
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_pinned_to_pillow",
        staticmethod(materialize),
    )

    results = NvImageCodecBackend.decode_many(
        data,
        batch_size=5,
        pipeline_depth=2,
    )

    assert all(result is not None for result in results)
    decode_positions = [
        index for index, event in enumerate(events) if event[0:2] == ("decode", "gpu")
    ]
    first_sync = next(index for index, event in enumerate(events) if event[0] == "sync")
    first_materialize = next(
        index for index, event in enumerate(events) if event[0] == "materialize"
    )
    assert len(decode_positions) == 3
    assert decode_positions[1] < first_sync
    assert decode_positions[2] < first_materialize
    assert [event[1] for event in events if event[0] == "materialize"] == list(
        range(12)
    )


def test_pipeline_drains_instead_of_blocking_when_pool_fits_one_chunk(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)

    class TrackingPool(MultiModalGPUMemoryPool):
        def __init__(self, total_bytes: int):
            super().__init__(total_bytes)
            self.exhausted_attempts = 0

        def try_acquire(self, nbytes: int):
            lease = super().try_acquire(nbytes)
            if lease is None:
                self.exhausted_attempts += 1
            return lease

    # One five-image RGB chunk consumes exactly 5 * 8 * 4 * 3 bytes.
    pool = TrackingPool(480)
    set_mm_gpu_ipc_pool(pool)

    results = NvImageCodecBackend.decode_many(
        data,
        batch_size=5,
        pipeline_depth=4,
    )

    assert all(result is not None for result in results)
    assert pool.exhausted_attempts >= 2
    assert pool.available_bytes == pool.total_bytes
    sizes = [len(event[2]) for event in events if event[0:2] == ("decode", "gpu")]
    assert sizes == [5, 5, 2]


def test_pipeline_depth_one_uses_legacy_conversion_path(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pinned",
        staticmethod(lambda *_args: pytest.fail("depth one used pinned staging")),
    )

    assert all(
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=1,
        )
    )


def test_pipeline_submission_failure_releases_all_leases_and_invalidates_slot(
    monkeypatch,
):
    data = [bytes([index]) for index in range(12)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    pool = MultiModalGPUMemoryPool(4096)
    set_mm_gpu_ipc_pool(pool)

    def stage(output, item):
        if item.index == 5:
            raise RuntimeError("pinned staging failed")
        return output, np.zeros((item.height, item.width, 3), dtype=np.uint8)

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pinned",
        staticmethod(stage),
    )

    with pytest.raises(
        NvImageCodecBatchItemError, match="pinned staging failed"
    ) as exc_info:
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    assert exc_info.value.index == 5
    assert pool.available_bytes == pool.total_bytes
    slot = _nvimagecodec_decoder_pool.slots[0]
    assert slot.gpu_decoder is None
    assert slot.stream is None
    assert slot.extra_streams == []


@pytest.mark.parametrize("failure_point", ["params", "stream"])
def test_pipeline_setup_failure_releases_lease(monkeypatch, failure_point: str):
    data = [bytes([index]) for index in range(6)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    pool = MultiModalGPUMemoryPool(4096)
    set_mm_gpu_ipc_pool(pool)

    def fail(*_args, **_kwargs):
        raise RuntimeError(f"{failure_point} setup failed")

    monkeypatch.setattr(
        NvImageCodecDecoderSlot,
        "get_decode_params" if failure_point == "params" else "get_stream",
        fail,
    )

    with pytest.raises(RuntimeError, match=f"{failure_point} setup failed"):
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    assert pool.available_bytes == pool.total_bytes
    assert _nvimagecodec_decoder_pool.slots[0].gpu_decoder is None


def test_pipeline_conversion_failure_drops_device_outputs_before_lease(monkeypatch):
    import gc
    import weakref

    data = [bytes([index]) for index in range(6)]
    output_refs: list[Any] = []
    released_with_live_outputs: list[bool] = []

    class TrackedOutput:
        ndim = 2
        shape = (4, 8)
        dtype = np.uint8

    class TrackingPool(MultiModalGPUMemoryPool):
        def _release(self, lease: Any) -> None:
            gc.collect()
            released_with_live_outputs.append(
                any(output_ref() is not None for output_ref in output_refs)
            )
            super()._release(lease)

    def on_decode(kind, sources):
        if kind != "gpu":
            return None
        if sources[0].data == data[0]:
            return [None] * len(sources)
        outputs = [TrackedOutput() for _ in sources]
        output_refs.extend(weakref.ref(output) for output in outputs)
        return outputs

    real_stage = NvImageCodecBackend._decoded_to_pinned
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        on_decode=on_decode,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pinned",
        staticmethod(real_stage),
    )
    pool = TrackingPool(4096)
    set_mm_gpu_ipc_pool(pool)

    with pytest.raises(NvImageCodecBatchItemError, match="unexpected shape"):
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    gc.collect()
    assert released_with_live_outputs == [False, False]
    assert all(output_ref() is None for output_ref in output_refs)
    assert pool.available_bytes == pool.total_bytes


def test_pipeline_event_record_failure_releases_all_leases(monkeypatch):
    data = [bytes([index]) for index in range(6)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    pool = MultiModalGPUMemoryPool(4096)
    set_mm_gpu_ipc_pool(pool)
    record_count = 0

    class ImmediateEvent:
        def synchronize(self):
            pass

    def record_event(_stream):
        nonlocal record_count
        record_count += 1
        if record_count == 2:
            raise RuntimeError("event record failed")
        return ImmediateEvent()

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_record_stream_event",
        staticmethod(record_event),
    )

    with pytest.raises(RuntimeError, match="event record failed"):
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    assert pool.available_bytes == pool.total_bytes
    assert _nvimagecodec_decoder_pool.slots[0].gpu_decoder is None


def test_pipeline_event_sync_failure_discards_entire_ring(monkeypatch):
    data = [bytes([index]) for index in range(6)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    pool = MultiModalGPUMemoryPool(4096)
    set_mm_gpu_ipc_pool(pool)
    record_count = 0
    failed_syncs = 0

    class Event:
        def __init__(self, fail: bool):
            self.fail = fail

        def synchronize(self):
            nonlocal failed_syncs
            if self.fail:
                failed_syncs += 1
                raise RuntimeError("event sync failed")

    def record_event(_stream):
        nonlocal record_count
        record_count += 1
        return Event(fail=record_count == 1)

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_record_stream_event",
        staticmethod(record_event),
    )

    with pytest.raises(RuntimeError, match="event sync failed"):
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    assert failed_syncs == 2
    assert pool.available_bytes == pool.total_bytes
    assert _nvimagecodec_decoder_pool.slots[0].gpu_decoder is None


def test_pipeline_pillow_failure_closes_earlier_result_and_drains_ring(monkeypatch):
    data = [bytes([index]) for index in range(6)]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    pool = MultiModalGPUMemoryPool(4096)
    set_mm_gpu_ipc_pool(pool)
    first_image = Image.new("RGB", (8, 4))

    def materialize(_host_buffer, item):
        if item.index == 1:
            raise RuntimeError("Pillow conversion failed")
        return first_image if item.index == 0 else Image.new("RGB", (8, 4))

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_pinned_to_pillow",
        staticmethod(materialize),
    )

    with pytest.raises(
        NvImageCodecBatchItemError, match="Pillow conversion failed"
    ) as exc_info:
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )

    assert exc_info.value.index == 1
    with pytest.raises(ValueError):
        first_image.getpixel((0, 0))
    assert pool.available_bytes == pool.total_bytes
    assert _nvimagecodec_decoder_pool.slots[0].gpu_decoder is None


def test_pipeline_gpu_misses_fall_back_by_native_chunk(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        outcomes={
            ("gpu", data[1]): False,
            ("gpu", data[10]): False,
        },
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    assert all(
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=2,
        )
    )
    gpu_batches = [event[2] for event in events if event[0:2] == ("decode", "gpu")]
    cpu_batches = [event[2] for event in events if event[0:2] == ("decode", "cpu")]
    assert [len(batch) for batch in gpu_batches] == [5, 5, 2]
    assert cpu_batches == [[data[1]], [data[10]]]


def test_cpu_codec_batches_never_enter_gpu_pipeline(monkeypatch):
    data = [bytes([index]) for index in range(12)]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("png") for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec, with_gpu_stream=False)
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pinned",
        staticmethod(lambda *_args: pytest.fail("CPU codec used pinned GPU staging")),
    )

    assert all(
        NvImageCodecBackend.decode_many(
            data,
            batch_size=5,
            pipeline_depth=4,
        )
    )
    assert [len(event[2]) for event in events if event[0:2] == ("decode", "cpu")] == [
        5,
        5,
        2,
    ]


def test_aggregate_encoded_bytes_chunks_native_calls(monkeypatch):
    data = [b"aaaaaa", b"bbbbbb", b"cccccc"]
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        events=events,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    monkeypatch.setattr(
        "vllm.multimodal.image_decoders.nvimagecodec.NVIMAGECODEC_MAX_ENCODED_BYTES",
        10,
    )
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(4096))

    NvImageCodecBackend.decode_many(data, batch_size=5)

    batches = [event[2] for event in events if event[0:2] == ("decode", "gpu")]
    assert batches == [[data[0]], [data[1]], [data[2]]]


def test_aggregate_raw_bytes_chunks_and_leases_exact_bytes(monkeypatch):
    data = [b"a", b"b", b"c"]
    pool = MultiModalGPUMemoryPool(200)
    seen_available: list[int] = []

    def on_decode(kind, sources):
        if kind == "gpu":
            seen_available.append(pool.available_bytes)

    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data},
        events=events,
        on_decode=on_decode,
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(pool)

    NvImageCodecBackend.decode_many(data, batch_size=5)

    # Each 8x4 RGB image is 96 bytes, so the 200-byte pool admits 2 then 1.
    assert seen_available == [8, 104]
    assert pool.available_bytes == pool.total_bytes


def test_single_gpu_item_above_memory_pool_preserves_capacity_error(monkeypatch):
    data = b"jpeg"
    nvimgcodec = _fake_nvimgcodec({data: _metadata("jpeg", width=8, height=4)})
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(95))

    with pytest.raises(ValueError, match="Increase --mm-ipc-gpu-memory-gb"):
        NvImageCodecBackend.decode_many([data])


@pytest.mark.parametrize(
    "record",
    [
        _metadata("jpeg", precision=9),
        _metadata("jpeg", precision=0),
        _metadata("jpeg", channels=0),
        _metadata("jpeg", channels=5),
        _metadata("jpeg", width=0),
    ],
)
def test_metadata_outside_uint8_hwc_contract_falls_back(monkeypatch, record):
    data = b"image"
    nvimgcodec = _fake_nvimgcodec({data: record})
    _install_fake_backend(monkeypatch, nvimgcodec)

    assert NvImageCodecBackend.decode_many([data]) == [None]


@pytest.mark.parametrize(
    "metadata_error",
    [
        RuntimeError("parser failure"),
        ValueError("invalid metadata value"),
        TypeError("unexpected metadata type"),
        OverflowError("metadata value is too large"),
    ],
)
def test_metadata_binding_errors_fall_back_positionally(monkeypatch, metadata_error):
    data = b"image"
    nvimgcodec = _fake_nvimgcodec(
        {data: {**_metadata("jpeg"), "metadata_error": metadata_error}}
    )
    _install_fake_backend(monkeypatch, nvimgcodec)

    assert NvImageCodecBackend.decode_many([data]) == [None]


def test_global_pixel_limit_is_checked_before_decoder_creation(monkeypatch):
    import vllm.envs as envs

    data = b"image"
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec(
        {data: _metadata("jpeg", width=20, height=20)}, events=events
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)

    with pytest.raises(ValueError, match="exceed"):
        NvImageCodecBackend.decode_many([data])
    assert not any(event[0] == "decoder" for event in events)


def test_calibrated_pixel_limit_returns_positional_none(monkeypatch):
    import vllm.envs as envs

    data = b"image"
    nvimgcodec = _fake_nvimgcodec(
        {data: _metadata("jpeg", width=NVIMAGECODEC_MAX_PIXELS + 1, height=1)}
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)

    assert NvImageCodecBackend.decode_many([data]) == [None]


def test_cpu_only_codec_does_not_require_gpu_pool_or_stream(monkeypatch):
    data = b"png"
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec({data: _metadata("png")}, events=events)
    _install_fake_backend(monkeypatch, nvimgcodec, with_gpu_stream=False)

    assert NvImageCodecBackend.decode_many([data])[0] is not None
    assert _nvimagecodec_decoder_pool.slots[0].stream is None
    assert not any(event[0:2] == ("decoder", "gpu") for event in events)


def test_gpu_codec_requires_memory_pool(monkeypatch):
    data = b"jpeg"
    nvimgcodec = _fake_nvimgcodec({data: _metadata("jpeg")})
    _install_fake_backend(monkeypatch, nvimgcodec)

    with pytest.raises(RuntimeError, match="mm-ipc"):
        NvImageCodecBackend.decode_many([data])


def test_unexpected_batch_result_count_invalidates_slot(monkeypatch):
    import weakref

    data = b"jpeg"
    output_refs: list[Any] = []
    released_with_live_outputs: list[bool] = []

    class TrackedOutput:
        pass

    class TrackingPool(MultiModalGPUMemoryPool):
        def _release(self, lease: Any) -> None:
            released_with_live_outputs.append(
                any(output_ref() is not None for output_ref in output_refs)
            )
            super()._release(lease)

    def on_decode(kind, sources):
        if kind != "gpu":
            return None
        outputs = [TrackedOutput(), TrackedOutput()]
        output_refs.extend(weakref.ref(output) for output in outputs)
        return outputs

    nvimgcodec = _fake_nvimgcodec({data: _metadata("jpeg")}, on_decode=on_decode)
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(TrackingPool(1024))

    with pytest.raises(RuntimeError, match="result count"):
        NvImageCodecBackend.decode_many([data])

    assert released_with_live_outputs == [False]
    assert all(output_ref() is None for output_ref in output_refs)
    slot = _nvimagecodec_decoder_pool.slots[0]
    assert slot.gpu_decoder is None
    assert slot.cpu_decoder is None
    assert slot.stream is None


def test_runtime_failure_restores_lease_and_invalidates_slot(monkeypatch):
    data = b"jpeg"
    pool = MultiModalGPUMemoryPool(1024)

    def on_decode(kind, sources):
        if kind == "gpu":
            raise RuntimeError("CUDA failure")

    nvimgcodec = _fake_nvimgcodec({data: _metadata("jpeg")}, on_decode=on_decode)
    _install_fake_backend(monkeypatch, nvimgcodec)
    set_mm_gpu_ipc_pool(pool)

    with pytest.raises(RuntimeError, match="CUDA failure"):
        NvImageCodecBackend.decode_many([data])

    assert pool.available_bytes == pool.total_bytes
    assert _nvimagecodec_decoder_pool.slots[0].gpu_decoder is None


@pytest.mark.parametrize("conversion_raises", [False, True])
def test_device_outputs_are_released_before_memory_lease(
    monkeypatch, conversion_raises
):
    import weakref

    data = [b"a", b"b"]
    output_refs: list[Any] = []
    released_with_live_outputs: list[bool] = []

    class TrackedOutput:
        ndim = 3
        shape = (4, 8, 3)
        dtype = np.uint8

        def cpu(self):
            height = 5 if conversion_raises else 4
            return np.zeros((height, 8, 3), dtype=np.uint8)

    class TrackingPool(MultiModalGPUMemoryPool):
        def _release(self, lease: Any) -> None:
            released_with_live_outputs.append(
                any(output_ref() is not None for output_ref in output_refs)
            )
            super()._release(lease)

    def on_decode(kind, sources):
        if kind != "gpu":
            return None
        outputs = [TrackedOutput() for _ in sources]
        output_refs.extend(weakref.ref(output) for output in outputs)
        return outputs

    decoded_to_pillow = NvImageCodecBackend._decoded_to_pillow
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("jpeg") for item in data}, on_decode=on_decode
    )
    _install_fake_backend(monkeypatch, nvimgcodec)
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pillow",
        staticmethod(decoded_to_pillow),
    )
    pool = TrackingPool(1024)
    set_mm_gpu_ipc_pool(pool)

    if conversion_raises:
        with pytest.raises(ValueError, match="unexpected shape"):
            NvImageCodecBackend.decode_many(data)
    else:
        assert all(NvImageCodecBackend.decode_many(data))

    assert released_with_live_outputs == [False]
    assert all(output_ref() is None for output_ref in output_refs)
    assert pool.available_bytes == pool.total_bytes


def test_late_conversion_failure_closes_earlier_pillow_results(monkeypatch):
    data = [b"first", b"second"]
    nvimgcodec = _fake_nvimgcodec(
        {item: _metadata("png") for item in data},
    )
    _install_fake_backend(monkeypatch, nvimgcodec, with_gpu_stream=False)
    first_image = Image.new("RGB", (8, 4))

    def convert(_output, item):
        if item.index == 1:
            raise ValueError("late conversion failure")
        return first_image

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_decoded_to_pillow",
        staticmethod(convert),
    )

    with pytest.raises(ValueError, match="late conversion failure"):
        NvImageCodecBackend.decode_many(data)

    with pytest.raises(ValueError):
        first_image.getpixel((0, 0))


def test_decoder_configuration_is_process_wide():
    NvImageCodecBackend._configure_decoder_slots(2, 5, 2)
    NvImageCodecBackend._configure_decoder_slots(2, 5, 2)

    with pytest.raises(RuntimeError, match="already configured"):
        NvImageCodecBackend._configure_decoder_slots(3, 5)
    with pytest.raises(RuntimeError, match="already configured"):
        NvImageCodecBackend._configure_decoder_slots(2, 8)
    with pytest.raises(RuntimeError, match="already configured"):
        NvImageCodecBackend._configure_decoder_slots(2, 5, 3)


def test_decode_params_disable_orientation_and_depth_conversion():
    events: list[tuple[Any, ...]] = []
    nvimgcodec = _fake_nvimgcodec({}, events=events)
    slot = NvImageCodecDecoderSlot()

    rgb_params = slot.get_decode_params(nvimgcodec, "RGB")
    rgba_params = slot.get_decode_params(nvimgcodec, "RGBA")

    assert rgb_params.apply_exif_orientation is False
    assert rgba_params.apply_exif_orientation is False
    assert rgb_params.allow_any_depth is False
    assert rgba_params.allow_any_depth is False
    assert rgb_params.sample_format == "rgb"
    assert rgba_params.sample_format == "rgba"


@pytest.mark.parametrize("raises", [False, True])
def test_torch_stream_context_restores_caller_device_and_stream(monkeypatch, raises):
    import torch

    caller_stream = SimpleNamespace(device=SimpleNamespace(index=1), name="caller")
    target_stream = SimpleNamespace(device=SimpleNamespace(index=0), name="decode")
    previous_target_stream = SimpleNamespace(
        device=SimpleNamespace(index=0), name="target"
    )

    class FakeAccelerator:
        def __init__(self):
            self.device_index = 1
            self.streams = {0: previous_target_stream, 1: caller_stream}

        def current_device_index(self):
            return self.device_index

        def current_stream(self):
            return self.streams[self.device_index]

        def set_device_index(self, device_index):
            self.device_index = device_index

        def set_stream(self, stream):
            self.streams[stream.device.index] = stream
            self.device_index = stream.device.index

    accelerator = FakeAccelerator()
    monkeypatch.setattr(torch, "accelerator", accelerator)

    expectation = pytest.raises(RuntimeError) if raises else nullcontext()
    with expectation, NvImageCodecBackend._torch_stream_context(target_stream):
        assert accelerator.current_device_index() == 0
        assert accelerator.current_stream() is target_stream
        if raises:
            raise RuntimeError("decode failed")

    assert accelerator.current_device_index() == 1
    assert accelerator.current_stream() is caller_stream
    assert accelerator.streams[0] is previous_target_stream


def test_decoder_slots_bound_concurrent_native_batches(monkeypatch):
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_create_decoder_slot",
        classmethod(lambda cls: NvImageCodecDecoderSlot()),
    )
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker():
        nonlocal active, max_active
        with NvImageCodecBackend._borrow_decoder_slot():
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    NvImageCodecBackend._configure_decoder_slots(2, 5)
    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert max_active == 2


def test_decoder_pool_grows_before_reusing_idle_slots(monkeypatch):
    created_slots: list[NvImageCodecDecoderSlot] = []

    def create_slot(cls):
        slot = NvImageCodecDecoderSlot()
        created_slots.append(slot)
        return slot

    monkeypatch.setattr(
        NvImageCodecBackend,
        "_create_decoder_slot",
        classmethod(create_slot),
    )
    NvImageCodecBackend._configure_decoder_slots(2, 5)

    with NvImageCodecBackend._borrow_decoder_slot() as first:
        pass
    with NvImageCodecBackend._borrow_decoder_slot() as second:
        pass

    assert created_slots == [first, second]
    assert first is not second
    assert _nvimagecodec_decoder_pool.active == 2
    assert len(_nvimagecodec_decoder_pool.slots) == 2


def test_decoder_pool_shutdown_wakes_existing_and_late_borrowers(monkeypatch):
    pool = _nvimagecodec_decoder_pool
    monkeypatch.setattr(
        NvImageCodecBackend,
        "_create_decoder_slot",
        classmethod(lambda cls: NvImageCodecDecoderSlot()),
    )
    NvImageCodecBackend._configure_decoder_slots(1, 1)
    holder_started = threading.Event()
    release_holder = threading.Event()
    shutdown_done = threading.Event()
    existing_done = threading.Event()
    late_done = threading.Event()
    errors: list[BaseException] = []

    def holder():
        with NvImageCodecBackend._borrow_decoder_slot():
            holder_started.set()
            assert release_holder.wait(timeout=2)

    def borrower(done: threading.Event):
        try:
            with NvImageCodecBackend._borrow_decoder_slot():
                pytest.fail("borrower entered a decoder generation during shutdown")
        except RuntimeError as error:
            errors.append(error)
        finally:
            done.set()

    def shutdown():
        shutdown_nvimagecodec_decoder_pool()
        shutdown_done.set()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holder_started.wait(timeout=2)

    existing_thread = threading.Thread(target=borrower, args=(existing_done,))
    existing_thread.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with pool.cond:
            if pool.cond._waiters:
                break
        time.sleep(0.001)
    else:
        pytest.fail("borrower did not wait for the occupied decoder slot")

    shutdown_thread = threading.Thread(target=shutdown)
    shutdown_thread.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with pool.cond:
            if pool.closing:
                break
        time.sleep(0.001)
    else:
        pytest.fail("decoder pool did not enter shutdown")

    late_thread = threading.Thread(target=borrower, args=(late_done,))
    late_thread.start()
    assert existing_done.wait(timeout=2)
    assert late_done.wait(timeout=2)
    assert not shutdown_done.is_set()

    release_holder.set()
    for thread in (
        holder_thread,
        existing_thread,
        late_thread,
        shutdown_thread,
    ):
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert shutdown_done.is_set()
    assert len(errors) == 2
    assert all("shutting down" in str(error) for error in errors)
    assert pool.max_slots is None
    assert not pool.closing


def test_pristine_decoder_pool_reinitializes_after_fork(monkeypatch):
    pool = _nvimagecodec_decoder_pool
    original_condition = pool.cond
    child_pid = pool.owner_pid + 1
    monkeypatch.setattr(
        "vllm.multimodal.image_decoders.nvimagecodec.os.getpid",
        lambda: child_pid,
    )

    pool.check_pid()

    assert pool.owner_pid == child_pid
    assert pool.cond is not original_condition
    assert pool.max_slots is None


def test_initialized_decoder_pool_rejects_fork(monkeypatch):
    pool = _nvimagecodec_decoder_pool
    pool.configure(2, 5)
    child_pid = pool.owner_pid + 1
    monkeypatch.setattr(
        "vllm.multimodal.image_decoders.nvimagecodec.os.getpid",
        lambda: child_pid,
    )

    with pytest.raises(RuntimeError, match="cannot be reused after fork"):
        pool.check_pid()


def test_shutdown_releases_retained_slots_and_configuration():
    pool = _nvimagecodec_decoder_pool
    slot = NvImageCodecDecoderSlot()
    slot.gpu_decoder = object()
    slot.cpu_decoder = object()
    pool.configure(1, 5)
    pool.slots = [slot]
    pool.active = 1

    shutdown_nvimagecodec_decoder_pool()

    assert slot.gpu_decoder is None
    assert slot.cpu_decoder is None
    assert pool.slots == []
    assert pool.active == 0
    assert pool.max_slots is None
    assert pool.batch_size is None
    assert pool.pipeline_depth is None


def test_scalar_wrapper_delegates_to_native_batch(monkeypatch):
    expected = Image.new("RGBA", (2, 2))
    calls = []

    def fake_decode_many(cls, data, **kwargs):
        calls.append((data, kwargs))
        return [expected]

    monkeypatch.setattr(
        NvImageCodecBackend,
        "decode_many",
        classmethod(fake_decode_many),
    )

    actual = decode_image_nvimagecodec(
        b"image",
        output_mode="RGBA",
        decoders=3,
        batch_size=8,
        pipeline_depth=4,
    )

    assert actual is expected
    assert calls == [
        (
            [b"image"],
            {
                "output_modes": ["RGBA"],
                "decoders": 3,
                "batch_size": 8,
                "pipeline_depth": 4,
            },
        )
    ]


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_decoded_to_pillow_validates_and_copies_hwc_uint8(mode):
    channels = len(mode)
    item = SimpleNamespace(
        output_mode=mode,
        width=8,
        height=4,
    )
    host = np.zeros((4, 8, channels), dtype=np.uint8)
    device = SimpleNamespace(
        ndim=3,
        shape=(4, 8, channels),
        dtype=np.uint8,
        cpu=lambda: host,
    )

    result = NvImageCodecBackend._decoded_to_pillow(device, item)

    assert result.mode == mode
    assert result.size == (8, 4)


def test_decoded_to_pillow_copies_padded_rows():
    item = SimpleNamespace(output_mode="RGB", width=8, height=4)
    padded = np.zeros((4, 11, 3), dtype=np.uint8)
    host = padded[:, :8]
    host[:] = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    assert not host.flags.c_contiguous
    decoded = SimpleNamespace(
        ndim=3,
        shape=(4, 8, 3),
        dtype=np.uint8,
        cpu=lambda: host,
    )

    result = NvImageCodecBackend._decoded_to_pillow(decoded, item)

    np.testing.assert_array_equal(np.asarray(result), host)


def test_decoded_to_pillow_rejects_unexpected_shape():
    item = SimpleNamespace(output_mode="RGB", width=8, height=4)
    device = SimpleNamespace(ndim=3, shape=(5, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="unexpected shape"):
        NvImageCodecBackend._decoded_to_pillow(device, item)
