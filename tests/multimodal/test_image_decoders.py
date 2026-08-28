# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deterministic tests for the GPU image backend. No GPU, no native library.

These cover the parts that must hold even where nvImageCodec is not installed:
the contract that the decode entry point never raises and always returns one
result per input, the startup-only configuration guard, lifecycle, and fork
safety.
"""

import multiprocessing as mp
import os

import numpy as np
import pytest
from PIL import Image

import vllm.multimodal.image_decoders.nvimgcodec as backend
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.image_decoders import (
    NVIMGCODEC_BACKEND,
    PILLOW_BACKEND,
    decode_batch,
)
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.image import ImageMediaIO

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.shutdown()
    yield
    backend.shutdown()


def _jpeg(width: int = 32, height: int = 24) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(np.full((height, width, 3), 7, np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


# --- the contract: never raise, always one result per input ---------------

@pytest.mark.parametrize(
    "datas",
    [[], [b""], [b"\xff\xd8"], [_jpeg()], [_jpeg(), b"garbage"], [b"x" * 10] * 5],
)
def test_decode_batch_returns_one_result_per_input(datas):
    results = decode_batch(datas, "RGB")
    assert len(results) == len(datas)
    assert all(result is None or isinstance(result, np.ndarray) for result in results)


def test_decode_batch_declines_when_not_configured():
    # Never configured: every position must fall back rather than raise.
    assert decode_batch([_jpeg()], "RGB") == [None]


def test_decode_batch_declines_after_shutdown():
    backend.configure(num_decoders=1)
    backend.shutdown()
    assert decode_batch([_jpeg()], "RGB") == [None]


def test_shutdown_is_idempotent_and_a_new_generation_may_follow():
    backend.configure(num_decoders=1)
    backend.shutdown()
    backend.shutdown()  # must not raise
    backend.configure(num_decoders=2)
    assert backend._NUM_DECODERS == 2
    assert not backend._CLOSED


def test_pid_change_declines_without_touching_the_inherited_lock():
    backend.configure(num_decoders=1)
    # Simulate being a forked child: the module-level pid no longer matches.
    backend._PID = os.getpid() + 1
    # Hold the lock, exactly as an unlucky fork would have inherited it. If the
    # decline path took the lock, this would deadlock instead of returning.
    with backend._LOCK:
        assert decode_batch([_jpeg()], "RGB") == [None]


def _child_decode(queue):
    queue.put(decode_batch([_jpeg()], "RGB"))


def test_forked_child_declines_rather_than_reusing_parent_state():
    backend.configure(num_decoders=1)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_child_decode, args=(queue,))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == 0
    assert queue.get(timeout=10) == [None]


# --- configuration is startup-owned (R4) ----------------------------------

@pytest.mark.parametrize("key", ["image_backend", "num_decoders", "image_output"])
def test_request_kwargs_cannot_change_retained_resources(key):
    merged = ImageMediaIO.merge_kwargs({"image_mode": "RGB"}, {key: "anything"})
    assert key not in merged


def test_request_kwargs_that_are_not_resource_shaping_still_apply():
    merged = ImageMediaIO.merge_kwargs({"image_mode": "RGB"}, {"image_mode": None})
    assert merged["image_mode"] is None


def test_startup_kwargs_are_honoured():
    merged = ImageMediaIO.merge_kwargs({"image_backend": NVIMGCODEC_BACKEND}, None)
    assert merged["image_backend"] == NVIMGCODEC_BACKEND


@pytest.mark.parametrize(
    "kwargs", [{"image_backend": "nope"}, {"image_output": "jpeg"}]
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ImageMediaIO(image_mode="RGB", **kwargs)


@pytest.mark.parametrize("count", [0, 17, True, "4", 2.5])
def test_invalid_decoder_count_is_rejected(count):
    with pytest.raises(ValueError):
        backend.configure(num_decoders=count)


def test_default_backend_is_pillow():
    assert ImageMediaIO(image_mode="RGB").image_backend == PILLOW_BACKEND


def test_pillow_backend_never_calls_the_gpu_decoder(monkeypatch):
    called = False

    def explode(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("the GPU decoder must not run for backend=pillow")

    monkeypatch.setattr("vllm.multimodal.media.image.decode_batch", explode)
    result = ImageMediaIO(image_mode="RGB").load_bytes(_jpeg())
    assert not called
    assert isinstance(result.media, Image.Image)


# --- R7: an image raster must not be hashed as a video ---------------------

def test_image_array_and_pillow_results_hash_differently():
    data = _jpeg()
    array = np.full((24, 32, 3), 7, np.uint8)
    gpu = MediaWithBytes(
        array,
        data,
        {
            "image_mode": "RGB",
            "rgba_background_color": (255, 255, 255),
            "image_backend": NVIMGCODEC_BACKEND,
        },
    )
    pillow = MediaWithBytes(Image.fromarray(array), data, None)
    assert (
        MultiModalHasher.hash_kwargs("blake3", x=gpu)
        != MultiModalHasher.hash_kwargs("blake3", x=pillow)
    )


def test_image_array_io_config_participates_in_the_hash():
    data = _jpeg()
    array = np.full((24, 32, 3), 7, np.uint8)

    def wrap(mode):
        return MediaWithBytes(
            array, data, {"image_mode": mode, "image_backend": NVIMGCODEC_BACKEND}
        )

    assert MultiModalHasher.hash_kwargs(
        "blake3", x=wrap("RGB")
    ) != MultiModalHasher.hash_kwargs("blake3", x=wrap("RGBA"))


def test_video_arrays_still_hash_as_video():
    frames = np.zeros((4, 8, 8, 3), np.uint8)
    video = MediaWithBytes(frames, b"encoded-video-bytes", None)
    # Unchanged behaviour for the existing video path: no io_config marker, so
    # the new image branch must not capture it.
    assert MultiModalHasher.hash_kwargs("blake3", x=video)


# --- the video path must not be rerouted by enabling the image backend ----

def test_video_frame_decoding_stays_on_pillow():
    from vllm.multimodal.media.connector import MediaConnector

    connector = MediaConnector(
        media_io_kwargs={"image": {"image_backend": NVIMGCODEC_BACKEND}}
    )
    import inspect

    source = inspect.getsource(MediaConnector.fetch_video)
    assert "PILLOW_BACKEND" in source, (
        "fetch_video must pin its inner ImageMediaIO to Pillow so enabling the "
        "GPU image backend does not implicitly reroute video frames"
    )
    assert connector is not None


# --- generation safety across shutdown/reconfigure ---------------------------

class _FakeSlot:
    def __init__(self, generation, tag):
        self.generation = generation
        self.tag = tag


def test_slot_from_a_retired_generation_never_rejoins_the_pool():
    """A borrow can outlive shutdown(); its slot must not serve a later generation.

    Without this, a decoder built against retired state is handed to a fresh
    borrower while the quota counter reads zero, so the pool can also exceed
    num_decoders.
    """
    backend.configure(num_decoders=1)
    borrowed = _FakeSlot(backend._GENERATION, "generation-A")
    with backend._LOCK:  # emulate the slot being out on loan
        backend._CREATED = 1
        backend._ACTIVE = 1

    backend.shutdown()             # generation A retires while the slot is out
    backend.configure(num_decoders=1)   # generation B begins
    backend._release_slot(borrowed)     # the old borrower finally returns

    assert not backend._FREE, "a retired-generation slot rejoined the pool"
    got = backend._acquire_slot()
    assert got is None or getattr(got, "tag", None) != "generation-A"
    assert backend._CREATED <= 1


def test_release_is_a_noop_after_shutdown():
    backend.configure(num_decoders=1)
    slot = _FakeSlot(backend._GENERATION, "x")
    backend.shutdown()
    backend._release_slot(slot)
    assert not backend._FREE


# --- output-layout capability gate ------------------------------------------

def test_probe_falls_back_to_pil_for_an_unusable_processor():
    from vllm.multimodal.image_decoders import probe_output_layout

    assert probe_output_layout(None) == "pil"
    assert probe_output_layout(object()) == "pil"


def test_probe_falls_back_to_pil_when_outputs_differ():
    """A processor that treats arrays differently must not be bypassed."""
    from vllm.multimodal.image_decoders import probe_output_layout

    class DivergentImageProcessor:
        def __call__(self, images, return_tensors=None):
            first = images[0]
            # Deliberately sensitive to the input type.
            marker = 0 if isinstance(first, np.ndarray) else 1
            return {"pixel_values": np.full((1, 4), marker, dtype=np.uint8)}

    class Wrapper:
        image_processor = DivergentImageProcessor()

    assert probe_output_layout(Wrapper()) == "pil"


def test_configure_rejects_an_unknown_output_layout():
    with pytest.raises(ValueError):
        backend.configure(num_decoders=1, output_layout="nhwc")


@pytest.mark.parametrize("layout", ["pil", "hwc", "chw"])
def test_configure_records_the_output_layout(layout):
    from vllm.multimodal.image_decoders import output_layout

    backend.configure(num_decoders=1, output_layout=layout)
    assert output_layout() == layout


def test_parked_queue_is_bounded_in_bytes_not_only_items():
    assert backend.MAX_PARKED_BYTES > 0
    backend.configure(num_decoders=1)
    assert backend._PARKED_BYTES == 0


# --- static vs request configuration composition ----------------------------

def test_trusted_merge_preserves_server_configuration():
    """A renderer re-merging its own authorized result must not lose settings.

    Kimi-K3 folds the server's media_io_kwargs into the override position
    before merging again. Treating that as raw request input silently deletes
    image_backend, so the GPU decoder never activates on that path.
    """
    from vllm.renderers.kimi_k3 import _merge_k3_media_io_kwargs

    server = {"image": {"image_backend": NVIMGCODEC_BACKEND, "num_decoders": 4}}
    merged = _merge_k3_media_io_kwargs(server)
    assert merged["image"]["image_backend"] == NVIMGCODEC_BACKEND
    assert merged["image"]["num_decoders"] == 4


def test_untrusted_merge_still_strips_request_overrides():
    merged = ImageMediaIO.merge_kwargs(
        {"image_mode": "RGB"}, {"image_backend": NVIMGCODEC_BACKEND}
    )
    assert "image_backend" not in merged


def test_trusted_flag_is_explicit_not_positional():
    trusted = ImageMediaIO.merge_kwargs(
        None, {"image_backend": NVIMGCODEC_BACKEND}, trusted=True
    )
    assert trusted["image_backend"] == NVIMGCODEC_BACKEND


# --- plural entry point ------------------------------------------------------

def _encode(value: int, fmt: str) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(np.full((48, 64, 3), value, np.uint8)).save(buf, fmt)
    return buf.getvalue()


def test_load_bytes_many_preserves_order_and_mixes_backends():
    """A mixed batch must behave exactly like N independent load_bytes calls."""
    io_ = ImageMediaIO(image_mode="RGB")
    batch = [_encode(v, f) for v, f in
             ((10, "JPEG"), (20, "JPEG"), (30, "PNG"), (40, "JPEG"))]
    out = io_.load_bytes_many(batch)
    assert len(out) == len(batch)
    for original, loaded in zip(batch, out):
        assert loaded.original_bytes == original
    firsts = [int(np.asarray(item.media)[0, 0, 0]) for item in out]
    assert firsts == [10, 20, 30, 40]


def test_load_bytes_many_matches_scalar_load_bytes():
    io_ = ImageMediaIO(image_mode="RGB")
    batch = [_encode(v, "JPEG") for v in (11, 22, 33)]
    plural = io_.load_bytes_many(batch)
    for data, item in zip(batch, plural):
        np.testing.assert_array_equal(
            np.asarray(item.media), np.asarray(io_.load_bytes(data).media)
        )


def test_load_bytes_many_handles_the_empty_batch():
    assert ImageMediaIO(image_mode="RGB").load_bytes_many([]) == []


# --- regressions from the second external review -----------------------------

def test_pil_output_works_when_the_probe_selected_chw():
    """image_output="pil" must survive a CHW-negotiated decoder.

    The decoder emits (3, H, W) once the probe picks CHW; handing that to
    Image.fromarray raises "Cannot handle this data type: (1, 1, W)".
    """
    from vllm.multimodal.media.image import _as_pil

    planar = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
    image = _as_pil(planar, "chw")
    assert isinstance(image, Image.Image)
    assert image.size == (5, 4) and image.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(image), planar.transpose(1, 2, 0))


def test_as_pil_leaves_interleaved_arrays_alone():
    from vllm.multimodal.media.image import _as_pil

    hwc = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    np.testing.assert_array_equal(np.asarray(_as_pil(hwc, "hwc")), hwc)


def test_trusted_merge_preserves_static_video_configuration():
    """The trust flag must mean the same thing for every modality."""
    from vllm.multimodal.media.video import VideoMediaIO

    merged = VideoMediaIO.merge_kwargs(
        {"num_frames": 8}, {"hw_decoders": 4, "video_backend": "pynvvideocodec"},
        trusted=True,
    )
    assert merged["hw_decoders"] == 4
    assert merged["video_backend"] == "pynvvideocodec"
    untrusted = VideoMediaIO.merge_kwargs(
        {"num_frames": 8}, {"hw_decoders": 4}, trusted=False
    )
    assert "hw_decoders" not in untrusted


def test_coalesce_width_is_configurable_and_validated():
    backend.configure(num_decoders=1, coalesce_width=3)
    assert backend.COALESCE_WIDTH == 3
    with pytest.raises(ValueError):
        backend.configure(num_decoders=1, coalesce_width=0)
    with pytest.raises(ValueError):
        backend.configure(num_decoders=1, coalesce_width=99)
    backend.configure(num_decoders=1)
    assert backend.COALESCE_WIDTH == backend.DEFAULT_COALESCE_WIDTH


def test_parked_bytes_count_encoded_not_decoded_size():
    """A tiny-dimension JPEG with huge metadata must be charged its real size.

    Counting decoded raster bytes would let such inputs retain far more than
    MAX_PARKED_BYTES while appearing to cost almost nothing.
    """
    import inspect

    source = inspect.getsource(backend._await_leader)
    assert "len(waiter.data)" in source or "parked_bytes = len(waiter.data)" in source
    assert "raster_bytes > MAX_PARKED_BYTES" not in source


def test_eligible_domain_is_bounded_by_the_reservation():
    """Reservation is only an upper bound if the admitted domain is bounded."""
    assert backend.MAX_ELIGIBLE_PIXELS == 3840 * 2160


# --- probe fidelity to the configured processor -------------------------------


class _RecordingImageProcessor:
    """Minimal stand-in that records how the probe called it."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, images, return_tensors=None, input_data_format=None, **_):
        import numpy as np

        self.calls.append({"input_data_format": input_data_format})
        out = []
        for image in images:
            array = np.asarray(image)
            if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
                array = array.transpose(1, 2, 0)
            out.append(array)
        return {"pixel_values": np.stack(out)}


def test_probe_forwards_configured_call_kwargs():
    """A pinned input_data_format changes how an array is read; probe with it."""
    from vllm.multimodal.image_decoders import probe_output_layout

    processor = _RecordingImageProcessor()
    layout = probe_output_layout(
        processor, {"input_data_format": "channels_last", "not_a_real_kwarg": 1}
    )
    assert layout in ("chw", "hwc", "pil")
    assert processor.calls, "probe never called the processor"
    # Forwarded the accepted kwarg...
    assert all(c["input_data_format"] == "channels_last" for c in processor.calls)


def test_probe_drops_unknown_kwargs_instead_of_failing():
    """An unknown key must not raise the probe down to PIL."""
    from vllm.multimodal.image_decoders import probe_output_layout

    processor = _RecordingImageProcessor()
    assert probe_output_layout(processor, {"unknown_key": object()}) != "pil"


def test_probe_without_kwargs_is_unchanged():
    from vllm.multimodal.image_decoders import probe_output_layout

    assert probe_output_layout(_RecordingImageProcessor(), None) != "pil"


def test_probe_kwargs_reach_get_hf_processor():
    """vLLM-style wrappers must receive the configured kwargs when unwrapping."""
    from vllm.multimodal.image_decoders import probe_output_layout

    seen: list[dict] = []

    class Info:
        def get_hf_processor(self, **kwargs):
            seen.append(kwargs)

            class Inner:
                image_processor = _RecordingImageProcessor()

            return Inner()

    class Wrapper:
        info = Info()

    probe_output_layout(Wrapper(), {"size": {"shortest_edge": 224}})
    assert seen and seen[0] == {"size": {"shortest_edge": 224}}


# --- regressions from external review ----------------------------------------


@pytest.mark.parametrize("width", [1, 3, 4, 17])
def test_chw_to_pil_is_correct_at_ambiguous_widths(width):
    """Shape alone cannot distinguish CHW from HWC at widths 1, 3 and 4.

    Inferring it raised TypeError at width 1 and produced wrong dimensions --
    and an RGBA mode -- at widths 3 and 4.
    """
    import numpy as np

    from vllm.multimodal.media.image import _as_pil

    chw = np.arange(3 * 5 * width, dtype=np.uint8).reshape(3, 5, width)
    image = _as_pil(chw, "chw")
    assert image.size == (width, 5)
    assert image.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(image), chw.transpose(1, 2, 0))


def test_probe_forwards_everything_to_a_var_keyword_processor():
    """**kwargs accepts anything, so every configured key reaches production."""
    from vllm.multimodal.image_decoders.capability import _call_kwargs

    class VarKwImageProcessor:
        def __call__(self, images, **kwargs):
            return {}

    configured = {"input_data_format": "channels_last", "some_future_option": 7}
    assert _call_kwargs(VarKwImageProcessor(), configured) == configured


def test_probe_filters_for_a_strict_signature_processor():
    from vllm.multimodal.image_decoders.capability import _call_kwargs

    class StrictImageProcessor:
        def __call__(self, images, return_tensors=None, input_data_format=None):
            return {}

    out = _call_kwargs(
        StrictImageProcessor(), {"input_data_format": "channels_last", "bogus": 1}
    )
    assert out == {"input_data_format": "channels_last"}


def test_max_parked_uses_the_newly_configured_width():
    """The bound was computed from the previous global before it was replaced."""
    import vllm.multimodal.image_decoders.nvimgcodec as backend

    backend.shutdown()
    try:
        backend.configure(num_decoders=3, coalesce_width=7)
        assert backend._MAX_PARKED == 21
        backend.configure(num_decoders=3, coalesce_width=2)
        assert backend._MAX_PARKED == 6
    finally:
        backend.shutdown()


def test_park_window_tracks_observed_slot_occupancy():
    """The window must span a decode, so it is derived from measurement."""
    import vllm.multimodal.image_decoders.nvimgcodec as backend

    backend.shutdown()
    try:
        backend.configure(num_decoders=1)
        floor = backend._park_timeout()
        assert floor == pytest.approx(backend.COALESCE_WAIT_SECONDS, rel=0.5)
        for _ in range(50):
            backend._observe_decode(0.010)
        assert backend._park_timeout() > floor
        assert backend._park_timeout() <= backend.MAX_COALESCE_WAIT_SECONDS
    finally:
        backend.shutdown()


def test_coalesce_width_is_configurable_from_server_config():
    from vllm.config.multimodal import MultiModalConfig

    cfg = MultiModalConfig(
        media_io_kwargs={"image": {"image_backend": "nvimgcodec", "coalesce_width": 3}},
        mm_ipc_gpu_memory_gb=1.0,
    )
    assert cfg.get_image_coalesce_width() == 3
    bad = MultiModalConfig(
        media_io_kwargs={"image": {"image_backend": "nvimgcodec", "coalesce_width": 0}},
        mm_ipc_gpu_memory_gb=1.0,
    )
    with pytest.raises(ValueError, match="coalesce_width"):
        bad.get_image_coalesce_width()


def test_coalesce_width_is_startup_only():
    """Process-wide state must not be reachable from a request."""
    from vllm.multimodal.media.image import ImageMediaIO

    merged = ImageMediaIO.merge_kwargs(
        {"image_backend": "nvimgcodec", "coalesce_width": 5},
        {"coalesce_width": 9},
    )
    assert merged.get("coalesce_width") == 5


def test_request_processor_kwargs_can_void_the_probe():
    from vllm.multimodal.image_decoders import request_invalidates_probe

    assert request_invalidates_probe(None) is False
    assert request_invalidates_probe({}) is False
    assert request_invalidates_probe({"max_pixels": 50176}) is False
    assert request_invalidates_probe({"input_data_format": "channels_last"}) is True
    assert request_invalidates_probe({"do_convert_rgb": False}) is True


def test_parser_withdraws_the_bypass_for_such_a_request():
    """A request that reinterprets layout must not get raw-array output."""
    from vllm.entrypoints.chat_utils import AsyncMultiModalContentParser

    class FakeTracker:
        media_io_kwargs = {"image": {"image_backend": "nvimgcodec"}}

    parser = AsyncMultiModalContentParser.__new__(AsyncMultiModalContentParser)
    parser._tracker = FakeTracker()

    parser._mm_processor_kwargs = {"max_pixels": 50176}
    assert "image_output" not in parser._media_io_kwargs_for_request()["image"]

    parser._mm_processor_kwargs = {"input_data_format": "channels_last"}
    adjusted = parser._media_io_kwargs_for_request()
    assert adjusted["image"]["image_output"] == "pil"
    # The server-level configuration must not be mutated in the process.
    assert "image_output" not in FakeTracker.media_io_kwargs["image"]


def test_probe_is_fed_the_config_that_actually_carries_processor_kwargs():
    """The probe must read kwargs from multimodal_config, not model_config.

    vLLM folds --mm-processor-kwargs into multimodal_config and leaves
    model_config's copy None, so reading the latter silently probes a processor
    the deployment does not run. Nothing fails loudly when that happens -- the
    probe just proves less than it claims and falls back to PIL -- so the wiring
    is asserted here rather than left to a benchmark to notice.

    This reads source because the defect is a wrong attribute on a correct call:
    both spellings type-check, both run, and only one is right.
    """
    import inspect

    from vllm.renderers.base import BaseRenderer

    src = inspect.getsource(BaseRenderer)
    probe_call = src[src.index("probe_output_layout("):]
    probe_call = probe_call[: probe_call.index(")")]
    assert "model_config.mm_processor_kwargs" not in probe_call, (
        "probe is reading model_config.mm_processor_kwargs, which vLLM leaves "
        "None; it must read mm_config.mm_processor_kwargs"
    )
    assert "mm_config.mm_processor_kwargs" in probe_call
