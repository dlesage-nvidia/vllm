# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import binascii
import io
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pybase64
import pytest
from PIL import Image

import vllm.multimodal.media.video as video_module
import vllm.multimodal.video_decoders.pynvvideocodec as pynv_module
from vllm.assets.base import get_vllm_public_assets
from vllm.assets.video import (
    video_get_metadata,
    video_to_ndarrays,
    video_to_pil_images_list,
)
from vllm.multimodal.media import ImageMediaIO, MediaWithBytes, VideoMediaIO
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image_decode_service import (
    shutdown_nvimagecodec_decode_service,
)
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    VIDEO_LOADER_REGISTRY,
    VLLM_VIDEO_INPUT_DATA_FORMAT_KEY,
    VideoLoader,
    validate_video_processor_output_layout,
)
from vllm.multimodal.video_decoders.pynvvideocodec import (
    PyNvVideoCodecVideoBackendMixin,
    _pynvvc_frames_to_pinned_host,
)

from ..utils import cosine_similarity, create_video_from_image, normalize_image

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


@VIDEO_LOADER_REGISTRY.register("assert_10_frames_1_fps")
class Assert10Frames1FPSVideoLoader(VideoLoader):
    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, fps: float = -1.0, **kwargs
    ) -> npt.NDArray:
        assert num_frames == 10, "bad num_frames"
        assert fps == 1.0, "bad fps"
        return FAKE_OUTPUT_2


@VIDEO_LOADER_REGISTRY.register("test_layout_kwargs_passthrough")
class LayoutKwargsPassthroughVideoLoader(VideoLoader):
    received_kwargs: dict = {}

    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict]:
        cls.received_kwargs = kwargs
        return np.zeros((1, 2, 4, 3), dtype=np.uint8), {}


def test_video_media_io_kwargs(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "assert_10_frames_1_fps")
        imageio = ImageMediaIO()

        # Verify that different args pass/fail assertions as expected.
        videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 1.0})
        _ = videoio.load_bytes(b"test")

        videoio = VideoMediaIO(
            imageio, **{"num_frames": 10, "fps": 1.0, "not_used": "not_used"}
        )
        _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad num_frames"):
            videoio = VideoMediaIO(imageio, **{})
            _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad num_frames"):
            videoio = VideoMediaIO(imageio, **{"num_frames": 9, "fps": 1.0})
            _ = videoio.load_bytes(b"test")

        with pytest.raises(AssertionError, match="bad fps"):
            videoio = VideoMediaIO(imageio, **{"num_frames": 10, "fps": 2.0})
            _ = videoio.load_bytes(b"test")


@pytest.mark.parametrize("is_color", [True, False])
@pytest.mark.parametrize("fourcc, ext", [("mp4v", "mp4"), ("XVID", "avi")])
def test_opencv_video_io_colorspace(tmp_path, is_color: bool, fourcc: str, ext: str):
    """
    Test all functions that use OpenCV for video I/O return RGB format.
    Both RGB and grayscale videos are tested.
    """
    image_path = get_vllm_public_assets(
        filename="stop_sign.jpg", s3_prefix="vision_model_images"
    )
    image = Image.open(image_path)

    if not is_color:
        image_path = f"{tmp_path}/test_grayscale_image.png"
        image = image.convert("L")
        image.save(image_path)
        # Convert to gray RGB for comparison
        image = image.convert("RGB")
    video_path = f"{tmp_path}/test_RGB_video.{ext}"
    create_video_from_image(
        image_path,
        video_path,
        num_frames=2,
        is_color=is_color,
        fourcc=fourcc,
    )

    frames = video_to_ndarrays(video_path)
    for frame in frames:
        sim = cosine_similarity(
            normalize_image(np.array(frame)), normalize_image(np.array(image))
        )
        assert np.sum(np.isnan(sim)) / sim.size < 0.001
        assert np.nanmean(sim) > 0.99

    pil_frames = video_to_pil_images_list(video_path)
    for frame in pil_frames:
        sim = cosine_similarity(
            normalize_image(np.array(frame)), normalize_image(np.array(image))
        )
        assert np.sum(np.isnan(sim)) / sim.size < 0.001
        assert np.nanmean(sim) > 0.99

    io_frames, _ = VideoMediaIO(ImageMediaIO()).load_file(Path(video_path))
    for frame in io_frames:
        sim = cosine_similarity(
            normalize_image(np.array(frame)), normalize_image(np.array(image))
        )
        assert np.sum(np.isnan(sim)) / sim.size < 0.001
        assert np.nanmean(sim) > 0.99


def test_opencv_video_metadata_matches_sampled_frame_timeline(tmp_path):
    image_path = f"{tmp_path}/test_metadata_image.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_path)
    video_path = f"{tmp_path}/test_metadata_video.mp4"
    create_video_from_image(image_path, video_path, num_frames=10, fps=5.0)

    metadata = video_get_metadata(video_path, num_frames=4)

    assert metadata["fps"] == pytest.approx(5.0)
    assert metadata["duration"] == pytest.approx(2.0)
    assert metadata["frames_indices"] == [0, 3, 6, 9]
    assert metadata["total_num_frames"] == 4


NUM_FRAMES = 10
FAKE_OUTPUT_1 = np.random.rand(NUM_FRAMES, 1280, 720, 3)
FAKE_OUTPUT_2 = np.random.rand(NUM_FRAMES, 1280, 720, 3)


@VIDEO_LOADER_REGISTRY.register("test_video_backend_override_1")
class TestVideoBackendOverride1(VideoLoader):
    """Test loader that returns FAKE_OUTPUT_1 to verify backend selection."""

    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict]:
        return FAKE_OUTPUT_1, {"video_backend": "test_video_backend_override_1"}


@VIDEO_LOADER_REGISTRY.register("test_video_backend_override_2")
class TestVideoBackendOverride2(VideoLoader):
    """Test loader that returns FAKE_OUTPUT_2 to verify backend selection."""

    @classmethod
    def load_bytes(
        cls, data: bytes, num_frames: int = -1, **kwargs
    ) -> tuple[npt.NDArray, dict]:
        return FAKE_OUTPUT_2, {"video_backend": "test_video_backend_override_2"}


def test_video_media_io_backend_kwarg_override(monkeypatch: pytest.MonkeyPatch):
    """
    Test that video_backend kwarg can override the VLLM_VIDEO_LOADER_BACKEND
    environment variable.

    This allows users to dynamically select a different video backend
    via --media-io-kwargs without changing the global env var, which is
    useful when plugins set a default backend but a specific request
    needs a different one.
    """
    with monkeypatch.context() as m:
        # Set the env var to one backend
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_video_backend_override_1")

        imageio = ImageMediaIO()

        # Without video_backend kwarg, should use env var backend
        videoio_default = VideoMediaIO(imageio, num_frames=10)
        frames_default, metadata_default = videoio_default.load_bytes(b"test")
        np.testing.assert_array_equal(frames_default, FAKE_OUTPUT_1)
        assert metadata_default["video_backend"] == "test_video_backend_override_1"

        # With video_backend kwarg, should override env var
        videoio_override = VideoMediaIO(
            imageio, num_frames=10, video_backend="test_video_backend_override_2"
        )
        frames_override, metadata_override = videoio_override.load_bytes(b"test")
        np.testing.assert_array_equal(frames_override, FAKE_OUTPUT_2)
        assert metadata_override["video_backend"] == "test_video_backend_override_2"


def test_video_media_io_backend_kwarg_not_passed_to_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Test that video_backend kwarg is consumed by VideoMediaIO and NOT passed
    through to the underlying video loader's load_bytes method.

    This ensures the kwarg is properly popped from kwargs before forwarding.
    """

    @VIDEO_LOADER_REGISTRY.register("test_reject_video_backend_kwarg")
    class RejectVideoBackendKwargLoader(VideoLoader):
        """Test loader that fails if video_backend is passed through."""

        @classmethod
        def load_bytes(
            cls, data: bytes, num_frames: int = -1, **kwargs
        ) -> tuple[npt.NDArray, dict]:
            # This should never receive video_backend in kwargs
            if "video_backend" in kwargs:
                raise AssertionError(
                    "video_backend should be consumed by VideoMediaIO, "
                    "not passed to loader"
                )
            return FAKE_OUTPUT_1, {"received_kwargs": list(kwargs.keys())}

    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_reject_video_backend_kwarg")

        imageio = ImageMediaIO()

        # Even when video_backend is provided, it should NOT be passed to loader
        videoio = VideoMediaIO(
            imageio,
            num_frames=10,
            video_backend="test_reject_video_backend_kwarg",
            other_kwarg="should_pass_through",
        )

        # This should NOT raise AssertionError
        frames, metadata = videoio.load_bytes(b"test")
        np.testing.assert_array_equal(frames, FAKE_OUTPUT_1)
        # Verify other kwargs are still passed through
        assert "other_kwarg" in metadata["received_kwargs"]


def test_video_media_io_backend_env_var_fallback(monkeypatch: pytest.MonkeyPatch):
    """
    Test that when video_backend kwarg is None or not provided,
    VideoMediaIO falls back to VLLM_VIDEO_LOADER_BACKEND env var.
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_VIDEO_LOADER_BACKEND", "test_video_backend_override_2")

        imageio = ImageMediaIO()

        # Explicit None should fall back to env var
        videoio_none = VideoMediaIO(imageio, num_frames=10, video_backend=None)
        frames_none, metadata_none = videoio_none.load_bytes(b"test")
        np.testing.assert_array_equal(frames_none, FAKE_OUTPUT_2)
        assert metadata_none["video_backend"] == "test_video_backend_override_2"

        # Not providing video_backend should also fall back to env var
        videoio_missing = VideoMediaIO(imageio, num_frames=10)
        frames_missing, metadata_missing = videoio_missing.load_bytes(b"test")
        np.testing.assert_array_equal(frames_missing, FAKE_OUTPUT_2)
        assert metadata_missing["video_backend"] == "test_video_backend_override_2"


def _make_jpeg_b64_frames(n: int, width: int = 8, height: int = 8) -> list[str]:
    """Return *n* tiny base64-encoded JPEG frames."""
    frames: list[str] = []
    for i in range(n):
        img = Image.new("RGB", (width, height), color=(i % 256, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frames.append(pybase64.b64encode(buf.getvalue()).decode("ascii"))
    return frames


def test_load_base64_jpeg_returns_metadata():
    """Regression test: load_base64 with video/jpeg must return metadata.

    Previously, base64 JPEG frame sequences returned an empty dict for
    metadata, which broke downstream consumers that rely on fields like
    total_num_frames and fps. See PR #37301.
    """

    num_test_frames = 3

    b64_frames = _make_jpeg_b64_frames(num_test_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=num_test_frames)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    # Frames array shape: (num_frames, H, W, 3)
    assert frames.shape[0] == num_test_frames

    # All required metadata keys must be present
    required_keys = {
        "total_num_frames",
        "fps",
        "duration",
        "video_backend",
        "frames_indices",
        "do_sample_frames",
    }
    assert required_keys.issubset(metadata.keys()), (
        f"Missing metadata keys: {required_keys - metadata.keys()}"
    )

    assert metadata["total_num_frames"] == num_test_frames
    assert metadata["video_backend"] == "jpeg_sequence"
    assert metadata["frames_indices"] == list(range(num_test_frames))
    assert metadata["do_sample_frames"] is False
    # Default fps=1 → duration == num_frames
    assert metadata["fps"] == 1.0
    assert metadata["duration"] == float(num_test_frames)


def test_load_base64_jpeg_pillow_streams_and_preserves_frame_io_config(
    monkeypatch,
):
    b64_frames = _make_jpeg_b64_frames(2)
    imageio = ImageMediaIO()
    original_load = imageio.load_base64
    calls: list[tuple[str, str]] = []
    loaded_frames: list[MediaWithBytes[Image.Image]] = []

    def load_with_backend_config(media_type, frame_data):
        for prior in loaded_frames:
            with pytest.raises(ValueError):
                prior.media.getpixel((0, 0))
        calls.append((media_type, frame_data))
        frame = original_load(media_type, frame_data)
        frame.io_config = {"frame": len(calls) - 1}
        loaded_frames.append(frame)
        return frame

    monkeypatch.setattr(imageio, "load_base64", load_with_backend_config)
    videoio = VideoMediaIO(imageio, num_frames=2)
    loaded = videoio.load_base64("video/jpeg", ",".join(b64_frames))

    assert calls == [("image/jpeg", frame) for frame in b64_frames]
    for frame in loaded_frames:
        with pytest.raises(ValueError):
            frame.media.getpixel((0, 0))
    assert loaded.io_config == {
        "frame_io_configs": [
            {"frame": 0},
            {"frame": 1},
        ]
    }


def test_load_base64_jpeg_nvimagecodec_uses_decode_service(monkeypatch):
    b64_frames = _make_jpeg_b64_frames(2)
    encoded_frames = [pybase64.b64decode(frame) for frame in b64_frames]
    calls = []

    def load_with_service(image_io, items):
        calls.append((image_io, list(items)))
        return [
            MediaWithBytes(
                Image.open(io.BytesIO(item)).copy(),
                item,
                {"backend": "nvimagecodec"},
            )
            for item in items
        ]

    monkeypatch.setattr(video_module, "load_images_with_service", load_with_service)
    imageio = ImageMediaIO(backend="nvimagecodec")
    videoio = VideoMediaIO(imageio, num_frames=2)

    loaded = videoio.load_base64("video/jpeg", ",".join(b64_frames))

    assert calls == [(imageio, encoded_frames)]
    assert loaded.media[0].shape == (2, 8, 8, 3)
    assert loaded.io_config == {
        "frame_io_configs": [
            {"backend": "nvimagecodec"},
            {"backend": "nvimagecodec"},
        ]
    }


@pytest.mark.asyncio
async def test_nvimagecodec_large_jpeg_sequence_is_segmented(monkeypatch):
    shutdown_nvimagecodec_decode_service()
    b64_frames = _make_jpeg_b64_frames(9)
    calls: list[list[bytes]] = []
    imageio = ImageMediaIO(
        backend="nvimagecodec",
        decoders=1,
        batch_size=1,
        pipeline_depth=1,
        coalesce_timeout_ms=0,
    )

    def load_bytes_many(items):
        calls.append(list(items))
        return [
            MediaWithBytes(Image.open(io.BytesIO(item)).copy(), item) for item in items
        ]

    monkeypatch.setattr(imageio, "load_bytes_many", load_bytes_many)
    videoio = VideoMediaIO(imageio, num_frames=9)
    data = ",".join(b64_frames)

    try:
        sync_loaded = videoio.load_base64("video/jpeg", data)
        assert [len(call) for call in calls] == [4, 4, 1]
        assert sync_loaded.media[0].shape == (9, 8, 8, 3)

        shutdown_nvimagecodec_decode_service()
        calls.clear()
        with ThreadPoolExecutor(max_workers=1) as executor:
            async_loaded = await videoio.load_base64_async(
                "video/jpeg", data, executor=executor
            )
        assert [len(call) for call in calls] == [4, 4, 1]
        assert async_loaded.media[0].shape == (9, 8, 8, 3)
    finally:
        shutdown_nvimagecodec_decode_service()


@pytest.mark.asyncio
async def test_load_base64_jpeg_nvimagecodec_counts_frames_off_event_loop(
    monkeypatch,
):
    event_loop_thread = threading.get_ident()
    count_threads: list[int] = []
    data = ",".join(_make_jpeg_b64_frames(2))
    videoio = VideoMediaIO(ImageMediaIO(backend="nvimagecodec"), num_frames=2)
    original_count = videoio._jpeg_sequence_frame_count

    def recording_count(data):
        count_threads.append(threading.get_ident())
        return original_count(data)

    @asynccontextmanager
    async def reserve_request(image_io, item_count):
        assert image_io is videoio.image_io
        assert item_count == 2
        yield

    async def load_with_service(image_io, items, *, executor):
        assert image_io is videoio.image_io
        assert executor is not None
        return [
            MediaWithBytes(Image.open(io.BytesIO(item)).copy(), item) for item in items
        ]

    monkeypatch.setattr(videoio, "_jpeg_sequence_frame_count", recording_count)
    monkeypatch.setattr(
        video_module, "reserve_image_decode_request_async", reserve_request
    )
    monkeypatch.setattr(
        video_module, "load_images_with_service_async", load_with_service
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        loaded = await videoio.load_base64_async("video/jpeg", data, executor=executor)

    assert loaded.media[0].shape == (2, 8, 8, 3)
    assert len(count_threads) == 1
    assert count_threads[0] != event_loop_thread


def test_load_base64_jpeg_strictly_decodes_only_selected_frames():
    valid_frame = _make_jpeg_b64_frames(1)[0]

    videoio = VideoMediaIO(ImageMediaIO(), num_frames=1)
    frames, _ = videoio.load_base64("video/jpeg", f"{valid_frame},not-valid-base64")
    assert frames.shape[0] == 1

    with pytest.raises(binascii.Error):
        videoio.load_base64("video/jpeg", "not-valid-base64")


def test_load_base64_jpeg_enforces_num_frames_limit():
    """Frames beyond num_frames must be truncated in the video/jpeg path.

    Without the limit an attacker can send thousands of base64 JPEG frames
    in a single request and exhaust server memory (OOM).
    """
    num_frames_limit = 4
    sent_frames = 20

    b64_frames = _make_jpeg_b64_frames(sent_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=num_frames_limit)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    assert frames.shape[0] == num_frames_limit
    assert metadata["total_num_frames"] == num_frames_limit
    assert metadata["frames_indices"] == list(range(num_frames_limit))


def test_load_base64_jpeg_no_limit_when_num_frames_negative():
    """When num_frames is -1, all frames should be loaded without truncation."""
    sent_frames = 10

    b64_frames = _make_jpeg_b64_frames(sent_frames)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=-1)
    frames, metadata = videoio.load_base64("video/jpeg", data)

    assert frames.shape[0] == sent_frames
    assert metadata["total_num_frames"] == sent_frames
    assert metadata["frames_indices"] == list(range(sent_frames))


def test_load_base64_jpeg_raises_on_zero_num_frames():
    """num_frames=0 is invalid and should raise ValueError."""
    b64_frames = _make_jpeg_b64_frames(3)
    data = ",".join(b64_frames)

    imageio = ImageMediaIO()
    videoio = VideoMediaIO(imageio, num_frames=0)

    with pytest.raises(ValueError, match="num_frames must be greater than 0 or -1"):
        videoio.load_base64("video/jpeg", data)


def test_pynvvideocodec_unrelated_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakePyNvVCException(Exception):
        pass

    fake_nvc = SimpleNamespace(PyNvVCException=FakePyNvVCException)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvc)
    original_error = RuntimeError("GPU decoder unavailable")

    def raise_unrelated_error(cls, file_path, nvc):
        raise original_error

    monkeypatch.setattr(
        PyNvVideoCodecVideoBackendMixin,
        "_read_source_metadata",
        classmethod(raise_unrelated_error),
    )

    with pytest.raises(RuntimeError) as exc_info:
        PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec(
            None, b"video", None
        )

    assert exc_info.value is original_error


# ---------------------------------------------------------------------------
# GPU video backend policy tests
# ---------------------------------------------------------------------------


class TestMergeKwargsGpuBackendPolicy:
    """Verify that merge_kwargs blocks request-level GPU backend selection
    when the static (engine-level) config did not configure that backend."""

    def test_pynvvideocodec_requires_gpu(self):
        assert VIDEO_LOADER_REGISTRY.backend_requires_gpu(PYNVVIDEOCODEC_VIDEO_BACKEND)

    def test_strips_video_backend_pynv_when_not_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs=None,
            runtime_kwargs={"video_backend": "pynvvideocodec"},
        )
        assert "video_backend" not in result

    def test_strips_backend_pynv_when_not_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"num_frames": 16},
            runtime_kwargs={"backend": "pynvvideocodec"},
        )
        assert result.get("backend") != "pynvvideocodec"

    def test_preserves_video_backend_pynv_when_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"video_backend": "pynvvideocodec"},
            runtime_kwargs={"video_backend": "pynvvideocodec", "num_frames": 8},
        )
        assert result["video_backend"] == "pynvvideocodec"
        assert result["num_frames"] == 8

    def test_preserves_backend_pynv_when_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"backend": "pynvvideocodec"},
            runtime_kwargs={"backend": "pynvvideocodec"},
        )
        assert result["backend"] == "pynvvideocodec"

    def test_strips_request_level_hw_decoders_when_not_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"video_backend": "pynvvideocodec"},
            runtime_kwargs={"hw_decoders": 4},
        )
        assert "hw_decoders" not in result

    def test_prevents_request_level_hw_decoders_override(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={
                "video_backend": "pynvvideocodec",
                "hw_decoders": 2,
            },
            runtime_kwargs={"hw_decoders": 4},
        )
        assert result["hw_decoders"] == 2

    def test_prevents_request_level_output_layout_override(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={
                "backend": "pynvvideocodec",
                "output_layout": "tchw",
            },
            runtime_kwargs={"output_layout": "thwc"},
        )
        assert result["output_layout"] == "tchw"

    def test_prevents_request_level_gpu_resize_override(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={
                "backend": "pynvvideocodec",
                "gpu_resize": True,
            },
            runtime_kwargs={"gpu_resize": False},
        )
        assert result["gpu_resize"] is True

    def test_strips_request_level_gpu_resize_when_not_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"backend": "pynvvideocodec"},
            runtime_kwargs={"gpu_resize": True},
        )
        assert "gpu_resize" not in result

    @pytest.mark.parametrize("backend", ["opencv", "pyav", "torchcodec"])
    def test_cpu_codec_fallback_drops_static_pynv_options(self, backend: str):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={
                "backend": "pynvvideocodec",
                "hw_decoders": 2,
                "output_layout": "tchw",
                "gpu_resize": True,
            },
            runtime_kwargs={"backend": backend},
        )
        assert result["backend"] == backend
        assert "hw_decoders" not in result
        assert "output_layout" not in result
        assert "gpu_resize" not in result

    def test_custom_loader_output_layout_passes_through(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"video_backend": "test_layout_kwargs_passthrough"},
            runtime_kwargs={"output_layout": "custom-layout"},
        )
        assert result["output_layout"] == "custom-layout"

    def test_custom_loader_fallback_keeps_request_owned_output_layout(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={
                "backend": "pynvvideocodec",
                "hw_decoders": 2,
                "output_layout": "tchw",
            },
            runtime_kwargs={
                "video_backend": "test_layout_kwargs_passthrough",
                "backend": "opencv",
                "output_layout": "custom-layout",
            },
        )
        assert result == {
            "video_backend": "test_layout_kwargs_passthrough",
            "backend": "opencv",
            "output_layout": "custom-layout",
        }

    @pytest.mark.parametrize("backend", ["opencv", "pyav", "torchcodec"])
    def test_software_video_backend_passes_through(self, backend: str):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs=None,
            runtime_kwargs={"video_backend": backend},
        )
        assert result["video_backend"] == backend

    @pytest.mark.parametrize("backend", ["opencv", "pyav"])
    def test_software_codec_backend_passes_through(self, backend: str):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs=None,
            runtime_kwargs={"backend": backend},
        )
        assert result["backend"] == backend

    def test_strips_both_keys_independently(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs=None,
            runtime_kwargs={
                "video_backend": "pynvvideocodec",
                "backend": "pynvvideocodec",
                "num_frames": 4,
            },
        )
        assert "video_backend" not in result
        assert result.get("backend") != "pynvvideocodec"
        assert result["num_frames"] == 4

    def test_other_kwargs_preserved_when_gpu_backend_stripped(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"fps": 2},
            runtime_kwargs={
                "video_backend": "pynvvideocodec",
                "num_frames": 16,
            },
        )
        assert "video_backend" not in result
        assert result["num_frames"] == 16

    def test_static_pynv_with_different_runtime_gpu_backend(self):
        """If static sets pynv via video_backend but runtime tries to set it
        via the codec-level 'backend' key (without a static match), strip it."""
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"video_backend": "pynvvideocodec"},
            runtime_kwargs={"backend": "pynvvideocodec"},
        )
        assert result.get("backend") != "pynvvideocodec"
        assert result["video_backend"] == "pynvvideocodec"

    def test_deepstream_requires_gpu(self):
        assert VIDEO_LOADER_REGISTRY.backend_requires_gpu("deepstream")

    def test_strips_backend_deepstream_when_not_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs=None,
            runtime_kwargs={"backend": "deepstream"},
        )
        assert result.get("backend") != "deepstream"

    def test_preserves_backend_deepstream_when_static(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"backend": "deepstream"},
            runtime_kwargs={"backend": "deepstream", "num_frames": 8},
        )
        assert result["backend"] == "deepstream"
        assert result["num_frames"] == 8

    def test_strips_pool_size_from_runtime(self):
        result = VideoMediaIO.merge_kwargs(
            default_kwargs={"backend": "deepstream"},
            runtime_kwargs={"backend": "deepstream", "pool_size": 4},
        )
        assert "pool_size" not in result

    def test_unknown_backend_not_treated_as_gpu(self):
        assert not VIDEO_LOADER_REGISTRY.backend_requires_gpu("totally_unknown")


@pytest.mark.parametrize(
    ("output_layout", "frame_shape"),
    [("thwc", (4, 5, 3)), ("tchw", (3, 4, 5))],
)
def test_pynvvc_frames_copy_directly_to_pinned_host(
    monkeypatch: pytest.MonkeyPatch,
    output_layout: str,
    frame_shape: tuple[int, int, int],
):
    """Each decoded frame is copied once, without a stacked device batch."""
    torch = pytest.importorskip("torch")
    import weakref

    frame_size = math.prod(frame_shape)
    frames = [
        torch.arange(frame_size, dtype=torch.uint8).reshape(frame_shape) + index
        for index in range(2)
    ]
    events: list[object] = []
    wrapper_refs: list[weakref.ReferenceType] = []
    original_from_dlpack = torch.from_dlpack

    def tracked_from_dlpack(frame):
        wrapper = original_from_dlpack(frame)
        wrapper_refs.append(weakref.ref(wrapper))
        events.append("from_dlpack")
        return wrapper

    class FakeHostFrame:
        def __init__(self, batch: "FakeHostBatch", index: int) -> None:
            self.batch = batch
            self.index = index

        def copy_(self, frame, *, non_blocking: bool):
            assert non_blocking
            events.append(f"copy_{self.index}")
            self.batch.array[self.index] = frame.numpy()

    class FakeHostBatch:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.array = np.empty(shape, dtype=np.uint8)

        def __getitem__(self, index: int) -> FakeHostFrame:
            return FakeHostFrame(self, index)

        def numpy(self) -> npt.NDArray:
            events.append("numpy")
            return self.array

    def fake_empty(shape, *, dtype, device, pin_memory):
        assert dtype == torch.uint8
        assert device == "cpu"
        assert pin_memory
        events.append("allocate")
        return FakeHostBatch(tuple(shape))

    class FakeStream:
        device = torch.device("cpu")

        @staticmethod
        def synchronize() -> None:
            assert all(ref() is not None for ref in wrapper_refs)
            events.append("synchronize")

    monkeypatch.setattr(torch, "from_dlpack", tracked_from_dlpack)
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(
        torch,
        "stack",
        lambda *args, **kwargs: pytest.fail("torch.stack must not be called"),
    )

    output = _pynvvc_frames_to_pinned_host(
        frames,
        output_layout,
        FakeStream(),  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(output, np.stack([frame.numpy() for frame in frames]))
    assert output.flags.c_contiguous
    assert events == [
        "from_dlpack",
        "from_dlpack",
        "allocate",
        "copy_0",
        "copy_1",
        "synchronize",
        "numpy",
    ]


@pytest.mark.parametrize(
    ("output_layout", "frame_shape", "expected_shape", "cvcuda_layout"),
    [
        ("thwc", (4, 5, 3), (2, 3, 3), "HWC"),
        ("tchw", (3, 4, 5), (3, 2, 3), "CHW"),
    ],
)
def test_pynvvc_gpu_resize_uses_cvcuda_hqresize(
    monkeypatch: pytest.MonkeyPatch,
    output_layout: str,
    frame_shape: tuple[int, int, int],
    expected_shape: tuple[int, int, int],
    cvcuda_layout: str,
):
    torch = pytest.importorskip("torch")
    calls: list[tuple] = []
    cvstream = object()

    class FakeOutput:
        def __init__(self, tensor) -> None:
            self.tensor = tensor

        def cuda(self):
            return self.tensor

    class FakeCvCuda:
        class Interp:
            CUBIC = "cubic"

        @staticmethod
        def as_tensor(frame, layout):
            calls.append(("as_tensor", layout))
            return frame

        @staticmethod
        def hq_resize(
            frame,
            size,
            *,
            antialias: bool,
            interpolation,
            stream,
        ):
            calls.append(("hq_resize", size, antialias, interpolation, stream))
            height, width = size
            output = (
                frame[:, :height, :width]
                if output_layout == "tchw"
                else frame[:height, :width, :]
            )
            return FakeOutput(output.contiguous())

    monkeypatch.setattr(pynv_module, "_cvcuda", lambda: FakeCvCuda)
    with pynv_module._RESIZE_COUNTERS_LOCK:
        pynv_module._RESIZE_COUNTERS.clear()

    frames = [
        torch.arange(math.prod(frame_shape), dtype=torch.uint8).reshape(frame_shape)
    ]
    resized = pynv_module._resize_pynvvc_frames(
        frames,
        output_layout,
        (3, 2),
        cvstream=cvstream,
    )

    assert [tuple(frame.shape) for frame in resized] == [expected_shape]
    assert all(frame.is_contiguous() for frame in resized)
    assert calls == [
        ("as_tensor", cvcuda_layout),
        ("hq_resize", (2, 3), True, "cubic", cvstream),
    ]
    assert pynv_module.get_pynvvideocodec_resize_stats() == {"resize_cvcuda": 1}


@pytest.mark.parametrize(
    ("fault", "error_match"),
    [
        ("rank", "expected HWC"),
        ("dtype", "expected torch.uint8"),
        ("device", "expected meta"),
        ("contiguous", "non-contiguous"),
        ("shape", "inconsistent shapes"),
        ("empty_dim", "expected HWC"),
        ("layout", "expected HWC"),
    ],
)
def test_pynvvc_direct_copy_validates_all_frames_and_synchronizes(
    fault: str,
    error_match: str,
):
    torch = pytest.importorskip("torch")

    frames = [torch.zeros((4, 5, 3), dtype=torch.uint8) for _ in range(2)]
    stream_device = torch.device("cpu")
    if fault == "rank":
        frames[0] = torch.zeros((4, 5), dtype=torch.uint8)
    elif fault == "dtype":
        frames[0] = torch.zeros((4, 5, 3), dtype=torch.float32)
    elif fault == "device":
        stream_device = torch.device("meta")
    elif fault == "contiguous":
        frames[0] = torch.zeros((4, 3, 5), dtype=torch.uint8).permute(0, 2, 1)
    elif fault == "shape":
        frames[1] = torch.zeros((5, 5, 3), dtype=torch.uint8)
    elif fault == "empty_dim":
        frames[0] = torch.zeros((0, 5, 3), dtype=torch.uint8)
    elif fault == "layout":
        frames[0] = torch.zeros((3, 4, 5), dtype=torch.uint8)

    class FakeStream:
        device = stream_device
        synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

    stream = FakeStream()
    with pytest.raises(ValueError, match=error_match):
        _pynvvc_frames_to_pinned_host(
            frames,
            "thwc",
            stream,  # type: ignore[arg-type]
        )

    assert stream.synchronize_count == 1


def test_pynvvc_direct_copy_retains_partial_dlpack_results_until_sync(
    monkeypatch: pytest.MonkeyPatch,
):
    torch = pytest.importorskip("torch")
    import weakref

    frames = [torch.zeros((4, 5, 3), dtype=torch.uint8) for _ in range(2)]
    wrapper_refs: list[weakref.ReferenceType] = []
    original_from_dlpack = torch.from_dlpack
    conversion_count = 0

    def failing_from_dlpack(frame):
        nonlocal conversion_count
        conversion_count += 1
        if conversion_count == 2:
            raise RuntimeError("DLPack conversion failed")
        wrapper = original_from_dlpack(frame)
        wrapper_refs.append(weakref.ref(wrapper))
        return wrapper

    monkeypatch.setattr(torch, "from_dlpack", failing_from_dlpack)

    class FakeStream:
        device = torch.device("cpu")
        synchronize_count = 0

        def synchronize(self) -> None:
            assert all(ref() is not None for ref in wrapper_refs)
            self.synchronize_count += 1

    stream = FakeStream()
    with pytest.raises(RuntimeError, match="DLPack conversion failed"):
        _pynvvc_frames_to_pinned_host(
            frames,
            "thwc",
            stream,  # type: ignore[arg-type]
        )

    assert conversion_count == 2
    assert stream.synchronize_count == 1


def test_pynvvc_direct_copy_synchronizes_empty_decode():
    torch = pytest.importorskip("torch")

    class FakeStream:
        device = torch.device("cpu")
        synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

    stream = FakeStream()
    output = _pynvvc_frames_to_pinned_host(
        [],
        "thwc",
        stream,  # type: ignore[arg-type]
    )

    assert output.shape == (0,)
    assert output.dtype == np.uint8
    assert stream.synchronize_count == 1


def test_pynvvc_direct_copy_synchronizes_after_partial_copy_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    torch = pytest.importorskip("torch")
    frames = [torch.zeros((4, 5, 3), dtype=torch.uint8) for _ in range(2)]
    copied_indices: list[int] = []

    class FakeHostFrame:
        def __init__(self, index: int) -> None:
            self.index = index

        def copy_(self, frame, *, non_blocking: bool):
            assert non_blocking
            copied_indices.append(self.index)
            if self.index == 1:
                raise RuntimeError("copy failed")

    class FakeHostBatch:
        def __getitem__(self, index: int) -> FakeHostFrame:
            return FakeHostFrame(index)

    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: FakeHostBatch())

    class FakeStream:
        device = torch.device("cpu")
        synchronize_count = 0

        def synchronize(self) -> None:
            self.synchronize_count += 1

    stream = FakeStream()
    with pytest.raises(RuntimeError, match="copy failed"):
        _pynvvc_frames_to_pinned_host(
            frames,
            "thwc",
            stream,  # type: ignore[arg-type]
        )

    assert copied_indices == [0, 1]
    assert stream.synchronize_count == 1


def test_pynvvideocodec_tchw_requires_audited_processor():
    with pytest.raises(ValueError, match="not supported by video processor"):
        validate_video_processor_output_layout(
            "Qwen2VLVideoProcessor",
            "tchw",
        )


@pytest.mark.parametrize(
    "video_processor",
    ["Qwen3VLVideoProcessor", "Cosmos3EdgeVideoProcessor"],
)
def test_pynvvideocodec_tchw_accepts_audited_processor(video_processor: str):
    validate_video_processor_output_layout(video_processor, "tchw")


def test_pynvvideocodec_processor_gate_runs_in_connector():
    connector = MediaConnector(
        media_io_kwargs={
            "video": {
                "backend": "pynvvideocodec",
                "output_layout": "tchw",
            }
        }
    )

    with pytest.raises(ValueError, match="not supported by video processor"):
        connector.fetch_video("unused", video_processor="Qwen2VLVideoProcessor")


def test_custom_loader_receives_layout_and_video_processor_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    connector = MediaConnector(
        media_io_kwargs={
            "video": {
                "video_backend": "test_layout_kwargs_passthrough",
                "output_layout": "custom-layout",
                "video_processor": "custom-value",
            }
        }
    )
    monkeypatch.setattr(
        connector,
        "load_from_url",
        lambda _url, media_io, **_kwargs: media_io.load_bytes(b"video"),
    )

    connector.fetch_video("unused", video_processor="Qwen3VLVideoProcessor")

    assert LayoutKwargsPassthroughVideoLoader.received_kwargs == {
        "output_layout": "custom-layout",
        "video_processor": "custom-value",
    }


def test_pynvvideocodec_tchw_marks_media_hash_config():
    frames = np.zeros((4, 3, 8, 10), dtype=np.uint8)
    video_io = VideoMediaIO(
        ImageMediaIO(),
        backend="pynvvideocodec",
        output_layout="tchw",
    )
    video_io.video_loader = SimpleNamespace(
        load_bytes=lambda *args, **kwargs: (
            frames,
            {VLLM_VIDEO_INPUT_DATA_FORMAT_KEY: "channels_first"},
        )
    )

    loaded = video_io.load_bytes(b"encoded-video")

    assert loaded.io_config == {"pynvvideocodec_input_data_format": "channels_first"}


def test_pynvvideocodec_gpu_resize_marks_media_hash_config():
    frames = np.zeros((4, 3, 8, 10), dtype=np.uint8)
    video_io = VideoMediaIO(
        ImageMediaIO(),
        backend="pynvvideocodec",
        output_layout="tchw",
        gpu_resize=True,
    )
    video_io.video_loader = SimpleNamespace(
        load_bytes=lambda *args, **kwargs: (
            frames,
            {
                VLLM_VIDEO_INPUT_DATA_FORMAT_KEY: "channels_first",
                video_module.VLLM_VIDEO_GPU_RESIZE_KEY: True,
            },
        )
    )

    loaded = video_io.load_bytes(b"encoded-video")

    assert loaded.io_config == {
        "pynvvideocodec_input_data_format": "channels_first",
        "pynvvideocodec_gpu_resize": True,
    }
    assert video_module.VLLM_VIDEO_GPU_RESIZE_KEY not in loaded.media[1]


def test_encode_base64_preserves_thwc_with_tchw_decoder_config():
    captured: list[np.ndarray] = []
    image_io = ImageMediaIO()

    def capture_frame(image: Image.Image, **_kwargs) -> str:
        captured.append(np.asarray(image))
        return "encoded"

    image_io.encode_base64 = capture_frame  # type: ignore[method-assign]
    video_io = VideoMediaIO(
        image_io,
        backend="pynvvideocodec",
        output_layout="tchw",
    )
    thwc = np.arange(2 * 5 * 7 * 3, dtype=np.uint8).reshape(2, 5, 7, 3)

    encoded = video_io.encode_base64(thwc)

    assert encoded == "encoded,encoded"
    np.testing.assert_array_equal(captured, thwc)


@pytest.mark.parametrize("shape", [(2, 5, 7), (2, 5, 7, 4)])
def test_encode_base64_preserves_legacy_frame_shape(shape: tuple[int, ...]):
    captured: list[np.ndarray] = []
    image_io = ImageMediaIO()

    def capture_frame(image: Image.Image, **_kwargs) -> str:
        captured.append(np.asarray(image))
        return "encoded"

    image_io.encode_base64 = capture_frame  # type: ignore[method-assign]
    video_io = VideoMediaIO(image_io)
    video = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    encoded = video_io.encode_base64(video)

    assert encoded == "encoded,encoded"
    np.testing.assert_array_equal(captured, video)
