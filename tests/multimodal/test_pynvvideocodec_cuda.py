# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from contextlib import contextmanager

import numpy as np
import pybase64
import pytest

from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.video import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    PyNvVideoCodecVideoBackend,
)
from vllm.multimodal.video_decoders.pynvvideocodec import _pynv_decoder_pool
from vllm.platforms import current_platform

from .utils import create_long_gop_video


@contextmanager
def _fresh_decoder_pool():
    pool = _pynv_decoder_pool
    old_state = pool.slots, pool.active, pool.cond, pool.max_slots
    pool.slots = []
    pool.active = 0
    pool.cond = threading.Condition()
    pool.max_slots = None
    try:
        yield
    finally:
        for slot in pool.slots:
            slot.invalidate()
        pool.slots, pool.active, pool.cond, pool.max_slots = old_state


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_rgbp_matches_rgb_on_pinned_host():
    pytest.importorskip("PyNvVideoCodec")
    video = create_long_gop_video(num_frames=10, width=320, height=240, fps=5)
    video_url = f"data:video/mp4;base64,{pybase64.b64encode(video).decode()}"

    with _fresh_decoder_pool():
        thwc, thwc_metadata = PyNvVideoCodecVideoBackend.load_bytes(
            video, num_frames=4, hw_decoders=1
        )
        connector = MediaConnector(
            media_io_kwargs={
                "video": {"backend": PYNVVIDEOCODEC_VIDEO_BACKEND, "hw_decoders": 1}
            }
        )
        tchw, tchw_metadata = connector.fetch_video(
            video_url, video_processor="Qwen3VLVideoProcessor"
        )

    assert thwc.shape == (4, 240, 320, 3)
    assert tchw.shape == (4, 3, 240, 320)
    assert thwc.flags.c_contiguous
    assert tchw.flags.c_contiguous
    np.testing.assert_array_equal(tchw.transpose(0, 2, 3, 1), thwc)
    assert tchw_metadata["frames_indices"] == thwc_metadata["frames_indices"]
