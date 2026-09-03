# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.multimodal.video import Qwen3VLVideoBackend
from vllm.multimodal.video_decoders import pynvvideocodec
from vllm.platforms import current_platform

from .utils import create_long_gop_video


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_pynvvideocodec_rgbp_matches_rgb_on_pinned_host(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("PyNvVideoCodec")
    video = create_long_gop_video(num_frames=10, width=320, height=240, fps=5)
    decode_kwargs = dict(
        fps=2,
        min_frames=4,
        max_frames=4,
        hw_decoders=1,
        backend="pynvvideocodec",
    )
    pool = pynvvideocodec._PyNvDecoderPool()
    monkeypatch.setattr(pynvvideocodec, "_pynv_decoder_pool", pool)

    try:
        tchw, tchw_metadata = Qwen3VLVideoBackend.load_bytes(video, **decode_kwargs)
        monkeypatch.setattr(Qwen3VLVideoBackend, "_pynvvideocodec_use_rgbp", False)
        thwc, thwc_metadata = Qwen3VLVideoBackend.load_bytes(video, **decode_kwargs)
    finally:
        for slot in pool.slots:
            slot.invalidate()

    assert thwc.shape == (4, 240, 320, 3)
    assert tchw.shape == (4, 3, 240, 320)
    assert thwc.flags.c_contiguous
    assert tchw.flags.c_contiguous
    assert torch.from_numpy(thwc).is_pinned()
    assert torch.from_numpy(tchw).is_pinned()
    np.testing.assert_array_equal(tchw.transpose(0, 2, 3, 1), thwc)
    assert tchw_metadata["frames_indices"] == thwc_metadata["frames_indices"]
