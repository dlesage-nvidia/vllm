# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import pytest

from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.gpu_ipc_memory import (
    MultiModalGPUMemoryPool,
    get_mm_gpu_ipc_pool,
    maybe_init_mm_gpu_ipc_pool,
    reserve_mm_ipc_gpu_memory,
    set_mm_gpu_ipc_pool,
)
from vllm.multimodal.video_decoders import PYNVVIDEOCODEC_VIDEO_BACKEND
from vllm.multimodal.video_decoders.pynvvideocodec import (
    PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES,
    PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES,
)
from vllm.utils.mem_constants import GiB_bytes


def _mm_config(
    *,
    mm_ipc_gpu_memory_gb: float = 0,
    video_backend: str | None = None,
    hw_decoders: int | None = None,
) -> MultiModalConfig:
    video_kwargs: dict[str, object] = (
        {} if video_backend is None else {"video_backend": video_backend}
    )
    if hw_decoders is not None:
        video_kwargs["hw_decoders"] = hw_decoders

    return MultiModalConfig(
        mm_ipc_gpu_memory_gb=mm_ipc_gpu_memory_gb,
        media_io_kwargs={"video": video_kwargs} if video_kwargs else {},
    )


def _pynvvideocodec_decoder_budget(
    api_process_count: int = 1,
    hw_decoders: int = 2,
) -> int:
    return api_process_count * (
        PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES * hw_decoders
        + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
    )


def test_acquire_release_accounting():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    assert pool.available_bytes == 100

    lease = pool.acquire(40)
    assert pool.available_bytes == 60

    lease.release()
    assert pool.available_bytes == 100


def test_acquire_too_large_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(ValueError):
        pool.acquire(101)
    # Nothing should have been reserved.
    assert pool.available_bytes == 100


def test_negative_acquire_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(ValueError):
        pool.acquire(-1)


def test_double_release_is_noop():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    lease = pool.acquire(50)
    lease.release()
    assert pool.available_bytes == 100
    # Releasing again must not inflate the pool past its capacity.
    lease.release()
    assert pool.available_bytes == 100


def test_context_manager_releases_on_exception():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    with pytest.raises(RuntimeError), pool.acquire(50):
        assert pool.available_bytes == 50
        raise RuntimeError("boom")
    assert pool.available_bytes == 100


def test_acquire_blocks_until_release():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    first = pool.acquire(80)

    acquired = threading.Event()

    def waiter():
        # Needs 50 bytes but only 20 are free; must block until `first`
        # is released.
        with pool.acquire(50):
            acquired.set()

    t = threading.Thread(target=waiter)
    t.start()

    # The waiter cannot proceed yet.
    assert not acquired.wait(timeout=0.2)

    # Releasing the first lease frees enough budget to unblock the waiter.
    first.release()
    assert acquired.wait(timeout=2.0)
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert pool.available_bytes == 100


def test_concurrent_acquires_serialize():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    # Each task needs 60 bytes, so only one can hold the budget at a time.
    in_section = []
    max_concurrent = 0
    lock = threading.Lock()

    def task():
        nonlocal max_concurrent
        with pool.acquire(60):
            with lock:
                in_section.append(1)
                max_concurrent = max(max_concurrent, len(in_section))
            time.sleep(0.05)
            with lock:
                in_section.pop()

    threads = [threading.Thread(target=task) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert max_concurrent == 1
    assert pool.available_bytes == 100


def test_zero_total_bytes_rejected():
    with pytest.raises(ValueError):
        MultiModalGPUMemoryPool(total_bytes=0)


def test_global_pool_accessor():
    try:
        assert maybe_init_mm_gpu_ipc_pool(0) is None
        assert get_mm_gpu_ipc_pool() is None

        pool = maybe_init_mm_gpu_ipc_pool(2)
        assert pool is not None
        assert get_mm_gpu_ipc_pool() is pool
        assert pool.total_bytes == 2 * GiB_bytes
    finally:
        set_mm_gpu_ipc_pool(None)


def test_global_pool_splits_budget_across_api_processes():
    try:
        pool = maybe_init_mm_gpu_ipc_pool(2, api_process_count=4)
        assert pool is not None
        assert get_mm_gpu_ipc_pool() is pool
        assert pool.total_bytes == GiB_bytes // 2
    finally:
        set_mm_gpu_ipc_pool(None)


def test_global_pool_rejects_invalid_api_process_count():
    with pytest.raises(ValueError):
        maybe_init_mm_gpu_ipc_pool(2, api_process_count=0)


@pytest.mark.parametrize("video_backend", [None, "opencv"])
def test_reserve_mm_ipc_gpu_memory_raw_frame_budget_only(
    monkeypatch: pytest.MonkeyPatch,
    video_backend: str | None,
):
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        video_backend=video_backend,
    )

    assert reserve_mm_ipc_gpu_memory(GiB_bytes, mm_config) == int(0.75 * GiB_bytes)


def test_reserve_mm_ipc_gpu_memory_includes_pynvvideocodec_decoder_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        video_backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
    )
    available_bytes = 4 * GiB_bytes

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes - int(0.25 * GiB_bytes) - _pynvvideocodec_decoder_budget()
    )


def test_reserve_mm_ipc_gpu_memory_uses_env_video_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", PYNVVIDEOCODEC_VIDEO_BACKEND)
    available_bytes = 4 * GiB_bytes

    assert reserve_mm_ipc_gpu_memory(available_bytes, _mm_config()) == (
        available_bytes - _pynvvideocodec_decoder_budget()
    )


def test_reserve_mm_ipc_gpu_memory_scales_decoder_budget_by_api_servers(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", PYNVVIDEOCODEC_VIDEO_BACKEND)
    available_bytes = 8 * GiB_bytes

    assert reserve_mm_ipc_gpu_memory(
        available_bytes,
        _mm_config(),
        api_process_count=3,
    ) == available_bytes - _pynvvideocodec_decoder_budget(api_process_count=3)


def test_reserve_mm_ipc_gpu_memory_uses_configured_hw_decoders(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    available_bytes = 4 * GiB_bytes
    mm_config = _mm_config(
        video_backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
        hw_decoders=3,
    )

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes - _pynvvideocodec_decoder_budget(hw_decoders=3)
    )


# --- GPU image-decoder reservation ------------------------------------------


def _image_mm_config(**overrides):
    from vllm.config.multimodal import MultiModalConfig

    kwargs = dict(
        media_io_kwargs={"image": {"image_backend": "nvimgcodec", "num_decoders": 4}},
        mm_ipc_gpu_memory_gb=1.0,
    )
    kwargs.update(overrides)
    return MultiModalConfig(**kwargs)


def test_image_backend_reserves_workspace_and_context():
    """The reservation must actually shrink the KV cache, by the stated amount."""
    from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory
    from vllm.multimodal.image_decoders import (
        CUDA_CONTEXT_BYTES,
        DECODER_WORKSPACE_BYTES,
    )

    available = 40 * GiB_bytes
    remaining = reserve_mm_ipc_gpu_memory(available, _image_mm_config(), 1)
    expected = available - (
        int(1.0 * GiB_bytes) + 4 * DECODER_WORKSPACE_BYTES + CUDA_CONTEXT_BYTES
    )
    assert remaining == expected


def test_image_reservation_scales_with_api_processes():
    from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory

    available = 60 * GiB_bytes
    one = available - reserve_mm_ipc_gpu_memory(available, _image_mm_config(), 1)
    two = available - reserve_mm_ipc_gpu_memory(available, _image_mm_config(), 2)
    # The raw-frame budget is split across processes; the decoder term is not.
    assert two > one


def test_pillow_backend_reserves_nothing_extra():
    from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory

    available = 40 * GiB_bytes
    pillow = _image_mm_config(media_io_kwargs={}, mm_ipc_gpu_memory_gb=0.0)
    assert reserve_mm_ipc_gpu_memory(available, pillow, 1) == available


def test_reservation_covers_the_whole_eligible_domain():
    """The workspace figure is only an upper bound if eligibility is bounded."""
    from vllm.multimodal.image_decoders import DECODER_WORKSPACE_BYTES
    from vllm.multimodal.image_decoders.nvimgcodec import MAX_ELIGIBLE_PIXELS

    # Largest admitted raster must fit comfortably inside the per-slot figure.
    assert MAX_ELIGIBLE_PIXELS * 3 < DECODER_WORKSPACE_BYTES * 8


def test_startup_validation_rejects_a_zero_gpu_budget():
    import pytest

    with pytest.raises(ValueError, match="mm-ipc-gpu-memory-gb"):
        _image_mm_config(mm_ipc_gpu_memory_gb=0.0).validate_image_backend_available()


def test_explicit_num_decoders_is_honoured():
    from vllm.config.multimodal import MultiModalConfig

    cfg = MultiModalConfig(
        media_io_kwargs={"image": {"image_backend": "nvimgcodec", "num_decoders": 16}},
        mm_ipc_gpu_memory_gb=1.0,
    )
    assert cfg.get_image_decoder_count() == 16


def test_reservation_covers_the_retained_decoder_buffer():
    """Buffer reuse retains a raster per slot; the reservation must include it.

    nvImageCodec allocates it from its own device allocator, so torch's
    accounting cannot see it -- unreserved, it comes silently out of the KV
    cache, which is the failure this reservation exists to prevent.
    """
    from vllm.multimodal.image_decoders.nvimgcodec import (
        DECODER_RETAINED_RASTER_BYTES,
        DECODER_WORKSPACE_BYTES,
        MAX_ELIGIBLE_PIXELS,
    )
    from vllm.multimodal.gpu_ipc_memory import reserve_mm_ipc_gpu_memory

    assert DECODER_RETAINED_RASTER_BYTES == MAX_ELIGIBLE_PIXELS * 3
    per_slot = DECODER_WORKSPACE_BYTES + DECODER_RETAINED_RASTER_BYTES
    available = 40 * GiB_bytes
    taken = available - reserve_mm_ipc_gpu_memory(available, _image_mm_config(), 1)
    assert taken >= 4 * per_slot, "retained rasters are not covered"
