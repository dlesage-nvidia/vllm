# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import pytest

import vllm.multimodal.gpu_ipc_memory as gpu_ipc_memory
from vllm.config.multimodal import MultiModalConfig
from vllm.multimodal.gpu_ipc_memory import (
    MultiModalGPUMemoryLease,
    MultiModalGPUMemoryPool,
    get_mm_gpu_ipc_pool,
    maybe_init_mm_gpu_ipc_pool,
    reserve_mm_ipc_gpu_memory,
    set_mm_gpu_ipc_pool,
)
from vllm.multimodal.image_decoders import (
    NVIMAGECODEC_CUDA_CONTEXT_BYTES,
    NVIMAGECODEC_DECODER_WORKSPACE_BYTES,
    NVIMAGECODEC_IMAGE_BACKEND,
    NVIMAGECODEC_MAX_CHANNELS,
    NVIMAGECODEC_MAX_PIXELS,
    get_nvimagecodec_decoder_gpu_memory_bytes,
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
    image_backend: str | None = None,
    image_decoders: int | None = None,
) -> MultiModalConfig:
    video_kwargs: dict[str, object] = (
        {} if video_backend is None else {"video_backend": video_backend}
    )
    if hw_decoders is not None:
        video_kwargs["hw_decoders"] = hw_decoders

    image_kwargs: dict[str, object] = (
        {} if image_backend is None else {"backend": image_backend}
    )
    if image_decoders is not None:
        image_kwargs["decoders"] = image_decoders

    media_io_kwargs = {}
    if video_kwargs:
        media_io_kwargs["video"] = video_kwargs
    if image_kwargs:
        media_io_kwargs["image"] = image_kwargs

    return MultiModalConfig(
        mm_ipc_gpu_memory_gb=mm_ipc_gpu_memory_gb,
        media_io_kwargs=media_io_kwargs,
    )


def _pynvvideocodec_decoder_budget(
    api_process_count: int = 1,
    hw_decoders: int = 2,
) -> int:
    return api_process_count * (
        PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES * hw_decoders
        + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
    )


def _nvimagecodec_decoder_budget(
    api_process_count: int = 1,
    image_decoders: int = 2,
) -> int:
    return api_process_count * (
        get_nvimagecodec_decoder_gpu_memory_bytes() * image_decoders
        + NVIMAGECODEC_CUDA_CONTEXT_BYTES
    )


def test_acquire_release_accounting():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    assert pool.available_bytes == 100

    lease = pool.acquire(40)
    assert pool.available_bytes == 60

    lease.release()
    assert pool.available_bytes == 100


def test_try_acquire_success():
    pool = MultiModalGPUMemoryPool(total_bytes=100)

    lease = pool.try_acquire(40)

    assert lease is not None
    assert lease.nbytes == 40
    assert pool.available_bytes == 60
    lease.release()


def test_try_acquire_insufficient_available_returns_none():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    first = pool.acquire(80)

    assert pool.try_acquire(30) is None
    assert pool.available_bytes == 20

    first.release()
    assert pool.available_bytes == 100


def test_try_acquire_too_large_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)

    with pytest.raises(ValueError, match="exceeds the total pool size"):
        pool.try_acquire(101)

    assert pool.available_bytes == 100


def test_try_acquire_negative_raises():
    pool = MultiModalGPUMemoryPool(total_bytes=100)

    with pytest.raises(ValueError, match="Cannot acquire negative bytes"):
        pool.try_acquire(-1)

    assert pool.available_bytes == 100


def test_try_acquire_is_atomic_for_one_lease():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    contender_count = 8
    start = threading.Barrier(contender_count + 1)
    leases: list[MultiModalGPUMemoryLease | None] = [None] * contender_count

    def contend(index: int):
        start.wait()
        leases[index] = pool.try_acquire(100)

    threads = [
        threading.Thread(target=contend, args=(index,))
        for index in range(contender_count)
    ]
    for thread in threads:
        thread.start()

    start.wait()
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    assert sum(lease is not None for lease in leases) == 1
    assert pool.available_bytes == 0

    winner = next(lease for lease in leases if lease is not None)
    winner.release()
    assert pool.available_bytes == 100


def test_try_acquire_release_accounting():
    pool = MultiModalGPUMemoryPool(total_bytes=100)
    lease = pool.try_acquire(100)
    assert lease is not None
    assert pool.try_acquire(1) is None

    lease.release()
    assert pool.available_bytes == 100

    next_lease = pool.try_acquire(100)
    assert next_lease is not None
    assert pool.available_bytes == 0
    next_lease.release()
    next_lease.release()
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


def test_reserve_mm_ipc_gpu_memory_includes_nvimagecodec_decoder_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    available_bytes = 8 * GiB_bytes
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        image_backend=NVIMAGECODEC_IMAGE_BACKEND,
    )

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes - int(0.25 * GiB_bytes) - _nvimagecodec_decoder_budget()
    )


def test_reserve_mm_ipc_gpu_memory_uses_configured_image_decoders(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    available_bytes = 8 * GiB_bytes
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        image_backend=NVIMAGECODEC_IMAGE_BACKEND,
        image_decoders=3,
    )

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes
        - int(0.25 * GiB_bytes)
        - _nvimagecodec_decoder_budget(image_decoders=3)
    )


def test_nvimagecodec_decoder_budget_scales_with_api_processes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    available_bytes = 16 * GiB_bytes
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        image_backend=NVIMAGECODEC_IMAGE_BACKEND,
    )

    assert reserve_mm_ipc_gpu_memory(
        available_bytes, mm_config, api_process_count=3
    ) == (
        available_bytes
        - int(0.25 * GiB_bytes)
        - _nvimagecodec_decoder_budget(api_process_count=3)
    )


def test_nvimagecodec_reservation_uses_lower_global_pixel_limit(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)

    assert get_nvimagecodec_decoder_gpu_memory_bytes() == (
        NVIMAGECODEC_DECODER_WORKSPACE_BYTES + 100 * NVIMAGECODEC_MAX_CHANNELS
    )


@pytest.mark.parametrize("configured_limit", [0, NVIMAGECODEC_MAX_PIXELS + 1])
def test_nvimagecodec_reservation_caps_unbounded_global_pixel_limit(
    monkeypatch, configured_limit: int
):
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", configured_limit)

    assert get_nvimagecodec_decoder_gpu_memory_bytes() == (
        NVIMAGECODEC_DECODER_WORKSPACE_BYTES
        + NVIMAGECODEC_MAX_PIXELS * NVIMAGECODEC_MAX_CHANNELS
    )


def test_pillow_reservation_allows_unlimited_image_pixels(monkeypatch):
    import vllm.envs as envs

    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)
    available_bytes = 2 * GiB_bytes
    mm_config = _mm_config(mm_ipc_gpu_memory_gb=0.25)

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes - int(0.25 * GiB_bytes)
    )


def test_raw_frame_only_log_reports_zero_decoder_budget_per_server(monkeypatch):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    log_calls = []

    def record_info_once(message, *args):
        log_calls.append((message, args))

    monkeypatch.setattr(gpu_ipc_memory.logger, "info_once", record_info_once)

    reserve_mm_ipc_gpu_memory(
        2 * GiB_bytes,
        _mm_config(mm_ipc_gpu_memory_gb=0.25),
        api_process_count=2,
    )

    assert len(log_calls) == 1
    _, args = log_calls[0]
    assert args[2] == args[4] == "0.0"


def test_gpu_image_and_video_backends_reserve_both_context_estimates(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")
    monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", "opencv")
    available_bytes = 8 * GiB_bytes
    mm_config = _mm_config(
        mm_ipc_gpu_memory_gb=0.25,
        video_backend=PYNVVIDEOCODEC_VIDEO_BACKEND,
        image_backend=NVIMAGECODEC_IMAGE_BACKEND,
    )
    decoder_bytes = (
        2 * PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES
        + 2 * get_nvimagecodec_decoder_gpu_memory_bytes()
        + PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES
        + NVIMAGECODEC_CUDA_CONTEXT_BYTES
    )

    assert reserve_mm_ipc_gpu_memory(available_bytes, mm_config) == (
        available_bytes - int(0.25 * GiB_bytes) - decoder_bytes
    )
