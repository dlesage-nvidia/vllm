# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture(scope="session")
def decode_batch():
    """The real GPU decode entry point, wired to a real GPU memory pool.

    Deliberately not a fake. The parity suite's whole value is that its
    reference comes from really decoding with Pillow and its subject really
    decodes on the GPU; substituting a fake here would let metadata-transfer
    and aliasing bugs pass, which is exactly how they survive elsewhere.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for image-decode parity tests")
    pytest.importorskip("nvidia.nvimgcodec")

    from vllm.multimodal.gpu_ipc_memory import (
        MultiModalGPUMemoryPool,
        get_mm_gpu_ipc_pool,
        set_mm_gpu_ipc_pool,
    )
    from vllm.multimodal.image_decoders import configure, shutdown

    previous = get_mm_gpu_ipc_pool()
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(2 * 1024**3))
    configure(num_decoders=2)
    try:
        from vllm.multimodal.image_decoders import decode_batch as real_decode_batch

        yield real_decode_batch
    finally:
        shutdown()
        set_mm_gpu_ipc_pool(previous)
