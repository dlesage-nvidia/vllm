# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.utils.mem_constants import GiB_bytes

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"
NVIMAGECODEC_MAX_PIXELS = 3840 * 2160
NVIMAGECODEC_DECODER_THREADS = 8

# Eight concurrent 4K decodes with nvImageCodec 0.9.0.20 and nvJPEG 13.2.1.68
# used 768 MiB in process-level measurements. Keep headroom for other GPUs.
NVIMAGECODEC_GPU_MEMORY_BYTES = GiB_bytes
