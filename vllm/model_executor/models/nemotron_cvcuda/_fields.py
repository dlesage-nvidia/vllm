# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native wire-compatible raw RGBX fields without collation copies."""

RAW_RGBX_CONTRACT_VERSION = 2


def preserve_raw_batch(batch):
    import torch

    if not isinstance(batch, list) or not batch:
        raise ValueError("Borrowed RGBX collation requires a nonempty image list")
    if any(not isinstance(group, list) or len(group) != 1 for group in batch):
        raise ValueError("Raw RGBX transport requires one tensor per image group")
    pixels_batch = [group[0] for group in batch]
    if any(
        not isinstance(pixels, torch.Tensor)
        or pixels.device.type != "cpu"
        or pixels.dtype != torch.uint8
        or pixels.ndim != 3
        or pixels.shape[-1] != 4
        or not pixels.is_contiguous()
        for pixels in pixels_batch
    ):
        raise ValueError("Borrowed collation requires contiguous CPU HWC RGBX uint8")
    return pixels_batch


def borrowed_raw_config():
    from vllm.multimodal.inputs import MultiModalFieldConfig

    return MultiModalFieldConfig.batched("image", keep_on_cpu=True)
