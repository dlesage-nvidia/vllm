# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in zero-copy raw RGBX collation; other PR fields retain native handling."""

import functools


def preserve_raw_batch(batch):
    import torch

    if not isinstance(batch, list) or not batch:
        raise ValueError("Borrowed RGBX collation requires a nonempty image list")
    if any(
        not isinstance(pixels, torch.Tensor)
        or pixels.device.type != "cpu"
        or pixels.dtype != torch.uint8
        or pixels.ndim != 3
        or pixels.shape[-1] != 4
        or not pixels.is_contiguous()
        for pixels in batch
    ):
        raise ValueError("Borrowed collation requires contiguous CPU HWC RGBX uint8")
    return batch


@functools.lru_cache(maxsize=1)
def borrowed_field_type():
    from vllm.multimodal.inputs import MultiModalBatchedField

    class BorrowedRGBXField(MultiModalBatchedField):
        def _reduce_data(self, batch, *, pin_memory):
            if pin_memory:
                raise ValueError("Borrowed raw pixels must not enter pinned staging")
            return preserve_raw_batch(batch)

    return BorrowedRGBXField


def borrowed_raw_config():
    from vllm.multimodal.inputs import MultiModalFieldConfig

    return MultiModalFieldConfig(
        field=borrowed_field_type()(keep_on_cpu=True),
        modality="image",
    )
