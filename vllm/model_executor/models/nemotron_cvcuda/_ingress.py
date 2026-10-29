# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Draft opt-in PR ingress using D's qualified, architecture-neutral pieces."""

import copy
import types

from vllm.logger import init_logger

logger = init_logger(__name__)


def install_processor(processor):
    import torch

    from ._arrow import export_rgbx
    from ._fields import RAW_RGBX_CONTRACT_VERSION

    if processor.do_cpu_normalize or not processor.dynamic_tiler:
        raise ValueError(
            "CV-CUDA vision requires dynamic images and device normalization"
        )
    original = processor._preprocess_image
    processor._d_cvcuda_borrow_raw = True

    def preprocess(module, text, images, max_num_tiles):
        if not images:
            return original(text, images, max_num_tiles)
        if any(max(image.size) > 8192 for image in images):
            logger.warning_once(
                "CV-CUDA image preprocessing uses native PR behavior for "
                "image extents above its qualified 8192-pixel limit."
            )
            return original(text, images, max_num_tiles)
        deferred = copy.copy(module)
        deferred.dynamic_tiler = copy.copy(module.dynamic_tiler)
        sizes = []

        def apply_params(tiler, params, dtype=None):
            view = export_rgbx(params.media, expected_size=params.media.size)
            pixels = torch.from_numpy(view.pixels)
            pixels._d_cvcuda_owner = view
            sizes.append(
                (
                    params.patch_size[1] * tiler._patch_size,
                    params.patch_size[0] * tiler._patch_size,
                )
            )
            return [pixels]

        deferred.dynamic_tiler.apply_params = types.MethodType(
            apply_params, deferred.dynamic_tiler
        )
        result_text, fields = type(module)._preprocess_image(
            deferred, text, images, max_num_tiles
        )
        fields["pixel_values_raw"] = [
            [pixels] for pixels in fields.pop("pixel_values_flat")
        ]
        fields["imgs_sizes"] = sizes
        fields["d_cvcuda_contract"] = [RAW_RGBX_CONTRACT_VERSION] * len(sizes)
        return result_text, fields

    processor._preprocess_image = types.MethodType(preprocess, processor)


def install_model(model):
    import cvcuda
    import torch

    from vllm.model_executor.models.nano_nemotron_vl import (
        NanoNemotronVLImagePixelInputsDynamic,
    )

    from ._encoder import cached_positions, identity_blocks
    from ._fast import RawRGBXUploader
    from ._fields import RAW_RGBX_CONTRACT_VERSION, preserve_raw_batch
    from ._fused import NormalizedPackedResize, TrustedNormalized

    if (
        model.input_norm is None
        or not model.dynamic_resolution
        or model.llm_dtype != torch.bfloat16
    ):
        raise ValueError(
            "CV-CUDA vision requires dynamic BF16 device-normalized images"
        )
    native_parse = model._parse_and_validate_image_input
    native_multimodal = model._parse_and_validate_multimodal_inputs
    native_normalize = model._normalize_pixel_values
    native_vision = model.vision_model.forward
    resizer = None
    uploader = None
    prepared = set()
    trusted = TrustedNormalized()
    from ._leads import PortableLeads

    options = model._cvcuda_encoder_options
    model._cvcuda_leads = (
        PortableLeads(gelu=options["gelu"]) if any(options.values()) else None
    )

    def normalized(module, pixels):
        return pixels if trusted.contains(pixels) else native_normalize(pixels)

    def parse(module, **kwargs):
        nonlocal resizer, uploader
        raw = kwargs.pop("pixel_values_raw", None)
        marker = kwargs.pop("d_cvcuda_contract", None)
        if raw is None:
            if marker is not None:
                raise ValueError("Raw-image marker has no pixels")
            return native_parse(**kwargs)
        raw = preserve_raw_batch(raw)
        if (
            kwargs.get("pixel_values_flat") is not None
            or kwargs.get("image_embeds") is not None
        ):
            raise ValueError("Ambiguous raw/resized image input")
        sizes = kwargs.get("imgs_sizes")
        tokens = kwargs.get("num_tokens_per_image")
        if (
            not isinstance(raw, list)
            or not raw
            or not isinstance(sizes, list)
            or not isinstance(tokens, list)
            or len(raw) != len(sizes)
            or len(raw) != len(tokens)
            or marker != [RAW_RGBX_CONTRACT_VERSION] * len(raw)
        ):
            raise ValueError("Invalid raw-image contract")
        if resizer is None:
            resizer = NormalizedPackedResize(
                cvcuda, model.patch_size, model.input_norm.weight, model.input_norm.bias
            )
            uploader = RawRGBXUploader()
        groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for index, (pixels, size) in enumerate(zip(raw, sizes, strict=True)):
            groups.setdefault((tuple(pixels.shape), tuple(size)), []).append(index)
        output = [None] * len(raw)
        packed_groups = []
        for (shape, size), indices in groups.items():
            key = (shape, size)
            if key not in prepared:
                resizer.prepare(shape[:2], size, channels=4)
                prepared.add(key)
            pixels = resizer(uploader([raw[index] for index in indices]), size)
            resizer.retire(pixels)
            packed_groups.append(pixels)
            if len(groups) > 1:
                patches = (size[0] // model.patch_size) * (size[1] // model.patch_size)
                for position, index in enumerate(indices):
                    output[index] = pixels[
                        0, position * patches : (position + 1) * patches
                    ]
        packed = (
            packed_groups[0]
            if len(groups) == 1
            else torch.cat(output, dim=0).unsqueeze(0)
        )
        trusted.add(packed)
        return NanoNemotronVLImagePixelInputsDynamic(
            pixel_values_flat=packed, imgs_sizes=sizes, num_tokens_per_image=tokens
        )

    def multimodal(module, **kwargs):
        if kwargs.get("pixel_values_raw") is not None:
            kwargs = {"pixel_values_flat": None, **kwargs}
        return native_multimodal(**kwargs)

    def vision(module, *inputs, **kwargs):
        if module.training or torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "Portable CV-CUDA vision is qualified for eager inference only"
            )
        from contextlib import ExitStack

        from ._encoder import exact_gelu_blocks, residual_norm_encoder

        with identity_blocks(model), cached_positions(model), ExitStack() as stack:
            if options["norm"]:
                stack.enter_context(residual_norm_encoder(model, model._cvcuda_leads))
            if options["gelu"]:
                stack.enter_context(exact_gelu_blocks(model, model._cvcuda_leads))
            return native_vision(*inputs, **kwargs)

    model._normalize_pixel_values = types.MethodType(normalized, model)
    model._parse_and_validate_image_input = types.MethodType(parse, model)
    model._parse_and_validate_multimodal_inputs = types.MethodType(multimodal, model)
    model.vision_model.forward = types.MethodType(vision, model.vision_model)
