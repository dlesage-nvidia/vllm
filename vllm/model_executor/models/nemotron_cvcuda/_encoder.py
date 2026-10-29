# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import functools
import types


@functools.cache
def identity_flags(module):
    import torch

    return tuple(
        scale.dtype == torch.bfloat16 and bool(torch.all(scale == 1).item())
        for scale in (module.ls1, module.ls2)
    )


@contextlib.contextmanager
def identity_blocks(model):
    modules = [
        module
        for module in model.vision_model.modules()
        if module.__class__.__name__ == "RadioVisionEncoderLayer"
    ]
    if len(modules) != 32:
        raise ValueError("expected32 native RADIO blocks")
    originals = [(module, module.forward) for module in modules]
    flags = {module: identity_flags(module) for module in modules}

    def forward(module, hidden_states, mask_meta=None):
        first, second = flags[module]
        branch = module.attn(module.norm1(hidden_states), mask_meta=mask_meta)
        hidden_states = hidden_states + (branch if first else branch * module.ls1)
        branch = module.mlp(module.norm2(hidden_states))
        return hidden_states + (branch if second else branch * module.ls2)

    try:
        for module, _ in originals:
            module.forward = types.MethodType(forward, module)
        yield {
            "modules": 32,
            "skipped_identity_multiplications": sum(
                sum(values) for values in flags.values()
            ),
            "verified_actual_loaded_scales": True,
            "layernorm": "unchanged",
            "attention": "unchanged_native_backend",
        }
    finally:
        for module, original in originals:
            module.forward = original


def apply_pos_enc_cached(generator, patches, imgs_sizes):
    import torch

    if not generator.abs_pos:
        return patches, None
    positions = {}
    pieces = []
    encodings = []
    offset = 0
    for size in imgs_sizes:
        size = tuple(size)
        length = (size[0] // generator.patch_size) * (size[1] // generator.patch_size)
        if size not in positions:
            positions[size] = generator.get_pos_enc(patches.shape[0], input_size=size)
        position = positions[size]
        end = offset + length
        pieces.append(patches[:, offset:end] + position)
        encodings.append(position)
        offset += length
    if offset != patches.shape[1]:
        raise ValueError(
            "position specialization requires complete packed image coverage"
        )
    return torch.cat(pieces, dim=1), torch.cat(encodings, dim=1)


@contextlib.contextmanager
def cached_positions(model):
    modules = [
        module
        for module in model.vision_model.modules()
        if module.__class__.__name__ == "ViTPatchGenerator"
    ]
    if len(modules) != 1:
        raise ValueError("expected one native patch generator")
    module = modules[0]
    original = module.apply_pos_enc_dynamic
    module.apply_pos_enc_dynamic = types.MethodType(apply_pos_enc_cached, module)
    try:
        yield {
            "cache_scope": "single_forward_per_unique_size",
            "full_packed_cat_rewrites": 1,
            "unchanged_native_interpolation": True,
        }
    finally:
        module.apply_pos_enc_dynamic = original


@contextlib.contextmanager
def residual_norm_encoder(model, leads):
    encoders = [
        module
        for module in model.vision_model.modules()
        if module.__class__.__name__ == "RadioVisionEncoder"
    ]
    if len(encoders) != 1 or len(encoders[0].layers) != 32:
        raise ValueError("Portable norm requires all 32 native RADIO blocks")
    encoder = encoders[0]
    if encoder.training or not all(
        all(identity_flags(layer)) for layer in encoder.layers
    ):
        raise ValueError("Portable norm requires inference and loaded identity scales")
    original = encoder.forward

    def forward(module, inputs_embeds, mask_meta=None):
        hidden = inputs_embeds
        normalized = module.layers[0].norm1(hidden)
        for index, layer in enumerate(module.layers):
            hidden, normalized = leads.norm(
                hidden,
                layer.attn(normalized, mask_meta=mask_meta),
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )
            branch = layer.mlp(normalized)
            if index + 1 == len(module.layers):
                hidden = hidden + branch
            else:
                norm = module.layers[index + 1].norm1
                hidden, normalized = leads.norm(
                    hidden, branch, norm.weight, norm.bias, norm.eps
                )
        return hidden

    encoder.forward = types.MethodType(forward, encoder)
    try:
        yield
    finally:
        encoder.forward = original


@contextlib.contextmanager
def exact_gelu_blocks(model, leads):
    layers = [
        module
        for module in model.vision_model.modules()
        if module.__class__.__name__ == "RadioVisionEncoderLayer"
    ]
    if len(layers) != 32 or any(layer.training for layer in layers):
        raise ValueError("Exact GELU requires all 32 inference RADIO blocks")
    originals = [(layer.mlp, layer.mlp.forward) for layer in layers]

    def forward(module, inputs):
        preactivation, bias = module.fc1(inputs)
        if bias is not None:
            raise ValueError("Exact GELU requires native FC1 with applied bias")
        output, bias = module.fc2(leads.gelu(preactivation))
        if bias is not None:
            raise ValueError("Exact GELU requires native FC2 with applied bias")
        return output

    try:
        for module, original in originals:
            module.forward = types.MethodType(forward, module)
        yield
    finally:
        for module, original in originals:
            module.forward = original
