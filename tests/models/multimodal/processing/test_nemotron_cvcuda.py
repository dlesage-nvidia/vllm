# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def cvcuda_cpu_helpers():
    """Exercise real CPU helpers without loading CUDA or model dependencies."""
    package = (
        Path(__file__).resolve().parents[4]
        / "vllm/model_executor/models/nemotron_cvcuda"
    )
    modules = {}
    for name in ("_arrow", "_resize"):
        module_name = "nemotron_cvcuda_cpu_test" + name
        specification = importlib.util.spec_from_file_location(
            module_name, package / (name + ".py")
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        modules[name] = module
    yield modules
    for module in modules.values():
        sys.modules.pop(module.__name__)


@pytest.mark.parametrize("shape", [(1, 1), (9, 13), (31, 17)])
def test_arrow_pixels_remain_exact_and_owned_after_exporter_release(
    cvcuda_cpu_helpers, shape
):
    height, width = shape
    expected = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    image = Image.fromarray(expected)
    owner = cvcuda_cpu_helpers["_arrow"].export_rgbx(
        image, expected_size=(width, height)
    )
    pixels = owner.pixels[:, :, :3]
    assert np.shares_memory(pixels, owner.pixels)
    assert not pixels.flags.writeable
    del owner, image
    gc.collect()
    np.testing.assert_array_equal(pixels, expected)
    with pytest.raises(ValueError, match="read-only"):
        pixels[0, 0, 0] = 0


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_arrow_rejects_non_rgb_without_reinterpreting_channels(
    cvcuda_cpu_helpers, mode
):
    image = Image.new(mode, (13, 9))
    with pytest.raises(ValueError, match="RGB image"):
        cvcuda_cpu_helpers["_arrow"].export_rgbx(image, expected_size=(13, 9))


def test_arrow_rejects_mismatched_expected_geometry(cvcuda_cpu_helpers):
    image = Image.new("RGB", (13, 9))
    with pytest.raises(ValueError, match="RGB image"):
        cvcuda_cpu_helpers["_arrow"].export_rgbx(image, expected_size=(9, 13))


@pytest.mark.parametrize("extent", [1, 17, 128])
def test_integer_resize_identity_preserves_each_uint8_value(cvcuda_cpu_helpers, extent):
    starts, counts, weights, precision = cvcuda_cpu_helpers[
        "_resize"
    ].axis_coefficients(extent, extent)
    source = np.arange(extent, dtype=np.int64) % 256
    actual = []
    for destination in range(extent):
        start = starts[destination]
        count = counts[destination]
        accumulated = np.dot(
            source[start : start + count], weights[destination, :count]
        )
        actual.append((accumulated + (1 << (precision - 1))) >> precision)
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize("sizes", [(0, 1), (1, 0), (-1, 17), (8193, 1)])
def test_resize_rejects_extents_outside_the_frozen_contract(cvcuda_cpu_helpers, sizes):
    with pytest.raises(ValueError, match="positive resize extents up to 8192"):
        cvcuda_cpu_helpers["_resize"].axis_coefficients(*sizes)
