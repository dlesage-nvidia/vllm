# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from enum import Enum

import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

from vllm.transformers_utils.processors.cvcuda_resize_plan import (
    ALLOCATION_OVERHEAD_BYTES,
    NATIVE_SCRATCH_BYTES,
    resize_plan,
)


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
def test_cvcuda_uint8_supported_versions(version):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_release_version,
    )

    _validate_cvcuda_release_version(version)


@pytest.mark.parametrize(
    "version",
    [
        "0.18.0",
        "0.19.0b0",
        "0.19.0+custom",
        "0.20.0-beta",
        "0.20.0b0",
        "0.20.0rc1",
        "0.20.0.dev1",
        "0.20.0+custom",
        "0.20.1",
        "0.21.0",
    ],
)
def test_cvcuda_uint8_rejects_unreviewed_versions(version):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_release_version,
    )

    with pytest.raises(RuntimeError, match="requires CV-CUDA"):
        _validate_cvcuda_release_version(version)


@pytest.fixture
def cvcuda_api_exports():
    return {
        "tvresize_into": len,
        "reformat_into": len,
        "as_tensor": len,
        "as_stream": len,
        "Interp": Enum("Interpolation", {"CUBIC": 3}),
    }


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
def test_cvcuda_uint8_public_api_contract(version, cvcuda_api_exports):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_api,
    )

    cvcuda_api_exports["__version__"] = version
    _validate_cvcuda_api(version, cvcuda_api_exports)


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
@pytest.mark.parametrize(
    "name", ["tvresize_into", "reformat_into", "as_tensor", "as_stream"]
)
@pytest.mark.parametrize("present", [False, True])
def test_cvcuda_uint8_rejects_missing_or_noncallable_api(
    version, name, present, cvcuda_api_exports
):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_api,
    )

    cvcuda_api_exports["__version__"] = version
    if present:
        cvcuda_api_exports[name] = None
    else:
        del cvcuda_api_exports[name]
    with pytest.raises(RuntimeError, match=name):
        _validate_cvcuda_api(version, cvcuda_api_exports)


@pytest.mark.parametrize("native_version", [None, "0.19.0", "0.20.0-beta"])
def test_cvcuda_uint8_rejects_mismatched_native_version(
    native_version, cvcuda_api_exports
):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_api,
    )

    cvcuda_api_exports["__version__"] = native_version
    with pytest.raises(RuntimeError, match="module and distribution"):
        _validate_cvcuda_api("0.20.0", cvcuda_api_exports)


@pytest.mark.parametrize("present", [False, True])
def test_cvcuda_uint8_rejects_missing_cubic_interpolation(present, cvcuda_api_exports):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        _validate_cvcuda_api,
    )

    cvcuda_api_exports["__version__"] = "0.20.0"
    if present:
        cvcuda_api_exports["Interp"] = Enum("Interpolation", {"LINEAR": 1})
    else:
        del cvcuda_api_exports["Interp"]
    with pytest.raises(RuntimeError, match="cubic interpolation"):
        _validate_cvcuda_api("0.20.0", cvcuda_api_exports)


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize(
    "target,budget,mode",
    [
        ((48, 64), 512 * 1024, "frames"),
        ((48, 64), 64 * 1024, "middle"),
        ((48, 64), 20 * 1024, "strips"),
        ((67, 64), 20 * 1024, "rows"),
        ((48, 97), 20 * 1024, "columns"),
        ((67, 97), 20 * 1024, "identity"),
    ],
)
def test_plan_bounds_tiles_without_cropping_filtered_axis(
    target, budget, mode, cpu_source
):
    plan = resize_plan((67, 97), target, 3, budget, 4096, cpu_source)
    assert plan.mode == mode
    assert plan.temporary_bytes <= budget
    assert plan.native_bytes <= NATIVE_SCRATCH_BYTES
    assert plan.output_bytes == 3 * 3 * target[0] * target[1]
    assert 1 <= plan.row_step <= 67
    assert 1 <= plan.column_step <= target[1]
    if cpu_source:
        assert 2 * plan.upload_capacity + 4096 <= plan.temporary_bytes
    else:
        assert plan.upload_capacity == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.__setattr__("frame_step", 7)


@pytest.mark.parametrize("cpu_source", [False, True])
def test_long_video_charges_full_output_separately_without_growing_temporary(
    cpu_source,
):
    short = resize_plan((1080, 1920), (672, 896), 1, 32 * 1024**2, 8192, cpu_source)
    long = resize_plan((1080, 1920), (672, 896), 1024, 32 * 1024**2, 8192, cpu_source)
    assert long.temporary_bytes == short.temporary_bytes
    assert long.output_bytes == 1024 * short.output_bytes
    assert long.output_bytes > 32 * 1024**2


def test_minimum_full_axis_and_tables_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="filtered-axis"):
        resize_plan((67, 97), (48, 64), 1, ALLOCATION_OVERHEAD_BYTES + 1024)
    with pytest.raises(ValueError, match="filtered-axis"):
        resize_plan((67, 97), (48, 64), 1, 64 * 1024, 64 * 1024)
    with pytest.raises(ValueError, match="signed int64"):
        resize_plan((67, 97), (48, 64), 2**63, 64 * 1024)


def test_preferred_single_frame_does_not_reduce_actual_temporary_budget_to_one_byte():
    plan = resize_plan((67, 97), (48, 64), 8, 384 * 1024**2, preferred_frame_step=1)
    assert plan.mode == "frames" and plan.frame_step == 1
    assert plan.temporary_bytes > 1


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize("default_stream", [False, True])
@pytest.mark.parametrize("budget", [24 * 1024, 64 * 1024])
def test_gpu_forced_tiling_is_exact_v2_for_cpu_and_cuda_inputs(
    cpu_source, default_stream, budget
):
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        cvcuda_bicubic_resize,
        reserve_cvcuda_workspace,
    )

    pixels = torch.randint(
        0,
        256,
        (1, 67, 97, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1234),
    )
    target = (48, 64)
    expected = tvf.resize(
        pixels.permute(0, 3, 1, 2),
        target,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    device = torch.device("cuda", 0)
    reserve = reserve_cvcuda_workspace(device)
    torch.accelerator.synchronize(device)
    del reserve
    stream = (
        torch.cuda.default_stream(device)
        if default_stream
        else torch.cuda.Stream(device)
    )
    with torch.cuda.stream(stream):
        source = pixels if cpu_source else pixels.to(device)
        baseline = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        outputs = [
            cvcuda_bicubic_resize(source, target, device, budget) for _ in range(2)
        ]
        retained = outputs[-1].clone()
        del outputs[-1]
        torch.empty_like(retained).fill_(99)
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    for actual in (*outputs, retained):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    output_bytes = expected.numel()
    assert (
        torch.accelerator.max_memory_allocated(device)
        <= baseline + 3 * output_bytes + budget
    )
