# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inject a benchmark-only nvImageCodec decoder helper-thread count."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import sys
from pathlib import Path

import benchmark_host_cpu_topology as cpu_topology
import benchmark_image_decode_cell as base_cell

from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecBackend,
    NvImageCodecDecoderSlot,
)

EXPECTED_GET_GPU_DECODER_SHA256 = (
    "4c6a74bfc6e89ea3e86adb8a27b29ee863f256039256388cc6b2fc16740a2852"
)


def make_injected_get_gpu_decoder(helper_threads: int):
    """Build the benchmark replacement while preserving product backends."""

    def injected_get_gpu_decoder(self, nvimgcodec):
        if self.gpu_decoder is None:
            self.gpu_decoder = nvimgcodec.Decoder(
                device_id=NvImageCodecBackend._DEVICE_INDEX,
                max_num_cpu_threads=helper_threads,
                options=":num_cuda_streams=1",
                backends=[
                    nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY),
                    nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
                    nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
                ],
            )
        return self.gpu_decoder

    return injected_get_gpu_decoder


def _option_value(arguments: list[str], option: str) -> str:
    for index, value in enumerate(arguments):
        if value == option and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(option + "="):
            return value.split("=", maxsplit=1)[1]
    raise ValueError(f"missing required base-cell option {option}")


def _write_annotated_result(
    output: Path,
    *,
    helper_threads: int,
    source_method_sha256: str,
    injected_method_sha256: str,
) -> None:
    result = json.loads(output.read_text())
    wrapper = Path(__file__).resolve()
    base = Path(base_cell.__file__).resolve()
    topology = Path(cpu_topology.__file__).resolve()
    injection = {
        "kind": "benchmark-only method replacement",
        "patched_symbol": (
            "vllm.multimodal.image_decoders.nvimagecodec."
            "NvImageCodecDecoderSlot.get_gpu_decoder"
        ),
        "max_num_cpu_threads": helper_threads,
        "num_cuda_streams": 1,
        "gpu_backends": ["HW_GPU_ONLY", "GPU_ONLY", "HYBRID_CPU_GPU"],
        "cpu_fallback_decoder_unchanged": True,
        "source_method_sha256": source_method_sha256,
        "expected_source_method_sha256": EXPECTED_GET_GPU_DECODER_SHA256,
        "injected_method_sha256": injected_method_sha256,
        "wrapper": {
            "path": str(wrapper),
            "sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        },
        "base_cell": {
            "path": str(base),
            "sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        },
        "cpu_topology_helper": {
            "path": str(topology),
            "sha256": hashlib.sha256(topology.read_bytes()).hexdigest(),
        },
    }
    result["benchmark_injection"] = injection
    result["cpu_topology"] = cpu_topology.capture_cpu_topology()
    configuration = result["configuration"]
    configuration["nvimagecodec_max_num_cpu_threads"] = helper_threads
    configuration["nominal_decoder_helper_thread_budget"] = (
        int(configuration["decoders"]) * helper_threads
    )
    result["validation"].update(
        {
            "benchmark_injection_recorded": True,
            "source_method_fingerprint_exact": (
                source_method_sha256 == EXPECTED_GET_GPU_DECODER_SHA256
            ),
        }
    )
    temporary = output.with_suffix(output.suffix + ".annotating")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--helper-threads", type=int, required=True)
    injected_args, remaining = parser.parse_known_args()
    if injected_args.helper_threads not in (1, 2, 4):
        parser.error("--helper-threads must be one of 1, 2, or 4")
    if _option_value(remaining, "--backend") != "nvimagecodec":
        parser.error("helper-thread injection requires --backend=nvimagecodec")
    if "--help" in remaining or "-h" in remaining:
        sys.argv = [sys.argv[0], *remaining]
        asyncio.run(base_cell.main())
        return
    output = Path(_option_value(remaining, "--output"))

    original = NvImageCodecDecoderSlot.get_gpu_decoder
    original_source = inspect.getsource(original)
    original_sha256 = hashlib.sha256(original_source.encode()).hexdigest()
    if original_sha256 != EXPECTED_GET_GPU_DECODER_SHA256:
        raise RuntimeError(
            "refusing to patch an unexpected get_gpu_decoder implementation: "
            f"{original_sha256}"
        )

    helper_threads = injected_args.helper_threads
    injected_get_gpu_decoder = make_injected_get_gpu_decoder(helper_threads)

    injected_sha256 = hashlib.sha256(
        inspect.getsource(injected_get_gpu_decoder).encode()
    ).hexdigest()
    NvImageCodecDecoderSlot.get_gpu_decoder = injected_get_gpu_decoder
    sys.argv = [sys.argv[0], *remaining]
    try:
        asyncio.run(base_cell.main())
    finally:
        NvImageCodecDecoderSlot.get_gpu_decoder = original

    _write_annotated_result(
        output,
        helper_threads=helper_threads,
        source_method_sha256=original_sha256,
        injected_method_sha256=injected_sha256,
    )
    print(
        f"benchmark-only helper-thread injection recorded: t{helper_threads}",
        flush=True,
    )


if __name__ == "__main__":
    main()
