# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

COMMITS = {
    "upstream": "d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302",
    "pr-base": "bc8abf31fef015339473f6071eda0de0305dd9b2",
    "pr-head": "30d917599b104423e452fa718890af01c4ff4d39",
}
TREES = {
    "upstream": "9cc26997991af6f8f38150c9631d482d18b1bd2c",
    "pr-base": "09423356278c6c4bd871ccda98499474fad78bdd",
    "pr-head": "66c4849eb21973b9ca391b7b0911968f4aa63dac",
}
FRAMES = 32
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PIXEL_BUDGET = (1024, 576)
MAX_PIXELS_PER_FRAME = PIXEL_BUDGET[0] * PIXEL_BUDGET[1]
MAX_PIXELS_TOTAL = MAX_PIXELS_PER_FRAME * FRAMES
PROMPT = "<|vision_start|><|video_pad|><|vision_end|>"
VIDEO_SHA256 = "b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d"
VIDEO_BYTES = 13_267_543
TRANSFORMERS_INIT_SHA256 = (
    "67b01cb68df95d42da0661ea120535f33bb618225622e9523bd32e3b7741f9e1"
)
CHAT_COMPLETION_PROTOCOL_ARTIFACTS = {
    "vllm/entrypoints/openai/chat_completion/protocol.py": (
        "47e5d710fd66886bc25946414f5c8e6e3a665cee7910feb5eacd4a17f3331da7"
    ),
    "vllm/entrypoints/openai/chat_completion/serving.py": (
        "9982953285e9df469032a82fffa4095d0e9d86278bede6e2b91d03d02373d182"
    ),
}
PYNV_RUNTIME_ARTIFACT_SHA256 = {
    "PyNvVideoCodec_121.cpython-312-x86_64-linux-gnu.so": (
        "2fb85f8bcd33c13e240ef2a8c6277f4d5a0260b629ecf9a242a04f1403f582a8"
    ),
    "PyNvVideoCodec_130.cpython-312-x86_64-linux-gnu.so": (
        "14f12a7977c2f681fb01693e41434308bfb5cf0e2c31ed2c29d1176337c86462"
    ),
    "VersionCheck.cpython-312-x86_64-linux-gnu.so": (
        "3800377df84245d3a41ce17433ccaab9e5f12636ab6f889165d2adb21e42eac2"
    ),
    "__init__.py": ("b613c6fad0629ad1b63538a2905938fd9c00eec36402e3b58faa840e744e83d7"),
}


def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def output(command: Sequence[str]) -> str:
    completed = run(command, capture_output=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def worker(args: argparse.Namespace) -> None:
    import importlib.metadata

    import numpy as np
    import torch
    import transformers
    import vllm

    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY
    from vllm.utils.torch_utils import set_default_torch_num_threads

    def tensor_sha256(tensor: torch.Tensor) -> str:
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    def tensor_signature(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "contiguous": tensor.is_contiguous(),
            "sha256": tensor_sha256(tensor),
        }

    backend_kwargs: dict[str, Any] = {"hw_decoders": 2}
    if args.variant == "pr-head":
        backend_kwargs["output_layout"] = "tchw"
    loader = VIDEO_LOADER_REGISTRY.load("pynvvideocodec")
    frames, metadata = loader.load_bytes(
        args.video.read_bytes(),
        num_frames=FRAMES,
        **backend_kwargs,
    )
    expected_shape_kind = "tchw" if args.variant == "pr-head" else "thwc"
    if frames.ndim != 4:
        raise RuntimeError(f"unexpected frame rank: {frames.shape}")
    if expected_shape_kind == "thwc":
        if frames.shape[-1] != 3:
            raise RuntimeError(f"{args.variant} is not THWC: {frames.shape}")
        canonical = np.ascontiguousarray(frames)
    else:
        if frames.shape[1] != 3:
            raise RuntimeError(f"{args.variant} is not TCHW: {frames.shape}")
        canonical = np.ascontiguousarray(frames.transpose(0, 2, 3, 1))
    if frames.dtype != np.uint8 or canonical.dtype != np.uint8:
        raise RuntimeError(f"unexpected dtype: {frames.dtype}")
    if frames.shape[0] != FRAMES:
        raise RuntimeError(f"unexpected frame count: {frames.shape[0]}")

    device_normalize = args.variant == "pr-head"
    media_video_kwargs = {
        "video_backend": "qwen3_vl",
        "min_frames": FRAMES,
        "max_frames": FRAMES,
        "backend": "pynvvideocodec",
        **backend_kwargs,
    }
    model_config = ModelConfig(
        model=MODEL,
        tokenizer=MODEL,
        revision=REVISION,
        tokenizer_revision=REVISION,
        dtype="auto",
        seed=0,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 1},
        media_io_kwargs={"video": media_video_kwargs},
        mm_processor_cache_gb=0,
        mm_processor_kwargs={"max_pixels": MAX_PIXELS_TOTAL},
        mm_device_do_normalize=device_normalize,
    )
    if model_config.multimodal_config.mm_device_do_normalize is not device_normalize:
        raise RuntimeError("resolved device-normalization mode mismatch")
    with set_default_torch_num_threads():
        processor = MULTIMODAL_REGISTRY.create_processor(model_config, cache=None)
    tokenizer = processor.info.get_tokenizer()
    input_prompt_token_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    with set_default_torch_num_threads():
        processed = processor(
            input_prompt_token_ids,
            mm_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
            mm_uuid_items={"video": ["endpoint-pixel-parity"]},
        )
    mm_data = processed["mm_kwargs"].get_data()
    production_pixels = mm_data["pixel_values_videos"]
    production_grid = mm_data["video_grid_thw"]
    output_prompt_token_ids = processed["prompt_token_ids"]
    placeholders = processed["mm_placeholders"].get("video", [])
    if not isinstance(production_pixels, torch.Tensor):
        raise RuntimeError("production processor did not return video pixels")
    if not isinstance(production_grid, torch.Tensor):
        raise RuntimeError("production processor did not return video grid")
    expected_production_dtype = torch.uint8 if device_normalize else torch.bfloat16
    if production_pixels.dtype != expected_production_dtype:
        raise RuntimeError(
            "production processor dtype mismatch: "
            f"{production_pixels.dtype} != {expected_production_dtype}"
        )
    if len(placeholders) != 1:
        raise RuntimeError(f"expected one video placeholder, got {len(placeholders)}")

    video_processor = processor.info.get_video_processor()
    with set_default_torch_num_threads():
        raw_processed = processor(
            input_prompt_token_ids,
            mm_items=processor.info.parse_mm_data({"video": [(frames, metadata)]}),
            mm_uuid_items={"video": ["endpoint-pixel-parity-raw"]},
            hf_processor_mm_kwargs={
                "do_normalize": False,
                "do_rescale": False,
            },
        )
    raw_mm_data = raw_processed["mm_kwargs"].get_data()
    raw_processor_pixels = raw_mm_data["pixel_values_videos"]
    raw_grid = raw_mm_data["video_grid_thw"]
    if raw_processor_pixels.dtype != torch.uint8:
        raise RuntimeError(
            f"raw processor pixels are not uint8: {raw_processor_pixels.dtype}"
        )
    if not torch.equal(raw_grid, production_grid):
        raise RuntimeError("raw and production processor grids differ")
    if raw_processed["prompt_token_ids"] != output_prompt_token_ids:
        raise RuntimeError("raw and production processor prompt tokens differ")
    raw_placeholders = raw_processed["mm_placeholders"].get("video", [])
    if len(raw_placeholders) != 1:
        raise RuntimeError("raw processor did not return one video placeholder")
    if (
        raw_placeholders[0].offset != placeholders[0].offset
        or raw_placeholders[0].length != placeholders[0].length
        or not torch.equal(raw_placeholders[0].is_embed, placeholders[0].is_embed)
    ):
        raise RuntimeError("raw and production processor placeholders differ")
    if device_normalize and not torch.equal(raw_processor_pixels, production_pixels):
        raise RuntimeError("PR-head production processor pixels differ from raw probe")

    visual_dtype = model_config.dtype
    if not isinstance(visual_dtype, torch.dtype):
        raise RuntimeError(f"unexpected model dtype: {visual_dtype!r}")
    if device_normalize:
        from vllm.model_executor.models.vision import FusedInputNorm

        input_norm = FusedInputNorm.from_model_config(model_config).to("cuda")
        model_visible_pixels = input_norm(
            production_pixels.to("cuda", non_blocking=True), visual_dtype
        ).cpu()
    else:
        model_visible_pixels = production_pixels.to(
            device="cuda", dtype=visual_dtype, non_blocking=True
        ).cpu()
    torch.cuda.synchronize()

    grid_values = production_grid.tolist()
    if grid_values != [[16, 36, 64]]:
        raise RuntimeError(
            "processor did not resolve 32 frames at exact 1024x576: "
            f"grid={grid_values}"
        )
    resized_height = int(grid_values[0][1]) * int(video_processor.patch_size)
    resized_width = int(grid_values[0][2]) * int(video_processor.patch_size)
    if (resized_width, resized_height) != PIXEL_BUDGET:
        raise RuntimeError(
            f"processed resolution {(resized_width, resized_height)} != "
            f"{PIXEL_BUDGET}"
        )
    tensor_artifact = args.output.with_suffix(".tensors.pt")
    torch.save(
        {
            "raw_processor_pixels": raw_processor_pixels.cpu().contiguous(),
            "model_visible_pixels": model_visible_pixels.cpu().contiguous(),
            "video_grid_thw": production_grid.cpu().contiguous(),
            "output_prompt_token_ids": torch.tensor(
                output_prompt_token_ids, dtype=torch.int64
            ),
            "placeholder_is_embed": placeholders[0].is_embed.cpu().contiguous(),
        },
        tensor_artifact,
    )
    import PyNvVideoCodec as nvc

    pynv_package_root = Path(nvc.__file__).resolve().parent
    pynv_runtime_artifacts = {
        name: sha256_file(pynv_package_root / name)
        for name in PYNV_RUNTIME_ARTIFACT_SHA256
    }
    result = {
        "variant": args.variant,
        "commit": output(["git", "-C", str(args.root), "rev-parse", "HEAD^{commit}"]),
        "backend_kwargs": backend_kwargs,
        "native_layout": expected_shape_kind,
        "native": {
            "dtype": str(frames.dtype),
            "shape": list(frames.shape),
            "strides_bytes": list(frames.strides),
            "c_contiguous": bool(frames.flags.c_contiguous),
        },
        "canonical_thwc": {
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
            "strides_bytes": list(canonical.strides),
            "c_contiguous": bool(canonical.flags.c_contiguous),
            "sha256": hashlib.sha256(memoryview(canonical)).hexdigest(),
            "bytes": int(canonical.nbytes),
        },
        "metadata": metadata,
        "processor": {
            "model": MODEL,
            "revision": REVISION,
            "device_normalize": device_normalize,
            "model_dtype": str(visual_dtype),
            "configured_max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
            "configured_max_pixels_total": MAX_PIXELS_TOTAL,
            "processed_width": resized_width,
            "processed_height": resized_height,
            "processed_pixels_per_frame": resized_width * resized_height,
            "production_pixel_values": tensor_signature(production_pixels),
            "raw_processor_pixel_values": tensor_signature(raw_processor_pixels),
            "model_visible_pixel_values": tensor_signature(model_visible_pixels),
            "video_grid_thw": production_grid.tolist(),
            "video_grid_thw_sha256": tensor_sha256(production_grid),
            "input_prompt_token_count": len(input_prompt_token_ids),
            "output_prompt_token_count": len(output_prompt_token_ids),
            "output_prompt_token_ids_sha256": tensor_sha256(
                torch.tensor(output_prompt_token_ids, dtype=torch.int64)
            ),
            "placeholder": {
                "offset": placeholders[0].offset,
                "length": placeholders[0].length,
                "is_embed_sha256": tensor_sha256(placeholders[0].is_embed),
            },
            "tensor_artifact": {
                "path": str(tensor_artifact),
                "sha256": sha256_file(tensor_artifact),
            },
        },
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "vllm_origin": str(Path(vllm.__file__).resolve()),
            "transformers_origin": str(Path(transformers.__file__).resolve()),
            "transformers_distribution": importlib.metadata.version("transformers"),
            "transformers_init_sha256": sha256_file(
                Path(transformers.__file__).resolve()
            ),
            "pynvvideocodec_origin": str(Path(nvc.__file__).resolve()),
            "pynvvideocodec_module_version": getattr(nvc, "__version__", None),
            "pynvvideocodec_runtime_artifacts": pynv_runtime_artifacts,
            "pynvvideocodec_distribution": importlib.metadata.version("PyNvVideoCodec"),
        },
    }
    atomic_json(args.output, result)


def orchestrator(args: argparse.Namespace) -> None:
    args.root = args.root.resolve()
    args.python = args.python.absolute()
    args.transformers_root = args.transformers_root.resolve()
    args.video = args.video.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if (
        args.video.stat().st_size != VIDEO_BYTES
        or sha256_file(args.video) != VIDEO_SHA256
    ):
        raise RuntimeError("canonical video content mismatch")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": os.pathsep.join(
                [str(args.root), str(args.transformers_root)]
            ),
        }
    )
    result: dict[str, Any] = {
        "schema": "pynv-three-arm-pixel-parity-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.root),
        "video": {
            "path": str(args.video),
            "bytes": args.video.stat().st_size,
            "sha256": sha256_file(args.video),
        },
        "frames": FRAMES,
        "model": MODEL,
        "revision": REVISION,
        "pixel_budget_per_frame": {
            "width": PIXEL_BUDGET[0],
            "height": PIXEL_BUDGET[1],
            "max_pixels": MAX_PIXELS_PER_FRAME,
        },
        "max_pixels_total": MAX_PIXELS_TOTAL,
        "commits": COMMITS,
        "variants": {},
    }
    atomic_json(args.output, result)
    worker_script = Path(__file__).resolve()
    for variant, commit in COMMITS.items():
        checked_out = run(
            ["git", "-C", str(args.root), "checkout", "--quiet", "--detach", commit],
            capture_output=True,
        )
        if checked_out.returncode:
            raise RuntimeError(checked_out.stderr or checked_out.stdout)
        actual_commit = output(
            ["git", "-C", str(args.root), "rev-parse", "HEAD^{commit}"]
        )
        actual_tree = output(["git", "-C", str(args.root), "rev-parse", "HEAD^{tree}"])
        if actual_commit != commit or actual_tree != TREES[variant]:
            raise RuntimeError(
                f"{variant} source identity mismatch: " f"{actual_commit}/{actual_tree}"
            )
        status = output(
            [
                "git",
                "-C",
                str(args.root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ]
        )
        if status:
            raise RuntimeError(f"source is dirty at {variant}:\n{status}")
        bytecode_paths = sorted(
            path.relative_to(args.root).as_posix()
            for path in (args.root / "vllm").rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        )
        if bytecode_paths:
            raise RuntimeError(
                f"{variant} source contains ignored Python bytecode/cache: "
                + ", ".join(bytecode_paths[:64])
            )
        source_harness = (
            args.root / "benchmarks/multimodal/benchmark_pynvvideocodec_e2e.py"
        )
        if source_harness.is_file():
            raise RuntimeError(
                f"{variant} unexpectedly contains experiment harness: {source_harness}"
            )
        for (
            relative_path,
            expected_sha256,
        ) in CHAT_COMPLETION_PROTOCOL_ARTIFACTS.items():
            if sha256_file(args.root / relative_path) != expected_sha256:
                raise RuntimeError(
                    f"{variant} chat protocol artifact changed: {relative_path}"
                )
        help_command = [
            str(args.python),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            "--help=mm-device-do-normalize",
        ]
        help_completed = run(
            help_command,
            cwd=args.root,
            env=environment,
            capture_output=True,
        )
        if help_completed.returncode:
            raise RuntimeError(
                f"{variant} serve help failed:\n"
                f"{help_completed.stdout}\n{help_completed.stderr}"
            )
        if "--mm-device-do-normalize" not in help_completed.stdout:
            raise RuntimeError(f"{variant} serve help lacks --mm-device-do-normalize")
        worker_output = args.output.with_name(
            f"{args.output.stem}-{variant}-worker.json"
        )
        completed = run(
            [
                str(args.python),
                str(worker_script),
                "--worker",
                "--variant",
                variant,
                "--root",
                str(args.root),
                "--video",
                str(args.video),
                "--output",
                str(worker_output),
            ],
            cwd=args.root,
            env=environment,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{variant} worker failed:\n{completed.stdout}\n{completed.stderr}"
            )
        variant_result = json.loads(worker_output.read_text())
        if variant_result["commit"] != commit:
            raise RuntimeError(f"{variant} worker commit mismatch")
        if variant_result["runtime"].get("pynvvideocodec_distribution") != "2.0.4":
            raise RuntimeError(f"{variant} did not use PyNvVideoCodec 2.0.4")
        if (
            Path(variant_result["runtime"]["vllm_origin"]).resolve()
            != (args.root / "vllm/__init__.py").resolve()
        ):
            raise RuntimeError(f"{variant} imported vLLM from the wrong root")
        if (
            Path(variant_result["runtime"]["transformers_origin"]).resolve()
            != (args.transformers_root / "transformers/__init__.py").resolve()
        ):
            raise RuntimeError(f"{variant} imported Transformers from wrong root")
        if (
            variant_result["runtime"]["transformers_distribution"] != "5.14.1"
            or variant_result["runtime"]["transformers_init_sha256"]
            != TRANSFORMERS_INIT_SHA256
        ):
            raise RuntimeError(f"{variant} Transformers artifact mismatch")
        if (
            variant_result["runtime"]["pynvvideocodec_runtime_artifacts"]
            != PYNV_RUNTIME_ARTIFACT_SHA256
        ):
            raise RuntimeError(f"{variant} PyNv runtime artifact mismatch")
        post_commit = output(
            ["git", "-C", str(args.root), "rev-parse", "HEAD^{commit}"]
        )
        post_tree = output(["git", "-C", str(args.root), "rev-parse", "HEAD^{tree}"])
        post_status = output(
            [
                "git",
                "-C",
                str(args.root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ]
        )
        post_bytecode_paths = sorted(
            path.relative_to(args.root).as_posix()
            for path in (args.root / "vllm").rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        )
        if (
            post_commit != commit
            or post_tree != TREES[variant]
            or post_status
            or post_bytecode_paths
        ):
            raise RuntimeError(
                f"{variant} source changed during pixel worker: "
                f"commit={post_commit}, tree={post_tree}, status={post_status!r}, "
                f"bytecode={post_bytecode_paths[:64]}"
            )
        variant_result["source"] = {
            "commit": post_commit,
            "tree": post_tree,
            "status": post_status,
            "experiment_harness_exists": False,
            "chat_completion_protocol_artifacts": (CHAT_COMPLETION_PROTOCOL_ARTIFACTS),
            "ignored_python_bytecode_scan": {
                "scope": "vllm/**",
                "excluded_shared_venv": ".venv",
                "matched_paths": post_bytecode_paths,
                "passed": True,
            },
            "post_worker_validation": True,
        }
        if variant_result["source"]["tree"] != TREES[variant]:
            raise RuntimeError(f"{variant} source tree mismatch")
        variant_result["serve_help"] = {
            "command": help_command,
            "returncode": help_completed.returncode,
            "stdout_sha256": hashlib.sha256(help_completed.stdout.encode()).hexdigest(),
            "supports_mm_device_do_normalize_flag": True,
        }
        result["variants"][variant] = variant_result
        atomic_json(args.output, result)
    import torch

    variant_results = result["variants"]
    variant_tensors = {}
    for variant, variant_result in variant_results.items():
        tensor_artifact = variant_result["processor"]["tensor_artifact"]
        if sha256_file(Path(tensor_artifact["path"])) != tensor_artifact["sha256"]:
            raise RuntimeError("processor tensor artifact hash mismatch")
        variant_tensors[variant] = torch.load(
            tensor_artifact["path"], map_location="cpu", weights_only=True
        )

    reference_variant = "upstream"
    reference_result = variant_results[reference_variant]
    reference_tensors = variant_tensors[reference_variant]
    reference_raw = reference_tensors["raw_processor_pixels"]
    reference_model_visible = reference_tensors["model_visible_pixels"]
    model_visible_atol = 2**-15
    model_visible_pairwise = {}
    for variant in COMMITS:
        if variant == reference_variant:
            continue
        candidate = variant_tensors[variant]["model_visible_pixels"]
        if reference_model_visible.shape != candidate.shape:
            raise RuntimeError(
                f"model-visible tensor shape differs for {variant}: "
                f"{candidate.shape} != {reference_model_visible.shape}"
            )
        difference = (reference_model_visible.float() - candidate.float()).abs()
        model_visible_pairwise[variant] = {
            "candidate_variant": variant,
            "candidate_dtype": str(candidate.dtype),
            "exact": torch.equal(reference_model_visible, candidate),
            "mismatch_count": int(
                torch.count_nonzero(reference_model_visible != candidate)
            ),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "allclose": torch.allclose(
                reference_model_visible,
                candidate,
                rtol=0.0,
                atol=model_visible_atol,
            ),
        }

    def all_equal_tensor(key: str) -> bool:
        reference = reference_tensors[key]
        return all(
            torch.equal(reference, tensors[key])
            for variant, tensors in variant_tensors.items()
            if variant != reference_variant
        )

    parity_fields = {
        "canonical_thwc_sha256_exact_all_variants": (
            len(
                {
                    variant_result["canonical_thwc"]["sha256"]
                    for variant_result in variant_results.values()
                }
            )
            == 1
        ),
        "canonical_thwc_shape_exact_all_variants": (
            len(
                {
                    tuple(variant_result["canonical_thwc"]["shape"])
                    for variant_result in variant_results.values()
                }
            )
            == 1
        ),
        "sampled_frame_indices_exact_all_variants": (
            all(
                variant_result["metadata"]["frames_indices"]
                == reference_result["metadata"]["frames_indices"]
                for variant, variant_result in variant_results.items()
                if variant != reference_variant
            )
        ),
        "source_frame_count_exact_all_variants": (
            all(
                variant_result["metadata"]["total_num_frames"]
                == reference_result["metadata"]["total_num_frames"]
                for variant, variant_result in variant_results.items()
                if variant != reference_variant
            )
        ),
        "processor_raw_resized_pixels_exact_all_variants": all(
            torch.equal(reference_raw, tensors["raw_processor_pixels"])
            for variant, tensors in variant_tensors.items()
            if variant != reference_variant
        ),
        "processor_video_grid_thw_exact_all_variants": all_equal_tensor(
            "video_grid_thw"
        ),
        "processor_output_prompt_token_ids_exact_all_variants": all_equal_tensor(
            "output_prompt_token_ids"
        ),
        "processor_placeholder_metadata_exact_all_variants": (
            all(
                variant_result["processor"]["placeholder"]
                == reference_result["processor"]["placeholder"]
                for variant, variant_result in variant_results.items()
                if variant != reference_variant
            )
            and all_equal_tensor("placeholder_is_embed")
        ),
        "processor_resolution_exact_1024x576_all_variants": all(
            variant_result["processor"]["processed_width"] == PIXEL_BUDGET[0]
            and variant_result["processor"]["processed_height"] == PIXEL_BUDGET[1]
            for variant_result in variant_results.values()
        ),
        "processor_pixel_budget_exact_all_variants": all(
            variant_result["processor"]["configured_max_pixels_per_frame"]
            == MAX_PIXELS_PER_FRAME
            and variant_result["processor"]["configured_max_pixels_total"]
            == MAX_PIXELS_TOTAL
            for variant_result in variant_results.values()
        ),
        "model_visible_bfloat16_allclose_all_variants": all(
            comparison["allclose"] for comparison in model_visible_pairwise.values()
        ),
    }
    if not all(parity_fields.values()):
        result["status"] = "failed"
        result["parity"] = parity_fields
        atomic_json(args.output, result)
        raise RuntimeError(f"endpoint pixel parity failed: {parity_fields}")
    result.update(
        {
            "status": "passed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "parity": parity_fields,
            "model_visible_comparison": {
                "reference_variant": reference_variant,
                "reference_dtype": str(reference_model_visible.dtype),
                "variant_dtypes": {
                    variant: str(tensors["model_visible_pixels"].dtype)
                    for variant, tensors in variant_tensors.items()
                },
                "shape": list(reference_model_visible.shape),
                "exact_all_variants": all(
                    comparison["exact"]
                    for comparison in model_visible_pairwise.values()
                ),
                "allclose": all(
                    comparison["allclose"]
                    for comparison in model_visible_pairwise.values()
                ),
                "pairwise_to_reference": model_visible_pairwise,
                "rtol": 0.0,
                "atol": model_visible_atol,
            },
        }
    )
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=COMMITS)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--transformers-root", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        if args.variant is None:
            parser.error("--variant is required with --worker")
        worker(args)
    else:
        if args.python is None or args.transformers_root is None:
            parser.error("--python and --transformers-root are required")
        orchestrator(args)


if __name__ == "__main__":
    main()
