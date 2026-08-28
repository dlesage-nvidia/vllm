# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import stat
import statistics
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

UTC = timezone.utc

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
SCHEDULE = [
    (1, [8, 16, 32], ["upstream", "pr-base", "pr-head"]),
    (2, [32, 16, 8], ["pr-head", "pr-base", "upstream"]),
    (3, [16, 32, 8], ["pr-base", "pr-head", "upstream"]),
    (4, [8, 32, 16], ["upstream", "pr-head", "pr-base"]),
    (5, [32, 8, 16], ["pr-head", "upstream", "pr-base"]),
    (6, [16, 8, 32], ["pr-base", "upstream", "pr-head"]),
]
REPETITIONS = 6
PAIRWISE_COMPARISONS = (
    ("upstream_to_pr_base", "upstream", "pr-base"),
    ("pr_base_to_pr_head", "pr-base", "pr-head"),
    ("upstream_to_pr_head", "upstream", "pr-head"),
)
WORKLOAD_PARITY_CONFIGURATION_FIELDS = (
    "model",
    "revision",
    "prompt",
    "prompt_sha256",
    "output_len",
    "seed",
    "frame_target",
    "video_count",
    "video_cycle_policy",
    "video_pixel_budget",
    "backend_argument",
    "request_media_io_kwargs",
    "server_mm_processor_kwargs",
    "server_limit_mm_per_prompt",
    "warmup_requests_by_concurrency",
    "measured_requests_per_concurrency",
    "concurrency_order",
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "mm_ipc_gpu_memory_gb",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "mm_processor_cache_gb",
    "prefix_caching",
    "request_timeout_seconds",
    "startup_timeout_seconds",
    "shutdown_timeout_seconds",
    "settle_seconds",
    "client_http_protocol",
)
ENDPOINT_TREATMENT_CONFIGURATION_FIELDS = (
    "backend_kwargs",
    "server_media_io_kwargs",
    "video_kwargs_for_metric_derivation",
    "extra_server_argv",
)
T_CRITICAL_95_DF5 = 2.570581835636314
MEASURED_REQUESTS = {8: 64, 16: 128, 32: 256}
WARMUP_REQUESTS = {8: 24, 16: 48, 32: 96}
CAMPAIGN_HARNESS_SHA256 = (
    "d6da18d1fd77df44476a66aadfb7767174906ce58b4da9b972b38d052255bcf6"
)
GPU_MONITOR_SHA256 = "239bcbbd0e635a8b44e46588142f336a8879067750aeee0d649faa8e62e950bc"
IDLE_GATE_SHA256 = "0a7119e7d0c40e3274ea9846db0b4e7213e7c1beeb4ddecce3a18d9641c5b02e"
GUARD_HELPER_SHA256 = "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
PREFLIGHT_RUNNER_FILENAME = "run_pynv_persistent_three_arm_high_concurrency_pilots.py"
PREFLIGHT_RUNNER_SHA256 = (
    "77aa45391de3a3436168827025ac6c5b7812bcc977e121d99fd244a5159f68b0"
)
PIXEL_PREFLIGHT_FILENAME = "preflight_pynv_persistent_three_arm_pixel_parity.py"
PIXEL_PREFLIGHT_SHA256 = (
    "af4cadebcfada425997baf3f773b0f81d4103e981d87222cc355bc884db0a2d8"
)
MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS = 1.0
INGRESS_IDLE_SECONDS = 1200.0
INGRESS_IDLE_TIMEOUT_SECONDS = 21600.0
APPROVED_MONITOR_WATCHDOG_PAIRS = frozenset({(1200.0, 120.0), (3600.0, 120.0)})
TIMING_MONITOR_WATCHDOG_PAIR = (3600.0, 120.0)
PIXEL_MONITOR_WATCHDOG_PAIR = (1200.0, 120.0)
CELL_IDLE_SECONDS = 30.0
CELL_IDLE_TIMEOUT_SECONDS = 1800.0
EXPECTED_PROMPT_TOKEN_IDS_SHA256 = (
    "35e94b098ceca2dcafba9847470cf7ced3bd823189b070806531ec8fb6db7818"
)
VIDEO_SHA256 = "b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d"
VIDEO_BYTES = 13_267_543
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
TRANSFORMERS_INIT_SHA256 = (
    "67b01cb68df95d42da0661ea120535f33bb618225622e9523bd32e3b7741f9e1"
)
CHAT_COMPLETION_PROTOCOL_SHA256 = (
    "47e5d710fd66886bc25946414f5c8e6e3a665cee7910feb5eacd4a17f3331da7"
)
CHAT_COMPLETION_SERVING_SHA256 = (
    "9982953285e9df469032a82fffa4095d0e9d86278bede6e2b91d03d02373d182"
)
TRANSFORMERS_OVERLAY_BASENAME = "vllm-pynv-e2e-transformers-5.14.1-20260827"
RUNTIME_MANIFEST_TOOL_SHA256 = (
    "d4edac7bc314aba8ceedc799b9d9b1c64ac880d340dff70b949db50066f1981a"
)
RUNTIME_MANIFEST_TEST_SHA256 = (
    "d0ab0fcf324f6bc1042610a0ac5a970fe5fcf3fe31927aa904d0bb0ea76e0366"
)
RUNTIME_MANIFEST_EXPECTATIONS = {
    "transformers_overlay": {
        "kind": "transformers-overlay",
        "sha256": ("91e8b5660cb228e78f48fe931bc28dd68f92d751475e9d76b93590246a255bb0"),
        "manifest_bytes": 418_598,
        "regular_file_count": 2_750,
        "logical_total_bytes": 51_873_110,
    },
    "transformers_package": {
        "kind": "transformers",
        "sha256": ("a33471c896d571395e22d4d4f1fa58f6b4fee7c0b66f281fceabaab1a804241a"),
        "manifest_bytes": 381_637,
        "regular_file_count": 2_740,
        "logical_total_bytes": 51_553_351,
    },
    "hf_snapshot": {
        "kind": "hf-snapshot",
        "sha256": ("5a2020450ee3804b0e3c5e8be0b1bf33eab679796706ca416e36517a10c3baf3"),
        "manifest_bytes": 2_277,
        "regular_file_count": 12,
        "logical_total_bytes": 4_266_648_961,
    },
}
TRANSFORMERS_OVERLAY_TREE_MANIFEST = RUNTIME_MANIFEST_EXPECTATIONS[
    "transformers_overlay"
]
TRANSFORMERS_TREE_MANIFEST = RUNTIME_MANIFEST_EXPECTATIONS["transformers_package"]
HF_SNAPSHOT_TREE_MANIFEST = RUNTIME_MANIFEST_EXPECTATIONS["hf_snapshot"]
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROMPT = "Describe this video concisely and factually."
SERVED_MODEL_NAME = "qwen3-vl-video-throughput"
PIXEL_BUDGET = (1024, 576)
FRAMES = 32
TOTAL_MAX_PIXELS = PIXEL_BUDGET[0] * PIXEL_BUDGET[1] * FRAMES
OUTPUT_LENGTH = 32
EXPECTED_PROMPT_TOKENS = 9375
MAX_NUM_SEQS = 32
KV_CACHE_MEMORY_BYTES = 40 * 1024**3


def validate_monitor_watchdog_pair(
    watchdog_pair: tuple[float, float],
) -> tuple[float, float]:
    pair = (float(watchdog_pair[0]), float(watchdog_pair[1]))
    if pair not in APPROVED_MONITOR_WATCHDOG_PAIRS:
        raise ValueError(f"unapproved monitor watchdog pair: {pair}")
    return pair


def build_monitored_command(
    *,
    python: Path,
    monitor: Path,
    output: Path,
    child_command: Sequence[str],
    watchdog_pair: tuple[float, float],
    conflicting_controller_roots: Sequence[Path],
) -> list[str]:
    timeout_seconds, grace_seconds = validate_monitor_watchdog_pair(watchdog_pair)
    if not child_command:
        raise ValueError("monitored child command cannot be empty")
    command = [
        str(python),
        str(monitor),
        "--output",
        str(output),
        "--device-index",
        "0",
        "--timeout-seconds",
        f"{timeout_seconds:g}",
        "--timeout-grace-seconds",
        f"{grace_seconds:g}",
    ]
    for root in conflicting_controller_roots:
        command.extend(["--conflicting-controller-root", str(root)])
    command.extend(["--", *map(str, child_command)])
    return command


def validate_cell_idle_pair(seconds: float, timeout: float) -> tuple[float, float]:
    pair = (float(seconds), float(timeout))
    expected = (CELL_IDLE_SECONDS, CELL_IDLE_TIMEOUT_SECONDS)
    if pair != expected:
        raise ValueError(
            f"per-attempt idle pair must remain exactly {expected}: {pair}"
        )
    return pair


def build_idle_gate_command(
    *,
    python: Path,
    idle_gate: Path,
    output: Path,
    seconds: float,
    timeout: float,
    conflicting_controller_roots: Sequence[Path],
) -> list[str]:
    command = [
        str(python),
        str(idle_gate),
        "--seconds",
        f"{float(seconds):g}",
        "--timeout",
        f"{float(timeout):g}",
        "--output",
        str(output),
    ]
    for root in conflicting_controller_roots:
        command.extend(["--conflicting-controller-root", str(root)])
    return command


def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def output(command: Sequence[str]) -> str:
    result = run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rehash_artifact(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Hash a bound regular file while proving its pathname and inode stay stable."""
    path_value = binding.get("path")
    resolved_value = binding.get("resolved_path")
    if not isinstance(path_value, str) or not isinstance(resolved_value, str):
        raise RuntimeError(f"{label} artifact path binding is missing")
    path = Path(path_value)
    claimed_resolved = Path(resolved_value)
    if not path.is_absolute() or not claimed_resolved.is_absolute():
        raise RuntimeError(f"{label} artifact paths are not absolute")
    if os.path.normpath(path_value) != path_value:
        raise RuntimeError(f"{label} artifact path is not normalized")
    try:
        path_lstat_before = path.lstat()
        resolved_before = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} artifact path cannot be resolved") from error
    if (
        resolved_before != claimed_resolved
        or claimed_resolved.resolve(strict=True) != claimed_resolved
        or claimed_resolved.is_symlink()
    ):
        raise RuntimeError(f"{label} artifact resolved-path binding mismatch")
    try:
        target_lstat_before = claimed_resolved.lstat()
    except OSError as error:
        raise RuntimeError(f"{label} resolved artifact cannot be inspected") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(claimed_resolved, flags)
    except OSError as error:
        raise RuntimeError(f"{label} artifact stable open failed") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} artifact is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"{label} artifact changed while hashing")
    try:
        path_lstat_after = path.lstat()
        target_lstat_after = claimed_resolved.lstat()
        resolved_after = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} artifact path changed after hashing") from error
    if (
        resolved_after != resolved_before
        or any(
            getattr(path_lstat_before, field) != getattr(path_lstat_after, field)
            for field in stable_fields
        )
        or any(
            getattr(target_lstat_before, field) != getattr(target_lstat_after, field)
            for field in stable_fields
        )
        or any(
            getattr(before, field) != getattr(target_lstat_before, field)
            or getattr(after, field) != getattr(target_lstat_after, field)
            for field in stable_fields
        )
    ):
        raise RuntimeError(f"{label} artifact pathname was substituted")
    actual_sha256 = digest.hexdigest()
    if binding.get("bytes") != before.st_size or binding.get("sha256") != actual_sha256:
        raise RuntimeError(f"{label} artifact size/SHA-256 claim mismatch")
    return {
        "path": str(path),
        "resolved_path": str(resolved_before),
        "basename": resolved_before.name,
        "bytes": before.st_size,
        "sha256": actual_sha256,
        "device": before.st_dev,
        "inode": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "stable_open_fstat_before_after": True,
        "path_identity_before_after": True,
    }


def revalidate_runtime_artifact_manifest(
    python_provenance: Mapping[str, Any], *, source_root: Path
) -> dict[str, Any]:
    artifacts = python_provenance.get("runtime_artifacts")
    origins = python_provenance.get("module_origins")
    native_origins = python_provenance.get("native_module_origins")
    executable = python_provenance.get("executable")
    if (
        not isinstance(artifacts, list)
        or not isinstance(origins, Mapping)
        or not isinstance(native_origins, Mapping)
        or not isinstance(executable, str)
    ):
        raise RuntimeError("runtime artifact provenance is incomplete")
    claimed_by_resolved: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(artifacts):
        if not isinstance(binding, Mapping):
            raise RuntimeError("runtime artifact binding is not an object")
        path_value = binding.get("path")
        resolved_value = binding.get("resolved_path")
        if (
            not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or os.path.normpath(path_value) != path_value
            or not isinstance(resolved_value, str)
            or not Path(resolved_value).is_absolute()
        ):
            raise RuntimeError(f"runtime artifact {index} path is invalid")
        if str(Path(path_value).resolve(strict=True)) != resolved_value:
            raise RuntimeError(f"runtime artifact {index} path binding is inconsistent")
        if resolved_value in claimed_by_resolved:
            raise RuntimeError("runtime artifact list contains a duplicate path")
        claimed_by_resolved[resolved_value] = binding

    required_paths: dict[str, str] = {"python": executable}
    for name in ("torch", "numpy", "transformers", "PyNvVideoCodec"):
        origin = origins.get(name)
        if not isinstance(origin, str):
            raise RuntimeError(f"required runtime module origin is missing: {name}")
        required_paths[f"module:{name}"] = origin
    if set(native_origins) != {"torch._C", "numpy._core._multiarray_umath"}:
        raise RuntimeError("required native module origins are incomplete")
    for name, origin in native_origins.items():
        if not isinstance(origin, str):
            raise RuntimeError(f"native runtime origin is invalid: {name}")
        required_paths[f"native:{name}"] = origin

    required_bindings: dict[str, str] = {}
    for label, path_value in required_paths.items():
        resolved = str(Path(path_value).resolve(strict=True))
        if resolved not in claimed_by_resolved:
            raise RuntimeError(f"required runtime artifact is not bound: {label}")
        required_bindings[label] = resolved

    source_root = source_root.resolve()
    vllm_package_root = source_root / "vllm"
    claimed_vllm_bindings = sorted(
        (str(Path(str(binding["path"]))), resolved)
        for resolved, binding in claimed_by_resolved.items()
        if Path(str(binding["path"])).is_relative_to(vllm_package_root)
        and Path(str(binding["path"])).suffix in {".so", ".pyd"}
    )
    if not claimed_vllm_bindings:
        raise RuntimeError("runtime artifact manifest lacks vLLM native extensions")
    discovered_vllm_bindings = sorted(
        (str(path), str(path.resolve(strict=True)))
        for pattern in ("*.so", "*.pyd")
        for path in vllm_package_root.rglob(pattern)
        if path.is_file()
    )
    if claimed_vllm_bindings != discovered_vllm_bindings:
        raise RuntimeError("runtime artifact manifest omits a vLLM native extension")
    vllm_native = sorted({resolved for _, resolved in claimed_vllm_bindings})
    pynv_origin = Path(required_paths["module:PyNvVideoCodec"]).resolve(strict=True)
    pynv_package = pynv_origin.parent
    pynv_native = sorted(
        resolved
        for resolved in claimed_by_resolved
        if Path(resolved).parent == pynv_package
        and (Path(resolved).match("*.so*") or Path(resolved).suffix == ".pyd")
    )
    if not pynv_native:
        raise RuntimeError("runtime artifact manifest lacks PyNv native extensions")
    discovered_pynv_native = sorted(
        str(path.resolve(strict=True))
        for pattern in ("*.so*", "*.pyd")
        for path in pynv_package.glob(pattern)
        if path.is_file()
    )
    if pynv_native != discovered_pynv_native:
        raise RuntimeError("runtime artifact manifest omits a PyNv native extension")

    selected_paths = {
        *required_bindings.values(),
        *pynv_native,
        *vllm_native,
    }
    live_artifacts = [
        stable_rehash_artifact(
            claimed_by_resolved[resolved], label=f"runtime artifact {resolved}"
        )
        for resolved in sorted(selected_paths)
    ]
    by_resolved = {item["resolved_path"]: item for item in live_artifacts}
    if set(by_resolved) != selected_paths:
        raise RuntimeError("live runtime artifact path set changed during rehash")

    torch_runtime = python_provenance.get("torch_runtime")
    if not isinstance(torch_runtime, Mapping):
        raise RuntimeError("torch runtime provenance is missing")
    nvcc_audit = None
    nvcc = torch_runtime.get("nvcc")
    if nvcc is not None:
        if not isinstance(nvcc, Mapping):
            raise RuntimeError("nvcc artifact binding is malformed")
        nvcc_audit = stable_rehash_artifact(nvcc, label="nvcc")
    canonical_manifest = {
        "artifacts": live_artifacts,
        "required_bindings": dict(sorted(required_bindings.items())),
        "pynv_native_paths": pynv_native,
        "vllm_native_paths": vllm_native,
        "nvcc": nvcc_audit,
    }
    return {
        "schema": "pynv-live-runtime-artifact-manifest-v1",
        **canonical_manifest,
        "sha256": sha256_json(canonical_manifest),
    }


def capture_live_runtime_artifact_manifest(
    *,
    harness: Path,
    python: Path,
    source_root: Path,
    pythonpath_extras: Sequence[Path],
    environment: Mapping[str, str],
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("pynv_frozen_harness_probe", harness)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import frozen harness: {harness}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe_environment = dict(environment)
    probe_environment.update(
        {
            "PATH": f"{python.parent}{os.pathsep}{probe_environment.get('PATH', '')}",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONPATH": os.pathsep.join(
                [str(source_root), *(str(path) for path in pythonpath_extras)]
            ),
        }
    )
    provenance = module.python_provenance(python, source_root, probe_environment)
    return revalidate_runtime_artifact_manifest(provenance, source_root=source_root)


def revalidate_live_runtime_artifact_manifest_binding(
    manifest: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if manifest.get("schema") != "pynv-live-runtime-artifact-manifest-v1":
        raise RuntimeError(f"{label} live artifact manifest schema mismatch")
    artifacts = manifest.get("artifacts")
    required = manifest.get("required_bindings")
    pynv_paths = manifest.get("pynv_native_paths")
    vllm_paths = manifest.get("vllm_native_paths")
    if (
        not isinstance(artifacts, list)
        or not artifacts
        or not isinstance(required, Mapping)
        or not isinstance(pynv_paths, list)
        or not isinstance(vllm_paths, list)
        or not pynv_paths
        or not vllm_paths
    ):
        raise RuntimeError(f"{label} live artifact manifest is incomplete")
    if (
        not all(isinstance(path, str) for path in pynv_paths)
        or not all(isinstance(path, str) for path in vllm_paths)
        or not all(isinstance(path, str) for path in required.values())
    ):
        raise RuntimeError(f"{label} live artifact path set is malformed")
    expected_required_keys = {
        "python",
        "module:torch",
        "module:numpy",
        "module:transformers",
        "module:PyNvVideoCodec",
        "native:torch._C",
        "native:numpy._core._multiarray_umath",
    }
    if set(required) != expected_required_keys:
        raise RuntimeError(f"{label} live required artifact set is incomplete")
    if pynv_paths != sorted(set(pynv_paths)) or vllm_paths != sorted(set(vllm_paths)):
        raise RuntimeError(f"{label} live native artifact paths are not canonical")
    pynv_origin = required.get("module:PyNvVideoCodec")
    if not isinstance(pynv_origin, str):
        raise RuntimeError(f"{label} live artifact manifest lacks PyNv origin")
    pynv_package = Path(pynv_origin).resolve(strict=True).parent
    discovered_pynv = sorted(
        str(path.resolve(strict=True))
        for pattern in ("*.so*", "*.pyd")
        for path in pynv_package.glob(pattern)
        if path.is_file()
    )
    artifacts_by_resolved: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"{label} live artifact {index} is malformed")
        logical_path = artifact.get("path")
        resolved_path = artifact.get("resolved_path")
        if not isinstance(logical_path, str) or not isinstance(resolved_path, str):
            raise RuntimeError(f"{label} live artifact path set is malformed")
        if (
            not Path(logical_path).is_absolute()
            or not Path(resolved_path).is_absolute()
        ):
            raise RuntimeError(f"{label} live artifact paths are not absolute")
        if os.path.normpath(logical_path) != logical_path:
            raise RuntimeError(f"{label} live artifact path is not normalized")
        if os.path.normpath(resolved_path) != resolved_path:
            raise RuntimeError(f"{label} live resolved path is not normalized")
        if resolved_path in artifacts_by_resolved:
            raise RuntimeError(f"{label} live artifact path set is malformed")
        artifacts_by_resolved[resolved_path] = artifact
    if not set(vllm_paths).issubset(artifacts_by_resolved):
        raise RuntimeError(f"{label} vLLM native artifact bindings are incomplete")
    logical_vllm_paths = [
        Path(str(artifacts_by_resolved[path].get("path", ""))) for path in vllm_paths
    ]
    vllm_package_roots = {
        path.parent for path in logical_vllm_paths if path.parent.name == "vllm"
    }
    if len(vllm_package_roots) != 1:
        raise RuntimeError(f"{label} vLLM logical package root is ambiguous")
    vllm_package = next(iter(vllm_package_roots))
    if not all(
        path.is_absolute() and path.is_relative_to(vllm_package)
        for path in logical_vllm_paths
    ):
        raise RuntimeError(f"{label} vLLM logical native paths escaped package root")
    claimed_vllm_bindings = sorted(
        (str(logical), resolved)
        for logical, resolved in zip(logical_vllm_paths, vllm_paths)
    )
    discovered_vllm_bindings = sorted(
        (str(path), str(path.resolve(strict=True)))
        for pattern in ("*.so", "*.pyd")
        for path in vllm_package.rglob(pattern)
        if path.is_file()
    )
    if claimed_vllm_bindings != discovered_vllm_bindings:
        raise RuntimeError(f"{label} live vLLM native artifact set changed")
    if sorted(pynv_paths) != discovered_pynv:
        raise RuntimeError(f"{label} live native artifact set changed")
    expected_artifact_paths = {
        *required.values(),
        *pynv_paths,
        *vllm_paths,
    }
    claimed_artifact_paths = {
        item.get("resolved_path") for item in artifacts if isinstance(item, Mapping)
    }
    if claimed_artifact_paths != expected_artifact_paths:
        raise RuntimeError(f"{label} live artifact manifest path set mismatch")
    rehashed = [
        stable_rehash_artifact(item, label=f"{label} artifact {index}")
        for index, item in enumerate(artifacts)
        if isinstance(item, Mapping)
    ]
    if len(rehashed) != len(artifacts) or rehashed != artifacts:
        raise RuntimeError(f"{label} live artifact manifest changed")
    nvcc = manifest.get("nvcc")
    if nvcc is not None:
        if (
            not isinstance(nvcc, Mapping)
            or stable_rehash_artifact(nvcc, label=f"{label} nvcc") != nvcc
        ):
            raise RuntimeError(f"{label} nvcc live artifact changed")
    canonical = {
        "artifacts": artifacts,
        "required_bindings": dict(required),
        "pynv_native_paths": pynv_paths,
        "vllm_native_paths": vllm_paths,
        "nvcc": nvcc,
    }
    if manifest.get("sha256") != sha256_json(canonical):
        raise RuntimeError(f"{label} live artifact manifest hash mismatch")
    return dict(manifest)


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_bound_file(
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha256_key: str,
    expected_parent: Path,
    bytes_key: str | None = None,
) -> Path:
    path = Path(str(record.get(path_key, ""))).resolve()
    if path.parent != expected_parent.resolve() or not path.is_file():
        raise RuntimeError(f"bound artifact path is invalid: {path}")
    if record.get(sha256_key) != sha256_file(path):
        raise RuntimeError(f"bound artifact hash changed: {path}")
    if bytes_key is not None and record.get(bytes_key) != path.stat().st_size:
        raise RuntimeError(f"bound artifact size changed: {path}")
    return path


def validate_runtime_manifest_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_label: str,
    expected_manifests: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        checkpoint.get("status") != "passed"
        or checkpoint.get("label") != expected_label
        or checkpoint.get("manifests") != expected_manifests
        or checkpoint.get("evidence_sha256") != sha256_json(expected_manifests)
        or not isinstance(checkpoint.get("validated_utc"), str)
    ):
        raise RuntimeError(f"runtime-manifest checkpoint mismatch: {expected_label}")
    return {
        "label": expected_label,
        "evidence_sha256": checkpoint["evidence_sha256"],
    }


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def hf_cache_environment(snapshot_root: Path) -> dict[str, str]:
    snapshot_root = snapshot_root.resolve()
    expected_repository_directory = "models--" + MODEL.replace("/", "--")
    if (
        snapshot_root.name != REVISION
        or snapshot_root.parent.name != "snapshots"
        or snapshot_root.parent.parent.name != expected_repository_directory
    ):
        raise RuntimeError(
            "HF snapshot root is not the exact model/revision cache layout: "
            f"{snapshot_root}"
        )
    hub_cache = snapshot_root.parent.parent.parent
    hf_home = hub_cache.parent
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hub_cache),
        "HUGGINGFACE_HUB_CACHE": str(hub_cache),
    }


def validate_runtime_tree_manifest(
    *,
    python: Path,
    tool: Path,
    kind: str,
    root: Path,
    jsonl: Path,
    summary_path: Path,
    expected: Mapping[str, Any],
    expected_root_basename: str | None = None,
    anchor_relative_path: str | None = None,
    anchor_sha256: str | None = None,
) -> dict[str, Any]:
    if sha256_file(jsonl) != expected["sha256"]:
        raise RuntimeError(f"{kind} canonical JSONL hash mismatch")
    if jsonl.stat().st_size != expected["manifest_bytes"]:
        raise RuntimeError(f"{kind} canonical JSONL byte count mismatch")
    summary = json.loads(summary_path.read_text())
    manifest = summary.get("manifest", {})
    expected_summary_values = {
        "manifest_sha256": expected["sha256"],
        "manifest_bytes": expected["manifest_bytes"],
        "regular_file_count": expected["regular_file_count"],
        "logical_total_bytes": expected["logical_total_bytes"],
    }
    for field, value in expected_summary_values.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"{kind} summary {field} mismatch: {manifest.get(field)} != {value}"
            )
    command = [
        str(python),
        str(tool),
        "--kind",
        kind,
        "--root",
        str(root),
        "--output-jsonl",
        str(jsonl),
        "--output-summary",
        str(summary_path),
        "--validate-existing",
    ]
    if kind == "hf-snapshot":
        command.extend(["--model", MODEL, "--revision", REVISION])
    else:
        if anchor_sha256 is None:
            raise RuntimeError(f"{kind} manifest requires an anchor SHA256")
        command.extend(["--anchor-sha256", anchor_sha256])
        if expected_root_basename is not None:
            command.extend(["--expected-root-basename", expected_root_basename])
        if anchor_relative_path is not None:
            command.extend(["--anchor-relative-path", anchor_relative_path])
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if kind == "hf-snapshot":
        environment.update(hf_cache_environment(root))
    completed = run(command, env=environment, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"{kind} full-byte revalidation failed: "
            f"{completed.stderr or completed.stdout}"
        )
    validation = json.loads(completed.stdout)
    if (
        validation.get("status") != "passed"
        or validation.get("actual_resolved_bytes_rehashed") is not True
        or validation.get("manifest_sha256") != expected["sha256"]
        or validation.get("regular_file_count") != expected["regular_file_count"]
        or validation.get("logical_total_bytes") != expected["logical_total_bytes"]
    ):
        raise RuntimeError(f"{kind} validation report mismatch: {validation}")
    return {
        "kind": kind,
        "root": str(root),
        "jsonl": {
            "path": str(jsonl),
            "bytes": jsonl.stat().st_size,
            "sha256": sha256_file(jsonl),
        },
        "summary": {
            "path": str(summary_path),
            "bytes": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        },
        "validation_command": command,
        "validation": validation,
    }


def validate_all_runtime_tree_manifests(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "transformers_overlay": validate_runtime_tree_manifest(
            python=args.python,
            tool=args.runtime_manifest_tool,
            kind="transformers-overlay",
            root=args.transformers_root,
            jsonl=args.transformers_overlay_manifest_jsonl,
            summary_path=args.transformers_overlay_manifest_summary,
            expected=TRANSFORMERS_OVERLAY_TREE_MANIFEST,
            expected_root_basename=TRANSFORMERS_OVERLAY_BASENAME,
            anchor_relative_path="transformers/__init__.py",
            anchor_sha256=TRANSFORMERS_INIT_SHA256,
        ),
        "transformers_package": validate_runtime_tree_manifest(
            python=args.python,
            tool=args.runtime_manifest_tool,
            kind="transformers",
            root=args.transformers_root / "transformers",
            jsonl=args.transformers_manifest_jsonl,
            summary_path=args.transformers_manifest_summary,
            expected=TRANSFORMERS_TREE_MANIFEST,
            expected_root_basename="transformers",
            anchor_relative_path="__init__.py",
            anchor_sha256=TRANSFORMERS_INIT_SHA256,
        ),
        "hf_snapshot": validate_runtime_tree_manifest(
            python=args.python,
            tool=args.runtime_manifest_tool,
            kind="hf-snapshot",
            root=args.hf_snapshot_root,
            jsonl=args.hf_manifest_jsonl,
            summary_path=args.hf_manifest_summary,
            expected=HF_SNAPSHOT_TREE_MANIFEST,
        ),
    }


def validate_runtime_manifests(
    *,
    python: Path,
    tool: Path,
    transformers_root: Path,
    transformers_overlay_jsonl: Path,
    transformers_overlay_summary: Path,
    transformers_package_jsonl: Path,
    transformers_package_summary: Path,
    hf_snapshot_root: Path,
    hf_jsonl: Path,
    hf_summary: Path,
) -> dict[str, Any]:
    """Validate all runtime byte trees for preflight and campaign callers."""
    return validate_all_runtime_tree_manifests(
        argparse.Namespace(
            python=python,
            runtime_manifest_tool=tool,
            transformers_root=transformers_root,
            transformers_overlay_manifest_jsonl=transformers_overlay_jsonl,
            transformers_overlay_manifest_summary=transformers_overlay_summary,
            transformers_manifest_jsonl=transformers_package_jsonl,
            transformers_manifest_summary=transformers_package_summary,
            hf_snapshot_root=hf_snapshot_root,
            hf_manifest_jsonl=hf_jsonl,
            hf_manifest_summary=hf_summary,
        )
    )


def runtime_manifest_checkpoint(
    *,
    expected: Mapping[str, Any],
    label: str,
    validation_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    current = validate_runtime_manifests(**validation_kwargs)
    if current != expected:
        raise RuntimeError(f"runtime byte-tree manifest changed at {label}")
    return {
        "status": "passed",
        "label": label,
        "validated_utc": datetime.now(UTC).isoformat(),
        "evidence_sha256": sha256_json(current),
        "manifests": current,
    }


def huggingface_cache_environment(snapshot_root: Path) -> dict[str, str]:
    return hf_cache_environment(snapshot_root)


def attach_secondary_failure(primary: BaseException, secondary: BaseException) -> None:
    note = (
        "post-attempt integrity validation also failed with "
        f"{type(secondary).__name__}"
    )
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(note)
    else:
        setattr(
            primary, "post_attempt_integrity_failure_type", type(secondary).__name__
        )


def _source_after_attempt(
    root: Path, *, commit: str | None, variant: str | None
) -> dict[str, Any]:
    if commit is None or variant is None:
        return validate_source_at_any_endpoint(root)
    return validate_source(root, commit, variant=variant)


@contextmanager
def attempt_integrity_context(
    *,
    record_container: list[dict[str, Any]],
    record: dict[str, Any],
    state: Mapping[str, Any],
    state_path: Path,
    runtime_manifests: Mapping[str, Any],
    runtime_validation_kwargs: Mapping[str, Any],
    live_runtime_capture_kwargs: Mapping[str, Any],
    source_root: Path,
    stem: str,
    commit: str | None,
    variant: str | None,
    evidence_paths: Mapping[str, Path],
) -> Any:
    """Persist both sides of an attempt even when its gate/body raises."""

    record["started_utc"] = datetime.now(timezone.utc).isoformat()
    record["attempt_state"] = "starting"
    record_container.append(record)
    write_json(state_path, state)
    primary_failure: BaseException | None = None
    post_failure: BaseException | None = None
    try:
        record["live_runtime_artifacts_before"] = (
            capture_live_runtime_artifact_manifest(**live_runtime_capture_kwargs)
        )
        record["runtime_manifest_before"] = runtime_manifest_checkpoint(
            expected=runtime_manifests,
            label=f"{stem}:before_attempt",
            validation_kwargs=runtime_validation_kwargs,
        )
        record["attempt_state"] = "running"
        write_json(state_path, state)
        yield record
    except BaseException as error:
        primary_failure = error
        record["body_status"] = "failed"
        record["body_failure_type"] = type(error).__name__
        raise
    else:
        record["body_status"] = "completed"
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        post_errors: list[BaseException] = []
        for label, path in evidence_paths.items():
            try:
                exists = path.is_file()
                record[f"{label}_bytes"] = path.stat().st_size if exists else None
                record[f"{label}_sha256"] = sha256_file(path) if exists else None
            except BaseException as error:
                post_errors.append(error)
                record[f"{label}_bytes"] = None
                record[f"{label}_sha256"] = None
                record[f"{label}_evidence_failure_type"] = type(error).__name__
        try:
            live_runtime_after = capture_live_runtime_artifact_manifest(
                **live_runtime_capture_kwargs
            )
            record["live_runtime_artifacts_after"] = live_runtime_after
            live_runtime_before = record.get("live_runtime_artifacts_before")
            if not isinstance(live_runtime_before, Mapping) or live_runtime_after.get(
                "sha256"
            ) != live_runtime_before.get("sha256"):
                raise RuntimeError("live runtime artifacts changed across attempt")
        except BaseException as error:
            post_errors.append(error)
            record["live_runtime_artifacts_after"] = {
                "status": "failed",
                "failure_type": type(error).__name__,
            }
        try:
            record["runtime_manifest_after"] = runtime_manifest_checkpoint(
                expected=runtime_manifests,
                label=f"{stem}:after_attempt",
                validation_kwargs=runtime_validation_kwargs,
            )
        except BaseException as error:
            post_errors.append(error)
            record["runtime_manifest_after"] = {
                "status": "failed",
                "failure_type": type(error).__name__,
            }
        try:
            record["source_after_attempt"] = _source_after_attempt(
                source_root, commit=commit, variant=variant
            )
        except BaseException as error:
            post_errors.append(error)
            record["source_after_attempt"] = {
                "status": "failed",
                "failure_type": type(error).__name__,
            }
        record["post_attempt_integrity_status"] = (
            "passed" if not post_errors else "failed"
        )
        record["attempt_state"] = "finished"
        try:
            write_json(state_path, state)
        except BaseException as error:
            post_errors.append(error)
        if post_errors:
            post_failure = RuntimeError(
                "post-attempt integrity validation failed: "
                + ", ".join(type(error).__name__ for error in post_errors)
            )
            if primary_failure is not None:
                attach_secondary_failure(primary_failure, post_failure)
            else:
                raise post_failure


def validate_jsonl_binding(
    report: Mapping[str, Any], *, expected_path: Path, expected_suffix: str
) -> dict[str, Any]:
    binding = report.get("sample_log")
    if not isinstance(binding, Mapping):
        raise RuntimeError("sample-log binding is missing")
    path = Path(str(binding.get("path", ""))).resolve()
    if (
        path != expected_path.resolve()
        or not path.name.endswith(expected_suffix)
        or not path.is_file()
    ):
        raise RuntimeError(f"sample-log path is invalid: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if binding.get("bytes") != actual_bytes or binding.get("sha256") != actual_sha256:
        raise RuntimeError(f"sample-log hash/size binding mismatch: {path}")
    parsed_count = 0
    samples: list[dict[str, Any]] = []
    first_time_ns = None
    last_time_ns = None
    last_monotonic_ns = None
    maximum_gap = 0.0
    with path.open() as source:
        for line in source:
            sample = json.loads(line)
            if sample.get("sample_index") != parsed_count:
                raise RuntimeError(
                    f"non-contiguous sample index in {path}: "
                    f"{sample.get('sample_index')} != {parsed_count}"
                )
            time_ns = sample.get("time_ns")
            monotonic_ns = sample.get("monotonic_ns")
            if not isinstance(time_ns, int) or not isinstance(monotonic_ns, int):
                raise RuntimeError(f"sample lacks integer timestamps: {path}")
            if first_time_ns is None:
                first_time_ns = time_ns
            if last_monotonic_ns is not None:
                gap = (monotonic_ns - last_monotonic_ns) / 1e9
                if gap < 0:
                    raise RuntimeError(f"sample monotonic timestamps regress: {path}")
                if not math.isclose(
                    float(sample.get("sample_gap_seconds")),
                    gap,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(f"sample gap field mismatch: {path}")
                maximum_gap = max(maximum_gap, gap)
            elif sample.get("sample_gap_seconds") is not None:
                raise RuntimeError(f"first sample gap must be null: {path}")
            last_time_ns = time_ns
            last_monotonic_ns = monotonic_ns
            samples.append(sample)
            parsed_count += 1
    if parsed_count != report.get("sample_count"):
        raise RuntimeError(
            f"sample count mismatch: {parsed_count} != {report.get('sample_count')}"
        )
    embedded_samples = report.get("samples")
    if embedded_samples is not None and embedded_samples != samples:
        raise RuntimeError(f"embedded monitor samples differ from JSONL: {path}")
    quiet = report.get("quiet_interval")
    if quiet is not None:
        if not isinstance(quiet, Mapping) or report.get("passed") is not True:
            raise RuntimeError(f"invalid quiet interval/report status: {path}")
        start_index = int(quiet["sample_start_index"])
        end_index = int(quiet["sample_end_index_inclusive"])
        if not (0 <= start_index < end_index == parsed_count - 1):
            raise RuntimeError(f"quiet interval boundaries are not terminal: {path}")
        quiet_samples = samples[start_index : end_index + 1]
        if len(quiet_samples) != quiet.get("sample_count"):
            raise RuntimeError(f"quiet interval sample count mismatch: {path}")
        if any(
            sample.get("reset_reasons") != []
            or sample.get("sample_error") is not None
            or sample.get("gpu") is None
            or sample.get("cpu", {}).get("conflicts") != []
            or sample.get("cpu", {}).get("errors") != []
            for sample in quiet_samples
        ):
            raise RuntimeError(f"quiet interval contains a dirty sample: {path}")
        first_quiet = quiet_samples[0]
        last_quiet = quiet_samples[-1]
        if (
            quiet.get("started_time_ns") != first_quiet["time_ns"]
            or quiet.get("finished_time_ns") != last_quiet["time_ns"]
            or quiet.get("started_utc") != first_quiet["utc"]
            or quiet.get("finished_utc") != last_quiet["utc"]
            or report.get("finished_time_ns") != last_quiet["time_ns"]
            or report.get("finished_utc") != last_quiet["utc"]
        ):
            raise RuntimeError(f"quiet interval timestamp binding mismatch: {path}")
        quiet_gaps = [
            (right["monotonic_ns"] - left["monotonic_ns"]) / 1e9
            for left, right in zip(quiet_samples, quiet_samples[1:])
        ]
        quiet_maximum_gap = max(quiet_gaps)
        quiet_duration = (
            last_quiet["monotonic_ns"] - first_quiet["monotonic_ns"]
        ) / 1e9
        if (
            not math.isclose(
                float(quiet["maximum_sample_gap_seconds"]),
                quiet_maximum_gap,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(quiet["duration_seconds"]),
                quiet_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or quiet_maximum_gap
            > float(report["configuration"]["maximum_sample_gap_seconds"])
        ):
            raise RuntimeError(
                f"quiet interval monotonic gap/duration mismatch: {path}"
            )
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "sample_count": parsed_count,
        "first_time_ns": first_time_ns,
        "last_time_ns": last_time_ns,
        "maximum_monotonic_gap_seconds": maximum_gap,
        "embedded_samples_exact": embedded_samples is not None,
        "quiet_interval_recomputed": quiet is not None,
    }


def validate_idle_gate_report(
    report: Mapping[str, Any],
    *,
    required_idle_seconds: float,
    required_timeout_seconds: float,
) -> None:
    configuration = report.get("configuration", {})
    device = report.get("device", {})
    quiet_interval = report.get("quiet_interval", {})
    if (
        report.get("status") != "passed"
        or report.get("passed") is not True
        or report.get("process", {}).get("script_sha256") != IDLE_GATE_SHA256
        or report.get("guard_helper", {}).get("sha256") != GUARD_HELPER_SHA256
        or device.get("index") != 0
        or configuration.get("device_index") != 0
        or float(configuration.get("required_idle_seconds", 0)) != required_idle_seconds
        or float(configuration.get("timeout_seconds", 0)) != required_timeout_seconds
        or configuration.get("sample_interval_seconds") != 0.2
        or configuration.get("maximum_sample_gap_seconds")
        != MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS
        or configuration.get("idle_memory_ceiling_mib") != 1024
        or configuration.get("idle_max_load_1m_per_cpu") != 0.25
        or quiet_interval.get("all_samples_clean") is not True
        or float(quiet_interval.get("duration_seconds", 0)) < required_idle_seconds
    ):
        raise RuntimeError(
            "idle gate report/configuration mismatch: "
            f"required_idle_seconds={required_idle_seconds}, report={report}"
        )


def validate_idle_gate_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_seconds: float,
    expected_timeout: float,
    conflicting_controller_roots: Sequence[Path],
) -> dict[str, Any]:
    report_path = Path(str(evidence.get("report_path", ""))).resolve()
    if not report_path.is_file():
        raise RuntimeError(f"idle-gate report is missing: {report_path}")
    report = json.loads(report_path.read_text())
    configuration = report.get("configuration", {})
    expected_roots = [
        str(path.resolve(strict=False)) for path in conflicting_controller_roots
    ]
    if (
        evidence.get("report") != report
        or evidence.get("report_sha256") != sha256_file(report_path)
        or report.get("status") != "passed"
        or report.get("passed") is not True
        or report.get("guard_helper", {}).get("sha256") != GUARD_HELPER_SHA256
        or report.get("process", {}).get("script_sha256") != IDLE_GATE_SHA256
        or configuration.get("device_index") != 0
        or configuration.get("required_idle_seconds") != float(expected_seconds)
        or configuration.get("timeout_seconds") != float(expected_timeout)
        or configuration.get("sample_interval_seconds") != 0.2
        or configuration.get("maximum_sample_gap_seconds") != 1.0
        or configuration.get("idle_memory_ceiling_mib") != 1024
        or configuration.get("idle_max_load_1m_per_cpu") != 0.25
        or configuration.get("conflicting_controller_roots") != expected_roots
    ):
        raise RuntimeError(f"idle-gate evidence/configuration mismatch: {report_path}")
    sample_audit = validate_jsonl_binding(
        report,
        expected_path=report_path.with_name(report_path.stem + ".samples.jsonl"),
        expected_suffix="-idle-gate.samples.jsonl",
    )
    if evidence.get("sample_log_audit") != sample_audit:
        raise RuntimeError(f"idle-gate sample audit mismatch: {report_path}")
    return {
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "sample_log_audit": sample_audit,
    }


def validate_monitor_evidence(
    report_path: Path,
    *,
    expected_command: Sequence[str],
    watchdog_pair: tuple[float, float],
    conflicting_controller_roots: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text())
    timeout, grace = validate_monitor_watchdog_pair(watchdog_pair)
    expected_roots = [
        str(path.resolve(strict=False)) for path in conflicting_controller_roots
    ]
    if (
        report.get("status") != "passed"
        or report.get("returncode") != 0
        or report.get("timed_out") is not False
        or report.get("contaminated") is not False
        or report.get("foreign_events") != []
        or report.get("monitor_errors") != []
        or report.get("command") != list(expected_command)
        or report.get("timeout_seconds") != timeout
        or report.get("timeout_grace_seconds") != grace
        or report.get("guard_helper", {}).get("sha256") != GUARD_HELPER_SHA256
        or report.get("process", {}).get("script_sha256") != GPU_MONITOR_SHA256
        or report.get("configuration", {}).get("device_index") != 0
        or report.get("configuration", {}).get("maximum_sample_gap_seconds") != 1.0
        or report.get("configuration", {}).get("conflicting_controller_roots")
        != expected_roots
        or len(report.get("post_exit_sample_indices", [])) < 2
        or report.get("post_popen_adopted_child_audit", {}).get("ran") is not True
        or report.get("post_popen_adopted_child_audit", {}).get(
            "survivors_before_cleanup"
        )
        != []
        or report.get("post_popen_adopted_child_audit", {}).get(
            "process_groups_alive_before_cleanup"
        )
        != []
    ):
        raise RuntimeError(f"GPU-monitor evidence mismatch: {report_path}")
    sample_audit = validate_jsonl_binding(
        report,
        expected_path=report_path.with_name(report_path.stem + ".samples.jsonl"),
        expected_suffix="-gpu-monitor.samples.jsonl",
    )
    validate_terminal_post_exit_samples(report)
    return report, sample_audit


def _expected_foreign_event_for_sample(
    sample: Mapping[str, Any], *, initial_contamination: bool
) -> dict[str, Any] | None:
    external_gpu = sample.get("external_gpu_processes")
    cpu_conflicts = sample.get("cpu_conflicts")
    if not isinstance(external_gpu, list) or not isinstance(cpu_conflicts, list):
        raise RuntimeError("monitor sample lacks foreign-process telemetry lists")
    cpu_apps: list[dict[str, Any]] = []
    for conflict in cpu_conflicts:
        if not isinstance(conflict, Mapping):
            raise RuntimeError("monitor sample has malformed CPU conflict evidence")
        argv = conflict.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(argument, str) for argument in argv
        ):
            raise RuntimeError("monitor CPU conflict lacks an argv list")
        cpu_apps.append(
            {
                **dict(conflict),
                "process_name": " ".join(argv),
                "used_memory_mib": -1,
            }
        )
    apps = [*external_gpu, *cpu_apps]
    if not apps:
        return None
    sample_index = sample.get("sample_index")
    if type(sample_index) is not int:
        raise RuntimeError("monitor foreign sample index is not an integer")
    event: dict[str, Any] = {
        "sample_index": sample_index,
        "utc": sample.get("utc"),
        "apps": apps,
    }
    if initial_contamination:
        reasons = sample.get("monitor_errors")
        if not isinstance(reasons, list) or not reasons:
            raise RuntimeError("initial contamination lacks telemetry reasons")
        event["reasons"] = reasons
    else:
        event["monotonic_ns"] = sample.get("monotonic_ns")
        if sample.get("post_exit_telemetry") is True:
            event["phase"] = "post_exit"
    return event


def validate_terminal_post_exit_samples(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    samples = report.get("samples")
    indices = report.get("post_exit_sample_indices")
    if not isinstance(samples, list) or len(samples) < 3:
        raise RuntimeError("monitor lacks two terminal post-exit samples")
    expected_indices = [len(samples) - 2, len(samples) - 1]
    if indices != expected_indices:
        raise RuntimeError("post-exit samples are not the final two JSONL rows")
    terminal = [samples[index] for index in expected_indices]
    if any(
        not isinstance(sample, Mapping)
        or sample.get("sample_index") != expected_indices[ordinal]
        or sample.get("post_exit_telemetry") is not True
        or sample.get("post_exit_ordinal") != ordinal
        for ordinal, sample in enumerate(terminal)
    ):
        raise RuntimeError("terminal post-exit sample identity/order mismatch")
    first_monotonic = terminal[0].get("monotonic_ns")
    second_monotonic = terminal[1].get("monotonic_ns")
    if type(first_monotonic) is not int or type(second_monotonic) is not int:
        raise RuntimeError("terminal post-exit samples lack monotonic timestamps")
    gap_seconds = (second_monotonic - first_monotonic) / 1e9
    if not 0.19 <= gap_seconds <= MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS:
        raise RuntimeError(
            f"terminal post-exit samples are not independently scheduled: {gap_seconds}"
        )
    return {
        "sample_indices": expected_indices,
        "ordinals": [0, 1],
        "monotonic_gap_seconds": gap_seconds,
        "terminal": True,
    }


def validate_contamination_retry_evidence(
    *,
    wrapper_returncode: int,
    report_path: Path,
    expected_wrapper_command: Sequence[str],
    expected_child_command: Sequence[str],
    watchdog_pair: tuple[float, float],
    conflicting_controller_roots: Sequence[Path],
) -> dict[str, Any]:
    """Prove that telemetry, rather than a workload failure, permits a retry."""
    if type(wrapper_returncode) is not int or wrapper_returncode != 99:
        raise RuntimeError("contamination retry requires monitor wrapper exit code 99")
    wrapper_command = list(expected_wrapper_command)
    child_command = list(expected_child_command)
    if len(wrapper_command) < 3 or "--" not in wrapper_command:
        raise RuntimeError("expected monitor wrapper command is malformed")
    delimiter = wrapper_command.index("--")
    if wrapper_command[delimiter + 1 :] != child_command:
        raise RuntimeError("monitor wrapper/child command binding mismatch")
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise RuntimeError(f"contamination monitor report is missing: {report_path}")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"contamination monitor report is unreadable: {report_path}"
        ) from error
    if not isinstance(report, Mapping):
        raise RuntimeError("contamination monitor report is not an object")

    timeout, grace = validate_monitor_watchdog_pair(watchdog_pair)
    expected_roots = [
        str(path.resolve(strict=False)) for path in conflicting_controller_roots
    ]
    expected_configuration = {
        "sample_interval_seconds": 0.2,
        "maximum_sample_gap_seconds": 1.0,
        "initial_idle_memory_ceiling_mib": 1024,
        "device_index": 0,
        "conflicting_controller_roots": expected_roots,
        "telemetry": "direct NVML; no external telemetry commands or pgrep",
        "workload_ownership": (
            "new process session/group plus PID/start_ticks ancestry"
        ),
    }
    if (
        report.get("status") != "contaminated"
        or report.get("contaminated") is not True
        or report.get("timed_out") is not False
        or report.get("monitor_errors") != []
        or report.get("termination_signal") is not None
        or report.get("command") != child_command
        or report.get("timeout_seconds") != timeout
        or report.get("timeout_grace_seconds") != grace
        or report.get("configuration") != expected_configuration
    ):
        raise RuntimeError("monitor report is not strict contamination evidence")

    process = report.get("process")
    helper = report.get("guard_helper")
    if not isinstance(process, Mapping) or not isinstance(helper, Mapping):
        raise RuntimeError("monitor script/helper provenance is missing")
    expected_python = Path(wrapper_command[0]).resolve()
    expected_monitor = Path(wrapper_command[1]).resolve()
    expected_helper = expected_monitor.with_name("pynv_gpu_guard.py")
    script_path = Path(str(process.get("script_path", ""))).resolve()
    helper_path = Path(str(helper.get("path", ""))).resolve()
    if (
        script_path != expected_monitor
        or helper_path != expected_helper
        or not expected_python.is_file()
        or not script_path.is_file()
        or not helper_path.is_file()
        or Path(str(process.get("executable", ""))).resolve() != expected_python
        or process.get("argv") != wrapper_command[1:]
    ):
        raise RuntimeError("monitor executable/script/helper path binding mismatch")
    script_sha256 = sha256_file(script_path)
    helper_sha256 = sha256_file(helper_path)
    if (
        process.get("script_sha256") != script_sha256
        or script_sha256 != GPU_MONITOR_SHA256
        or helper.get("sha256") != helper_sha256
        or helper_sha256 != GUARD_HELPER_SHA256
    ):
        raise RuntimeError("monitor script/helper SHA-256 binding mismatch")

    sample_audit = validate_jsonl_binding(
        report,
        expected_path=report_path.with_name(report_path.stem + ".samples.jsonl"),
        expected_suffix="-gpu-monitor.samples.jsonl",
    )
    samples = report.get("samples")
    events = report.get("foreign_events")
    if (
        not isinstance(samples, list)
        or not samples
        or not isinstance(events, list)
        or not events
    ):
        raise RuntimeError("contamination lacks rehashed foreign-process samples")

    process_was_launched = report.get("workload_identity") is not None
    initial_contamination = not process_was_launched
    post_exit_audit: dict[str, Any] | None = None
    expected_events = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or sample.get("sample_index") != index:
            raise RuntimeError("monitor sample sequence is malformed")
        expected_event = _expected_foreign_event_for_sample(
            sample,
            initial_contamination=initial_contamination and index == 0,
        )
        if expected_event is not None:
            expected_events.append(expected_event)
    if events != expected_events:
        raise RuntimeError("foreign events do not exactly match rehashed telemetry")

    child_returncode = report.get("returncode")
    child_semantics: str
    if initial_contamination:
        if (
            child_returncode is not None
            or report.get("command_pid") is not None
            or report.get("post_popen_adopted_child_audit") is not None
            or any(event.get("sample_index") != 0 for event in events)
        ):
            raise RuntimeError("initial contamination has inconsistent child state")
        child_semantics = "not_launched_due_to_initial_contamination"
    else:
        workload = report.get("workload_identity")
        post_audit = report.get("post_popen_adopted_child_audit")
        if (
            not isinstance(workload, Mapping)
            or type(report.get("command_pid")) is not int
            or workload.get("pid") != report.get("command_pid")
            or not isinstance(post_audit, Mapping)
            or post_audit.get("ran") is not True
            or post_audit.get("capture_errors") != []
            or post_audit.get("identity_errors") != []
            or post_audit.get("survivors_before_cleanup") != []
            or post_audit.get("process_groups_alive_before_cleanup") != []
            or type(child_returncode) is not int
            or child_returncode == 99
        ):
            raise RuntimeError("contamination has inconsistent launched-child state")
        post_exit_audit = validate_terminal_post_exit_samples(report)
        in_flight_events = [event for event in events if "phase" not in event]
        post_exit_events = [
            event for event in events if event.get("phase") == "post_exit"
        ]
        if in_flight_events:
            cleanup = report.get("timeout_cleanup")
            sent_signals = (
                {
                    action.get("signal")
                    for action in cleanup.get("signal_actions", [])
                    if isinstance(action, Mapping) and action.get("signal_sent") is True
                }
                if isinstance(cleanup, Mapping)
                else set()
            )
            valid_child_signal_status = (
                child_returncode in (-signal.SIGINT, 128 + signal.SIGINT)
                and "SIGINT" in sent_signals
            ) or (child_returncode == -signal.SIGKILL and "SIGKILL" in sent_signals)
            if (
                not isinstance(cleanup, Mapping)
                or cleanup.get("reason") != "foreign_workload_detected"
                or cleanup.get("completed") is not True
                or not valid_child_signal_status
            ):
                raise RuntimeError(
                    "in-flight contamination cleanup/child state mismatch"
                )
            child_semantics = "terminated_by_monitor_for_foreign_workload"
        elif post_exit_events:
            if child_returncode != 0:
                raise RuntimeError("post-exit contamination requires child success")
            child_semantics = "successful_child_then_post_exit_contamination"
        else:
            raise RuntimeError("launched-child contamination has no valid event phase")

    return {
        "status": "validated_contamination_retry",
        "wrapper_returncode": 99,
        "child_returncode": child_returncode,
        "child_semantics": child_semantics,
        "report": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "monitor_script": {
            "path": str(script_path),
            "bytes": script_path.stat().st_size,
            "sha256": script_sha256,
        },
        "guard_helper": {
            "path": str(helper_path),
            "bytes": helper_path.stat().st_size,
            "sha256": helper_sha256,
        },
        "configuration_sha256": sha256_json(expected_configuration),
        "wrapper_command_sha256": sha256_json(wrapper_command),
        "child_command_sha256": sha256_json(child_command),
        "foreign_event_sample_indices": [event["sample_index"] for event in events],
        "sample_log_audit": sample_audit,
        "terminal_post_exit_samples": (
            post_exit_audit if process_was_launched else None
        ),
    }


def monitor_coverage_audit(
    result: Mapping[str, Any], monitor: Mapping[str, Any]
) -> dict[str, Any]:
    sample_times = sorted(
        int(sample["monotonic_ns"])
        for sample in monitor.get("samples", [])
        if isinstance(sample, Mapping) and isinstance(sample.get("monotonic_ns"), int)
    )
    blocks = []
    for block in result["concurrency_blocks"]:
        concurrency = int(block["concurrency"])
        measured = block["measured"]
        started_ns, finished_ns = batch_monotonic_window(measured)
        in_window = [
            value for value in sample_times if started_ns <= value <= finished_ns
        ]
        if in_window:
            gaps = [
                (in_window[0] - started_ns) / 1e9,
                *(
                    (right - left) / 1e9
                    for left, right in zip(in_window, in_window[1:])
                ),
                (finished_ns - in_window[-1]) / 1e9,
            ]
        else:
            gaps = [(finished_ns - started_ns) / 1e9]
        maximum_gap = max(gaps)
        blocks.append(
            {
                "concurrency": concurrency,
                "sample_count": len(in_window),
                "maximum_boundary_inclusive_gap_seconds": maximum_gap,
                "maximum_allowed_gap_seconds": MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS,
                "passed": bool(in_window)
                and maximum_gap <= MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS,
            }
        )
    return {
        "policy": (
            "Every measured window must contain monitor samples and all boundary/"
            "adjacent gaps must be <=1 second, using only the shared host "
            "monotonic clock."
        ),
        "blocks": blocks,
        "passed": all(block["passed"] for block in blocks),
    }


def variant_backend_kwargs(variant: str) -> dict[str, Any]:
    if variant in {"upstream", "pr-base"}:
        return {"hw_decoders": 2}
    if variant == "pr-head":
        return {"hw_decoders": 2, "output_layout": "tchw"}
    raise ValueError(f"unknown variant: {variant}")


def variant_server_argv(variant: str) -> list[str]:
    if variant in {"upstream", "pr-base"}:
        return ["--no-mm-device-do-normalize"]
    if variant == "pr-head":
        return ["--mm-device-do-normalize"]
    raise ValueError(f"unknown variant: {variant}")


def validate_pixel_preflight_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    source_root: Path,
    transformers_root: Path,
    expected_video: Path,
    expected_gpu_name: str,
) -> dict[str, Any]:
    expected_top_level = {
        "schema": "pynv-three-arm-pixel-parity-v1",
        "status": "passed",
        "source_root": str(source_root),
        "frames": FRAMES,
        "model": MODEL,
        "revision": REVISION,
        "commits": COMMITS,
        "max_pixels_total": TOTAL_MAX_PIXELS,
    }
    for key, expected in expected_top_level.items():
        if result.get(key) != expected:
            raise RuntimeError(
                f"pixel preflight {key} mismatch: {result.get(key)!r} != {expected!r}"
            )
    if result.get("pixel_budget_per_frame") != {
        "width": PIXEL_BUDGET[0],
        "height": PIXEL_BUDGET[1],
        "max_pixels": PIXEL_BUDGET[0] * PIXEL_BUDGET[1],
    }:
        raise RuntimeError("pixel preflight per-frame budget mismatch")
    video = result.get("video", {})
    if (
        Path(str(video.get("path", ""))).resolve() != expected_video.resolve()
        or video.get("bytes") != VIDEO_BYTES
        or video.get("sha256") != VIDEO_SHA256
    ):
        raise RuntimeError("pixel preflight video provenance mismatch")
    parity = result.get("parity", {})
    if not isinstance(parity, Mapping) or not parity or not all(parity.values()):
        raise RuntimeError("pixel preflight parity is not fully passed")
    model_visible = result.get("model_visible_comparison", {})
    if (
        model_visible.get("reference_variant") != "upstream"
        or model_visible.get("reference_dtype") != "torch.bfloat16"
        or model_visible.get("allclose") is not True
        or model_visible.get("rtol") != 0.0
        or model_visible.get("atol") != 2**-15
    ):
        raise RuntimeError("pixel preflight model-visible comparison mismatch")
    variants = result.get("variants", {})
    if set(variants) != set(COMMITS):
        raise RuntimeError("pixel preflight variant set mismatch")
    import torch

    worker_artifacts = []
    runtime_fingerprints = {}
    loaded_tensors = {}
    for variant, commit in COMMITS.items():
        variant_result = variants[variant]
        source = variant_result.get("source", {})
        bytecode_scan = source.get("ignored_python_bytecode_scan", {})
        if (
            variant_result.get("commit") != commit
            or variant_result.get("backend_kwargs") != variant_backend_kwargs(variant)
            or variant_result.get("native_layout")
            != ("tchw" if variant == "pr-head" else "thwc")
            or source.get("commit") != commit
            or source.get("tree") != TREES[variant]
            or source.get("status") != ""
            or source.get("experiment_harness_exists") is not False
            or source.get("chat_completion_protocol_artifacts")
            != {
                "vllm/entrypoints/openai/chat_completion/protocol.py": (
                    CHAT_COMPLETION_PROTOCOL_SHA256
                ),
                "vllm/entrypoints/openai/chat_completion/serving.py": (
                    CHAT_COMPLETION_SERVING_SHA256
                ),
            }
            or bytecode_scan.get("passed") is not True
            or bytecode_scan.get("matched_paths") != []
        ):
            raise RuntimeError(f"pixel preflight {variant} source/treatment mismatch")
        runtime = variant_result.get("runtime", {})
        if (
            runtime.get("pynvvideocodec_distribution") != "2.0.4"
            or runtime.get("transformers_distribution") != "5.14.1"
            or runtime.get("transformers_init_sha256") != TRANSFORMERS_INIT_SHA256
            or Path(str(runtime.get("vllm_origin", ""))).resolve()
            != (source_root / "vllm/__init__.py").resolve()
            or Path(str(runtime.get("transformers_origin", ""))).resolve()
            != (transformers_root / "transformers/__init__.py").resolve()
            or runtime.get("pynvvideocodec_runtime_artifacts")
            != PYNV_RUNTIME_ARTIFACT_SHA256
            or runtime.get("gpu") != expected_gpu_name
        ):
            raise RuntimeError(f"pixel preflight {variant} runtime mismatch")
        runtime_fingerprints[variant] = {
            key: runtime.get(key)
            for key in ("python", "numpy", "torch", "torch_cuda", "gpu")
        }
        worker_path = result_path.with_name(
            f"{result_path.stem}-{variant}-worker.json"
        ).resolve()
        if worker_path.parent != result_path.parent or not worker_path.is_file():
            raise RuntimeError(
                f"pixel preflight worker artifact missing: {worker_path}"
            )
        if json.loads(worker_path.read_text()) != {
            key: value
            for key, value in variant_result.items()
            if key not in {"source", "serve_help"}
        }:
            raise RuntimeError(f"pixel preflight worker record changed: {worker_path}")
        tensor_binding = variant_result.get("processor", {}).get("tensor_artifact", {})
        tensor_path = Path(str(tensor_binding.get("path", ""))).resolve()
        if (
            tensor_path.parent != result_path.parent
            or tensor_path != worker_path.with_suffix(".tensors.pt")
            or not tensor_path.is_file()
            or tensor_binding.get("sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError(
                f"pixel preflight tensor artifact changed: {tensor_path}"
            )
        tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
        expected_tensor_keys = {
            "raw_processor_pixels",
            "model_visible_pixels",
            "video_grid_thw",
            "output_prompt_token_ids",
            "placeholder_is_embed",
        }
        if not isinstance(tensors, dict) or set(tensors) != expected_tensor_keys:
            raise RuntimeError(f"pixel tensor artifact keys changed: {tensor_path}")
        loaded_tensors[variant] = tensors
        worker_artifacts.append(
            {
                "variant": variant,
                "worker_result": {
                    "path": str(worker_path),
                    "bytes": worker_path.stat().st_size,
                    "sha256": sha256_file(worker_path),
                },
                "tensor_artifact": {
                    "path": str(tensor_path),
                    "bytes": tensor_path.stat().st_size,
                    "sha256": sha256_file(tensor_path),
                },
            }
        )
    if len({sha256_json(value) for value in runtime_fingerprints.values()}) != 1:
        raise RuntimeError("pixel preflight runtime/HW fingerprints differ by endpoint")
    parity_keys = {
        "canonical_thwc_sha256_exact_all_variants",
        "canonical_thwc_shape_exact_all_variants",
        "sampled_frame_indices_exact_all_variants",
        "source_frame_count_exact_all_variants",
        "processor_raw_resized_pixels_exact_all_variants",
        "processor_video_grid_thw_exact_all_variants",
        "processor_output_prompt_token_ids_exact_all_variants",
        "processor_placeholder_metadata_exact_all_variants",
        "processor_resolution_exact_1024x576_all_variants",
        "processor_pixel_budget_exact_all_variants",
        "model_visible_bfloat16_allclose_all_variants",
    }
    if set(parity) != parity_keys or any(
        parity.get(key) is not True for key in parity_keys
    ):
        raise RuntimeError("pixel preflight parity dimensions changed")

    def tensor_sha256(tensor: Any) -> str:
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    reference_variant = "upstream"
    reference_tensors = loaded_tensors[reference_variant]
    reference_model_visible = reference_tensors["model_visible_pixels"]
    reference_variant_result = variants[reference_variant]
    recomputed_pairwise = {}
    for variant, tensors in loaded_tensors.items():
        variant_result = variants[variant]
        processor = variant_result["processor"]
        signature_fields = {
            "raw_processor_pixels": "raw_processor_pixel_values",
            "model_visible_pixels": "model_visible_pixel_values",
        }
        for tensor_key, processor_key in signature_fields.items():
            tensor = tensors[tensor_key]
            expected_signature = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "contiguous": tensor.is_contiguous(),
                "sha256": tensor_sha256(tensor),
            }
            if processor.get(processor_key) != expected_signature:
                raise RuntimeError(
                    f"pixel processor tensor signature changed: {variant}/{tensor_key}"
                )
        grid = tensors["video_grid_thw"]
        if grid.tolist() != [[16, 36, 64]] or (
            processor.get("video_grid_thw") != grid.tolist()
            or processor.get("video_grid_thw_sha256") != tensor_sha256(grid)
            or processor.get("output_prompt_token_ids_sha256")
            != tensor_sha256(tensors["output_prompt_token_ids"])
            or processor.get("placeholder", {}).get("is_embed_sha256")
            != tensor_sha256(tensors["placeholder_is_embed"])
        ):
            raise RuntimeError(
                f"pixel processor metadata/tensor binding changed: {variant}"
            )
        candidate = tensors["model_visible_pixels"]
        if (
            candidate.dtype != torch.bfloat16
            or candidate.shape != reference_model_visible.shape
        ):
            raise RuntimeError(f"pixel model-visible dtype/shape changed: {variant}")
        if (
            processor.get("processed_width") != PIXEL_BUDGET[0]
            or processor.get("processed_height") != PIXEL_BUDGET[1]
            or processor.get("configured_max_pixels_per_frame")
            != PIXEL_BUDGET[0] * PIXEL_BUDGET[1]
            or processor.get("configured_max_pixels_total") != TOTAL_MAX_PIXELS
            or variant_result.get("canonical_thwc", {}).get("shape")
            != reference_variant_result.get("canonical_thwc", {}).get("shape")
            or variant_result.get("canonical_thwc", {}).get("sha256")
            != reference_variant_result.get("canonical_thwc", {}).get("sha256")
            or variant_result.get("metadata", {}).get("frames_indices")
            != reference_variant_result.get("metadata", {}).get("frames_indices")
            or variant_result.get("metadata", {}).get("total_num_frames")
            != reference_variant_result.get("metadata", {}).get("total_num_frames")
            or not torch.equal(
                reference_tensors["raw_processor_pixels"],
                tensors["raw_processor_pixels"],
            )
            or not torch.equal(
                reference_tensors["video_grid_thw"], tensors["video_grid_thw"]
            )
            or not torch.equal(
                reference_tensors["output_prompt_token_ids"],
                tensors["output_prompt_token_ids"],
            )
            or not torch.equal(
                reference_tensors["placeholder_is_embed"],
                tensors["placeholder_is_embed"],
            )
        ):
            raise RuntimeError(
                f"pixel raw/processor/canonical parity changed: {variant}"
            )
        if variant == reference_variant:
            continue
        difference = (reference_model_visible.float() - candidate.float()).abs()
        recomputed_pairwise[variant] = {
            "candidate_variant": variant,
            "candidate_dtype": str(candidate.dtype),
            "exact": torch.equal(reference_model_visible, candidate),
            "mismatch_count": int(
                torch.count_nonzero(reference_model_visible != candidate)
            ),
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "allclose": torch.allclose(
                reference_model_visible, candidate, rtol=0.0, atol=2**-15
            ),
        }
    expected_model_visible = {
        "reference_variant": reference_variant,
        "reference_dtype": str(reference_model_visible.dtype),
        "variant_dtypes": {
            variant: str(tensors["model_visible_pixels"].dtype)
            for variant, tensors in loaded_tensors.items()
        },
        "shape": list(reference_model_visible.shape),
        "exact_all_variants": all(
            comparison["exact"] for comparison in recomputed_pairwise.values()
        ),
        "allclose": all(
            comparison["allclose"] for comparison in recomputed_pairwise.values()
        ),
        "pairwise_to_reference": recomputed_pairwise,
        "rtol": 0.0,
        "atol": 2**-15,
    }
    if model_visible != expected_model_visible:
        raise RuntimeError("pixel model-visible comparison evidence changed")
    return {
        "status": "passed",
        "result": {
            "path": str(result_path),
            "bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
        },
        "worker_artifacts": worker_artifacts,
        "runtime_fingerprint": next(iter(runtime_fingerprints.values())),
        "runtime_fingerprint_sha256": sha256_json(
            next(iter(runtime_fingerprints.values()))
        ),
    }


def validate_source(
    root: Path, expected_commit: str, *, variant: str
) -> dict[str, Any]:
    actual_commit = output(["git", "-C", str(root), "rev-parse", "HEAD^{commit}"])
    if actual_commit != expected_commit:
        raise RuntimeError(f"checkout mismatch: {actual_commit} != {expected_commit}")
    actual_tree = output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"])
    if actual_tree != TREES[variant]:
        raise RuntimeError(f"source tree mismatch: {actual_tree} != {TREES[variant]}")
    status = output(
        [
            "git",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    if status:
        raise RuntimeError(f"source is dirty:\n{status}")
    bytecode_paths = sorted(
        (
            path.relative_to(root).as_posix()
            for path in (root / "vllm").rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        )
    )
    if bytecode_paths:
        raise RuntimeError(
            "source contains ignored Python bytecode/cache artifacts: "
            + ", ".join(bytecode_paths[:64])
        )
    source_harness = root / "benchmarks/multimodal/benchmark_pynvvideocodec_e2e.py"
    source_harness_exists = source_harness.is_file()
    if source_harness_exists:
        raise RuntimeError(
            f"source unexpectedly contains experiment harness: {source_harness}"
        )
    protocol_bindings = {
        "vllm/entrypoints/openai/chat_completion/protocol.py": (
            CHAT_COMPLETION_PROTOCOL_SHA256
        ),
        "vllm/entrypoints/openai/chat_completion/serving.py": (
            CHAT_COMPLETION_SERVING_SHA256
        ),
    }
    for relative_path, expected_sha256 in protocol_bindings.items():
        if sha256_file(root / relative_path) != expected_sha256:
            raise RuntimeError(f"chat protocol artifact changed: {relative_path}")
    return {
        "commit": actual_commit,
        "tree": actual_tree,
        "status": status,
        "source_harness_exists": source_harness_exists,
        "chat_completion_protocol_artifacts": protocol_bindings,
        "ignored_python_bytecode_scan": {
            "scope": "vllm/**",
            "excluded_shared_venv": ".venv",
            "matched_paths": bytecode_paths,
            "passed": True,
        },
    }


def validate_source_at_any_endpoint(root: Path) -> dict[str, Any]:
    actual_commit = output(["git", "-C", str(root), "rev-parse", "HEAD^{commit}"])
    matching_variants = [
        variant for variant, commit in COMMITS.items() if commit == actual_commit
    ]
    if len(matching_variants) != 1:
        raise RuntimeError(f"source is not at a campaign endpoint: {actual_commit}")
    variant = matching_variants[0]
    return {"variant": variant, **validate_source(root, actual_commit, variant=variant)}


def validate_recorded_source(
    record: Mapping[str, Any], *, expected_variant: str | None = None
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RuntimeError("recorded source evidence is missing")
    variant = expected_variant or record.get("variant")
    if variant not in COMMITS:
        raise RuntimeError(f"recorded source endpoint is invalid: {variant!r}")
    if "variant" in record and record.get("variant") != variant:
        raise RuntimeError("recorded source endpoint label mismatch")
    bytecode_scan = record.get("ignored_python_bytecode_scan")
    if (
        record.get("commit") != COMMITS[variant]
        or record.get("tree") != TREES[variant]
        or record.get("status") != ""
        or record.get("source_harness_exists") is not False
        or record.get("chat_completion_protocol_artifacts")
        != {
            "vllm/entrypoints/openai/chat_completion/protocol.py": (
                CHAT_COMPLETION_PROTOCOL_SHA256
            ),
            "vllm/entrypoints/openai/chat_completion/serving.py": (
                CHAT_COMPLETION_SERVING_SHA256
            ),
        }
        or not isinstance(bytecode_scan, Mapping)
        or bytecode_scan.get("scope") != "vllm/**"
        or bytecode_scan.get("excluded_shared_venv") != ".venv"
        or bytecode_scan.get("matched_paths") != []
        or bytecode_scan.get("passed") is not True
    ):
        raise RuntimeError(f"recorded source evidence changed for {variant}")
    return {"variant": variant, **dict(record)}


def parse_server_log(log: str) -> dict[str, Any]:
    kv_matches = [
        {
            "tokens": int(tokens.replace(",", "")),
            "per_request_tokens": int(per_request.replace(",", "")),
            "maximum_concurrency": float(maximum_concurrency),
            "line": line,
        }
        for line in log.splitlines()
        if (
            match := re.search(
                r"GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for "
                r"([0-9,]+) tokens per request: ([0-9.]+)x",
                line,
            )
        )
        for tokens, per_request, maximum_concurrency in [match.groups()]
    ]
    preemption_lines = [
        line
        for line in log.splitlines()
        if re.search(r"preempt(?:ion|ed|ing)", line, re.IGNORECASE)
    ]
    oom_lines = [
        line
        for line in log.splitlines()
        if re.search(
            r"CUDA out of memory|torch\.OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY",
            line,
            re.IGNORECASE,
        )
    ]
    return {
        "gpu_kv_capacity": kv_matches,
        "preemption_line_count": len(preemption_lines),
        "preemption_lines": preemption_lines,
        "oom_line_count": len(oom_lines),
        "oom_lines": oom_lines,
    }


def measured_window_vram(
    monitor: Mapping[str, Any], block: Mapping[str, Any]
) -> dict[str, Any]:
    measured = block["measured"]
    started_ns, finished_ns = batch_monotonic_window(measured)
    samples = [
        sample
        for sample in monitor.get("samples", [])
        if isinstance(sample, Mapping)
        and isinstance(sample.get("monotonic_ns"), int)
        and started_ns <= sample["monotonic_ns"] <= finished_ns
    ]
    if not samples:
        raise RuntimeError(
            f"no GPU monitor samples overlap c{block['concurrency']} measured window"
        )
    peak_compute_process_memory_mib_by_name: dict[str, int] = {}
    peak_mps_client_memory_mib_by_name: dict[str, int] = {}
    peak_non_mps_compute_memory_mib = 0
    peak_mps_client_memory_mib = 0
    for sample in samples:
        non_mps_memory_mib = 0
        for app in sample.get("compute_apps", []):
            name = str(app["process_name"])
            used_memory_mib = int(app["used_memory_mib"])
            peak_compute_process_memory_mib_by_name[name] = max(
                peak_compute_process_memory_mib_by_name.get(name, 0),
                used_memory_mib,
            )
            if Path(name).name != "nvidia-cuda-mps-server":
                non_mps_memory_mib += used_memory_mib
        mps_client_memory_mib = 0
        for app in sample.get("mps_compute_apps", []):
            name = str(app["process_name"])
            used_memory_mib = int(app["used_memory_mib"])
            if used_memory_mib < 0:
                continue
            peak_mps_client_memory_mib_by_name[name] = max(
                peak_mps_client_memory_mib_by_name.get(name, 0),
                used_memory_mib,
            )
            mps_client_memory_mib += used_memory_mib
        peak_non_mps_compute_memory_mib = max(
            peak_non_mps_compute_memory_mib, non_mps_memory_mib
        )
        peak_mps_client_memory_mib = max(
            peak_mps_client_memory_mib, mps_client_memory_mib
        )
    return {
        "measured_started_at": measured["started_at"],
        "measured_finished_at": measured["finished_at"],
        "measured_started_monotonic_ns": started_ns,
        "measured_finished_monotonic_ns": finished_ns,
        "clock_policy": "host monotonic clock only",
        "sample_count": len(samples),
        "first_sample_utc": samples[0]["utc"],
        "last_sample_utc": samples[-1]["utc"],
        "peak_total_gpu_memory_used_mib": max(
            int(sample["memory_used_mib"]) for sample in samples
        ),
        "peak_gpu_utilization_percent": max(
            int(sample["utilization_percent"]) for sample in samples
        ),
        "peak_non_mps_compute_process_memory_mib": (peak_non_mps_compute_memory_mib),
        "peak_mps_client_memory_mib": peak_mps_client_memory_mib,
        "peak_compute_process_memory_mib_by_name": (
            peak_compute_process_memory_mib_by_name
        ),
        "peak_mps_client_memory_mib_by_name": peak_mps_client_memory_mib_by_name,
    }


def batch_monotonic_window(batch: Mapping[str, Any]) -> tuple[int, int]:
    started_ns = batch.get("started_monotonic_ns")
    finished_ns = batch.get("finished_monotonic_ns")
    if (
        not isinstance(started_ns, int)
        or not isinstance(finished_ns, int)
        or finished_ns <= started_ns
    ):
        raise RuntimeError(f"invalid absolute monotonic batch window: {batch}")
    duration_seconds = (finished_ns - started_ns) / 1e9
    for reported_duration in (
        batch.get("measured_window_seconds"),
        batch.get("aggregate", {}).get("measured_window_seconds"),
    ):
        if (
            isinstance(reported_duration, bool)
            or not isinstance(reported_duration, (int, float))
            or float(reported_duration) != duration_seconds
        ):
            raise RuntimeError(
                "batch duration does not match its absolute monotonic boundaries"
            )
    return started_ns, finished_ns


def linear_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def require_numeric_match(
    actual: Any, expected: float | int | None, label: str
) -> None:
    if expected is None:
        if actual is not None:
            raise RuntimeError(f"{label} changed: {actual!r} != None")
        return
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or type(actual) is not type(expected)
        or actual != expected
    ):
        raise RuntimeError(f"{label} changed: {actual!r} != {expected!r}")


def validate_persistent_transport_audit(
    batch: Mapping[str, Any],
    *,
    phase: str,
    concurrency: int,
    prior_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = batch.get("records")
    aggregate = batch.get("aggregate")
    if not isinstance(records, list) or not isinstance(aggregate, Mapping):
        raise RuntimeError("persistent transport batch evidence is missing")
    current_transports: list[Mapping[str, Any]] = []
    for request_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("status") != "passed":
            raise RuntimeError("persistent transport record is not passed")
        transport = record.get("transport")
        if not isinstance(transport, Mapping):
            raise RuntimeError("persistent transport metadata is missing")
        expected_transport_keys = {
            "pool_slot_id",
            "phase",
            "seeded_first_wave",
            "connection_generation",
            "request_ordinal_on_generation",
            "connection_reused",
            "prewarmed_for_measurement",
            "request_connection_header",
            "response_http_version",
            "response_connection_header",
            "response_will_close",
            "response_persistent",
        }
        if set(transport) != expected_transport_keys:
            raise RuntimeError("persistent transport metadata schema changed")
        slot = transport.get("pool_slot_id")
        if (
            not isinstance(slot, int)
            or not 0 <= slot < concurrency
            or transport.get("phase") != phase
            or transport.get("seeded_first_wave") is not (request_index < concurrency)
            or (request_index < concurrency and slot != request_index)
            or transport.get("connection_generation") != 1
            or not isinstance(transport.get("request_ordinal_on_generation"), int)
            or int(transport["request_ordinal_on_generation"]) <= 0
            or transport.get("request_connection_header") != "keep-alive"
            or transport.get("response_http_version") != 11
            or transport.get("response_connection_header")
            not in {None, "keep-alive", "Keep-Alive"}
            or transport.get("response_will_close") is not False
            or transport.get("response_persistent") is not True
        ):
            raise RuntimeError("persistent transport request evidence changed")
        if phase == "warmup":
            expected_reused = int(transport["request_ordinal_on_generation"]) > 1
            if (
                transport.get("connection_reused") is not expected_reused
                or transport.get("prewarmed_for_measurement") is not False
            ):
                raise RuntimeError("persistent warmup connection evidence changed")
        elif phase == "measured":
            if (
                transport.get("connection_reused") is not True
                or transport.get("prewarmed_for_measurement") is not True
            ):
                raise RuntimeError("persistent measured connection evidence changed")
        else:
            raise RuntimeError(f"invalid persistent transport phase: {phase}")
        current_transports.append(transport)

    all_records = [*prior_records, *records]
    transports_by_slot: dict[int, list[Mapping[str, Any]]] = {
        slot: [] for slot in range(concurrency)
    }
    for record in all_records:
        if not isinstance(record, Mapping) or record.get("status") != "passed":
            raise RuntimeError("persistent transport history is not passed")
        transport = record.get("transport")
        if not isinstance(transport, Mapping):
            raise RuntimeError("persistent transport history is incomplete")
        slot = transport.get("pool_slot_id")
        if not isinstance(slot, int) or slot not in transports_by_slot:
            raise RuntimeError("persistent transport history slot changed")
        transports_by_slot[slot].append(transport)
    expected_snapshots = []
    for slot, transports in transports_by_slot.items():
        ordinals = sorted(
            int(transport["request_ordinal_on_generation"]) for transport in transports
        )
        if ordinals != list(range(1, len(transports) + 1)):
            raise RuntimeError("persistent connection ordinals are not contiguous")
        expected_snapshots.append(
            {
                "slot_id": slot,
                "current_generation": 1,
                "warmed_generation": 1,
                "request_ordinal_on_current_generation": len(transports),
                "open_count": 1,
                "reuse_count": len(transports) - 1,
                "close_count": 0,
                "close_reasons": {},
                "currently_open": True,
            }
        )
    expected_audit = {
        "status": "passed",
        "phase": phase,
        "pool_size": concurrency,
        "request_count": len(records),
        "used_slot_ids": list(range(concurrency)),
        "seeded_first_wave_request_to_slot": {
            str(slot): slot for slot in range(concurrency)
        },
        "reasons": [],
        "counts_at_phase_end": {
            "open_count": concurrency,
            "reuse_count": len(all_records) - concurrency,
            "close_count": 0,
        },
        "slot_snapshots_at_phase_end": expected_snapshots,
    }
    if aggregate.get("persistent_transport_audit") != expected_audit:
        raise RuntimeError("persistent transport aggregate audit changed")
    return {
        "status": "passed",
        "phase": phase,
        "record_count": len(records),
        "history_record_count": len(all_records),
        "evidence_sha256": sha256_json(expected_audit),
    }


def validate_result_integrity(result: Mapping[str, Any]) -> dict[str, Any]:
    videos = result.get("videos")
    if not isinstance(videos, list) or len(videos) != 8:
        raise RuntimeError("result lacks exactly eight video provenance records")
    videos_by_index: dict[int, Mapping[str, Any]] = {}
    for expected_video_index, video in enumerate(videos):
        if (
            not isinstance(video, Mapping)
            or video.get("video_index") != expected_video_index
            or not isinstance(video.get("path"), str)
            or not isinstance(video.get("file_uri"), str)
            or not isinstance(video.get("sha256"), str)
        ):
            raise RuntimeError("video provenance identity/order changed")
        videos_by_index[expected_video_index] = video
    payload_records = result.get("request_payloads_by_video", [])
    if not isinstance(payload_records, list) or len(payload_records) != 8:
        raise RuntimeError("result lacks exactly eight request payload records")
    payload_hash_by_video: dict[int, str] = {}
    for expected_video_index, payload_record in enumerate(payload_records):
        if (
            not isinstance(payload_record, Mapping)
            or payload_record.get("video_index") != expected_video_index
        ):
            raise RuntimeError("request payload video ordering changed")
        video = videos_by_index[expected_video_index]
        expected_payload = expected_chat_payload(str(video["file_uri"]))
        if (
            payload_record.get("video_path") != video["path"]
            or payload_record.get("payload") != expected_payload
        ):
            raise RuntimeError("request payload semantics/video binding changed")
        recomputed_payload_sha256 = sha256_json(payload_record.get("payload"))
        if payload_record.get("payload_sha256") != recomputed_payload_sha256:
            raise RuntimeError("request payload hash changed")
        payload_hash_by_video[expected_video_index] = recomputed_payload_sha256

    audited_response_count = 0
    audited_batch_count = 0
    expected_global_request_index = 0
    for block_index, block in enumerate(result.get("concurrency_blocks", [])):
        concurrency = int(block["concurrency"])
        warmup_records_for_transport: Sequence[Mapping[str, Any]] = []
        for phase in ("warmup", "measured"):
            batch = block[phase]
            started_ns, finished_ns = batch_monotonic_window(batch)
            elapsed_seconds = (finished_ns - started_ns) / 1e9
            records = batch.get("records")
            if not isinstance(records, list):
                raise RuntimeError("batch records are not a list")
            passed = []
            failed = []
            ordered_fingerprints = []
            for request_index, record in enumerate(records):
                if (
                    not isinstance(record, Mapping)
                    or record.get("phase") != phase
                    or record.get("block_index") != block_index
                    or record.get("concurrency") != concurrency
                    or record.get("request_index") != request_index
                    or record.get("global_request_index")
                    != expected_global_request_index
                ):
                    raise RuntimeError("request identity/order changed")
                expected_global_request_index += 1
                video_index = record.get("video_index")
                video = videos_by_index.get(video_index)
                if (
                    not isinstance(video_index, int)
                    or video_index != request_index % len(videos_by_index)
                    or video is None
                    or record.get("video_path") != video["path"]
                    or record.get("video_file_uri") != video["file_uri"]
                    or record.get("video_sha256") != video["sha256"]
                    or record.get("request_payload_sha256")
                    != payload_hash_by_video.get(video_index)
                    or record.get("payload") != payload_records[video_index]["payload"]
                ):
                    raise RuntimeError("request payload/video binding changed")
                probe = video.get("probe")
                if not isinstance(probe, Mapping):
                    raise RuntimeError("video probe provenance is missing")
                sampled_frames = min(FRAMES, int(probe["frame_count"]))
                expected_video_work = {
                    "source_width": int(probe["width"]),
                    "source_height": int(probe["height"]),
                    "source_frame_count": int(probe["frame_count"]),
                    "sampled_frames": sampled_frames,
                    "sampled_source_megapixels_estimate": (
                        sampled_frames
                        * int(probe["width"])
                        * int(probe["height"])
                        / 1_000_000
                    ),
                    "derivation": "Qwen3-VL equal min/max frame clamp",
                }
                if record.get("video_work") != expected_video_work:
                    raise RuntimeError("request video-work derivation changed")
                record_started_ns = record.get("started_monotonic_ns")
                record_finished_ns = record.get("finished_monotonic_ns")
                if (
                    not isinstance(record_started_ns, int)
                    or not isinstance(record_finished_ns, int)
                    or not started_ns <= record_started_ns < record_finished_ns
                    or record_finished_ns > finished_ns
                ):
                    raise RuntimeError("request absolute monotonic timing changed")
                expected_start_offset = (record_started_ns - started_ns) / 1e9
                expected_finish_offset = (record_finished_ns - started_ns) / 1e9
                expected_latency_seconds = (
                    record_finished_ns - record_started_ns
                ) / 1e9
                require_numeric_match(
                    record.get("start_offset_seconds"),
                    expected_start_offset,
                    "request start offset",
                )
                require_numeric_match(
                    record.get("finish_offset_seconds"),
                    expected_finish_offset,
                    "request finish offset",
                )
                require_numeric_match(
                    record.get("latency_seconds"),
                    expected_latency_seconds,
                    "request latency seconds",
                )
                require_numeric_match(
                    record.get("latency_ms"),
                    expected_latency_seconds * 1000.0,
                    "request latency",
                )
                if record.get("status") != "passed":
                    failed.append(record)
                    continue
                response = record.get("response")
                if not isinstance(response, Mapping):
                    raise RuntimeError("passing request lacks a response")
                prompt_ids = response.get("prompt_token_ids")
                completion_ids = response.get("completion_token_ids")
                if (
                    not isinstance(prompt_ids, list)
                    or not all(isinstance(token, int) for token in prompt_ids)
                    or not isinstance(completion_ids, list)
                    or not all(isinstance(token, int) for token in completion_ids)
                ):
                    raise RuntimeError("response token arrays are invalid")
                expected_hashes = {
                    "prompt_token_ids_sha256": sha256_json(prompt_ids),
                    "completion_token_ids_sha256": sha256_json(completion_ids),
                    "prompt_and_completion_token_ids_sha256": sha256_json(
                        {"prompt": prompt_ids, "completion": completion_ids}
                    ),
                    "text_sha256": sha256_json(response.get("text")),
                    "reasoning_content_sha256": sha256_json(
                        response.get("reasoning_content")
                    ),
                }
                if any(
                    response.get(key) != value for key, value in expected_hashes.items()
                ):
                    raise RuntimeError("response token/text hash changed")
                if response.get("prompt_token_count") != len(
                    prompt_ids
                ) or response.get("completion_token_count") != len(completion_ids):
                    raise RuntimeError("response token count changed")
                if not isinstance(response.get("text"), str) or not isinstance(
                    response.get("reasoning_content"), (str, type(None))
                ):
                    raise RuntimeError("response text/reasoning types changed")
                raw_response = response.get("raw_response")
                if not isinstance(raw_response, Mapping) or response.get(
                    "raw_response_sha256"
                ) != sha256_json(raw_response):
                    raise RuntimeError("raw HTTP response hash changed")
                raw_choices = raw_response.get("choices")
                raw_choice = (
                    raw_choices[0]
                    if isinstance(raw_choices, list) and len(raw_choices) == 1
                    else None
                )
                raw_message = (
                    raw_choice.get("message")
                    if isinstance(raw_choice, Mapping)
                    else None
                )
                raw_usage = raw_response.get("usage")
                if (
                    not isinstance(raw_choice, Mapping)
                    or not isinstance(raw_message, Mapping)
                    or not isinstance(raw_usage, Mapping)
                    or raw_response.get("prompt_token_ids") != prompt_ids
                    or raw_choice.get("token_ids") != completion_ids
                    or raw_message.get("content") != response.get("text")
                    or raw_message.get("reasoning") != response.get("reasoning_content")
                    or raw_choice.get("finish_reason") != response.get("finish_reason")
                    or raw_choice.get("stop_reason") != response.get("stop_reason")
                    or raw_response.get("id") != response.get("id")
                    or raw_response.get("model") != SERVED_MODEL_NAME
                    or response.get("model") != SERVED_MODEL_NAME
                    or raw_response.get("metrics") != response.get("server_metrics")
                    or raw_usage != response.get("usage")
                    or raw_usage.get("prompt_tokens") != len(prompt_ids)
                    or raw_usage.get("completion_tokens") != len(completion_ids)
                    or raw_usage.get("total_tokens")
                    != len(prompt_ids) + len(completion_ids)
                ):
                    raise RuntimeError("raw/extracted HTTP response binding changed")
                passed.append(record)
                audited_response_count += 1
                ordered_fingerprints.append(
                    {
                        "request_index": request_index,
                        "video_index": video_index,
                        "video_path": record.get("video_path"),
                        "prompt_token_ids_sha256": expected_hashes[
                            "prompt_token_ids_sha256"
                        ],
                        "completion_token_ids_sha256": expected_hashes[
                            "completion_token_ids_sha256"
                        ],
                        "prompt_and_completion_token_ids_sha256": expected_hashes[
                            "prompt_and_completion_token_ids_sha256"
                        ],
                    }
                )
            aggregate = batch.get("aggregate")
            if not isinstance(aggregate, Mapping):
                raise RuntimeError("batch aggregate is missing")
            prompt_tokens = sum(
                len(record["response"]["prompt_token_ids"]) for record in passed
            )
            completion_tokens = sum(
                len(record["response"]["completion_token_ids"]) for record in passed
            )
            sampled_source_megapixel_estimates = [
                float(record["video_work"]["sampled_source_megapixels_estimate"])
                for record in passed
                if isinstance(
                    record["video_work"].get("sampled_source_megapixels_estimate"),
                    (int, float),
                )
            ]
            video_work_missing = [
                {
                    "request_index": record["request_index"],
                    "video_index": record["video_index"],
                    "video_path": record["video_path"],
                    "reason": record["video_work"].get("derivation"),
                }
                for record in passed
                if not isinstance(
                    record["video_work"].get("sampled_source_megapixels_estimate"),
                    (int, float),
                )
            ]
            in_flight_events = [
                (float(record["start_offset_seconds"]), 1) for record in records
            ] + [(float(record["finish_offset_seconds"]), -1) for record in records]
            in_flight = 0
            peak_in_flight = 0
            for unused_offset, delta in sorted(
                in_flight_events, key=lambda event: (event[0], event[1])
            ):
                del unused_offset
                in_flight += delta
                peak_in_flight = max(peak_in_flight, in_flight)
            recomputed_failures = [
                {
                    "request_index": record.get("request_index"),
                    "global_request_index": record.get("global_request_index"),
                    "video_index": record.get("video_index"),
                    "video_path": record.get("video_path"),
                    "error": record.get("error"),
                }
                for record in failed
            ]
            exact_fields = {
                "status": "passed" if not failed else "failed",
                "attempted_requests": len(records),
                "successful_requests": len(passed),
                "failed_requests": len(failed),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": completion_tokens,
                "all_tokens": prompt_tokens + completion_tokens,
                "achieved_peak_in_flight_requests": peak_in_flight,
                "response_token_fingerprints_by_request": ordered_fingerprints,
                "ordered_response_token_fingerprints_sha256": sha256_json(
                    ordered_fingerprints
                ),
                "ordered_response_token_ids_sha256": sha256_json(
                    [
                        {
                            "prompt": item["prompt_token_ids_sha256"],
                            "completion": item["completion_token_ids_sha256"],
                        }
                        for item in ordered_fingerprints
                    ]
                ),
                "completion_token_ids_sha256_counts": dict(
                    sorted(
                        Counter(
                            item["completion_token_ids_sha256"]
                            for item in ordered_fingerprints
                        ).items()
                    )
                ),
                "video_megapixel_estimate_method": (
                    "sum(estimated Qwen3-VL sampled frames * externally probed "
                    "encoded source width * encoded source height) / measured client "
                    "wall time; this is not a count of frames actually decoded by "
                    "the codec"
                ),
                "video_megapixel_estimate_unavailable": video_work_missing,
                "failures": recomputed_failures,
            }
            if any(aggregate.get(key) != value for key, value in exact_fields.items()):
                raise RuntimeError("batch aggregate token/count/hash fields changed")
            numeric_fields = {
                "measured_window_seconds": elapsed_seconds,
                "attempted_request_throughput_per_second": (
                    len(records) / elapsed_seconds
                ),
                "request_throughput_per_second": len(passed) / elapsed_seconds,
                "prompt_token_throughput_per_second": prompt_tokens / elapsed_seconds,
                "generated_token_throughput_per_second": (
                    completion_tokens / elapsed_seconds
                ),
                "all_token_throughput_per_second": (
                    (prompt_tokens + completion_tokens) / elapsed_seconds
                ),
                "achieved_mean_in_flight_requests": (
                    sum(float(record["latency_seconds"]) for record in passed)
                    / elapsed_seconds
                ),
                "sampled_source_megapixels_estimate": (
                    sum(sampled_source_megapixel_estimates)
                    if passed and not video_work_missing
                    else None
                ),
                "sampled_source_megapixels_estimate_per_second": (
                    sum(sampled_source_megapixel_estimates) / elapsed_seconds
                    if passed and not video_work_missing
                    else None
                ),
            }
            expected_aggregate_keys = {
                *exact_fields,
                *numeric_fields,
                "latency_ms",
                "persistent_transport_audit",
            }
            if set(aggregate) != expected_aggregate_keys:
                raise RuntimeError(
                    "batch aggregate schema changed: "
                    f"{sorted(set(aggregate) ^ expected_aggregate_keys)}"
                )
            for key, expected in numeric_fields.items():
                require_numeric_match(aggregate.get(key), expected, f"aggregate {key}")
            latency_values = [float(record["latency_ms"]) for record in passed]
            latency = aggregate.get("latency_ms")
            if not isinstance(latency, Mapping) or not latency_values:
                raise RuntimeError("aggregate latency evidence is missing")
            expected_latency = {
                "count": len(latency_values),
                "min": min(latency_values),
                "mean": statistics.fmean(latency_values),
                "median": statistics.median(latency_values),
                "p50": linear_percentile(latency_values, 0.50),
                "p90": linear_percentile(latency_values, 0.90),
                "p95": linear_percentile(latency_values, 0.95),
                "p99": linear_percentile(latency_values, 0.99),
                "max": max(latency_values),
                "population_stdev": statistics.pstdev(latency_values),
            }
            for key, expected in expected_latency.items():
                require_numeric_match(latency.get(key), expected, f"latency {key}")
            if latency.get("percentile_method") != (
                "linear interpolation at (n - 1) * fraction"
            ):
                raise RuntimeError("latency percentile method changed")
            validate_persistent_transport_audit(
                batch,
                phase=phase,
                concurrency=concurrency,
                prior_records=warmup_records_for_transport,
            )
            if phase == "warmup":
                warmup_records_for_transport = records
            audited_batch_count += 1
        measured_aggregate = block.get("measured", {}).get("aggregate")
        if block.get("aggregate") != measured_aggregate:
            raise RuntimeError(
                f"c{concurrency} duplicate measured aggregate evidence changed"
            )
    return {
        "status": "passed",
        "payload_count": len(payload_records),
        "batch_count": audited_batch_count,
        "response_count": audited_response_count,
        "policy": "all payload, token/text, aggregate hash/count fields recomputed",
    }


def validate_full_server_log_binding(
    result: Mapping[str, Any], server_log_path: Path
) -> dict[str, Any]:
    server_log_record = result["server"]["log"]
    if (
        Path(server_log_record.get("path", "")).resolve() != server_log_path.resolve()
        or server_log_record.get("storage") != "append-only full server-log sidecar"
        or server_log_record.get("bytes") != server_log_path.stat().st_size
        or server_log_record.get("sha256") != sha256_file(server_log_path)
    ):
        raise RuntimeError(f"full server log binding mismatch: {server_log_record}")
    server_log_audit = parse_server_log(server_log_path.read_text(errors="replace"))
    if not server_log_audit["gpu_kv_capacity"]:
        raise RuntimeError("server log lacks GPU KV cache capacity provenance")
    expected_kv = {
        "tokens": 336560,
        "per_request_tokens": 32768,
        "maximum_concurrency": 10.27,
    }
    if not any(
        all(capacity[key] == value for key, value in expected_kv.items())
        for capacity in server_log_audit["gpu_kv_capacity"]
    ):
        raise RuntimeError(
            "server log lacks exact expected GPU KV capacity: "
            f"{expected_kv}; got {server_log_audit['gpu_kv_capacity']}"
        )
    if server_log_audit["oom_line_count"]:
        raise RuntimeError("server log contains an out-of-memory event")
    if server_log_audit["preemption_line_count"]:
        raise RuntimeError("server log contains a preemption event")
    return {
        "server_log_audit": server_log_audit,
        "full_server_log": {
            "path": str(server_log_path),
            "bytes": server_log_path.stat().st_size,
            "sha256": sha256_file(server_log_path),
            "full_file_scanned": True,
        },
    }


def validate_batch_token_evidence(
    batch: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    records = batch.get("records")
    if not isinstance(records, list):
        raise RuntimeError(f"{context} records are not a list")
    ordered_fingerprints: list[dict[str, Any]] = []
    completion_hash_counts: Counter[str] = Counter()
    prompt_token_total = 0
    completion_token_total = 0
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") != "passed":
            raise RuntimeError(f"{context} contains a non-passed token record")
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or record.get(
            "request_payload_sha256"
        ) != sha256_json(payload):
            raise RuntimeError(f"{context} request payload hash mismatch")
        response = record.get("response")
        if not isinstance(response, Mapping):
            raise RuntimeError(f"{context} response record is missing")
        prompt_ids = response.get("prompt_token_ids")
        completion_ids = response.get("completion_token_ids")
        if (
            not isinstance(prompt_ids, list)
            or not all(isinstance(token, int) for token in prompt_ids)
            or not isinstance(completion_ids, list)
            or not all(isinstance(token, int) for token in completion_ids)
        ):
            raise RuntimeError(f"{context} token arrays are invalid")
        expected_response_fields = {
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": sha256_json(prompt_ids),
            "completion_token_count": len(completion_ids),
            "completion_token_ids_sha256": sha256_json(completion_ids),
            "prompt_and_completion_token_ids_sha256": sha256_json(
                {"prompt": prompt_ids, "completion": completion_ids}
            ),
            "text_sha256": sha256_json(response.get("text")),
            "reasoning_content_sha256": sha256_json(response.get("reasoning_content")),
        }
        for field, expected in expected_response_fields.items():
            if response.get(field) != expected:
                raise RuntimeError(
                    f"{context} response {field} mismatch: "
                    f"{response.get(field)!r} != {expected!r}"
                )
        raw_response = response.get("raw_response")
        if not isinstance(raw_response, Mapping) or response.get(
            "raw_response_sha256"
        ) != sha256_json(raw_response):
            raise RuntimeError(f"{context} raw response hash mismatch")
        raw_choices = raw_response.get("choices")
        if not isinstance(raw_choices, list) or len(raw_choices) != 1:
            raise RuntimeError(f"{context} raw response choice mismatch")
        raw_choice = raw_choices[0]
        raw_message = (
            raw_choice.get("message") if isinstance(raw_choice, Mapping) else None
        )
        if (
            not isinstance(raw_message, Mapping)
            or raw_response.get("prompt_token_ids") != prompt_ids
            or raw_choice.get("token_ids") != completion_ids
            or raw_message.get("content") != response.get("text")
            or raw_message.get("reasoning") != response.get("reasoning_content")
            or raw_choice.get("finish_reason") != response.get("finish_reason")
            or raw_choice.get("stop_reason") != response.get("stop_reason")
            or raw_response.get("id") != response.get("id")
            or raw_response.get("model") != response.get("model")
        ):
            raise RuntimeError(f"{context} raw/normalized response mismatch")
        prompt_token_total += len(prompt_ids)
        completion_token_total += len(completion_ids)
        fingerprint = {
            "request_index": record["request_index"],
            "video_index": record["video_index"],
            "video_path": record["video_path"],
            "prompt_token_ids_sha256": expected_response_fields[
                "prompt_token_ids_sha256"
            ],
            "completion_token_ids_sha256": expected_response_fields[
                "completion_token_ids_sha256"
            ],
            "prompt_and_completion_token_ids_sha256": expected_response_fields[
                "prompt_and_completion_token_ids_sha256"
            ],
        }
        ordered_fingerprints.append(fingerprint)
        completion_hash_counts[fingerprint["completion_token_ids_sha256"]] += 1

    aggregate = batch.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError(f"{context} aggregate is missing")
    expected_aggregate_fields = {
        "prompt_tokens": prompt_token_total,
        "generated_tokens": completion_token_total,
        "all_tokens": prompt_token_total + completion_token_total,
        "response_token_fingerprints_by_request": ordered_fingerprints,
        "ordered_response_token_fingerprints_sha256": sha256_json(ordered_fingerprints),
        "ordered_response_token_ids_sha256": sha256_json(
            [
                {
                    "prompt": item["prompt_token_ids_sha256"],
                    "completion": item["completion_token_ids_sha256"],
                }
                for item in ordered_fingerprints
            ]
        ),
        "completion_token_ids_sha256_counts": dict(
            sorted(completion_hash_counts.items())
        ),
    }
    for field, expected in expected_aggregate_fields.items():
        if aggregate.get(field) != expected:
            raise RuntimeError(f"{context} aggregate {field} mismatch")
    return {
        "status": "passed",
        "record_count": len(records),
        "prompt_token_total": prompt_token_total,
        "completion_token_total": completion_token_total,
        "evidence_sha256": sha256_json(expected_aggregate_fields),
    }


PASSED_REQUEST_RECORD_FIELDS = frozenset(
    {
        "phase",
        "block_index",
        "concurrency",
        "request_index",
        "global_request_index",
        "video_index",
        "video_path",
        "video_file_uri",
        "video_sha256",
        "video_work",
        "request_payload_sha256",
        "payload",
        "status",
        "http_status",
        "transport",
        "response",
        "started_at",
        "finished_at",
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "start_offset_seconds",
        "finish_offset_seconds",
        "latency_seconds",
        "latency_ms",
    }
)
RAW_CHAT_RESPONSE_FIELDS = frozenset(
    {
        "id",
        "object",
        "created",
        "model",
        "choices",
        "service_tier",
        "system_fingerprint",
        "usage",
        "prompt_logprobs",
        "prompt_token_ids",
        "prompt_text",
        "kv_transfer_params",
        "ec_transfer_params",
        "metrics",
    }
)
RAW_CHAT_CHOICE_FIELDS = frozenset(
    {
        "index",
        "message",
        "logprobs",
        "finish_reason",
        "stop_reason",
        "token_ids",
        "routed_experts",
    }
)
RAW_CHAT_MESSAGE_FIELDS = frozenset(
    {
        "role",
        "content",
        "refusal",
        "annotations",
        "audio",
        "function_call",
        "reasoning",
    }
)
RAW_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "total_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
)
RAW_SERVER_METRICS_FIELDS = frozenset(
    {
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
        "speculative_decoding",
    }
)
NORMALIZED_RESPONSE_FIELDS = frozenset(
    {
        "id",
        "model",
        "prompt_token_count",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "completion_token_count",
        "completion_token_ids",
        "completion_token_ids_sha256",
        "prompt_and_completion_token_ids_sha256",
        "text",
        "text_sha256",
        "reasoning_content",
        "reasoning_content_sha256",
        "finish_reason",
        "stop_reason",
        "usage",
        "server_metrics",
        "raw_response_sha256",
        "raw_response",
    }
)
PERSISTENT_TRANSPORT_FIELDS = frozenset(
    {
        "pool_slot_id",
        "phase",
        "seeded_first_wave",
        "connection_generation",
        "request_ordinal_on_generation",
        "connection_reused",
        "prewarmed_for_measurement",
        "request_connection_header",
        "response_http_version",
        "response_connection_header",
        "response_will_close",
        "response_persistent",
    }
)
PERSISTENT_PHASE_AUDIT_FIELDS = frozenset(
    {
        "status",
        "phase",
        "pool_size",
        "request_count",
        "used_slot_ids",
        "seeded_first_wave_request_to_slot",
        "reasons",
        "counts_at_phase_end",
        "slot_snapshots_at_phase_end",
    }
)
PERSISTENT_SLOT_SNAPSHOT_FIELDS = frozenset(
    {
        "slot_id",
        "current_generation",
        "warmed_generation",
        "request_ordinal_on_current_generation",
        "open_count",
        "reuse_count",
        "close_count",
        "close_reasons",
        "currently_open",
    }
)
PERSISTENT_POOL_FIELDS = frozenset(
    {
        "implementation",
        "pool_size",
        "connection_scope",
        "phase_scope",
        "request_streaming",
        "request_retry_count",
        "counts",
        "closed",
        "slots",
        "phase_audits",
    }
)


def expected_chat_payload(video_file_uri: str) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_file_uri},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "max_completion_tokens": OUTPUT_LENGTH,
        "ignore_eos": True,
        "n": 1,
        "seed": 0,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_token_ids": True,
    }


def _expected_slot_snapshots(
    records: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    closed: bool,
) -> list[dict[str, Any]]:
    snapshots = []
    for slot_id in range(concurrency):
        transports = [
            record["transport"]
            for record in records
            if record["transport"]["pool_slot_id"] == slot_id
        ]
        ordinals = sorted(
            int(transport["request_ordinal_on_generation"]) for transport in transports
        )
        if ordinals != list(range(1, len(transports) + 1)):
            raise RuntimeError(
                f"persistent slot {slot_id} request ordinals are not unique/contiguous"
            )
        snapshots.append(
            {
                "slot_id": slot_id,
                "current_generation": 1,
                "warmed_generation": 1,
                "request_ordinal_on_current_generation": len(transports),
                "open_count": 1,
                "reuse_count": len(transports) - 1,
                "close_count": 1 if closed else 0,
                "close_reasons": {"pool_close": 1} if closed else {},
                "currently_open": not closed,
            }
        )
    return snapshots


def validate_persistent_transport_evidence(
    block: Mapping[str, Any],
    *,
    concurrency: int,
    phase_records: Mapping[str, Sequence[Mapping[str, Any]]],
    context: str,
) -> dict[str, Any]:
    if set(phase_records) != {"warmup", "measured"}:
        raise RuntimeError(f"{context} transport phases are incomplete")
    cumulative: list[Mapping[str, Any]] = []
    phase_audits: dict[str, Mapping[str, Any]] = {}
    for phase in ("warmup", "measured"):
        records = list(phase_records[phase])
        cumulative.extend(records)
        aggregate = block[phase]["aggregate"]
        audit = aggregate.get("persistent_transport_audit")
        if (
            not isinstance(audit, Mapping)
            or set(audit) != PERSISTENT_PHASE_AUDIT_FIELDS
        ):
            raise RuntimeError(f"{context} {phase} transport audit schema mismatch")
        transports = [record["transport"] for record in records]
        used_slots = sorted({transport["pool_slot_id"] for transport in transports})
        seeded_mapping = {
            str(record["request_index"]): record["transport"]["pool_slot_id"]
            for record in records
            if record["transport"]["seeded_first_wave"] is True
        }
        expected_snapshots = _expected_slot_snapshots(
            cumulative, concurrency=concurrency, closed=False
        )
        for snapshot in audit.get("slot_snapshots_at_phase_end", []):
            if not isinstance(snapshot, Mapping) or set(snapshot) != (
                PERSISTENT_SLOT_SNAPSHOT_FIELDS
            ):
                raise RuntimeError(
                    f"{context} {phase} transport slot snapshot schema mismatch"
                )
        expected_audit = {
            "status": "passed",
            "phase": phase,
            "pool_size": concurrency,
            "request_count": len(records),
            "used_slot_ids": used_slots,
            "seeded_first_wave_request_to_slot": seeded_mapping,
            "reasons": [],
            "counts_at_phase_end": {
                "open_count": concurrency,
                "reuse_count": len(cumulative) - concurrency,
                "close_count": 0,
            },
            "slot_snapshots_at_phase_end": expected_snapshots,
        }
        if audit != expected_audit:
            raise RuntimeError(
                f"{context} {phase} transport audit recomputation mismatch"
            )
        phase_audits[phase] = audit

    pool = block.get("persistent_http_pool")
    if not isinstance(pool, Mapping) or set(pool) != PERSISTENT_POOL_FIELDS:
        raise RuntimeError(f"{context} persistent pool schema mismatch")
    final_snapshots = _expected_slot_snapshots(
        cumulative, concurrency=concurrency, closed=True
    )
    for snapshot in pool.get("slots", []):
        if not isinstance(snapshot, Mapping) or set(snapshot) != (
            PERSISTENT_SLOT_SNAPSHOT_FIELDS
        ):
            raise RuntimeError(f"{context} final pool slot schema mismatch")
    expected_pool = {
        "implementation": "stdlib http.client.HTTPConnection HTTP/1.1",
        "pool_size": concurrency,
        "connection_scope": "one pool per concurrency block",
        "phase_scope": "same slots span warmup, settle, and measured phases",
        "request_streaming": False,
        "request_retry_count": 0,
        "counts": {
            "open_count": concurrency,
            "reuse_count": len(cumulative) - concurrency,
            "close_count": concurrency,
        },
        "closed": True,
        "slots": final_snapshots,
        "phase_audits": phase_audits,
    }
    if pool != expected_pool:
        raise RuntimeError(f"{context} final persistent pool recomputation mismatch")
    return {
        "status": "passed",
        "record_count": len(cumulative),
        "phase_audits_sha256": sha256_json(phase_audits),
        "final_pool_sha256": sha256_json(expected_pool),
    }


def independently_expected_video_work(
    video: Mapping[str, Any], video_kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    probe = video.get("probe")
    if not isinstance(probe, Mapping):
        raise RuntimeError("video provenance lacks probe metadata")
    width = probe.get("width")
    height = probe.get("height")
    frame_count = probe.get("frame_count")
    min_frames = video_kwargs.get("min_frames", 4)
    max_frames = video_kwargs.get("max_frames", 768)
    target_fps = video_kwargs.get("fps", 2)
    source_fps = probe.get("frames_per_second")
    if not all(type(value) is int and value > 0 for value in (width, height)):
        raise RuntimeError("source width/height are not independently derivable")
    if type(frame_count) is not int or frame_count <= 0:
        raise RuntimeError("source frame count is not independently derivable")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in (min_frames, max_frames)
    ):
        raise RuntimeError("video min/max frame bounds are invalid")
    if min_frames > max_frames:
        raise RuntimeError("video min_frames exceeds max_frames")
    if min_frames == max_frames:
        sampled_frames = min(int(min_frames), frame_count)
        derivation = "Qwen3-VL equal min/max frame clamp"
    else:
        if not isinstance(source_fps, (int, float)) or source_fps <= 0:
            raise RuntimeError("source fps is not independently derivable")
        if not isinstance(target_fps, (int, float)) or target_fps <= 0:
            raise RuntimeError("target video fps is invalid")
        sampled_frames = int(frame_count / source_fps * target_fps)
        sampled_frames = min(
            max(sampled_frames, int(min_frames)), int(max_frames), frame_count
        )
        derivation = "Qwen3-VL fps sampling calculation"
    return {
        "source_width": width,
        "source_height": height,
        "source_frame_count": frame_count,
        "sampled_frames": sampled_frames,
        "sampled_source_megapixels_estimate": (
            sampled_frames * width * height / 1_000_000
        ),
        "derivation": derivation,
    }


def validate_exact_request_record(
    record: Mapping[str, Any],
    *,
    context: str,
    phase: str,
    block_index: int,
    concurrency: int,
    request_index: int,
    global_request_index: int,
    video: Mapping[str, Any],
    video_work: Mapping[str, Any],
) -> dict[str, Any]:
    if set(record) != PASSED_REQUEST_RECORD_FIELDS:
        raise RuntimeError(f"{context} request record schema mismatch")
    video_index = request_index % 8
    expected_identity = {
        "phase": phase,
        "block_index": block_index,
        "concurrency": concurrency,
        "request_index": request_index,
        "global_request_index": global_request_index,
        "video_index": video_index,
        "video_path": video["path"],
        "video_file_uri": video["file_uri"],
        "video_sha256": video["sha256"],
        "video_work": dict(video_work),
    }
    for field, expected in expected_identity.items():
        if record.get(field) != expected:
            raise RuntimeError(
                f"{context} independently reconstructed {field} mismatch"
            )
    actual_identity = {field: record[field] for field in expected_identity}
    if sha256_json(actual_identity) != sha256_json(expected_identity):
        raise RuntimeError(f"{context} request identity JSON-type mismatch")
    payload = expected_chat_payload(str(video["file_uri"]))
    actual_payload = record.get("payload")
    actual_video_work = record.get("video_work")
    if (
        not isinstance(actual_payload, Mapping)
        or set(actual_payload) != set(payload)
        or actual_payload != payload
        or record.get("request_payload_sha256") != sha256_json(payload)
        or not isinstance(actual_video_work, Mapping)
        or set(actual_video_work) != set(video_work)
        or sha256_json(actual_video_work) != sha256_json(video_work)
    ):
        raise RuntimeError(
            f"{context} independently reconstructed payload/video-work schema mismatch"
        )
    if record.get("status") != "passed" or record.get("http_status") != 200:
        raise RuntimeError(f"{context} request outcome schema mismatch")
    wall_times: dict[str, datetime] = {}
    for field in ("started_at", "finished_at"):
        value = record.get(field)
        try:
            parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
        except ValueError as error:
            raise RuntimeError(f"{context} {field} is not an ISO timestamp") from error
        if (
            parsed is None
            or parsed.tzinfo is None
            or parsed.utcoffset() != UTC.utcoffset(None)
        ):
            raise RuntimeError(f"{context} {field} is not an aware UTC timestamp")
        wall_times[field] = parsed
    if wall_times["finished_at"] < wall_times["started_at"]:
        raise RuntimeError(f"{context} wall-clock timestamp ordering mismatch")

    transport = record.get("transport")
    if (
        not isinstance(transport, Mapping)
        or set(transport) != PERSISTENT_TRANSPORT_FIELDS
    ):
        raise RuntimeError(f"{context} transport schema mismatch")
    first_wave = request_index < concurrency
    expected_reused = not (phase == "warmup" and first_wave)
    expected_transport = {
        "phase": phase,
        "seeded_first_wave": first_wave,
        "connection_generation": 1,
        "connection_reused": expected_reused,
        "prewarmed_for_measurement": phase == "measured",
        "request_connection_header": "keep-alive",
        "response_http_version": 11,
        "response_will_close": False,
        "response_persistent": True,
    }
    for field, expected in expected_transport.items():
        if transport.get(field) != expected or type(transport.get(field)) is not type(
            expected
        ):
            raise RuntimeError(
                f"{context} independently reconstructed transport mismatch"
            )
    slot_id = transport.get("pool_slot_id")
    request_ordinal = transport.get("request_ordinal_on_generation")
    if (
        type(slot_id) is not int
        or slot_id not in range(concurrency)
        or (first_wave and slot_id != request_index)
        or type(request_ordinal) is not int
        or request_ordinal <= 0
        or (phase == "warmup" and first_wave and request_ordinal != 1)
        or transport.get("response_connection_header") is not None
    ):
        raise RuntimeError(f"{context} persistent transport identity mismatch")

    response = record.get("response")
    if not isinstance(response, Mapping) or set(response) != NORMALIZED_RESPONSE_FIELDS:
        raise RuntimeError(f"{context} normalized response schema mismatch")
    raw_response = response.get("raw_response")
    if (
        not isinstance(raw_response, Mapping)
        or set(raw_response) != RAW_CHAT_RESPONSE_FIELDS
    ):
        raise RuntimeError(f"{context} raw response schema mismatch")
    raw_choices = raw_response.get("choices")
    raw_choice = (
        raw_choices[0]
        if isinstance(raw_choices, list) and len(raw_choices) == 1
        else None
    )
    if not isinstance(raw_choice, Mapping) or set(raw_choice) != (
        RAW_CHAT_CHOICE_FIELDS
    ):
        raise RuntimeError(f"{context} raw response choice schema mismatch")
    raw_message = raw_choice.get("message")
    if not isinstance(raw_message, Mapping) or set(raw_message) != (
        RAW_CHAT_MESSAGE_FIELDS
    ):
        raise RuntimeError(f"{context} raw response message schema mismatch")
    prompt_ids = response.get("prompt_token_ids")
    completion_ids = response.get("completion_token_ids")
    usage = response.get("usage")
    created = raw_response.get("created")
    if (
        type(created) is not int
        or created < int(wall_times["started_at"].timestamp())
        or created > int(wall_times["finished_at"].timestamp())
    ):
        raise RuntimeError(f"{context} raw response created timestamp mismatch")
    if not isinstance(usage, Mapping) or set(usage) != RAW_USAGE_FIELDS:
        raise RuntimeError(f"{context} raw response usage schema mismatch")
    server_metrics = raw_response.get("metrics")
    if server_metrics is not None:
        if not isinstance(server_metrics, Mapping) or set(server_metrics) != (
            RAW_SERVER_METRICS_FIELDS
        ):
            raise RuntimeError(f"{context} raw response metrics schema mismatch")
        raise RuntimeError(
            f"{context} raw response metrics unexpectedly enabled by fixed server config"
        )
    if (
        not isinstance(prompt_ids, list)
        or not all(type(token) is int for token in prompt_ids)
        or not isinstance(completion_ids, list)
        or not all(type(token) is int for token in completion_ids)
        or type(response.get("prompt_token_count")) is not int
        or response.get("prompt_token_count") != len(prompt_ids)
        or type(response.get("completion_token_count")) is not int
        or response.get("completion_token_count") != len(completion_ids)
        or len(completion_ids) != OUTPUT_LENGTH
        or response.get("model") != SERVED_MODEL_NAME
        or raw_response.get("model") != SERVED_MODEL_NAME
        or not isinstance(response.get("id"), str)
        or not response["id"].startswith("chatcmpl-")
        or raw_response.get("object") != "chat.completion"
        or raw_response.get("service_tier") is not None
        or raw_response.get("system_fingerprint") is not None
        or raw_response.get("prompt_logprobs") is not None
        or raw_response.get("prompt_text") is not None
        or raw_response.get("kv_transfer_params") is not None
        or raw_response.get("ec_transfer_params") is not None
        or type(raw_choice.get("index")) is not int
        or raw_choice.get("index") != 0
        or raw_choice.get("logprobs") is not None
        or raw_choice.get("finish_reason") != "length"
        or raw_choice.get("stop_reason") is not None
        or raw_choice.get("routed_experts") is not None
        or raw_message.get("role") != "assistant"
        or not isinstance(raw_message.get("content"), str)
        or raw_message.get("refusal") is not None
        or raw_message.get("annotations") is not None
        or raw_message.get("audio") is not None
        or raw_message.get("function_call") is not None
        or raw_message.get("reasoning") is not None
        or usage != raw_response.get("usage")
        or type(usage.get("prompt_tokens")) is not int
        or usage.get("prompt_tokens") != len(prompt_ids)
        or type(usage.get("completion_tokens")) is not int
        or usage.get("completion_tokens") != len(completion_ids)
        or type(usage.get("total_tokens")) is not int
        or usage.get("total_tokens") != len(prompt_ids) + len(completion_ids)
        or usage.get("prompt_tokens_details") is not None
        or usage.get("completion_tokens_details") is not None
        or response.get("server_metrics") is not None
    ):
        raise RuntimeError(f"{context} raw/normalized response crosslink mismatch")
    return {
        "request_index": request_index,
        "global_request_index": global_request_index,
        "video_index": video_index,
        "response_id": response["id"],
        "payload_sha256": sha256_json(payload),
        "video_work_sha256": sha256_json(video_work),
    }


def _recomputed_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("cannot recompute a percentile from no values")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _recomputed_latency_summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": _recomputed_percentile(values, 0.50),
        "p90": _recomputed_percentile(values, 0.90),
        "p95": _recomputed_percentile(values, 0.95),
        "p99": _recomputed_percentile(values, 0.99),
        "max": max(values),
        "population_stdev": statistics.pstdev(values),
        "percentile_method": "linear interpolation at (n - 1) * fraction",
    }


def independently_recompute_batch_aggregate(
    batch: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Recompute every aggregate field from raw records and monotonic clocks."""
    records = batch.get("records")
    aggregate = batch.get("aggregate")
    if not isinstance(records, list) or not isinstance(aggregate, Mapping):
        raise RuntimeError(f"{context} lacks records/aggregate")
    started_ns = batch.get("started_monotonic_ns")
    finished_ns = batch.get("finished_monotonic_ns")
    if (
        type(started_ns) is not int
        or type(finished_ns) is not int
        or finished_ns <= started_ns
    ):
        raise RuntimeError(f"{context} has invalid monotonic boundaries")
    elapsed_seconds = (finished_ns - started_ns) / 1e9
    if batch.get("measured_window_seconds") != elapsed_seconds:
        raise RuntimeError(f"{context} batch elapsed time mismatch")

    passed: list[Mapping[str, Any]] = []
    failed: list[Mapping[str, Any]] = []
    derived_latencies_ms: list[float] = []
    derived_latencies_seconds: list[float] = []
    events: list[tuple[int, int]] = []
    ordered_fingerprints: list[dict[str, Any]] = []
    completion_hash_counts: Counter[str] = Counter()
    prompt_tokens = 0
    generated_tokens = 0
    sampled_source_megapixels: list[float] = []
    video_work_missing: list[dict[str, Any]] = []
    for expected_request_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"{context} record is not an object")
        if record.get("request_index") != expected_request_index:
            raise RuntimeError(f"{context} records are not in request-index order")
        record_started = record.get("started_monotonic_ns")
        record_finished = record.get("finished_monotonic_ns")
        if (
            type(record_started) is not int
            or type(record_finished) is not int
            or not started_ns <= record_started <= record_finished <= finished_ns
        ):
            raise RuntimeError(f"{context} record monotonic boundaries mismatch")
        latency_seconds = (record_finished - record_started) / 1e9
        latency_ms = (record_finished - record_started) / 1e6
        start_offset = (record_started - started_ns) / 1e9
        finish_offset = (record_finished - started_ns) / 1e9
        for field, expected in (
            ("latency_seconds", latency_seconds),
            ("latency_ms", latency_ms),
            ("start_offset_seconds", start_offset),
            ("finish_offset_seconds", finish_offset),
        ):
            if record.get(field) != expected:
                raise RuntimeError(f"{context} record {field} mismatch")
        events.extend(((record_started, 1), (record_finished, -1)))
        if record.get("status") == "passed":
            passed.append(record)
            derived_latencies_ms.append(latency_ms)
            derived_latencies_seconds.append(latency_seconds)
            response = record.get("response")
            if not isinstance(response, Mapping):
                raise RuntimeError(f"{context} passed record lacks response")
            prompt_ids = response.get("prompt_token_ids")
            completion_ids = response.get("completion_token_ids")
            if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
                raise RuntimeError(f"{context} passed record lacks raw token arrays")
            prompt_tokens += len(prompt_ids)
            generated_tokens += len(completion_ids)
            fingerprint = {
                "request_index": record["request_index"],
                "video_index": record["video_index"],
                "video_path": record["video_path"],
                "prompt_token_ids_sha256": sha256_json(prompt_ids),
                "completion_token_ids_sha256": sha256_json(completion_ids),
                "prompt_and_completion_token_ids_sha256": sha256_json(
                    {"prompt": prompt_ids, "completion": completion_ids}
                ),
            }
            ordered_fingerprints.append(fingerprint)
            completion_hash_counts[fingerprint["completion_token_ids_sha256"]] += 1
            work = record.get("video_work")
            if not isinstance(work, Mapping):
                raise RuntimeError(f"{context} passed record lacks video work")
            megapixels = work.get("sampled_source_megapixels_estimate")
            if isinstance(megapixels, (int, float)) and not isinstance(
                megapixels, bool
            ):
                sampled_source_megapixels.append(float(megapixels))
            else:
                video_work_missing.append(
                    {
                        "request_index": record["request_index"],
                        "video_index": record["video_index"],
                        "video_path": record["video_path"],
                        "reason": work.get("derivation"),
                    }
                )
        else:
            failed.append(record)

    in_flight = 0
    peak_in_flight = 0
    for _timestamp, delta in sorted(events, key=lambda event: (event[0], event[1])):
        in_flight += delta
        if in_flight < 0:
            raise RuntimeError(f"{context} has inconsistent in-flight events")
        peak_in_flight = max(peak_in_flight, in_flight)
    if in_flight != 0:
        raise RuntimeError(f"{context} has unterminated in-flight events")

    attempted_throughput = len(records) / elapsed_seconds
    request_throughput = len(passed) / elapsed_seconds
    prompt_throughput = prompt_tokens / elapsed_seconds
    generated_throughput = generated_tokens / elapsed_seconds
    all_token_throughput = (prompt_tokens + generated_tokens) / elapsed_seconds
    has_all_video_work = bool(passed) and not video_work_missing
    expected_aggregate = {
        "status": "passed" if not failed else "failed",
        "attempted_requests": len(records),
        "successful_requests": len(passed),
        "failed_requests": len(failed),
        "measured_window_seconds": elapsed_seconds,
        "attempted_request_throughput_per_second": attempted_throughput,
        "request_throughput_per_second": request_throughput,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "all_tokens": prompt_tokens + generated_tokens,
        "prompt_token_throughput_per_second": prompt_throughput,
        "generated_token_throughput_per_second": generated_throughput,
        "all_token_throughput_per_second": all_token_throughput,
        "sampled_source_megapixels_estimate": (
            sum(sampled_source_megapixels) if has_all_video_work else None
        ),
        "sampled_source_megapixels_estimate_per_second": (
            sum(sampled_source_megapixels) / elapsed_seconds
            if has_all_video_work
            else None
        ),
        "video_megapixel_estimate_method": (
            "sum(estimated Qwen3-VL sampled frames * externally probed encoded "
            "source width * encoded source height) / measured client wall time; "
            "this is not a count of frames actually decoded by the codec"
        ),
        "video_megapixel_estimate_unavailable": video_work_missing,
        "latency_ms": _recomputed_latency_summary(derived_latencies_ms),
        "achieved_mean_in_flight_requests": (
            sum(derived_latencies_seconds) / elapsed_seconds
        ),
        "achieved_peak_in_flight_requests": peak_in_flight,
        "response_token_fingerprints_by_request": ordered_fingerprints,
        "ordered_response_token_fingerprints_sha256": sha256_json(ordered_fingerprints),
        "ordered_response_token_ids_sha256": sha256_json(
            [
                {
                    "prompt": fingerprint["prompt_token_ids_sha256"],
                    "completion": fingerprint["completion_token_ids_sha256"],
                }
                for fingerprint in ordered_fingerprints
            ]
        ),
        "completion_token_ids_sha256_counts": dict(
            sorted(completion_hash_counts.items())
        ),
        "failures": [
            {
                "request_index": record["request_index"],
                "global_request_index": record["global_request_index"],
                "video_index": record["video_index"],
                "video_path": record["video_path"],
                "error": record.get("error"),
            }
            for record in failed
        ],
    }
    expected_keys = {*expected_aggregate, "persistent_transport_audit"}
    if set(aggregate) != expected_keys:
        raise RuntimeError(f"{context} aggregate field set mismatch")
    for field, expected in expected_aggregate.items():
        if aggregate.get(field) != expected:
            raise RuntimeError(f"{context} independently recomputed {field} mismatch")
    return {
        "status": "passed",
        "record_count": len(records),
        "elapsed_seconds": elapsed_seconds,
        "aggregate_sha256": sha256_json(expected_aggregate),
        "recomputed_aggregate": expected_aggregate,
    }


def canonical_runtime_fingerprint(result: Mapping[str, Any]) -> dict[str, Any]:
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("result lacks provenance")
    python = provenance.get("python")
    hardware = provenance.get("hardware")
    server = result.get("server")
    performance_environment = (
        server.get("performance_environment") if isinstance(server, Mapping) else None
    )
    if (
        not isinstance(python, Mapping)
        or not isinstance(hardware, Mapping)
        or not isinstance(performance_environment, Mapping)
    ):
        raise RuntimeError("result runtime/hardware provenance is incomplete")
    artifacts = python.get("runtime_artifacts")
    origins = python.get("module_origins")
    native_origins = python.get("native_module_origins")
    packages = python.get("packages")
    torch_runtime = python.get("torch_runtime")
    if (
        not isinstance(artifacts, list)
        or not isinstance(origins, Mapping)
        or not isinstance(native_origins, Mapping)
        or not isinstance(packages, Mapping)
        or not isinstance(torch_runtime, Mapping)
    ):
        raise RuntimeError("Python runtime provenance is incomplete")
    source_root = Path(str(provenance["source"]["root"])).resolve()
    live_manifest = revalidate_runtime_artifact_manifest(
        python, source_root=source_root
    )
    artifacts_by_resolved = {
        item["resolved_path"]: item for item in live_manifest["artifacts"]
    }

    def artifact_identity(path: str, *, label: str) -> dict[str, Any]:
        resolved = str(Path(path).resolve())
        item = artifacts_by_resolved.get(resolved)
        if item is None:
            raise RuntimeError(f"runtime artifact for {label} is not hash-bound")
        return {
            "basename": Path(resolved).name,
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }

    selected_packages = {
        name: packages.get(name)
        for name in ("vllm", "torch", "numpy", "transformers", "PyNvVideoCodec")
    }
    if any(not isinstance(value, str) for value in selected_packages.values()):
        raise RuntimeError(
            f"required package versions are missing: {selected_packages}"
        )
    module_artifacts = {}
    for name in ("torch", "numpy", "transformers", "PyNvVideoCodec"):
        origin = origins.get(name)
        if not isinstance(origin, str):
            raise RuntimeError(f"required module origin is missing: {name}")
        module_artifacts[name] = artifact_identity(origin, label=name)
    expected_native_origins = {"torch._C", "numpy._core._multiarray_umath"}
    if set(native_origins) != expected_native_origins:
        raise RuntimeError(
            f"required native module origins are missing: {native_origins}"
        )
    native_module_artifacts = {
        name: artifact_identity(str(native_origins[name]), label=name)
        for name in sorted(expected_native_origins)
    }
    executable = python.get("executable")
    if not isinstance(executable, str):
        raise RuntimeError("Python executable provenance is missing")
    vllm_native_paths = set(live_manifest["vllm_native_paths"])
    vllm_compiled_artifacts = sorted(
        (
            {
                "basename": Path(str(item["resolved_path"])).name,
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in live_manifest["artifacts"]
            if item["resolved_path"] in vllm_native_paths
        ),
        key=lambda item: (item["basename"], item["sha256"]),
    )
    nvcc = torch_runtime.get("nvcc")
    live_nvcc = live_manifest["nvcc"]
    if (nvcc is None) != (live_nvcc is None):
        raise RuntimeError("nvcc live artifact binding mismatch")
    selected_torch_runtime = {
        "torch_version": torch_runtime.get("torch_version"),
        "compiled_cuda_version": torch_runtime.get("compiled_cuda_version"),
        "cudnn_version": torch_runtime.get("cudnn_version"),
        "nvcc": (
            None
            if nvcc is None
            else {
                "basename": live_nvcc["basename"],
                "bytes": live_nvcc["bytes"],
                "sha256": live_nvcc["sha256"],
                "version_output": nvcc["version_output"],
            }
        ),
    }
    if selected_torch_runtime["torch_version"] != selected_packages[
        "torch"
    ] or not isinstance(selected_torch_runtime["compiled_cuda_version"], str):
        raise RuntimeError("torch package/runtime CUDA provenance mismatch")
    gpu_output = hardware.get("nvidia_smi_output")
    if not isinstance(gpu_output, str) or "\n" in gpu_output.strip():
        raise RuntimeError("hardware provenance must contain exactly one GPU row")
    fields = [field.strip() for field in gpu_output.split(",")]
    if len(fields) != 10 or fields[0] != "0":
        raise RuntimeError(f"unexpected nvidia-smi provenance row: {gpu_output}")
    selected_hardware = {
        "index": fields[0],
        "name": fields[1],
        "uuid": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": fields[4],
        "compute_capability": fields[5],
        "pci_bus_id": fields[6],
        "logical_cpus": hardware.get("logical_cpus"),
        "cuda_visible_devices": hardware.get("cuda_visible_devices"),
    }
    selected_environment_names = {
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "TOKENIZERS_PARALLELISM",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
    required_environment = {
        name: performance_environment.get(name) for name in selected_environment_names
    }
    if any(value is None for value in required_environment.values()):
        raise RuntimeError(
            f"required performance environment is missing: {required_environment}"
        )
    selected_environment = {
        str(name): value
        for name, value in sorted(
            performance_environment.items(), key=lambda item: str(item[0])
        )
    }
    canonical = {
        "python": {
            "implementation": python.get("implementation"),
            "python_version": python.get("python_version"),
            "executable_artifact": artifact_identity(executable, label="python"),
            "packages": selected_packages,
            "module_artifacts": module_artifacts,
            "native_module_artifacts": native_module_artifacts,
            "vllm_compiled_artifacts": vllm_compiled_artifacts,
            "torch_runtime": selected_torch_runtime,
        },
        "hardware": selected_hardware,
        "performance_environment": selected_environment,
    }
    return {
        "schema": "pynv-runtime-hardware-fingerprint-v1",
        "canonical": canonical,
        "sha256": sha256_json(canonical),
        "live_runtime_artifact_manifest": live_manifest,
    }


def validate_pixel_monitor_report(
    monitor: Mapping[str, Any],
    *,
    monitor_path: Path,
    expected_child_command: list[str],
    wrapper_returncode: int,
    expected_device: Mapping[str, Any],
    watchdog_pair: tuple[float, float],
    conflicting_controller_roots: Sequence[Path],
) -> dict[str, Any]:
    accepted_monitor, sample_log_audit = validate_monitor_evidence(
        monitor_path,
        expected_command=expected_child_command,
        watchdog_pair=watchdog_pair,
        conflicting_controller_roots=conflicting_controller_roots,
    )
    if accepted_monitor != monitor:
        raise RuntimeError("accepted pixel monitor changed while validating")
    process = accepted_monitor.get("process", {})
    argv = process.get("argv")
    device = monitor.get("device")
    if (
        wrapper_returncode != 0
        or not isinstance(argv, list)
        or not argv
        or device != expected_device
    ):
        raise RuntimeError("accepted pixel monitor evidence changed")
    return {
        "status": "passed",
        "device": device,
        "process_argv": argv,
        "sample_log_audit": sample_log_audit,
    }


def validate_video_hardlink_identity(videos: Sequence[Mapping[str, Any]]) -> None:
    identities = {(video.get("device"), video.get("inode")) for video in videos}
    if len(videos) != 8 or len(identities) != 1 or (None, None) in identities:
        raise RuntimeError("video corpus entries are not one hard-linked source asset")


def validate_result(
    result: Mapping[str, Any],
    monitor: Mapping[str, Any],
    *,
    commit: str,
    variant: str,
    concurrency_order: list[int],
    harness: Path,
    harness_sha256: str,
    expected_monitor_command: list[str],
    monitor_path: Path,
    corpus: Path,
    transformers_root: Path,
    hf_snapshot_root: Path,
    source_root: Path,
    server_log_path: Path,
    warmup_requests: Mapping[int, int] = WARMUP_REQUESTS,
    measured_requests: Mapping[int, int] = MEASURED_REQUESTS,
    result_variant_label: str | None = None,
) -> dict[str, Any]:
    if result.get("status") != "passed":
        raise RuntimeError("benchmark result status is not passed")
    if result.get("schema") != (
        "vllm-qwen3-vl-video-e2e-throughput-v3-persistent-http"
    ):
        raise RuntimeError("benchmark result schema is not persistent HTTP v3")
    integrity_audit = validate_result_integrity(result)
    if monitor.get("contaminated") is not False:
        raise RuntimeError("GPU monitor did not record an uncontaminated run")
    if monitor.get("foreign_events") != []:
        raise RuntimeError("GPU monitor recorded foreign-process events")
    if monitor.get("returncode") != 0:
        raise RuntimeError(
            f"GPU monitor child return code: {monitor.get('returncode')}"
        )
    if monitor.get("timed_out") is not False:
        raise RuntimeError("GPU monitor timed_out is not false")
    if monitor.get("timeout_seconds") != 3600.0:
        raise RuntimeError("GPU monitor watchdog timeout mismatch")
    if monitor.get("timeout_grace_seconds") != 120.0:
        raise RuntimeError("GPU monitor watchdog grace mismatch")
    if monitor.get("command") != expected_monitor_command:
        raise RuntimeError("GPU monitor child command mismatch")
    monitor_process = monitor.get("process", {})
    monitor_script_path = Path(str(monitor_process.get("script_path", ""))).resolve()
    monitor_argv = monitor_process.get("argv")
    expected_monitor_argv = [
        str(monitor_script_path),
        "--output",
        str(monitor_path.resolve()),
        "--device-index",
        "0",
        "--timeout-seconds",
        "3600",
        "--timeout-grace-seconds",
        "120",
        "--",
        *expected_monitor_command,
    ]
    if (
        monitor_process.get("script_sha256") != GPU_MONITOR_SHA256
        or not isinstance(monitor_argv, list)
        or not monitor_argv
        or Path(str(monitor_argv[0])).resolve() != monitor_script_path
        or monitor_argv != expected_monitor_argv
    ):
        raise RuntimeError("GPU monitor script provenance mismatch")
    if monitor.get("status") != "passed" or monitor.get("monitor_errors") != []:
        raise RuntimeError("GPU monitor terminal status/health mismatch")
    if monitor.get("guard_helper", {}).get("sha256") != GUARD_HELPER_SHA256:
        raise RuntimeError("GPU monitor helper provenance mismatch")
    monitor_device = monitor.get("device")
    if not isinstance(monitor_device, Mapping):
        raise RuntimeError("GPU monitor lacks physical device identity")
    if monitor_device.get("index") != 0:
        raise RuntimeError(f"GPU monitor device index mismatch: {monitor_device}")
    monitor_configuration = monitor.get("configuration", {})
    if (
        monitor_configuration.get("device_index") != 0
        or monitor_configuration.get("sample_interval_seconds") != 0.2
        or monitor_configuration.get("maximum_sample_gap_seconds")
        != MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS
    ):
        raise RuntimeError("GPU monitor telemetry configuration mismatch")

    provenance = result["provenance"]
    source = provenance["source"]
    if Path(source["root"]).resolve() != source_root:
        raise RuntimeError("result source root mismatch")
    if source["commit"] != commit:
        raise RuntimeError(f"result commit mismatch: {source['commit']} != {commit}")
    if source["tree"] != TREES[variant]:
        raise RuntimeError(f"result tree mismatch: {source['tree']}")
    if source["tracked_diff_bytes"] != 0 or source["untracked_files"]:
        raise RuntimeError("result source provenance is not pristine")
    result_harness = provenance["harness"]
    if Path(result_harness["path"]).resolve() != harness:
        raise RuntimeError(f"result harness path mismatch: {result_harness['path']}")
    if result_harness["sha256"] != harness_sha256:
        raise RuntimeError(f"result harness hash mismatch: {result_harness['sha256']}")
    hardware = provenance["hardware"]
    if hardware.get("cuda_visible_devices") != "0":
        raise RuntimeError(
            "harness CUDA_VISIBLE_DEVICES mismatch: "
            f"{hardware.get('cuda_visible_devices')!r} != '0'"
        )
    gpu_lines = [
        line.strip()
        for line in str(hardware.get("nvidia_smi_output", "")).splitlines()
        if line.strip()
    ]
    if len(gpu_lines) != 1:
        raise RuntimeError(f"expected exactly one physical GPU row: {gpu_lines}")
    gpu_fields = [field.strip() for field in gpu_lines[0].split(",")]
    if len(gpu_fields) < 3:
        raise RuntimeError(f"invalid nvidia-smi provenance row: {gpu_lines[0]}")
    if (
        gpu_fields[0] != "0"
        or gpu_fields[1] != monitor_device.get("name")
        or gpu_fields[2] != monitor_device.get("uuid")
    ):
        raise RuntimeError(
            "monitor/harness physical GPU identity mismatch: "
            f"monitor={monitor_device}, nvidia-smi={gpu_fields[:3]}"
        )

    configuration = result["configuration"]
    expected_server_argv = variant_server_argv(variant)
    expected_backend_kwargs = variant_backend_kwargs(variant)
    expected_warmup_map = [
        {
            "concurrency": concurrency,
            "requested": warmup_requests[concurrency],
            "effective": warmup_requests[concurrency],
        }
        for concurrency in concurrency_order
    ]
    expected_measured_map = [
        {"concurrency": concurrency, "requests": measured_requests[concurrency]}
        for concurrency in concurrency_order
    ]
    expected_values = {
        "variant": result_variant_label or variant,
        "model": MODEL,
        "revision": REVISION,
        "prompt": PROMPT,
        "prompt_sha256": sha256_json(PROMPT),
        "output_len": OUTPUT_LENGTH,
        "frame_target": FRAMES,
        "warmup_requests_by_concurrency": expected_warmup_map,
        "measured_requests_per_concurrency": expected_measured_map,
        "concurrency_order": concurrency_order,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": 9216,
        "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
        "mm_ipc_gpu_memory_gb": 2.0,
        "backend_argument": "pynvvideocodec",
        "backend_kwargs": expected_backend_kwargs,
        "extra_server_argv": expected_server_argv,
        "dtype": "bfloat16",
        "seed": 0,
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "mm_processor_cache_gb": 0,
        "prefix_caching": False,
        "gpu_memory_utilization": None,
        "request_media_io_kwargs": {},
        "server_mm_processor_kwargs": {"max_pixels": TOTAL_MAX_PIXELS},
        "server_limit_mm_per_prompt": {"image": 0, "video": 1},
        "request_timeout_seconds": 1200.0,
        "startup_timeout_seconds": 600.0,
        "shutdown_timeout_seconds": 60.0,
        "settle_seconds": 1.0,
        "video_count": 8,
        "video_cycle_policy": (
            "video_index = phase-local request_index modulo video count; "
            "reset for each warmup and measured batch"
        ),
        "parity_reference": None,
        "pythonpath_extra": [str(transformers_root)],
        "allowed_local_media_path": str(corpus),
        "client_http_protocol": {
            "implementation": "stdlib http.client.HTTPConnection",
            "http_version": "HTTP/1.1",
            "streaming": False,
            "request_connection_header": "keep-alive",
            "pool_size": "exact requested concurrency for each block",
            "pool_lifetime": "warmup through settle and measurement; then close",
            "slot_seeding": (
                "the first wave leases every slot exactly once before any slot "
                "can be reused"
            ),
            "request_retries": 0,
            "measured_connection_requirement": (
                "same successful warmup generation, reused and persistent"
            ),
        },
    }
    for key, expected in expected_values.items():
        if configuration.get(key) != expected:
            raise RuntimeError(
                f"configuration {key} mismatch: {configuration.get(key)!r} != "
                f"{expected!r}"
            )
    packages = provenance["python"]["packages"]
    if packages.get("PyNvVideoCodec") != "2.0.4":
        raise RuntimeError(
            "PyNvVideoCodec distribution mismatch: "
            f"{packages.get('PyNvVideoCodec')!r} != '2.0.4'"
        )
    if packages.get("transformers") != "5.14.1":
        raise RuntimeError(
            f"Transformers distribution mismatch: {packages.get('transformers')!r}"
        )
    module_origins = provenance["python"]["module_origins"]
    expected_transformers_origin = (
        transformers_root / "transformers/__init__.py"
    ).resolve()
    if Path(module_origins["transformers"]).resolve() != expected_transformers_origin:
        raise RuntimeError("Transformers module origin mismatch")
    expected_vllm_origin = (Path(source["root"]) / "vllm/__init__.py").resolve()
    if Path(module_origins["vllm"]).resolve() != expected_vllm_origin:
        raise RuntimeError("vLLM module origin mismatch")
    runtime_artifacts = provenance["python"]["runtime_artifacts"]
    transformers_artifacts = [
        artifact
        for artifact in runtime_artifacts
        if Path(artifact["path"]).resolve() == expected_transformers_origin
    ]
    if (
        len(transformers_artifacts) != 1
        or transformers_artifacts[0]["sha256"] != TRANSFORMERS_INIT_SHA256
    ):
        raise RuntimeError("Transformers runtime artifact mismatch")
    pynv_artifacts = {
        Path(artifact["path"]).name: artifact["sha256"]
        for artifact in runtime_artifacts
        if "PyNvVideoCodec" in Path(artifact["path"]).parts
        and Path(artifact["path"]).name in PYNV_RUNTIME_ARTIFACT_SHA256
    }
    if pynv_artifacts != PYNV_RUNTIME_ARTIFACT_SHA256:
        raise RuntimeError(f"PyNv runtime artifact mismatch: {pynv_artifacts}")
    server_command = result["server"]["command"]
    normalize_arguments = {
        "--mm-device-do-normalize",
        "--no-mm-device-do-normalize",
    }
    present_normalize_arguments = [
        argument for argument in server_command if argument in normalize_arguments
    ]
    if present_normalize_arguments != expected_server_argv:
        raise RuntimeError(
            "server normalization arguments mismatch: "
            f"{present_normalize_arguments!r} != {expected_server_argv!r}"
        )
    pixel_budget = configuration["video_pixel_budget"]
    expected_pixel_budget = {
        "reference_width": PIXEL_BUDGET[0],
        "reference_height": PIXEL_BUDGET[1],
        "max_pixels_per_sampled_frame": PIXEL_BUDGET[0] * PIXEL_BUDGET[1],
        "sampled_frames": FRAMES,
        "max_pixels_total": TOTAL_MAX_PIXELS,
    }
    for key, expected in expected_pixel_budget.items():
        if pixel_budget.get(key) != expected:
            raise RuntimeError(
                f"pixel budget {key} mismatch: {pixel_budget.get(key)!r} != {expected!r}"
            )
    expected_server_media_video = {
        "video_backend": "qwen3_vl",
        "min_frames": FRAMES,
        "max_frames": FRAMES,
        "backend": "pynvvideocodec",
        **expected_backend_kwargs,
    }
    if configuration.get("server_media_io_kwargs") != {
        "video": expected_server_media_video
    }:
        raise RuntimeError("server media-I/O configuration mismatch")
    if configuration.get("video_kwargs_for_metric_derivation") != (
        expected_server_media_video
    ):
        raise RuntimeError("derived video configuration mismatch")
    expected_runtime_environment = {
        **huggingface_cache_environment(hf_snapshot_root),
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    performance_environment = result["server"].get("performance_environment", {})
    for name, expected in expected_runtime_environment.items():
        if performance_environment.get(name) != expected:
            raise RuntimeError(
                f"server runtime environment {name} mismatch: "
                f"{performance_environment.get(name)!r} != {expected!r}"
            )

    videos = result["videos"]
    if len(videos) != 8:
        raise RuntimeError(f"expected eight videos, got {len(videos)}")
    for index, video in enumerate(videos):
        expected_path = (corpus / f"traffic1080-{index:02d}.mp4").resolve()
        expected_stat = expected_path.stat()
        if (
            set(video)
            != {
                "video_index",
                "path",
                "file_uri",
                "bytes",
                "sha256",
                "device",
                "inode",
                "mtime_ns",
                "probe",
            }
            or video.get("video_index") != index
            or Path(video["path"]).resolve() != expected_path
            or video.get("file_uri") != expected_path.as_uri()
            or video.get("device") != expected_stat.st_dev
            or video.get("inode") != expected_stat.st_ino
            or video.get("mtime_ns") != expected_stat.st_mtime_ns
        ):
            raise RuntimeError(f"video {index} path mismatch")
        if video["sha256"] != VIDEO_SHA256 or video["bytes"] != VIDEO_BYTES:
            raise RuntimeError(f"video {index} content mismatch")
        probe = video["probe"]
        if (
            probe["width"],
            probe["height"],
            probe["frame_count"],
        ) != (1920, 1080, 914):
            raise RuntimeError(f"video {index} probe mismatch")
    validate_video_hardlink_identity(videos)

    expected_video_works = [
        independently_expected_video_work(video, expected_server_media_video)
        for video in videos
    ]
    expected_payloads_by_video = [
        {
            "video_index": index,
            "video_path": video["path"],
            "payload": (payload := expected_chat_payload(video["file_uri"])),
            "payload_sha256": sha256_json(payload),
        }
        for index, video in enumerate(videos)
    ]
    if result.get("request_payloads_by_video") != expected_payloads_by_video:
        raise RuntimeError("top-level request payload graph mismatch")

    blocks = result["concurrency_blocks"]
    actual_order = [int(block["concurrency"]) for block in blocks]
    if actual_order != concurrency_order:
        raise RuntimeError(
            f"concurrency order mismatch: {actual_order} != {concurrency_order}"
        )
    block_summaries = []
    expected_global_request_index = 0
    seen_response_ids: set[str] = set()
    for block_index, block in enumerate(blocks):
        concurrency = int(block["concurrency"])
        if block.get("block_index") != block_index:
            raise RuntimeError(f"c{concurrency} block index mismatch")
        if block["status"] != "passed":
            raise RuntimeError(f"c{concurrency} block did not pass")
        if block["requested_warmup_requests"] != warmup_requests[concurrency]:
            raise RuntimeError(f"c{concurrency} requested warmup mismatch")
        if block["effective_warmup_requests"] != warmup_requests[concurrency]:
            raise RuntimeError(f"c{concurrency} effective warmup mismatch")
        if block["requested_measured_requests"] != measured_requests[concurrency]:
            raise RuntimeError(f"c{concurrency} requested measured count mismatch")
        actual_settle_seconds = block.get("actual_settle_seconds")
        if (
            isinstance(actual_settle_seconds, bool)
            or not isinstance(actual_settle_seconds, (int, float))
            or not math.isfinite(float(actual_settle_seconds))
            or float(actual_settle_seconds) < 1.0
        ):
            raise RuntimeError(f"c{concurrency} settle interval mismatch")

        phase_summaries: dict[str, Any] = {}
        phase_records: dict[str, Sequence[Mapping[str, Any]]] = {}
        for phase, expected_count in (
            ("warmup", warmup_requests[concurrency]),
            ("measured", measured_requests[concurrency]),
        ):
            batch = block[phase]
            aggregate = batch["aggregate"]
            if batch["status"] != "passed":
                raise RuntimeError(f"c{concurrency} {phase} batch did not pass")
            started_monotonic_ns = batch.get("started_monotonic_ns")
            finished_monotonic_ns = batch.get("finished_monotonic_ns")
            if (
                not isinstance(started_monotonic_ns, int)
                or not isinstance(finished_monotonic_ns, int)
                or finished_monotonic_ns <= started_monotonic_ns
                or not math.isclose(
                    float(batch["measured_window_seconds"]),
                    (finished_monotonic_ns - started_monotonic_ns) / 1e9,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise RuntimeError(
                    f"c{concurrency} {phase} monotonic boundary mismatch"
                )
            if batch["requested_concurrency"] != concurrency:
                raise RuntimeError(
                    f"c{concurrency} {phase} requested concurrency mismatch"
                )
            if batch["effective_client_workers"] != concurrency:
                raise RuntimeError(f"c{concurrency} {phase} client worker mismatch")
            if not isinstance(batch.get("records"), list):
                raise RuntimeError(f"c{concurrency} {phase} records are missing")
            phase_records[phase] = batch["records"]
            for field, expected in (
                ("attempted_requests", expected_count),
                ("successful_requests", expected_count),
                ("failed_requests", 0),
            ):
                if aggregate[field] != expected:
                    raise RuntimeError(
                        f"c{concurrency} {phase} {field} mismatch: "
                        f"{aggregate[field]} != {expected}"
                    )
            if aggregate["achieved_peak_in_flight_requests"] != concurrency:
                raise RuntimeError(
                    f"c{concurrency} {phase} peak in-flight mismatch: "
                    f"{aggregate['achieved_peak_in_flight_requests']}"
                )
            achieved_mean = float(aggregate["achieved_mean_in_flight_requests"])
            if not 0.0 < achieved_mean <= concurrency + 1e-6:
                raise RuntimeError(
                    f"c{concurrency} {phase} invalid mean in-flight: {achieved_mean}"
                )
            transport_audit = aggregate.get("persistent_transport_audit", {})
            expected_seeded_mapping = {str(slot): slot for slot in range(concurrency)}
            if (
                transport_audit.get("status") != "passed"
                or transport_audit.get("phase") != phase
                or transport_audit.get("pool_size") != concurrency
                or transport_audit.get("request_count") != expected_count
                or transport_audit.get("used_slot_ids") != list(range(concurrency))
                or transport_audit.get("seeded_first_wave_request_to_slot")
                != expected_seeded_mapping
                or transport_audit.get("reasons") != []
            ):
                raise RuntimeError(
                    f"c{concurrency} {phase} persistent transport audit mismatch: "
                    f"{transport_audit}"
                )
            record_spec_audits = []
            for request_index, record in enumerate(batch["records"]):
                video_index = request_index % len(videos)
                record_spec_audit = validate_exact_request_record(
                    record,
                    context=f"c{concurrency} {phase} request {request_index}",
                    phase=phase,
                    block_index=block_index,
                    concurrency=concurrency,
                    request_index=request_index,
                    global_request_index=expected_global_request_index,
                    video=videos[video_index],
                    video_work=expected_video_works[video_index],
                )
                response_id = record_spec_audit["response_id"]
                if response_id in seen_response_ids:
                    raise RuntimeError(
                        f"duplicate response ID across accepted records: {response_id}"
                    )
                seen_response_ids.add(response_id)
                record_spec_audits.append(record_spec_audit)
                expected_global_request_index += 1
                if (
                    not isinstance(record.get("started_monotonic_ns"), int)
                    or not isinstance(record.get("finished_monotonic_ns"), int)
                    or not started_monotonic_ns
                    <= record["started_monotonic_ns"]
                    <= record["finished_monotonic_ns"]
                    <= finished_monotonic_ns
                ):
                    raise RuntimeError(
                        f"c{concurrency} {phase} request monotonic boundary mismatch"
                    )
                transport = record.get("transport", {})
                if (
                    transport.get("request_connection_header") != "keep-alive"
                    or transport.get("response_http_version") != 11
                    or transport.get("response_will_close") is not False
                    or transport.get("response_persistent") is not True
                    or int(transport.get("pool_slot_id", -1)) not in range(concurrency)
                ):
                    raise RuntimeError(
                        f"c{concurrency} {phase} invalid request transport: "
                        f"{transport}"
                    )
                if phase == "measured" and (
                    transport.get("connection_reused") is not True
                    or transport.get("prewarmed_for_measurement") is not True
                ):
                    raise RuntimeError(
                        f"c{concurrency} measured request was not prewarmed/reused"
                    )
            token_evidence_audit = validate_batch_token_evidence(
                batch, context=f"c{concurrency} {phase}"
            )
            aggregate_recomputation_audit = independently_recompute_batch_aggregate(
                batch, context=f"c{concurrency} {phase}"
            )
            prompt_counts = {
                int(record["response"]["prompt_token_count"])
                for record in batch["records"]
            }
            if prompt_counts != {EXPECTED_PROMPT_TOKENS}:
                raise RuntimeError(
                    f"c{concurrency} {phase} prompt token counts: {prompt_counts}"
                )
            prompt_hashes = {
                str(record["response"]["prompt_token_ids_sha256"])
                for record in batch["records"]
            }
            if prompt_hashes != {EXPECTED_PROMPT_TOKEN_IDS_SHA256}:
                raise RuntimeError(
                    f"c{concurrency} {phase} prompt token ID hashes: "
                    f"{sorted(prompt_hashes)}"
                )
            completion_counts = {
                int(record["response"]["completion_token_count"])
                for record in batch["records"]
            }
            finish_stop_pairs = {
                (
                    record["response"].get("finish_reason"),
                    record["response"].get("stop_reason"),
                )
                for record in batch["records"]
            }
            if completion_counts != {OUTPUT_LENGTH} or finish_stop_pairs != {
                ("length", None)
            }:
                raise RuntimeError(
                    f"c{concurrency} {phase} output-length/finish policy changed"
                )
            phase_summaries[phase] = {
                "attempted_requests": aggregate["attempted_requests"],
                "successful_requests": aggregate["successful_requests"],
                "failed_requests": aggregate["failed_requests"],
                "effective_client_workers": batch["effective_client_workers"],
                "achieved_mean_in_flight_requests": achieved_mean,
                "achieved_peak_in_flight_requests": aggregate[
                    "achieved_peak_in_flight_requests"
                ],
                "prompt_token_count_values": sorted(prompt_counts),
                "prompt_token_ids_sha256_values": sorted(prompt_hashes),
                "persistent_transport_audit": transport_audit,
                "token_evidence_audit": token_evidence_audit,
                "aggregate_recomputation_audit": aggregate_recomputation_audit,
                "record_specification_audit_sha256": sha256_json(record_spec_audits),
            }

        persistent_transport_recomputation = validate_persistent_transport_evidence(
            block,
            concurrency=concurrency,
            phase_records=phase_records,
            context=f"c{concurrency}",
        )

        persistent_pool = block.get("persistent_http_pool", {})
        expected_pool_counts = {
            "open_count": concurrency,
            "reuse_count": (
                warmup_requests[concurrency]
                + measured_requests[concurrency]
                - concurrency
            ),
            "close_count": concurrency,
        }
        if (
            persistent_pool.get("implementation")
            != "stdlib http.client.HTTPConnection HTTP/1.1"
            or persistent_pool.get("pool_size") != concurrency
            or persistent_pool.get("request_streaming") is not False
            or persistent_pool.get("request_retry_count") != 0
            or persistent_pool.get("closed") is not True
            or persistent_pool.get("counts") != expected_pool_counts
        ):
            raise RuntimeError(
                f"c{concurrency} persistent pool summary mismatch: {persistent_pool}"
            )
        slots = persistent_pool.get("slots", [])
        if len(slots) != concurrency:
            raise RuntimeError(f"c{concurrency} persistent pool slot count mismatch")
        for slot_id, slot in enumerate(slots):
            if (
                slot.get("slot_id") != slot_id
                or slot.get("current_generation") != 1
                or slot.get("warmed_generation") != 1
                or slot.get("open_count") != 1
                or slot.get("close_count") != 1
                or slot.get("currently_open") is not False
                or slot.get("close_reasons") != {"pool_close": 1}
            ):
                raise RuntimeError(
                    f"c{concurrency} persistent slot {slot_id} mismatch: {slot}"
                )

        aggregate = block["aggregate"]
        if aggregate != block["measured"]["aggregate"]:
            raise RuntimeError(
                f"c{concurrency} duplicate measured aggregate evidence changed"
            )
        latency = aggregate["latency_ms"]
        block_summaries.append(
            {
                "concurrency": concurrency,
                "warmup": phase_summaries["warmup"],
                "measured": phase_summaries["measured"],
                "request_throughput_per_second": aggregate[
                    "request_throughput_per_second"
                ],
                "generated_token_throughput_per_second": aggregate[
                    "generated_token_throughput_per_second"
                ],
                "e2e_latency_ms": {
                    "p50": latency["p50"],
                    "p95": latency["p95"],
                },
                "persistent_transport_recomputation": (
                    persistent_transport_recomputation
                ),
                "measured_window_vram": measured_window_vram(monitor, block),
            }
        )

    server_log_record = result["server"]["log"]
    if (
        Path(server_log_record.get("path", "")).resolve() != server_log_path.resolve()
        or server_log_record.get("storage") != "append-only full server-log sidecar"
        or server_log_record.get("bytes") != server_log_path.stat().st_size
        or server_log_record.get("sha256") != sha256_file(server_log_path)
    ):
        raise RuntimeError(f"full server log binding mismatch: {server_log_record}")
    full_server_log = server_log_path.read_text(errors="replace")
    server_log_audit = parse_server_log(full_server_log)
    if not server_log_audit["gpu_kv_capacity"]:
        raise RuntimeError("server log lacks GPU KV cache capacity provenance")
    expected_kv = {
        "tokens": 336560,
        "per_request_tokens": 32768,
        "maximum_concurrency": 10.27,
    }
    if not any(
        all(capacity[key] == value for key, value in expected_kv.items())
        for capacity in server_log_audit["gpu_kv_capacity"]
    ):
        raise RuntimeError(
            "server log lacks exact expected GPU KV capacity: "
            f"{expected_kv}; got {server_log_audit['gpu_kv_capacity']}"
        )
    if server_log_audit["oom_line_count"]:
        raise RuntimeError("server log contains an out-of-memory event")
    if server_log_audit["preemption_line_count"]:
        raise RuntimeError("server log contains a preemption event")
    runtime_fingerprint = canonical_runtime_fingerprint(result)
    return {
        "result_integrity_audit": integrity_audit,
        "blocks": block_summaries,
        "server_log_audit": server_log_audit,
        "full_server_log": {
            "path": str(server_log_path),
            "bytes": server_log_path.stat().st_size,
            "sha256": sha256_file(server_log_path),
            "full_file_scanned": True,
        },
        "whole_run_peak_total_gpu_memory_used_mib": monitor["peak_memory_used_mib"],
        "monitor_sample_count": monitor["sample_count"],
        "runtime_hardware_fingerprint": runtime_fingerprint,
    }


def collection_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, dict[str, list[float]]]] = {
        variant: {
            str(concurrency): {
                "request_throughput_per_second": [],
                "generated_token_throughput_per_second": [],
                "e2e_latency_p50_ms": [],
                "e2e_latency_p95_ms": [],
                "achieved_mean_in_flight_requests": [],
                "measured_window_peak_vram_mib": [],
            }
            for concurrency in MEASURED_REQUESTS
        }
        for variant in COMMITS
    }
    for cell in cells:
        if cell["status"] != "passed":
            raise RuntimeError("collection summary received a non-passing cell")
        for block in cell["validated_metrics"]["blocks"]:
            destination = metrics[str(cell["variant"])][str(block["concurrency"])]
            destination["request_throughput_per_second"].append(
                float(block["request_throughput_per_second"])
            )
            destination["generated_token_throughput_per_second"].append(
                float(block["generated_token_throughput_per_second"])
            )
            destination["e2e_latency_p50_ms"].append(
                float(block["e2e_latency_ms"]["p50"])
            )
            destination["e2e_latency_p95_ms"].append(
                float(block["e2e_latency_ms"]["p95"])
            )
            destination["achieved_mean_in_flight_requests"].append(
                float(block["measured"]["achieved_mean_in_flight_requests"])
            )
            destination["measured_window_peak_vram_mib"].append(
                float(block["measured_window_vram"]["peak_total_gpu_memory_used_mib"])
            )

    aggregates: dict[str, Any] = {}
    for variant, by_concurrency in metrics.items():
        aggregates[variant] = {}
        for concurrency, by_metric in by_concurrency.items():
            aggregates[variant][concurrency] = {}
            for metric, values in by_metric.items():
                if len(values) != REPETITIONS:
                    raise RuntimeError(
                        f"missing {variant} c{concurrency} {metric} samples: {values}"
                    )
                aggregates[variant][concurrency][metric] = {
                    "values": values,
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "sample_stdev": statistics.stdev(values),
                }
    return {
        "status": "collection_passed",
        "model": MODEL,
        "revision": REVISION,
        "protocol": (
            "pooled persistent non-streaming HTTP/1.1 chat/completions; "
            "E2E latency only; no TTFT"
        ),
        "sampled_frames": FRAMES,
        "pixel_budget_per_sampled_frame": {
            "width": PIXEL_BUDGET[0],
            "height": PIXEL_BUDGET[1],
            "max_pixels": PIXEL_BUDGET[0] * PIXEL_BUDGET[1],
        },
        "max_pixels_total": TOTAL_MAX_PIXELS,
        "measured_requests_by_concurrency": MEASURED_REQUESTS,
        "warmup_requests_by_concurrency": WARMUP_REQUESTS,
        "warmup_waves": 3,
        "max_num_seqs": MAX_NUM_SEQS,
        "repetitions": REPETITIONS,
        "aggregates": aggregates,
        "paired_endpoint_comparisons": paired_endpoint_summaries(cells),
    }


def expected_treatment_configuration(variant: str) -> dict[str, Any]:
    backend_kwargs = variant_backend_kwargs(variant)
    media_video = {
        "video_backend": "qwen3_vl",
        "min_frames": FRAMES,
        "max_frames": FRAMES,
        "backend": "pynvvideocodec",
        **backend_kwargs,
    }
    return {
        "backend_kwargs": backend_kwargs,
        "server_media_io_kwargs": {"video": media_video},
        "video_kwargs_for_metric_derivation": media_video,
        "extra_server_argv": variant_server_argv(variant),
    }


def configuration_fingerprint(
    configuration: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    values = {field: configuration.get(field) for field in fields}
    return {
        "fields": list(fields),
        "values": values,
        "sha256": sha256_json(values),
    }


def validate_three_way_pilot_parity(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(results) != set(COMMITS):
        raise RuntimeError("pilot parity result set mismatch")
    compared_pairs = 0
    comparisons = {}
    input_mismatch_count = 0
    generation_mismatch_count = 0
    mismatch_details = []
    for comparison_name, baseline_variant, candidate_variant in PAIRWISE_COMPARISONS:
        baseline = results[baseline_variant]
        candidate = results[candidate_variant]
        comparison_pairs = 0
        baseline_blocks = baseline.get("concurrency_blocks", [])
        candidate_blocks = candidate.get("concurrency_blocks", [])
        if len(baseline_blocks) != len(candidate_blocks):
            raise RuntimeError("pilot parity block count mismatch")
        for baseline_block, candidate_block in zip(
            baseline_blocks, candidate_blocks, strict=True
        ):
            if baseline_block.get("concurrency") != candidate_block.get("concurrency"):
                raise RuntimeError("pilot parity concurrency mismatch")
            for phase in ("warmup", "measured"):
                baseline_records = baseline_block[phase]["records"]
                candidate_records = candidate_block[phase]["records"]
                if len(baseline_records) != len(candidate_records):
                    raise RuntimeError("pilot parity response count mismatch")
                for baseline_record, candidate_record in zip(
                    baseline_records, candidate_records, strict=True
                ):
                    differing_input_fields = []
                    for field in (
                        "phase",
                        "block_index",
                        "concurrency",
                        "request_index",
                        "video_index",
                        "video_sha256",
                        "request_payload_sha256",
                        "status",
                    ):
                        if baseline_record.get(field) != candidate_record.get(field):
                            differing_input_fields.append(field)
                    baseline_response = baseline_record.get("response", {})
                    candidate_response = candidate_record.get("response", {})
                    for field in ("prompt_token_ids", "prompt_token_ids_sha256"):
                        if baseline_response.get(field) != candidate_response.get(
                            field
                        ):
                            differing_input_fields.append(f"response.{field}")
                    differing_generation_fields = []
                    for field in (
                        "completion_token_ids",
                        "completion_token_ids_sha256",
                        "text",
                        "text_sha256",
                        "reasoning_content",
                        "reasoning_content_sha256",
                        "finish_reason",
                        "stop_reason",
                    ):
                        if baseline_response.get(field) != candidate_response.get(
                            field
                        ):
                            differing_generation_fields.append(f"response.{field}")
                    if differing_input_fields or differing_generation_fields:
                        input_mismatch_count += bool(differing_input_fields)
                        generation_mismatch_count += bool(differing_generation_fields)
                        if len(mismatch_details) < 256:
                            mismatch_details.append(
                                {
                                    "comparison": comparison_name,
                                    "concurrency": baseline_block["concurrency"],
                                    "phase": phase,
                                    "request_index": baseline_record.get(
                                        "request_index"
                                    ),
                                    "input_fields": differing_input_fields,
                                    "generation_fields": (differing_generation_fields),
                                }
                            )
                    compared_pairs += 1
                    comparison_pairs += 1
        comparisons[comparison_name] = {
            "baseline": baseline_variant,
            "candidate": candidate_variant,
            "compared_response_pair_count": comparison_pairs,
        }
    status = (
        "failed_input_parity"
        if input_mismatch_count
        else (
            "completion_or_text_mismatch"
            if generation_mismatch_count
            else "passed_exact"
        )
    )
    return {
        "status": status,
        "compared_response_pair_count": compared_pairs,
        "input_mismatch_count": input_mismatch_count,
        "generation_mismatch_count": generation_mismatch_count,
        "mismatch_details": mismatch_details,
        "mismatch_details_truncated": (
            input_mismatch_count + generation_mismatch_count > len(mismatch_details)
        ),
        "comparisons": comparisons,
        "comparison": "full prompt/completion arrays and text metadata",
    }


def strict_token_text_audit(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results_by_rep_variant: dict[tuple[int, str], dict[str, Any]] = {}
    result_artifacts: dict[str, Any] = {}
    configuration_mismatches: list[dict[str, Any]] = []
    configuration_mismatch_count = 0
    mismatch_counts = {
        "common_configuration": 0,
        "treatment_configuration": 0,
        "request_identity": 0,
        "prompt_token_ids": 0,
        "completion_token_ids": 0,
        "text_sha256": 0,
        "reasoning_content_sha256": 0,
        "finish_reason": 0,
        "stop_reason": 0,
    }
    for cell in cells:
        rep = int(cell["rep"])
        variant = str(cell["variant"])
        result_path = Path(str(cell["output"])).resolve()
        encoded = result_path.read_bytes()
        winning_attempt = int(cell["winning_attempt"])
        accepted_attempts = [
            attempt
            for attempt in cell["attempts"]
            if int(attempt["attempt"]) == winning_attempt
            and attempt.get("accepted") is True
        ]
        if len(accepted_attempts) != 1:
            raise RuntimeError(
                f"rep {rep} {variant} lacks exactly one accepted winning attempt"
            )
        recorded_result_sha256 = accepted_attempts[0].get("result_sha256")
        actual_result_sha256 = hashlib.sha256(encoded).hexdigest()
        if recorded_result_sha256 != actual_result_sha256:
            raise RuntimeError(
                f"rep {rep} {variant} result changed after acceptance: "
                f"{actual_result_sha256} != {recorded_result_sha256}"
            )
        result = json.loads(encoded)
        if not isinstance(result, dict):
            raise RuntimeError(f"accepted result is not a JSON object: {result_path}")
        result_integrity_audit = validate_result_integrity(result)
        configuration = result.get("configuration")
        if not isinstance(configuration, Mapping):
            raise RuntimeError(f"accepted result lacks configuration: {result_path}")
        workload_fingerprint = configuration_fingerprint(
            configuration, WORKLOAD_PARITY_CONFIGURATION_FIELDS
        )
        actual_treatment = {
            field: configuration.get(field)
            for field in ENDPOINT_TREATMENT_CONFIGURATION_FIELDS
        }
        expected_treatment = expected_treatment_configuration(variant)
        treatment_matches = actual_treatment == expected_treatment
        if not treatment_matches:
            configuration_mismatch_count += 1
            mismatch_counts["treatment_configuration"] += 1
            if len(configuration_mismatches) < 256:
                configuration_mismatches.append(
                    {
                        "kind": "endpoint_treatment_configuration",
                        "rep": rep,
                        "variant": variant,
                        "actual": actual_treatment,
                        "expected": expected_treatment,
                    }
                )
        key = (rep, variant)
        if key in results_by_rep_variant:
            raise RuntimeError(f"duplicate accepted result for {key}")
        results_by_rep_variant[key] = result
        result_artifacts[f"r{rep:02d}:{variant}"] = {
            "path": str(result_path),
            "sha256": actual_result_sha256,
            "accepted_attempt_recorded_sha256": recorded_result_sha256,
            "bytes": len(encoded),
            "result_integrity_audit": result_integrity_audit,
            "workload_configuration_fingerprint": workload_fingerprint,
            "endpoint_treatment_configuration": {
                "fields": list(ENDPOINT_TREATMENT_CONFIGURATION_FIELDS),
                "actual": actual_treatment,
                "expected": expected_treatment,
                "matches_expected": treatment_matches,
            },
        }

    configuration_by_rep: dict[str, Any] = {}
    for rep in range(1, REPETITIONS + 1):
        fingerprints = {
            variant: result_artifacts[f"r{rep:02d}:{variant}"][
                "workload_configuration_fingerprint"
            ]
            for variant in COMMITS
        }
        reference_sha256 = fingerprints["upstream"]["sha256"]
        exact = all(
            fingerprint["sha256"] == reference_sha256
            for fingerprint in fingerprints.values()
        )
        if not exact:
            configuration_mismatch_count += 1
            mismatch_counts["common_configuration"] += 1
            if len(configuration_mismatches) < 256:
                configuration_mismatches.append(
                    {
                        "kind": "workload_configuration_fingerprint",
                        "rep": rep,
                        "fingerprints": fingerprints,
                    }
                )
        configuration_by_rep[str(rep)] = {
            "exact_across_endpoints": exact,
            "fingerprints": fingerprints,
        }

    comparisons: dict[str, Any] = {}
    total_record_pairs = 0
    record_mismatch_count = 0
    mismatch_details: list[dict[str, Any]] = []
    for comparison_name, baseline_variant, candidate_variant in PAIRWISE_COMPARISONS:
        comparison_record_pairs = 0
        comparison_mismatches = 0
        by_rep: dict[str, Any] = {}
        for rep in range(1, REPETITIONS + 1):
            baseline_result = results_by_rep_variant[(rep, baseline_variant)]
            candidate_result = results_by_rep_variant[(rep, candidate_variant)]
            baseline_blocks = {
                int(block["concurrency"]): block
                for block in baseline_result["concurrency_blocks"]
            }
            candidate_blocks = {
                int(block["concurrency"]): block
                for block in candidate_result["concurrency_blocks"]
            }
            if baseline_blocks.keys() != candidate_blocks.keys():
                raise RuntimeError(
                    f"{comparison_name} rep {rep} concurrency sets differ"
                )
            rep_record_pairs = 0
            rep_mismatches = 0
            by_concurrency: dict[str, Any] = {}
            for concurrency in MEASURED_REQUESTS:
                phase_summaries: dict[str, Any] = {}
                for phase in ("warmup", "measured"):
                    baseline_records = {
                        int(record["request_index"]): record
                        for record in baseline_blocks[concurrency][phase]["records"]
                    }
                    candidate_records = {
                        int(record["request_index"]): record
                        for record in candidate_blocks[concurrency][phase]["records"]
                    }
                    if baseline_records.keys() != candidate_records.keys():
                        raise RuntimeError(
                            f"{comparison_name} rep {rep} c{concurrency} {phase} "
                            "request index sets differ"
                        )
                    phase_mismatches = 0
                    for request_index, baseline_record in baseline_records.items():
                        candidate_record = candidate_records[request_index]
                        differing_fields = []
                        identity_differences = []
                        for field in (
                            "phase",
                            "block_index",
                            "concurrency",
                            "request_index",
                            "global_request_index",
                            "video_index",
                            "video_path",
                            "video_file_uri",
                            "video_sha256",
                            "request_payload_sha256",
                            "status",
                        ):
                            if baseline_record.get(field) != candidate_record.get(
                                field
                            ):
                                differing_fields.append(field)
                                identity_differences.append(field)
                        baseline_response = baseline_record["response"]
                        candidate_response = candidate_record["response"]
                        prompt_exact = all(
                            baseline_response.get(field)
                            == candidate_response.get(field)
                            for field in (
                                "prompt_token_ids",
                                "prompt_token_ids_sha256",
                            )
                        )
                        completion_exact = all(
                            baseline_response.get(field)
                            == candidate_response.get(field)
                            for field in (
                                "completion_token_ids",
                                "completion_token_ids_sha256",
                            )
                        )
                        text_metadata_fields = (
                            "text",
                            "text_sha256",
                            "reasoning_content",
                            "reasoning_content_sha256",
                            "finish_reason",
                            "stop_reason",
                        )
                        text_exact = all(
                            baseline_response.get(field)
                            == candidate_response.get(field)
                            for field in ("text", "text_sha256")
                        )
                        reasoning_exact = all(
                            baseline_response.get(field)
                            == candidate_response.get(field)
                            for field in (
                                "reasoning_content",
                                "reasoning_content_sha256",
                            )
                        )
                        finish_reason_exact = baseline_response.get(
                            "finish_reason"
                        ) == candidate_response.get("finish_reason")
                        stop_reason_exact = baseline_response.get(
                            "stop_reason"
                        ) == candidate_response.get("stop_reason")
                        for field in (
                            "prompt_token_ids",
                            "prompt_token_ids_sha256",
                            "completion_token_ids",
                            "completion_token_ids_sha256",
                            *text_metadata_fields,
                        ):
                            if baseline_response.get(field) != candidate_response.get(
                                field
                            ):
                                differing_fields.append(f"response.{field}")
                        if identity_differences:
                            mismatch_counts["request_identity"] += 1
                        if not prompt_exact:
                            mismatch_counts["prompt_token_ids"] += 1
                        if not completion_exact:
                            mismatch_counts["completion_token_ids"] += 1
                        if not text_exact:
                            mismatch_counts["text_sha256"] += 1
                        if not reasoning_exact:
                            mismatch_counts["reasoning_content_sha256"] += 1
                        if not finish_reason_exact:
                            mismatch_counts["finish_reason"] += 1
                        if not stop_reason_exact:
                            mismatch_counts["stop_reason"] += 1
                        if differing_fields:
                            phase_mismatches += 1
                            if len(mismatch_details) < 256:
                                mismatch_details.append(
                                    {
                                        "comparison": comparison_name,
                                        "baseline": baseline_variant,
                                        "candidate": candidate_variant,
                                        "rep": rep,
                                        "concurrency": concurrency,
                                        "phase": phase,
                                        "request_index": request_index,
                                        "video_index": baseline_record.get(
                                            "video_index"
                                        ),
                                        "differing_fields": differing_fields,
                                        "baseline_prompt_token_ids_sha256": (
                                            baseline_response.get(
                                                "prompt_token_ids_sha256"
                                            )
                                        ),
                                        "candidate_prompt_token_ids_sha256": (
                                            candidate_response.get(
                                                "prompt_token_ids_sha256"
                                            )
                                        ),
                                        "baseline_completion_token_ids_sha256": (
                                            baseline_response.get(
                                                "completion_token_ids_sha256"
                                            )
                                        ),
                                        "candidate_completion_token_ids_sha256": (
                                            candidate_response.get(
                                                "completion_token_ids_sha256"
                                            )
                                        ),
                                        "baseline_text_sha256": baseline_response.get(
                                            "text_sha256"
                                        ),
                                        "candidate_text_sha256": candidate_response.get(
                                            "text_sha256"
                                        ),
                                    }
                                )
                    record_pairs = len(baseline_records)
                    total_record_pairs += record_pairs
                    comparison_record_pairs += record_pairs
                    rep_record_pairs += record_pairs
                    record_mismatch_count += phase_mismatches
                    comparison_mismatches += phase_mismatches
                    rep_mismatches += phase_mismatches
                    phase_summaries[phase] = {
                        "record_pairs": record_pairs,
                        "mismatches": phase_mismatches,
                        "exact": phase_mismatches == 0,
                    }
                by_concurrency[str(concurrency)] = phase_summaries
            by_rep[str(rep)] = {
                "record_pairs": rep_record_pairs,
                "mismatches": rep_mismatches,
                "exact": rep_mismatches == 0,
                "by_concurrency": by_concurrency,
            }
        comparisons[comparison_name] = {
            "baseline": baseline_variant,
            "candidate": candidate_variant,
            "accepted_result_artifacts": {
                str(rep): {
                    "baseline": result_artifacts[f"r{rep:02d}:{baseline_variant}"],
                    "candidate": result_artifacts[f"r{rep:02d}:{candidate_variant}"],
                }
                for rep in range(1, REPETITIONS + 1)
            },
            "record_pairs": comparison_record_pairs,
            "mismatches": comparison_mismatches,
            "exact": comparison_mismatches == 0,
            "by_rep": by_rep,
        }
    input_mismatches = sum(
        mismatch_counts[key]
        for key in (
            "common_configuration",
            "treatment_configuration",
            "request_identity",
            "prompt_token_ids",
        )
    )
    generation_mismatches = sum(
        mismatch_counts[key]
        for key in (
            "completion_token_ids",
            "text_sha256",
            "reasoning_content_sha256",
            "finish_reason",
            "stop_reason",
        )
    )
    status = (
        "failed_input_parity"
        if input_mismatches
        else "completion_or_text_mismatch" if generation_mismatches else "passed_exact"
    )
    expected_record_pairs = (
        len(PAIRWISE_COMPARISONS)
        * REPETITIONS
        * sum(
            WARMUP_REQUESTS[concurrency] + MEASURED_REQUESTS[concurrency]
            for concurrency in MEASURED_REQUESTS
        )
    )
    if total_record_pairs != expected_record_pairs:
        raise RuntimeError(
            "strict token audit response-pair count mismatch: "
            f"{total_record_pairs} != {expected_record_pairs}"
        )
    total_mismatches = sum(mismatch_counts.values())
    first_generation_divergence = next(
        (
            detail
            for detail in mismatch_details
            if any(
                field.startswith(
                    (
                        "response.completion_token_ids",
                        "response.text",
                        "response.reasoning_content",
                        "response.finish_reason",
                        "response.stop_reason",
                    )
                )
                for field in detail["differing_fields"]
            )
        ),
        None,
    )
    return {
        "schema": "pynv-endpoint-strict-token-parity-v1",
        "status": status,
        "policy": (
            "Post-collection only: bind every accepted result SHA and intended "
            "endpoint treatment, require a common workload/protocol fingerprint "
            "within each repetition, and compare exact request identity, full "
            "prompt/completion token ID arrays, text hashes/content, reasoning "
            "content, and stop metadata for every warmup and measured request. "
            "A mismatch fails collection and never retries a timing cell."
        ),
        "timing_cell_acceptance_independent": True,
        "match_seeking_retry_count": 0,
        "timing_retry_on_mismatch": False,
        "accepted_result_count": len(result_artifacts),
        "accepted_result_artifacts": result_artifacts,
        "configuration_audit": {
            "workload_fields": list(WORKLOAD_PARITY_CONFIGURATION_FIELDS),
            "treatment_fields": list(ENDPOINT_TREATMENT_CONFIGURATION_FIELDS),
            "by_rep": configuration_by_rep,
            "mismatches": configuration_mismatch_count,
            "mismatch_details": configuration_mismatches,
            "mismatch_details_truncated": configuration_mismatch_count
            > len(configuration_mismatches),
        },
        "expected_response_pair_count": expected_record_pairs,
        "compared_response_pair_count": total_record_pairs,
        "expected_individual_response_references": expected_record_pairs * 2,
        "record_pairs": total_record_pairs,
        "record_mismatches": record_mismatch_count,
        "mismatch_counts": mismatch_counts,
        "mismatches": total_mismatches,
        "mismatch_details": mismatch_details,
        "mismatch_details_truncated": record_mismatch_count > len(mismatch_details),
        "comparisons": comparisons,
        "first_generation_divergence": first_generation_divergence,
        "untimed_top20_logprob_diagnostic_required": (
            first_generation_divergence is not None
        ),
    }


def terminal_accepted_sidecar_audit(
    cells: Sequence[Mapping[str, Any]],
    *,
    results_root: Path,
    python: Path,
    harness: Path,
    harness_sha256: str,
    monitor_script: Path,
    source_root: Path,
    transformers_root: Path,
    hf_snapshot_root: Path,
    corpus: Path,
    videos: list[Path],
    port: int,
    conflicting_controller_roots: Sequence[Path],
    runtime_manifests: Mapping[str, Any],
    expected_runtime_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Reopen and fully validate every accepted timing-cell sidecar."""

    expected_cells = REPETITIONS * len(COMMITS)
    if len(cells) != expected_cells:
        raise RuntimeError(
            f"terminal sidecar audit cell count changed: {len(cells)} != {expected_cells}"
        )
    audited_cells = []
    seen: set[tuple[int, str]] = set()
    for cell in cells:
        rep = int(cell["rep"])
        variant = str(cell["variant"])
        key = (rep, variant)
        if key in seen or variant not in COMMITS:
            raise RuntimeError(f"duplicate/invalid terminal sidecar cell: {key}")
        seen.add(key)
        if cell.get("status") != "passed":
            raise RuntimeError(f"terminal sidecar cell is not passed: {key}")
        winning_attempt = int(cell["winning_attempt"])
        accepted_attempts = [
            attempt
            for attempt in cell.get("attempts", [])
            if attempt.get("accepted") is True
        ]
        matching_attempts = [
            attempt
            for attempt in accepted_attempts
            if int(attempt.get("attempt", -1)) == winning_attempt
        ]
        if len(accepted_attempts) != 1 or len(matching_attempts) != 1:
            raise RuntimeError(f"cell lacks one accepted winning attempt: {key}")
        attempt = matching_attempts[0]
        if (
            attempt.get("validation_status") != "accepted"
            or attempt.get("post_attempt_integrity_status") != "passed"
            or attempt.get("body_status") != "completed"
        ):
            raise RuntimeError(f"accepted attempt lifecycle evidence changed: {key}")
        stem = str(attempt["stem"])
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_before", {}),
            expected_label=f"{stem}:before_attempt",
            expected_manifests=runtime_manifests,
        )
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_after", {}),
            expected_label=f"{stem}:after_attempt",
            expected_manifests=runtime_manifests,
        )
        live_runtime_before = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_before", {}),
            label=f"{stem}:before_attempt",
        )
        live_runtime_after = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_after", {}),
            label=f"{stem}:after_attempt",
        )
        if live_runtime_before["sha256"] != live_runtime_after["sha256"]:
            raise RuntimeError(f"accepted live runtime artifacts changed: {key}")
        validate_recorded_source(
            attempt.get("source_after_attempt", {}), expected_variant=variant
        )
        validate_recorded_source(
            cell.get("source_after_cell", {}), expected_variant=variant
        )
        bound_paths = {
            path_key: validate_bound_file(
                attempt,
                path_key=path_key,
                sha256_key=sha_key,
                expected_parent=results_root,
                bytes_key=bytes_key,
            )
            for path_key, sha_key, bytes_key in (
                ("idle_gate", "idle_gate_sha256", "idle_gate_bytes"),
                ("output", "result_sha256", "result_bytes"),
                ("server_log", "server_log_sha256", "server_log_bytes"),
                ("monitor", "monitor_sha256", "monitor_bytes"),
                ("log", "log_sha256", "log_bytes"),
            )
        }
        if (
            Path(str(cell["output"])).resolve() != bound_paths["output"]
            or Path(str(cell["monitor"])).resolve() != bound_paths["monitor"]
            or Path(str(cell["server_log"])).resolve() != bound_paths["server_log"]
        ):
            raise RuntimeError(f"cell/accepted-attempt artifact paths differ: {key}")

        idle_report = json.loads(bound_paths["idle_gate"].read_text())
        validate_idle_gate_report(
            idle_report,
            required_idle_seconds=30.0,
            required_timeout_seconds=1800.0,
        )
        idle_sample_path = Path(
            str(attempt.get("idle_gate_sample_log_audit", {}).get("path", ""))
        ).resolve()
        if idle_sample_path.parent != results_root.resolve():
            raise RuntimeError(f"idle sample log escaped results root: {key}")
        idle_sample_audit = validate_jsonl_binding(
            idle_report,
            expected_suffix="-idle-gate.samples.jsonl",
            expected_path=idle_sample_path,
        )
        if (
            idle_sample_audit != attempt.get("idle_gate_sample_log_audit")
            or attempt.get("idle_gate_sample_log_sha256") != idle_sample_audit["sha256"]
            or attempt.get("idle_gate_sample_log_bytes") != idle_sample_audit["bytes"]
        ):
            raise RuntimeError(f"accepted idle sample-log evidence changed: {key}")

        result = json.loads(bound_paths["output"].read_text())
        monitor = json.loads(bound_paths["monitor"].read_text())
        if idle_report.get("device") != monitor.get("device"):
            raise RuntimeError(f"accepted idle/monitor device identity changed: {key}")
        monitor_sample_path = Path(
            str(attempt.get("monitor_sample_log_audit", {}).get("path", ""))
        ).resolve()
        if monitor_sample_path.parent != results_root.resolve():
            raise RuntimeError(f"monitor sample log escaped results root: {key}")
        monitor_sample_audit = validate_jsonl_binding(
            monitor,
            expected_suffix="-gpu-monitor.samples.jsonl",
            expected_path=monitor_sample_path,
        )
        if (
            monitor_sample_audit != attempt.get("monitor_sample_log_audit")
            or attempt.get("monitor_sample_log_sha256")
            != monitor_sample_audit["sha256"]
            or attempt.get("monitor_sample_log_bytes") != monitor_sample_audit["bytes"]
        ):
            raise RuntimeError(f"accepted monitor sample-log evidence changed: {key}")

        concurrency_order = [int(value) for value in cell["concurrency_order"]]
        expected_harness_command = build_harness_command(
            python=python,
            harness=harness,
            source_root=source_root,
            transformers_root=transformers_root,
            variant=variant,
            corpus=corpus,
            videos=videos,
            concurrencies=concurrency_order,
            port=port,
            result_path=bound_paths["output"],
        )
        validated_monitor, validated_monitor_sample_audit = validate_monitor_evidence(
            bound_paths["monitor"],
            expected_command=expected_harness_command,
            watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
            conflicting_controller_roots=conflicting_controller_roots,
        )
        if (
            validated_monitor != monitor
            or validated_monitor_sample_audit != monitor_sample_audit
            or int(attempt["returncode"]) != 0
        ):
            raise RuntimeError(f"accepted monitor evidence changed: {key}")
        expected_wrapper_command = build_monitored_command(
            python=python,
            monitor=monitor_script,
            output=bound_paths["monitor"],
            child_command=expected_harness_command,
            watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
            conflicting_controller_roots=conflicting_controller_roots,
        )
        if attempt.get("command") != expected_wrapper_command:
            raise RuntimeError(f"accepted wrapper command changed: {key}")
        validated_metrics = validate_result(
            result,
            monitor,
            commit=COMMITS[variant],
            variant=variant,
            concurrency_order=concurrency_order,
            harness=harness,
            harness_sha256=harness_sha256,
            expected_monitor_command=expected_harness_command,
            monitor_path=bound_paths["monitor"],
            corpus=corpus,
            transformers_root=transformers_root,
            hf_snapshot_root=hf_snapshot_root,
            source_root=source_root,
            server_log_path=bound_paths["server_log"],
        )
        if (
            validated_metrics != cell.get("validated_metrics")
            or validated_metrics["runtime_hardware_fingerprint"]["sha256"]
            != expected_runtime_fingerprint_sha256
        ):
            raise RuntimeError(f"accepted result validation changed: {key}")
        coverage = monitor_coverage_audit(result, monitor)
        if (
            not coverage["passed"]
            or coverage != attempt.get("monitor_coverage_audit")
            or coverage != cell.get("monitor_coverage_audit")
        ):
            raise RuntimeError(f"accepted monitor coverage changed: {key}")
        audited_cells.append(
            {
                "rep": rep,
                "variant": variant,
                "attempt": winning_attempt,
                "artifacts": {
                    name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for name, path in bound_paths.items()
                },
                "idle_sample_log_audit": idle_sample_audit,
                "monitor_sample_log_audit": monitor_sample_audit,
                "monitor_coverage_audit": coverage,
                "runtime_hardware_fingerprint_sha256": (
                    validated_metrics["runtime_hardware_fingerprint"]["sha256"]
                ),
                "live_runtime_artifact_manifest_sha256": live_runtime_after["sha256"],
            }
        )
    return {
        "schema": "pynv-accepted-cell-sidecar-audit-v1",
        "status": "passed",
        "accepted_cell_count": len(audited_cells),
        "policy": (
            "terminal full rehash/reparse of each accepted result, idle/monitor "
            "report and JSONL, controller log, full server log, source/runtime "
            "checkpoints, monotonic coverage, and hardware fingerprint"
        ),
        "cells": audited_cells,
    }


def paired_endpoint_summaries(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cells_by_rep_variant = {
        (int(cell["rep"]), str(cell["variant"])): cell for cell in cells
    }
    metric_getters = {
        "request_throughput_per_second": lambda block: float(
            block["request_throughput_per_second"]
        ),
        "generated_token_throughput_per_second": lambda block: float(
            block["generated_token_throughput_per_second"]
        ),
        "e2e_latency_p50_ms": lambda block: float(block["e2e_latency_ms"]["p50"]),
        "e2e_latency_p95_ms": lambda block: float(block["e2e_latency_ms"]["p95"]),
        "measured_window_peak_vram_mib": lambda block: float(
            block["measured_window_vram"]["peak_total_gpu_memory_used_mib"]
        ),
    }
    comparisons: dict[str, Any] = {}
    for comparison_name, baseline_variant, candidate_variant in PAIRWISE_COMPARISONS:
        by_concurrency: dict[str, Any] = {}
        for concurrency in MEASURED_REQUESTS:
            per_metric: dict[str, Any] = {}
            for metric, getter in metric_getters.items():
                pairs = []
                for rep in range(1, REPETITIONS + 1):
                    values: dict[str, float] = {}
                    for variant in (baseline_variant, candidate_variant):
                        cell = cells_by_rep_variant[(rep, variant)]
                        block = next(
                            item
                            for item in cell["validated_metrics"]["blocks"]
                            if int(item["concurrency"]) == concurrency
                        )
                        values[variant] = getter(block)
                    baseline = values[baseline_variant]
                    candidate = values[candidate_variant]
                    ratio = candidate / baseline
                    pairs.append(
                        {
                            "rep": rep,
                            "baseline": baseline,
                            "candidate": candidate,
                            "candidate_minus_baseline": candidate - baseline,
                            "candidate_over_baseline": ratio,
                            "candidate_percent_delta": (ratio - 1.0) * 100.0,
                        }
                    )
                baselines = [pair["baseline"] for pair in pairs]
                candidates = [pair["candidate"] for pair in pairs]
                deltas = [pair["candidate_minus_baseline"] for pair in pairs]
                percent_deltas = [pair["candidate_percent_delta"] for pair in pairs]
                log_ratios = [
                    math.log(pair["candidate_over_baseline"]) for pair in pairs
                ]
                baseline_mean = statistics.fmean(baselines)
                candidate_mean = statistics.fmean(candidates)
                delta_mean = statistics.fmean(deltas)
                delta_standard_error = statistics.stdev(deltas) / math.sqrt(REPETITIONS)
                percent_delta_mean = statistics.fmean(percent_deltas)
                percent_delta_standard_error = statistics.stdev(
                    percent_deltas
                ) / math.sqrt(REPETITIONS)
                log_ratio_mean = statistics.fmean(log_ratios)
                log_ratio_standard_error = statistics.stdev(log_ratios) / math.sqrt(
                    REPETITIONS
                )
                per_metric[metric] = {
                    "higher_is_better": metric.endswith("throughput_per_second"),
                    "pairs": pairs,
                    "baseline_mean": baseline_mean,
                    "baseline_median": statistics.median(baselines),
                    "baseline_sample_stdev": statistics.stdev(baselines),
                    "candidate_mean": candidate_mean,
                    "candidate_median": statistics.median(candidates),
                    "candidate_sample_stdev": statistics.stdev(candidates),
                    "ratio_of_means": candidate_mean / baseline_mean,
                    "ratio_of_means_percent_delta": (
                        (candidate_mean / baseline_mean) - 1.0
                    )
                    * 100.0,
                    "paired_difference_mean": delta_mean,
                    "paired_difference_sample_stdev": statistics.stdev(deltas),
                    "paired_difference_95_percent_t_ci": {
                        "degrees_of_freedom": REPETITIONS - 1,
                        "t_critical": T_CRITICAL_95_DF5,
                        "low": delta_mean - T_CRITICAL_95_DF5 * delta_standard_error,
                        "high": delta_mean + T_CRITICAL_95_DF5 * delta_standard_error,
                    },
                    "paired_percent_delta_mean": percent_delta_mean,
                    "paired_percent_delta_median": statistics.median(percent_deltas),
                    "paired_percent_delta_sample_stdev": statistics.stdev(
                        percent_deltas
                    ),
                    "paired_percent_delta_95_percent_t_ci": {
                        "degrees_of_freedom": REPETITIONS - 1,
                        "t_critical": T_CRITICAL_95_DF5,
                        "low": percent_delta_mean
                        - T_CRITICAL_95_DF5 * percent_delta_standard_error,
                        "high": percent_delta_mean
                        + T_CRITICAL_95_DF5 * percent_delta_standard_error,
                    },
                    "paired_geomean_candidate_over_baseline": math.exp(log_ratio_mean),
                    "paired_geomean_ratio_95_percent_t_ci": {
                        "degrees_of_freedom": REPETITIONS - 1,
                        "t_critical": T_CRITICAL_95_DF5,
                        "low": math.exp(
                            log_ratio_mean
                            - T_CRITICAL_95_DF5 * log_ratio_standard_error
                        ),
                        "high": math.exp(
                            log_ratio_mean
                            + T_CRITICAL_95_DF5 * log_ratio_standard_error
                        ),
                    },
                }
            by_concurrency[str(concurrency)] = per_metric
        comparisons[comparison_name] = {
            "baseline": baseline_variant,
            "candidate": candidate_variant,
            "pairing": "same repetition and concurrency",
            "repetitions": REPETITIONS,
            "by_concurrency": by_concurrency,
        }
    return {
        "design": "six-sequence three-treatment Williams design",
        "schedule": SCHEDULE,
        "balance": (
            "all six endpoint permutations and all six concurrency-order "
            "permutations; each endpoint and concurrency occupies each position "
            "twice"
        ),
        "comparisons": comparisons,
    }


def build_harness_command(
    *,
    python: Path,
    harness: Path,
    source_root: Path,
    transformers_root: Path,
    variant: str,
    corpus: Path,
    videos: Sequence[Path],
    concurrencies: Sequence[int],
    port: int,
    result_path: Path,
    variant_label: str | None = None,
    warmup_requests: Mapping[int, int] = WARMUP_REQUESTS,
    measured_requests: Mapping[int, int] = MEASURED_REQUESTS,
) -> list[str]:
    command = [
        str(python),
        str(harness),
        "--source-root",
        str(source_root),
        "--python",
        str(python),
        "--pythonpath-extra",
        str(transformers_root),
        "--variant",
        variant_label or variant,
        "--allowed-local-media-path",
        str(corpus),
        "--backend",
        "pynvvideocodec",
        "--backend-kwargs",
        json.dumps(
            variant_backend_kwargs(variant), separators=(",", ":"), sort_keys=True
        ),
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--prompt",
        PROMPT,
        "--frames",
        str(FRAMES),
        "--video-pixel-budget",
        "1024x576",
        "--warmup-requests",
        "1",
        "--warmup-requests-by-concurrency",
        json.dumps(warmup_requests, separators=(",", ":"), sort_keys=True),
        "--requests",
        "1",
        "--requests-by-concurrency",
        json.dumps(measured_requests, separators=(",", ":"), sort_keys=True),
        "--output-len",
        str(OUTPUT_LENGTH),
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "9216",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--mm-ipc-gpu-memory-gb",
        "2",
        "--kv-cache-memory-bytes",
        str(KV_CACHE_MEMORY_BYTES),
        "--settle-seconds",
        "1.0",
        "--request-timeout",
        "1200",
        "--startup-timeout",
        "600",
        "--shutdown-timeout",
        "60",
        "--port",
        str(port),
        "--output",
        str(result_path),
    ]
    for server_argument in variant_server_argv(variant):
        command.append(f"--server-arg={server_argument}")
    for video in videos:
        command.extend(["--video", str(video)])
    for concurrency in concurrencies:
        command.extend(["--concurrency", str(concurrency)])
    return command


def validate_preflight_attempt_command_and_paths(
    attempt: Mapping[str, Any],
    *,
    kind: str,
    preflight_root: Path,
    python: Path,
    monitor: Path,
    source_root: Path,
    transformers_root: Path,
    pixel_preflight: Path,
    harness: Path,
    corpus: Path,
    videos: Sequence[Path],
    port: int,
    conflicting_controller_roots: Sequence[Path],
) -> dict[str, Any]:
    attempt_index = attempt.get("attempt")
    if not isinstance(attempt_index, int) or not 1 <= attempt_index <= 20:
        raise RuntimeError("preflight attempt index is invalid")
    if kind == "pixel":
        stem = f"pixel-parity-a{attempt_index:02d}"
        child_command = [
            str(python),
            str(pixel_preflight),
            "--root",
            str(source_root),
            "--python",
            str(python),
            "--transformers-root",
            str(transformers_root),
            "--video",
            str(videos[0]),
            "--output",
            str(preflight_root / f"{stem}.json"),
        ]
        expected_paths = {
            "idle_gate": preflight_root / f"{stem}-idle-gate.json",
            "result": preflight_root / f"{stem}.json",
            "monitor": preflight_root / f"{stem}-gpu-monitor.json",
            "log": preflight_root / f"{stem}.log",
        }
        watchdog_pair = PIXEL_MONITOR_WATCHDOG_PAIR
    elif kind == "pilot":
        variant = attempt.get("variant")
        if variant not in COMMITS or attempt.get("commit") != COMMITS[variant]:
            raise RuntimeError("preflight pilot endpoint identity changed")
        stem = f"pilot-{variant}-c1-8-32-a{attempt_index:02d}"
        expected_paths = {
            "idle_gate": preflight_root / f"{stem}-idle-gate.json",
            "result": preflight_root / f"{stem}.json",
            "server_log": preflight_root / f"{stem}.server.log",
            "monitor": preflight_root / f"{stem}-gpu-monitor.json",
            "log": preflight_root / f"{stem}.log",
        }
        child_command = build_harness_command(
            python=python,
            harness=harness,
            source_root=source_root,
            transformers_root=transformers_root,
            variant=str(variant),
            variant_label=f"pilot-{variant}",
            corpus=corpus,
            videos=videos,
            concurrencies=[1, 8, 32],
            port=port,
            result_path=expected_paths["result"],
            warmup_requests={1: 8, 8: 8, 32: 32},
            measured_requests={1: 8, 8: 8, 32: 32},
        )
        watchdog_pair = TIMING_MONITOR_WATCHDOG_PAIR
    else:
        raise RuntimeError(f"unknown preflight attempt kind: {kind}")
    if attempt.get("stem") != stem:
        raise RuntimeError(f"preflight {kind} attempt stem changed")
    for path_key, expected_path in expected_paths.items():
        if Path(str(attempt.get(path_key, ""))).resolve() != expected_path.resolve():
            raise RuntimeError(f"preflight {kind} attempt {path_key} path changed")
    idle_sample_path = preflight_root / f"{stem}-idle-gate.samples.jsonl"
    monitor_sample_path = preflight_root / f"{stem}-gpu-monitor.samples.jsonl"
    for audit_key, expected_path in (
        ("idle_gate_sample_log_audit", idle_sample_path),
        ("monitor_sample_log_audit", monitor_sample_path),
    ):
        audit = attempt.get(audit_key)
        if (
            not isinstance(audit, Mapping)
            or Path(str(audit.get("path", ""))).resolve() != expected_path.resolve()
        ):
            raise RuntimeError(f"preflight {kind} attempt {audit_key} path changed")
    wrapper_command = build_monitored_command(
        python=python,
        monitor=monitor,
        output=expected_paths["monitor"],
        child_command=child_command,
        watchdog_pair=watchdog_pair,
        conflicting_controller_roots=conflicting_controller_roots,
    )
    if attempt.get("command") != child_command:
        raise RuntimeError(f"preflight {kind} child command changed")
    return {
        "kind": kind,
        "stem": stem,
        "paths": {key: str(value) for key, value in expected_paths.items()},
        "idle_sample_log": str(idle_sample_path),
        "monitor_sample_log": str(monitor_sample_path),
        "child_command": child_command,
        "wrapper_command": wrapper_command,
    }


def _campaign_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-tool", type=Path, required=True)
    parser.add_argument("--runtime-manifest-test", type=Path, required=True)
    parser.add_argument(
        "--transformers-overlay-manifest-jsonl", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-overlay-manifest-summary", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-package-manifest-jsonl", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-package-manifest-summary", type=Path, required=True
    )
    parser.add_argument("--hf-snapshot-root", type=Path, required=True)
    parser.add_argument("--hf-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--hf-manifest-summary", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--idle-gate", type=Path, required=True)
    parser.add_argument("--guard-helper", type=Path, required=True)
    parser.add_argument("--expected-harness-sha256", required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument(
        "--conflicting-controller-root", type=Path, action="append", required=True
    )
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--idle-timeout", type=float, default=1800.0)
    parser.add_argument("--max-attempts", type=int, default=20)
    args = parser.parse_args()

    validate_cell_idle_pair(args.idle_seconds, args.idle_timeout)

    args.root = args.root.resolve()
    args.python = args.python.absolute()
    args.transformers_root = args.transformers_root.resolve()
    args.runtime_manifest_tool = args.runtime_manifest_tool.resolve()
    args.runtime_manifest_test = args.runtime_manifest_test.resolve()
    args.transformers_overlay_manifest_jsonl = (
        args.transformers_overlay_manifest_jsonl.resolve()
    )
    args.transformers_overlay_manifest_summary = (
        args.transformers_overlay_manifest_summary.resolve()
    )
    args.transformers_package_manifest_jsonl = (
        args.transformers_package_manifest_jsonl.resolve()
    )
    args.transformers_package_manifest_summary = (
        args.transformers_package_manifest_summary.resolve()
    )
    args.hf_snapshot_root = args.hf_snapshot_root.resolve()
    args.hf_manifest_jsonl = args.hf_manifest_jsonl.resolve()
    args.hf_manifest_summary = args.hf_manifest_summary.resolve()
    args.corpus = args.corpus.resolve()
    args.results = args.results.resolve()
    args.harness = args.harness.resolve()
    args.monitor = args.monitor.resolve()
    args.idle_gate = args.idle_gate.resolve()
    args.guard_helper = args.guard_helper.resolve()
    args.preflight_summary = args.preflight_summary.resolve()
    args.conflicting_controller_root = sorted(
        {path.resolve(strict=False) for path in args.conflicting_controller_root},
        key=str,
    )
    if args.max_attempts != 20 or args.port != 18600:
        raise ValueError(
            "campaign control values must remain max_attempts=20 and port=18600"
        )
    if args.results.exists():
        raise FileExistsError(f"results path already exists: {args.results}")

    harness_sha256 = sha256_file(args.harness)
    if args.expected_harness_sha256 != CAMPAIGN_HARNESS_SHA256:
        raise RuntimeError(
            "expected campaign harness SHA argument does not match frozen SHA"
        )
    if harness_sha256 != CAMPAIGN_HARNESS_SHA256:
        raise RuntimeError(
            f"campaign harness hash mismatch: {harness_sha256} != "
            f"{CAMPAIGN_HARNESS_SHA256}"
        )
    if sha256_file(args.monitor) != GPU_MONITOR_SHA256:
        raise RuntimeError("GPU monitor hash mismatch")
    if sha256_file(args.idle_gate) != IDLE_GATE_SHA256:
        raise RuntimeError("idle gate hash mismatch")
    if sha256_file(args.guard_helper) != GUARD_HELPER_SHA256:
        raise RuntimeError("GPU guard helper hash mismatch")
    if sha256_file(args.runtime_manifest_tool) != RUNTIME_MANIFEST_TOOL_SHA256:
        raise RuntimeError("runtime-manifest tool hash mismatch")
    if sha256_file(args.runtime_manifest_test) != RUNTIME_MANIFEST_TEST_SHA256:
        raise RuntimeError("runtime-manifest test hash mismatch")
    if (
        args.monitor.parent != args.guard_helper.parent
        or args.idle_gate.parent != args.guard_helper.parent
    ):
        raise RuntimeError("monitor, idle gate, and guard helper must share one tag")
    videos = [args.corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    if not all(video.is_file() for video in videos):
        raise FileNotFoundError("eight-video corpus is incomplete")
    for video in videos:
        if video.stat().st_size != VIDEO_BYTES or sha256_file(video) != VIDEO_SHA256:
            raise RuntimeError(f"video corpus mismatch: {video}")
    runtime_manifest_validation_kwargs = {
        "python": args.python,
        "tool": args.runtime_manifest_tool,
        "transformers_root": args.transformers_root,
        "transformers_overlay_jsonl": args.transformers_overlay_manifest_jsonl,
        "transformers_overlay_summary": args.transformers_overlay_manifest_summary,
        "transformers_package_jsonl": args.transformers_package_manifest_jsonl,
        "transformers_package_summary": args.transformers_package_manifest_summary,
        "hf_snapshot_root": args.hf_snapshot_root,
        "hf_jsonl": args.hf_manifest_jsonl,
        "hf_summary": args.hf_manifest_summary,
    }
    runtime_manifests = validate_runtime_manifests(**runtime_manifest_validation_kwargs)
    initial_source = validate_source_at_any_endpoint(args.root)
    preflight = json.loads(args.preflight_summary.read_text())
    preflight_root = args.preflight_summary.parent
    if (
        args.preflight_summary.name != "pilot-summary.json"
        or preflight.get("schema") != "pynv-three-arm-persistent-preflight-v1"
        or preflight.get("evidence_namespace")
        != {
            "root": str(preflight_root),
            "summary": str(args.preflight_summary),
            "fresh_at_collection_start": True,
            "cross_namespace_sidecars_forbidden": True,
        }
    ):
        raise RuntimeError("endpoint preflight namespace/schema mismatch")
    preflight_publication_status = preflight.get("status")
    if preflight_publication_status not in {
        "passed",
        "timing_passed_completion_mismatch",
    }:
        raise RuntimeError("endpoint preflight summary is not publication-valid")
    if preflight.get("diagnostic_required") is not (
        preflight_publication_status == "timing_passed_completion_mismatch"
    ):
        raise RuntimeError("endpoint preflight diagnostic status mismatch")
    if preflight.get("runner_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("endpoint preflight used a different matrix runner")
    if preflight.get("harness_sha256") != CAMPAIGN_HARNESS_SHA256:
        raise RuntimeError("endpoint preflight used a different campaign harness")
    preflight_artifacts = preflight.get("artifacts", {})
    expected_preflight_artifacts = {
        "driver": (
            Path(__file__).resolve(),
            sha256_file(Path(__file__).resolve()),
        ),
        "harness": (args.harness, CAMPAIGN_HARNESS_SHA256),
        "monitor": (args.monitor, GPU_MONITOR_SHA256),
        "idle_gate": (args.idle_gate, IDLE_GATE_SHA256),
        "guard_helper": (args.guard_helper, GUARD_HELPER_SHA256),
        "runtime_manifest_tool": (
            args.runtime_manifest_tool,
            RUNTIME_MANIFEST_TOOL_SHA256,
        ),
        "runtime_manifest_test": (
            args.runtime_manifest_test,
            RUNTIME_MANIFEST_TEST_SHA256,
        ),
    }
    for artifact_key, (
        expected_path,
        expected_sha256,
    ) in expected_preflight_artifacts.items():
        artifact = preflight_artifacts.get(artifact_key, {})
        if (
            Path(str(artifact.get("path", ""))).resolve() != expected_path.resolve()
            or artifact.get("sha256") != expected_sha256
            or sha256_file(expected_path) != expected_sha256
        ):
            raise RuntimeError(f"endpoint preflight {artifact_key} provenance mismatch")
    expected_preflight_programs = {
        "pilot_runner": (
            Path(__file__).resolve().with_name(PREFLIGHT_RUNNER_FILENAME),
            PREFLIGHT_RUNNER_SHA256,
        ),
        "pixel_preflight_artifact": (
            Path(__file__).resolve().with_name(PIXEL_PREFLIGHT_FILENAME),
            PIXEL_PREFLIGHT_SHA256,
        ),
    }
    for artifact_key, (
        expected_path,
        expected_sha256,
    ) in expected_preflight_programs.items():
        artifact = preflight.get(artifact_key, {})
        recorded_path = Path(str(artifact.get("path", ""))).resolve()
        if (
            recorded_path != expected_path
            or artifact.get("sha256") != expected_sha256
            or not expected_path.is_file()
            or sha256_file(expected_path) != expected_sha256
        ):
            raise RuntimeError(
                f"endpoint preflight {artifact_key} frozen provenance mismatch"
            )
    if not all(preflight["pixel_preflight"]["parity"].values()):
        raise RuntimeError("endpoint pixel preflight parity is not fully passed")
    if not preflight["pixel_preflight"]["model_visible_comparison"]["allclose"]:
        raise RuntimeError("endpoint model-visible preflight is not allclose")
    if not preflight["token_parity"]["all_prompt_token_ids_exact"]:
        raise RuntimeError("endpoint preflight prompt token parity failed")
    generation_parity_exact = (
        preflight["token_parity"]["all_completion_token_ids_exact"]
        and preflight["token_parity"]["all_text_exact"]
    )
    if generation_parity_exact != (preflight_publication_status == "passed"):
        raise RuntimeError("endpoint preflight generation-parity status mismatch")
    expected_preflight_configuration = {
        "model": MODEL,
        "revision": REVISION,
        "frames": FRAMES,
        "pixel_budget_per_frame": list(PIXEL_BUDGET),
        "max_pixels_total": TOTAL_MAX_PIXELS,
        "warmups": {"1": 8, "8": 8, "32": 32},
        "measured": {"1": 8, "8": 8, "32": 32},
        "concurrencies": [1, 8, 32],
        "max_num_seqs": MAX_NUM_SEQS,
    }
    if preflight["configuration"] != expected_preflight_configuration:
        raise RuntimeError("endpoint preflight configuration mismatch")
    validate_recorded_source(preflight.get("initial_source", {}))
    for attempt in preflight.get("pixel_attempts", []):
        validate_recorded_source(attempt.get("source_after_attempt", {}))
    for attempt in preflight.get("pilot_attempts", []):
        validate_recorded_source(
            attempt.get("source_after_attempt", {}),
            expected_variant=str(attempt.get("variant", "")),
        )
    validate_recorded_source(
        preflight.get("terminal_source_revalidation", {}),
        expected_variant="pr-head",
    )
    preflight_variants = {
        (pilot["variant"], pilot["commit"]) for pilot in preflight["pilots"]
    }
    if len(preflight["pilots"]) != len(COMMITS) or preflight_variants != set(
        COMMITS.items()
    ):
        raise RuntimeError("endpoint preflight variants/commits mismatch")
    ingress_audit = validate_idle_gate_evidence(
        preflight.get("ingress_idle_gate", {}),
        expected_seconds=INGRESS_IDLE_SECONDS,
        expected_timeout=INGRESS_IDLE_TIMEOUT_SECONDS,
        conflicting_controller_roots=args.conflicting_controller_root,
    )
    outer_quiet_path = Path(ingress_audit["report_path"])
    if outer_quiet_path != preflight_root / "preflight-ingress-idle-gate.json":
        raise RuntimeError("endpoint preflight ingress idle-gate namespace mismatch")
    outer_quiet_report = json.loads(outer_quiet_path.read_text())
    pixel_attempts = preflight.get("pixel_attempts", [])
    if not pixel_attempts:
        raise RuntimeError("endpoint preflight lacks a post-quiet pixel attempt")
    if [int(attempt.get("attempt", -1)) for attempt in pixel_attempts] != list(
        range(1, len(pixel_attempts) + 1)
    ):
        raise RuntimeError("endpoint preflight pixel attempts are not contiguous")
    first_pixel_idle_path = Path(str(pixel_attempts[0].get("idle_gate", ""))).resolve()
    if (
        first_pixel_idle_path.parent != args.preflight_summary.parent
        or not first_pixel_idle_path.is_file()
        or pixel_attempts[0].get("idle_gate_sha256")
        != sha256_file(first_pixel_idle_path)
    ):
        raise RuntimeError("endpoint first pixel idle-gate binding mismatch")
    first_pixel_idle_report = json.loads(first_pixel_idle_path.read_text())
    if int(first_pixel_idle_report.get("started_time_ns", 0)) < int(
        outer_quiet_report["finished_time_ns"]
    ):
        raise RuntimeError(
            "pixel/pilot work began before the outer quiet gate finished"
        )
    accepted_pixel_attempts = [
        attempt for attempt in pixel_attempts if attempt.get("accepted") is True
    ]
    if len(accepted_pixel_attempts) != 1:
        raise RuntimeError(
            "endpoint preflight lacks exactly one accepted pixel attempt"
        )
    accepted_pixel = accepted_pixel_attempts[0]
    if accepted_pixel is not pixel_attempts[-1]:
        raise RuntimeError("accepted pixel attempt is not terminal")
    accepted_pixel_result_path: Path | None = None
    for path_key, sha256_key in (
        ("result", "result_sha256"),
        ("monitor", "monitor_sha256"),
        ("log", "log_sha256"),
    ):
        bound_path = validate_bound_file(
            accepted_pixel,
            path_key=path_key,
            sha256_key=sha256_key,
            expected_parent=args.preflight_summary.parent,
        )
        if path_key == "result":
            accepted_pixel_result_path = bound_path
    assert accepted_pixel_result_path is not None
    accepted_pixel_monitor_path = Path(str(accepted_pixel["monitor"])).resolve()
    accepted_pixel_monitor = json.loads(accepted_pixel_monitor_path.read_text())
    accepted_pixel_idle_path = Path(str(accepted_pixel["idle_gate"])).resolve()
    accepted_pixel_idle_report = json.loads(accepted_pixel_idle_path.read_text())
    if accepted_pixel_idle_report.get("device") != outer_quiet_report.get("device"):
        raise RuntimeError("pixel idle/outer quiet physical device identity changed")
    pixel_preflight_path = Path(
        str(preflight["pixel_preflight_artifact"]["path"])
    ).resolve()
    preflight_attempt_layout_audits = [
        *(
            validate_preflight_attempt_command_and_paths(
                attempt,
                kind="pixel",
                preflight_root=args.preflight_summary.parent,
                python=args.python,
                monitor=args.monitor,
                source_root=args.root,
                transformers_root=args.transformers_root,
                pixel_preflight=pixel_preflight_path,
                harness=args.harness,
                corpus=args.corpus,
                videos=videos,
                port=18600,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
            for attempt in preflight.get("pixel_attempts", [])
        ),
        *(
            validate_preflight_attempt_command_and_paths(
                attempt,
                kind="pilot",
                preflight_root=args.preflight_summary.parent,
                python=args.python,
                monitor=args.monitor,
                source_root=args.root,
                transformers_root=args.transformers_root,
                pixel_preflight=pixel_preflight_path,
                harness=args.harness,
                corpus=args.corpus,
                videos=videos,
                port=18600,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
            for attempt in preflight.get("pilot_attempts", [])
        ),
    ]
    ordered_preflight_attempts = [
        *preflight.get("pixel_attempts", []),
        *preflight.get("pilot_attempts", []),
    ]
    if len(ordered_preflight_attempts) != len(preflight_attempt_layout_audits):
        raise RuntimeError("endpoint preflight attempt/layout count mismatch")
    for attempt, layout in zip(
        ordered_preflight_attempts, preflight_attempt_layout_audits, strict=True
    ):
        stem = layout["stem"]
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_before", {}),
            expected_label=f"{stem}:before_attempt",
            expected_manifests=runtime_manifests,
        )
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_after", {}),
            expected_label=f"{stem}:after_attempt",
            expected_manifests=runtime_manifests,
        )
        live_before = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_before", {}),
            label=f"{stem}:before_attempt",
        )
        live_after = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_after", {}),
            label=f"{stem}:after_attempt",
        )
        if live_before != live_after:
            raise RuntimeError(f"endpoint preflight live runtime changed: {stem}")
        validate_recorded_source(
            attempt.get("source_after_attempt", {}),
            expected_variant=(
                str(attempt["variant"]) if layout["kind"] == "pilot" else None
            ),
        )
        idle_audit = validate_idle_gate_evidence(
            attempt.get("idle_evidence", {}),
            expected_seconds=CELL_IDLE_SECONDS,
            expected_timeout=CELL_IDLE_TIMEOUT_SECONDS,
            conflicting_controller_roots=args.conflicting_controller_root,
        )
        if Path(idle_audit["report_path"]).resolve() != Path(
            layout["paths"]["idle_gate"]
        ).resolve() or idle_audit.get("sample_log_audit") != attempt.get(
            "idle_gate_sample_log_audit"
        ):
            raise RuntimeError(f"endpoint preflight idle evidence changed: {stem}")
        for artifact_name, artifact_path_text in layout["paths"].items():
            artifact_path = Path(artifact_path_text)
            if (
                artifact_name in {"idle_gate", "monitor", "log"}
                and not artifact_path.is_file()
            ):
                raise RuntimeError(
                    f"endpoint preflight required artifact missing: {artifact_path}"
                )
            if artifact_path.is_file() and (
                attempt.get(f"{artifact_name}_bytes") != artifact_path.stat().st_size
                or attempt.get(f"{artifact_name}_sha256") != sha256_file(artifact_path)
            ):
                raise RuntimeError(
                    f"endpoint preflight artifact binding changed: {artifact_path}"
                )
        monitor_path = Path(layout["paths"]["monitor"])
        monitor_report = json.loads(monitor_path.read_text())
        monitor_sample_audit = validate_jsonl_binding(
            monitor_report,
            expected_path=Path(layout["monitor_sample_log"]),
            expected_suffix="-gpu-monitor.samples.jsonl",
        )
        if monitor_sample_audit != attempt.get("monitor_sample_log_audit"):
            raise RuntimeError(
                f"endpoint preflight monitor sample binding changed: {stem}"
            )
        if attempt.get("accepted") is True:
            if attempt.get("validation_status") != "accepted":
                raise RuntimeError(f"accepted preflight status changed: {stem}")
            continue
        if attempt.get("validation_status") != "excluded_contamination":
            raise RuntimeError(
                f"non-winning preflight attempt is not contamination: {stem}"
            )
        contamination_audit = validate_contamination_retry_evidence(
            wrapper_returncode=attempt.get("returncode"),
            report_path=monitor_path,
            expected_wrapper_command=layout["wrapper_command"],
            expected_child_command=layout["child_command"],
            watchdog_pair=(
                PIXEL_MONITOR_WATCHDOG_PAIR
                if layout["kind"] == "pixel"
                else TIMING_MONITOR_WATCHDOG_PAIR
            ),
            conflicting_controller_roots=args.conflicting_controller_root,
        )
        if attempt.get("contamination_retry_evidence") != contamination_audit:
            raise RuntimeError(
                f"endpoint preflight contamination evidence changed: {stem}"
            )
    expected_pixel_child_command = [
        str(args.python),
        str(pixel_preflight_path),
        "--root",
        str(args.root),
        "--python",
        str(args.python),
        "--transformers-root",
        str(args.transformers_root),
        "--video",
        str(videos[0]),
        "--output",
        str(accepted_pixel_result_path),
    ]
    preflight_pixel_monitor_audit = validate_pixel_monitor_report(
        accepted_pixel_monitor,
        monitor_path=accepted_pixel_monitor_path,
        expected_child_command=expected_pixel_child_command,
        wrapper_returncode=int(accepted_pixel["returncode"]),
        expected_device=outer_quiet_report["device"],
        watchdog_pair=PIXEL_MONITOR_WATCHDOG_PAIR,
        conflicting_controller_roots=args.conflicting_controller_root,
    )
    accepted_pixel_result = json.loads(accepted_pixel_result_path.read_text())
    preflight_pixel_deep_audit = validate_pixel_preflight_result(
        accepted_pixel_result,
        result_path=accepted_pixel_result_path,
        source_root=args.root,
        transformers_root=args.transformers_root,
        expected_video=videos[0],
        expected_gpu_name=str(outer_quiet_report["device"]["name"]),
    )
    if preflight["pixel_preflight"].get("parity") != accepted_pixel_result.get(
        "parity"
    ) or preflight["pixel_preflight"].get(
        "model_visible_comparison"
    ) != accepted_pixel_result.get(
        "model_visible_comparison"
    ):
        raise RuntimeError("endpoint accepted pixel result summary differs")
    if any(
        preflight["pixel_preflight"].get(key) != accepted_pixel.get(key)
        for key in (
            "attempt",
            "idle_gate",
            "result",
            "result_sha256",
            "monitor",
            "monitor_sha256",
            "log",
            "log_sha256",
        )
    ):
        raise RuntimeError("endpoint accepted pixel artifact bindings differ")

    accepted_pilot_attempts = [
        attempt
        for attempt in preflight.get("pilot_attempts", [])
        if attempt.get("accepted") is True
    ]
    if len(accepted_pilot_attempts) != len(COMMITS):
        raise RuntimeError("endpoint preflight accepted pilot-attempt count mismatch")
    preflight_pilot_deep_audits = []
    preflight_pilot_results = {}
    preflight_pilot_runtime_fingerprints = {}
    for pilot_record in preflight["pilots"]:
        variant_attempts = [
            attempt
            for attempt in preflight.get("pilot_attempts", [])
            if attempt.get("variant") == pilot_record["variant"]
        ]
        if [int(attempt.get("attempt", -1)) for attempt in variant_attempts] != list(
            range(1, len(variant_attempts) + 1)
        ):
            raise RuntimeError("endpoint pilot attempts are not contiguous")
        matching_attempts = [
            attempt
            for attempt in accepted_pilot_attempts
            if attempt.get("variant") == pilot_record["variant"]
            and attempt.get("commit") == pilot_record["commit"]
            and attempt.get("attempt") == pilot_record["attempt"]
        ]
        if len(matching_attempts) != 1:
            raise RuntimeError("endpoint pilot lacks one matching accepted attempt")
        accepted_attempt = matching_attempts[0]
        if accepted_attempt is not variant_attempts[-1]:
            raise RuntimeError("accepted endpoint pilot attempt is not terminal")
        accepted_paths = {}
        for path_key, sha256_key in (
            ("result", "result_sha256"),
            ("monitor", "monitor_sha256"),
            ("log", "log_sha256"),
        ):
            accepted_path = validate_bound_file(
                accepted_attempt,
                path_key=path_key,
                sha256_key=sha256_key,
                expected_parent=args.preflight_summary.parent,
            )
            if pilot_record.get(path_key) != str(accepted_path) or pilot_record.get(
                sha256_key
            ) != accepted_attempt.get(sha256_key):
                raise RuntimeError("endpoint pilot artifact bindings differ")
            accepted_paths[path_key] = accepted_path
        accepted_server_log = validate_bound_file(
            accepted_attempt,
            path_key="server_log",
            sha256_key="server_log_sha256",
            expected_parent=args.preflight_summary.parent,
        )
        bound_server_log = validate_bound_file(
            pilot_record["full_server_log"],
            path_key="path",
            sha256_key="sha256",
            bytes_key="bytes",
            expected_parent=args.preflight_summary.parent,
        )
        if (
            bound_server_log != accepted_server_log
            or pilot_record["full_server_log"]["sha256"]
            != accepted_attempt["server_log_sha256"]
        ):
            raise RuntimeError("endpoint pilot server-log bindings differ")
        expected_pilot_harness_command = build_harness_command(
            python=args.python,
            harness=args.harness,
            source_root=args.root,
            transformers_root=args.transformers_root,
            variant=pilot_record["variant"],
            variant_label=f"pilot-{pilot_record['variant']}",
            corpus=args.corpus,
            videos=videos,
            concurrencies=[1, 8, 32],
            port=18600,
            result_path=accepted_paths["result"],
            warmup_requests={1: 8, 8: 8, 32: 32},
            measured_requests={1: 8, 8: 8, 32: 32},
        )
        pilot_result = json.loads(accepted_paths["result"].read_text())
        pilot_monitor = json.loads(accepted_paths["monitor"].read_text())
        pilot_monitor_sample_audit = validate_jsonl_binding(
            pilot_monitor,
            expected_suffix="-gpu-monitor.samples.jsonl",
            expected_path=Path(
                str(accepted_attempt["monitor_sample_log_audit"]["path"])
            ),
        )
        validated_pilot_monitor, validated_pilot_sample_audit = (
            validate_monitor_evidence(
                accepted_paths["monitor"],
                expected_command=expected_pilot_harness_command,
                watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
        )
        if (
            validated_pilot_monitor != pilot_monitor
            or validated_pilot_sample_audit != pilot_monitor_sample_audit
            or int(accepted_attempt["returncode"]) != 0
        ):
            raise RuntimeError("accepted endpoint pilot monitor evidence changed")
        validated_pilot = validate_result(
            pilot_result,
            pilot_monitor,
            commit=pilot_record["commit"],
            variant=pilot_record["variant"],
            concurrency_order=[1, 8, 32],
            harness=args.harness,
            harness_sha256=CAMPAIGN_HARNESS_SHA256,
            expected_monitor_command=expected_pilot_harness_command,
            monitor_path=accepted_paths["monitor"],
            corpus=args.corpus,
            transformers_root=args.transformers_root,
            hf_snapshot_root=args.hf_snapshot_root,
            source_root=args.root,
            server_log_path=accepted_server_log,
            warmup_requests={1: 8, 8: 8, 32: 32},
            measured_requests={1: 8, 8: 8, 32: 32},
            result_variant_label=f"pilot-{pilot_record['variant']}",
        )
        pilot_coverage = monitor_coverage_audit(pilot_result, pilot_monitor)
        if (
            not pilot_coverage["passed"]
            or pilot_record.get("blocks") != validated_pilot["blocks"]
            or pilot_record.get("monitor_coverage_audit") != pilot_coverage
            or pilot_record.get("full_server_log") != validated_pilot["full_server_log"]
            or pilot_record.get("runtime_hardware_fingerprint")
            != validated_pilot["runtime_hardware_fingerprint"]
        ):
            raise RuntimeError("endpoint pilot deep validation changed")
        preflight_pilot_deep_audits.append(
            {
                "variant": pilot_record["variant"],
                "commit": pilot_record["commit"],
                "result_integrity_audit": validated_pilot["result_integrity_audit"],
                "monitor_coverage_audit": pilot_coverage,
            }
        )
        preflight_pilot_results[pilot_record["variant"]] = pilot_result
        preflight_pilot_runtime_fingerprints[pilot_record["variant"]] = validated_pilot[
            "runtime_hardware_fingerprint"
        ]
    preflight_pilot_parity_audit = validate_three_way_pilot_parity(
        preflight_pilot_results
    )
    expected_preflight_pilot_status = (
        "passed_exact"
        if preflight_publication_status == "passed"
        else "completion_or_text_mismatch"
    )
    if preflight_pilot_parity_audit["status"] != expected_preflight_pilot_status:
        raise RuntimeError("endpoint pilot recomputed parity status changed")
    fingerprint_hashes = {
        fingerprint["sha256"]
        for fingerprint in preflight_pilot_runtime_fingerprints.values()
    }
    if len(fingerprint_hashes) != 1:
        raise RuntimeError(
            "endpoint preflight runtime/hardware fingerprints differ: "
            f"{preflight_pilot_runtime_fingerprints}"
        )
    reference_fingerprint = preflight_pilot_runtime_fingerprints["upstream"]
    runtime_hardware_fingerprint_contract = {
        "status": "passed",
        "schema": reference_fingerprint["schema"],
        "sha256": reference_fingerprint["sha256"],
        "canonical": reference_fingerprint["canonical"],
        "variants": {
            variant: fingerprint["sha256"]
            for variant, fingerprint in preflight_pilot_runtime_fingerprints.items()
        },
    }
    if (
        preflight.get("runtime_hardware_fingerprint_contract")
        != runtime_hardware_fingerprint_contract
    ):
        raise RuntimeError(
            "endpoint preflight runtime/hardware fingerprint contract changed"
        )
    for accepted_attempt in [accepted_pixel, *accepted_pilot_attempts]:
        idle_path = validate_bound_file(
            accepted_attempt,
            path_key="idle_gate",
            sha256_key="idle_gate_sha256",
            expected_parent=args.preflight_summary.parent,
        )
        idle_report = json.loads(idle_path.read_text())
        validate_idle_gate_report(
            idle_report,
            required_idle_seconds=30.0,
            required_timeout_seconds=1800.0,
        )
        idle_sample_path = Path(
            str(accepted_attempt["idle_gate_sample_log_audit"]["path"])
        ).resolve()
        if idle_sample_path.parent != args.preflight_summary.parent:
            raise RuntimeError("endpoint preflight idle sample-log path escaped tag")
        idle_sample_audit = validate_jsonl_binding(
            idle_report,
            expected_suffix="-idle-gate.samples.jsonl",
            expected_path=idle_sample_path,
        )
        if idle_sample_audit != accepted_attempt["idle_gate_sample_log_audit"]:
            raise RuntimeError("endpoint preflight idle sample-log audit changed")
        monitor_path = Path(str(accepted_attempt["monitor"])).resolve()
        monitor_report = json.loads(monitor_path.read_text())
        monitor_sample_path = Path(
            str(accepted_attempt["monitor_sample_log_audit"]["path"])
        ).resolve()
        if monitor_sample_path.parent != args.preflight_summary.parent:
            raise RuntimeError("endpoint preflight monitor sample-log path escaped tag")
        monitor_sample_audit = validate_jsonl_binding(
            monitor_report,
            expected_suffix="-gpu-monitor.samples.jsonl",
            expected_path=monitor_sample_path,
        )
        if monitor_sample_audit != accepted_attempt["monitor_sample_log_audit"]:
            raise RuntimeError("endpoint preflight monitor sample-log audit changed")

    if preflight.get("runtime_manifests") != runtime_manifests:
        raise RuntimeError(
            "endpoint preflight runtime-manifest evidence differs from matrix "
            "startup revalidation"
        )
    preflight_checkpoints = preflight.get("runtime_manifest_checkpoints", [])
    if len(preflight_checkpoints) != 2:
        raise RuntimeError("endpoint preflight terminal checkpoint count mismatch")
    validate_runtime_manifest_checkpoint(
        preflight_checkpoints[0],
        expected_label="preflight_start",
        expected_manifests=runtime_manifests,
    )
    validate_runtime_manifest_checkpoint(
        preflight_checkpoints[1],
        expected_label="preflight_end",
        expected_manifests=runtime_manifests,
    )
    for attempt in [
        *preflight.get("pixel_attempts", []),
        *preflight.get("pilot_attempts", []),
    ]:
        attempt_stem = Path(str(attempt.get("result", ""))).stem
        if not attempt_stem:
            raise RuntimeError("endpoint preflight attempt lacks an artifact stem")
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_before", {}),
            expected_label=f"{attempt_stem}:before_attempt",
            expected_manifests=runtime_manifests,
        )
        validate_runtime_manifest_checkpoint(
            attempt.get("runtime_manifest_after", {}),
            expected_label=f"{attempt_stem}:after_attempt",
            expected_manifests=runtime_manifests,
        )
        live_before = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_before", {}),
            label=f"{attempt_stem}:before_attempt",
        )
        live_after = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_after", {}),
            label=f"{attempt_stem}:after_attempt",
        )
        if live_before["sha256"] != live_after["sha256"]:
            raise RuntimeError(
                f"endpoint preflight live runtime artifacts changed: {attempt_stem}"
            )

    terminal_live_runtime = revalidate_live_runtime_artifact_manifest_binding(
        preflight.get("terminal_live_runtime_artifact_revalidation", {}),
        label="preflight terminal",
    )
    if not preflight.get("pilot_attempts") or terminal_live_runtime != preflight[
        "pilot_attempts"
    ][-1].get("live_runtime_artifacts_after"):
        raise RuntimeError("endpoint preflight terminal live runtime mismatch")

    args.results.mkdir(parents=True)

    runner = Path(__file__).resolve()
    artifacts = {
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "campaign_harness": {
            "path": str(args.harness),
            "sha256": harness_sha256,
        },
        "gpu_monitor": {
            "path": str(args.monitor),
            "sha256": sha256_file(args.monitor),
        },
        "idle_gate": {
            "path": str(args.idle_gate),
            "sha256": sha256_file(args.idle_gate),
        },
        "guard_helper": {
            "path": str(args.guard_helper),
            "sha256": sha256_file(args.guard_helper),
        },
        "preflight_summary": {
            "path": str(args.preflight_summary),
            "sha256": sha256_file(args.preflight_summary),
        },
        "runtime_manifest_tool": {
            "path": str(args.runtime_manifest_tool),
            "sha256": sha256_file(args.runtime_manifest_tool),
        },
        "runtime_manifest_test": {
            "path": str(args.runtime_manifest_test),
            "sha256": sha256_file(args.runtime_manifest_test),
        },
        "transformers_overlay_manifest": runtime_manifests["transformers_overlay"],
        "transformers_package_manifest": runtime_manifests["transformers_package"],
        "hf_snapshot_manifest": runtime_manifests["hf_snapshot"],
    }
    manifest: dict[str, Any] = {
        "status": "collection_running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(args.root),
        "initial_source": initial_source,
        "results_root": str(args.results),
        "commits": COMMITS,
        "schedule": SCHEDULE,
        "artifacts": artifacts,
        "configuration": {
            "model": MODEL,
            "revision": REVISION,
            "protocol": (
                "pooled persistent non-streaming HTTP/1.1 chat/completions; "
                "E2E latency only; no TTFT"
            ),
            "frames": FRAMES,
            "pixel_budget": list(PIXEL_BUDGET),
            "max_pixels_per_sampled_frame": PIXEL_BUDGET[0] * PIXEL_BUDGET[1],
            "max_pixels_total": TOTAL_MAX_PIXELS,
            "warmup_requests_by_concurrency": WARMUP_REQUESTS,
            "measured_requests_by_concurrency": MEASURED_REQUESTS,
            "warmup_waves": 3,
            "videos": [str(video) for video in videos],
            "output_len": OUTPUT_LENGTH,
            "expected_prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "max_model_len": 32768,
            "max_num_batched_tokens": 9216,
            "max_num_seqs": MAX_NUM_SEQS,
            "mm_ipc_gpu_memory_gb": 2,
            "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
            "settle_seconds": 1.0,
            "backend": "pynvvideocodec",
            "per_variant_configuration": {
                variant: {
                    "commit": commit,
                    "backend_kwargs": variant_backend_kwargs(variant),
                    "extra_server_argv": variant_server_argv(variant),
                }
                for variant, commit in COMMITS.items()
            },
            "strict_token_audit_policy": (
                "required after all cells; any paired warmup/measured token or text "
                "difference fails collection"
            ),
            "server_lifecycle": (
                "one fresh vLLM server per endpoint-by-repetition cell; that server "
                "spans the cell's three independently warmed concurrency blocks"
            ),
            "retry_policy": (
                "cell attempts retry only after telemetry-proven foreign "
                "contamination; HTTP requests never retry and parity mismatches "
                "never trigger timing retries"
            ),
            "cuda_visible_devices": "0",
            "physical_nvml_device_index": 0,
            "cell_watchdog_seconds": 3600,
            "cell_watchdog_grace_seconds": 120,
            "monitor_coverage_maximum_gap_seconds": (
                MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS
            ),
        },
        "cells": [],
        "runtime_manifest_checkpoints": [
            {
                "status": "passed",
                "label": "campaign_start",
                "validated_utc": datetime.now(timezone.utc).isoformat(),
                "evidence_sha256": sha256_json(runtime_manifests),
            }
        ],
        "preflight_evidence_audit": {
            "pixel": preflight_pixel_deep_audit,
            "pixel_monitor": preflight_pixel_monitor_audit,
            "pilots": preflight_pilot_deep_audits,
            "pilot_token_parity": preflight_pilot_parity_audit,
            "attempt_command_and_path_audits": preflight_attempt_layout_audits,
            "runtime_hardware_fingerprint_contract": (
                runtime_hardware_fingerprint_contract
            ),
        },
    }
    manifest_path = args.results / "matrix-manifest.json"
    write_json(manifest_path, manifest)

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            **huggingface_cache_environment(args.hf_snapshot_root),
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    position = 0
    for rep, concurrencies, variants in SCHEDULE:
        for variant in variants:
            position += 1
            commit = COMMITS[variant]
            checked_out = run(
                [
                    "git",
                    "-C",
                    str(args.root),
                    "checkout",
                    "--quiet",
                    "--detach",
                    commit,
                ],
                capture_output=True,
            )
            if checked_out.returncode:
                raise RuntimeError(checked_out.stderr or checked_out.stdout)
            source = validate_source(args.root, commit, variant=variant)
            cell: dict[str, Any] = {
                "rep": rep,
                "position": position,
                "variant": variant,
                "commit": commit,
                "concurrency_order": concurrencies,
                "source": source,
                "status": "running",
                "attempts": [],
            }
            manifest["cells"].append(cell)
            write_json(manifest_path, manifest)
            for attempt in range(1, args.max_attempts + 1):
                order_label = "-".join(str(item) for item in concurrencies)
                stem = (
                    f"r{rep:02d}-p{position:02d}-{variant}-"
                    f"c{order_label}-a{attempt:02d}"
                )
                idle_output = args.results / f"{stem}-idle-gate.json"
                idle_sample_log = args.results / f"{stem}-idle-gate.samples.jsonl"
                result_path = args.results / f"{stem}.json"
                server_log_path = args.results / f"{stem}.server.log"
                monitor_path = args.results / f"{stem}-gpu-monitor.json"
                monitor_sample_log = args.results / f"{stem}-gpu-monitor.samples.jsonl"
                log_path = args.results / f"{stem}.log"
                preexisting = [
                    path
                    for path in (
                        idle_output,
                        idle_sample_log,
                        result_path,
                        server_log_path,
                        monitor_path,
                        monitor_sample_log,
                        log_path,
                    )
                    if path.exists()
                ]
                if preexisting:
                    raise FileExistsError(
                        "refusing to overwrite append-only attempt evidence: "
                        + ", ".join(str(path) for path in preexisting)
                    )
                harness_command = build_harness_command(
                    python=args.python,
                    harness=args.harness,
                    source_root=args.root,
                    transformers_root=args.transformers_root,
                    variant=variant,
                    corpus=args.corpus,
                    videos=videos,
                    concurrencies=concurrencies,
                    port=args.port,
                    result_path=result_path,
                )
                command = build_monitored_command(
                    python=args.python,
                    monitor=args.monitor,
                    output=monitor_path,
                    child_command=harness_command,
                    watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
                    conflicting_controller_roots=args.conflicting_controller_root,
                )
                attempt_record: dict[str, Any] = {
                    "attempt": attempt,
                    "stem": stem,
                    "idle_gate": str(idle_output),
                    "command": command,
                    "log": str(log_path),
                    "monitor": str(monitor_path),
                    "output": str(result_path),
                    "server_log": str(server_log_path),
                }
                with attempt_integrity_context(
                    record_container=cell["attempts"],
                    record=attempt_record,
                    state=manifest,
                    state_path=manifest_path,
                    runtime_manifests=runtime_manifests,
                    runtime_validation_kwargs=runtime_manifest_validation_kwargs,
                    live_runtime_capture_kwargs={
                        "harness": args.harness,
                        "python": args.python,
                        "source_root": args.root,
                        "pythonpath_extras": [args.transformers_root],
                        "environment": environment,
                    },
                    source_root=args.root,
                    stem=stem,
                    commit=commit,
                    variant=variant,
                    evidence_paths={
                        "idle_gate": idle_output,
                        "idle_gate_sample_log": idle_sample_log,
                        "result": result_path,
                        "server_log": server_log_path,
                        "monitor": monitor_path,
                        "monitor_sample_log": monitor_sample_log,
                        "log": log_path,
                    },
                ):
                    print(f"IDLE_GATE {stem}", flush=True)
                    idle = run(
                        build_idle_gate_command(
                            python=args.python,
                            idle_gate=args.idle_gate,
                            output=idle_output,
                            seconds=args.idle_seconds,
                            timeout=args.idle_timeout,
                            conflicting_controller_roots=(
                                args.conflicting_controller_root
                            ),
                        ),
                        env=environment,
                        capture_output=True,
                    )
                    attempt_record["idle_gate_returncode"] = idle.returncode
                    if idle.returncode:
                        raise RuntimeError(idle.stderr or idle.stdout)
                    idle_report = json.loads(idle_output.read_text())
                    idle_sample_audit = validate_jsonl_binding(
                        idle_report,
                        expected_suffix="-idle-gate.samples.jsonl",
                        expected_path=idle_sample_log,
                    )
                    idle_evidence = {
                        "report_path": str(idle_output),
                        "report": idle_report,
                        "report_sha256": sha256_file(idle_output),
                        "sample_log_audit": idle_sample_audit,
                    }
                    validate_idle_gate_evidence(
                        idle_evidence,
                        expected_seconds=args.idle_seconds,
                        expected_timeout=args.idle_timeout,
                        conflicting_controller_roots=args.conflicting_controller_root,
                    )
                    attempt_record["idle_evidence"] = idle_evidence
                    attempt_record["idle_gate_sample_log_audit"] = idle_sample_audit
                    print(
                        f"RUN {stem} commit={commit} concurrencies={concurrencies}",
                        flush=True,
                    )
                    with log_path.open("x") as log:
                        completed = run(
                            command,
                            cwd=args.root,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    attempt_record["returncode"] = completed.returncode
                    monitor_result = (
                        json.loads(monitor_path.read_text())
                        if monitor_path.is_file()
                        else {}
                    )
                    if monitor_result.get("device") != idle_report.get("device"):
                        raise RuntimeError("idle gate/monitor device identity mismatch")
                    monitor_sample_audit = (
                        validate_jsonl_binding(
                            monitor_result,
                            expected_suffix="-gpu-monitor.samples.jsonl",
                            expected_path=monitor_sample_log,
                        )
                        if monitor_result
                        else None
                    )
                    attempt_record["contaminated"] = monitor_result.get("contaminated")
                    attempt_record["timed_out"] = monitor_result.get("timed_out")
                    attempt_record["monitor_sample_log_audit"] = monitor_sample_audit
                    write_json(manifest_path, manifest)
                    if monitor_result.get("timed_out"):
                        raise RuntimeError(f"cell watchdog expired for {stem}")
                    if monitor_result.get("contaminated") is True:
                        attempt_record["contamination_retry_evidence"] = (
                            validate_contamination_retry_evidence(
                                wrapper_returncode=completed.returncode,
                                report_path=monitor_path,
                                expected_wrapper_command=command,
                                expected_child_command=harness_command,
                                watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
                                conflicting_controller_roots=(
                                    args.conflicting_controller_root
                                ),
                            )
                        )
                        attempt_record["accepted"] = False
                        attempt_record["validation_status"] = "excluded_contamination"
                        print(f"CONTAMINATED {stem}; retrying", flush=True)
                        write_json(manifest_path, manifest)
                        continue
                    accepted_monitor, accepted_monitor_sample_audit = (
                        validate_monitor_evidence(
                            monitor_path,
                            expected_command=harness_command,
                            watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
                            conflicting_controller_roots=(
                                args.conflicting_controller_root
                            ),
                        )
                    )
                    if accepted_monitor_sample_audit != monitor_sample_audit:
                        raise RuntimeError(
                            f"accepted monitor sample audit changed for {stem}"
                        )
                    monitor_result = accepted_monitor
                    if completed.returncode:
                        tail = log_path.read_text(errors="replace")[-16000:]
                        raise RuntimeError(f"cell failed {stem}:\n{tail}")
                    result = json.loads(result_path.read_text())
                    validated_metrics = validate_result(
                        result,
                        monitor_result,
                        commit=commit,
                        variant=variant,
                        concurrency_order=concurrencies,
                        harness=args.harness,
                        harness_sha256=harness_sha256,
                        expected_monitor_command=harness_command,
                        monitor_path=monitor_path,
                        corpus=args.corpus,
                        transformers_root=args.transformers_root,
                        hf_snapshot_root=args.hf_snapshot_root,
                        source_root=args.root,
                        server_log_path=server_log_path,
                    )
                    expected_runtime_fingerprint_sha256 = (
                        runtime_hardware_fingerprint_contract["sha256"]
                    )
                    actual_runtime_fingerprint_sha256 = validated_metrics[
                        "runtime_hardware_fingerprint"
                    ]["sha256"]
                    if (
                        actual_runtime_fingerprint_sha256
                        != expected_runtime_fingerprint_sha256
                    ):
                        raise RuntimeError(
                            "cell runtime/hardware fingerprint differs from preflight: "
                            f"{actual_runtime_fingerprint_sha256} != "
                            f"{expected_runtime_fingerprint_sha256}"
                        )
                    coverage = monitor_coverage_audit(result, monitor_result)
                    if not coverage["passed"]:
                        attempt_record["accepted"] = False
                        attempt_record["validation_status"] = (
                            "rejected_monitor_coverage"
                        )
                        attempt_record["monitor_coverage_audit"] = coverage
                        write_json(manifest_path, manifest)
                        raise RuntimeError(
                            f"monitor coverage failed for {stem}: {coverage['blocks']}"
                        )
                    attempt_record["accepted"] = True
                    attempt_record["validation_status"] = "accepted"
                    attempt_record["monitor_coverage_audit"] = coverage
                    source_after_cell = validate_source(
                        args.root, commit, variant=variant
                    )
                    cell.update(
                        {
                            "status": "passed",
                            "winning_attempt": attempt,
                            "output": str(result_path),
                            "monitor": str(monitor_path),
                            "server_log": str(server_log_path),
                            "validated_metrics": validated_metrics,
                            "monitor_coverage_audit": coverage,
                            "source_after_cell": source_after_cell,
                        }
                    )
                    print(f"PASS {stem}", flush=True)
                    write_json(manifest_path, manifest)
                    break
            else:
                raise RuntimeError(
                    f"too many contaminated attempts for {variant} rep {rep}"
                )

    manifest["terminal_source"] = validate_source(args.root, commit, variant=variant)
    manifest["runtime_manifest_checkpoints"].append(
        runtime_manifest_checkpoint(
            expected=runtime_manifests,
            label="campaign_end",
            validation_kwargs=runtime_manifest_validation_kwargs,
        )
    )
    write_json(manifest_path, manifest)

    sidecar_audit_path = args.results / "accepted-cell-evidence-audit.json"
    if sidecar_audit_path.exists():
        raise FileExistsError(
            f"refusing to overwrite accepted sidecar audit: {sidecar_audit_path}"
        )
    sidecar_audit = terminal_accepted_sidecar_audit(
        manifest["cells"],
        results_root=args.results,
        python=args.python,
        harness=args.harness,
        harness_sha256=harness_sha256,
        monitor_script=args.monitor,
        source_root=args.root,
        transformers_root=args.transformers_root,
        hf_snapshot_root=args.hf_snapshot_root,
        corpus=args.corpus,
        videos=videos,
        port=args.port,
        conflicting_controller_roots=args.conflicting_controller_root,
        runtime_manifests=runtime_manifests,
        expected_runtime_fingerprint_sha256=(
            runtime_hardware_fingerprint_contract["sha256"]
        ),
    )
    write_json(sidecar_audit_path, sidecar_audit)
    sidecar_audit_binding = {
        "status": sidecar_audit["status"],
        "path": str(sidecar_audit_path),
        "bytes": sidecar_audit_path.stat().st_size,
        "sha256": sha256_file(sidecar_audit_path),
        "accepted_cell_count": sidecar_audit["accepted_cell_count"],
    }
    manifest["accepted_cell_evidence_audit"] = sidecar_audit_binding
    write_json(manifest_path, manifest)

    strict_token_audit_path = args.results / "token-parity.json"
    if strict_token_audit_path.exists():
        raise FileExistsError(
            f"refusing to overwrite token audit: {strict_token_audit_path}"
        )
    strict_token_audit = strict_token_text_audit(manifest["cells"])
    write_json(strict_token_audit_path, strict_token_audit)
    strict_token_audit_binding = {
        "status": strict_token_audit["status"],
        "path": str(strict_token_audit_path),
        "bytes": strict_token_audit_path.stat().st_size,
        "sha256": sha256_file(strict_token_audit_path),
        "accepted_result_count": strict_token_audit["accepted_result_count"],
        "mismatch_counts": strict_token_audit["mismatch_counts"],
        "untimed_top20_logprob_diagnostic_required": strict_token_audit[
            "untimed_top20_logprob_diagnostic_required"
        ],
    }
    summary = collection_summary(manifest["cells"])
    summary["accepted_cell_evidence_audit"] = sidecar_audit_binding
    summary["strict_token_audit"] = strict_token_audit_binding
    publication_status = {
        "passed_exact": "passed_exact",
        "failed_input_parity": "invalid_input_parity",
        "completion_or_text_mismatch": "timing_passed_completion_mismatch",
    }[strict_token_audit["status"]]
    summary["status"] = publication_status
    summary["guard_refinement_provenance"] = {
        "publication_clean_restart": True,
        "prior_cell_reuse_count": 0,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "guard_helper_sha256": GUARD_HELPER_SHA256,
        "idle_gate_sha256": IDLE_GATE_SHA256,
        "gpu_monitor_sha256": GPU_MONITOR_SHA256,
        "monitor_coverage_maximum_gap_seconds": (MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS),
    }
    summary_path = args.results / "collection-summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite summary: {summary_path}")
    write_json(summary_path, summary)
    manifest["status"] = publication_status
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["collection_summary"] = {
        "path": str(summary_path),
        "bytes": summary_path.stat().st_size,
        "sha256": sha256_file(summary_path),
    }
    manifest["strict_token_audit"] = strict_token_audit_binding
    manifest["terminal_source_after_audit"] = validate_source(
        args.root,
        COMMITS["pr-head"],
        variant="pr-head",
    )
    write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if publication_status == "invalid_input_parity":
        raise SystemExit(2)


def failure_category(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    if isinstance(error, FileExistsError):
        return "append_only_artifact_conflict"
    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return "timeout"
    if isinstance(error, RuntimeError):
        return "validation_or_workload_failure"
    if isinstance(error, OSError):
        return "io_failure"
    return "internal_failure"


def configured_results_root(argv: Sequence[str]) -> Path | None:
    values = []
    for index, value in enumerate(argv):
        if value == "--results" and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif value.startswith("--results="):
            values.append(value.partition("=")[2])
    if len(values) != 1 or not values[0]:
        return None
    return Path(values[0]).expanduser().resolve()


def record_terminal_failure(error: BaseException) -> None:
    results_root = configured_results_root(sys.argv[1:])
    if results_root is None or not results_root.is_dir():
        return
    manifest_path = results_root / "matrix-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}
    manifest.update(
        {
            "status": "collection_failed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "terminal_failure": {
                "category": failure_category(error),
                "exception_type": type(error).__name__,
                "message_recorded": False,
                "traceback_recorded": False,
            },
        }
    )
    write_json(manifest_path, manifest)


def main() -> None:
    try:
        _campaign_main()
    except SystemExit:
        raise
    except BaseException as error:
        try:
            record_terminal_failure(error)
        except BaseException as recording_error:
            setattr(
                error,
                "terminal_failure_recording_error_type",
                type(recording_error).__name__,
            )
        raise


if __name__ == "__main__":
    main()
