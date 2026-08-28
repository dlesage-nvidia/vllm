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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

UTC = timezone.utc

COMMITS = {
    "upstream": "d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302",
    "pr-head": "30d917599b104423e452fa718890af01c4ff4d39",
}
TREES = {
    "upstream": "9cc26997991af6f8f38150c9631d482d18b1bd2c",
    "pr-head": "66c4849eb21973b9ca391b7b0911968f4aa63dac",
}
SCHEDULE = [
    (1, [8, 16, 32], ["upstream", "pr-head"]),
    (2, [32, 16, 8], ["pr-head", "upstream"]),
    (3, [16, 32, 8], ["upstream", "pr-head"]),
    (4, [8, 32, 16], ["pr-head", "upstream"]),
    (5, [32, 8, 16], ["upstream", "pr-head"]),
    (6, [16, 8, 32], ["pr-head", "upstream"]),
]
MEASURED_REQUESTS = {8: 64, 16: 128, 32: 256}
WARMUP_REQUESTS = {8: 24, 16: 48, 32: 96}
CAMPAIGN_HARNESS_SHA256 = (
    "71adcc9ddb99e65e51d9531ed40728b8261f0f763c2fd1d89c2610a58fa3aa2b"
)
GPU_MONITOR_SHA256 = "239bcbbd0e635a8b44e46588142f336a8879067750aeee0d649faa8e62e950bc"
IDLE_GATE_SHA256 = "0a7119e7d0c40e3274ea9846db0b4e7213e7c1beeb4ddecce3a18d9641c5b02e"
GUARD_HELPER_SHA256 = "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
PREFLIGHT_RUNNER_SHA256 = (
    "924f7a6cf445678bc872ff5aa4d4cedfaac8db69ccf03fe7869ef0ba74b08636"
)
PIXEL_PREFLIGHT_SHA256 = (
    "e4cb333cd47f3015ccf3aa510e3f6c26364cc4947b63d89053decc0f8156addb"
)
MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS = 1.0
INGRESS_IDLE_SECONDS = 1200.0
INGRESS_IDLE_TIMEOUT_SECONDS = 21600.0
APPROVED_MONITOR_WATCHDOG_PAIRS = frozenset({(1200.0, 120.0), (3600.0, 120.0)})
TIMING_MONITOR_WATCHDOG_PAIR = (3600.0, 120.0)
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
RUNTIME_TREE_MANIFEST_TOOL_SHA256 = (
    "d4edac7bc314aba8ceedc799b9d9b1c64ac880d340dff70b949db50066f1981a"
)
RUNTIME_TREE_MANIFEST_TEST_SHA256 = (
    "d0ab0fcf324f6bc1042610a0ac5a970fe5fcf3fe31927aa904d0bb0ea76e0366"
)
TRANSFORMERS_OVERLAY_BASENAME = "vllm-pynv-e2e-transformers-5.14.1-20260827"
TRANSFORMERS_OVERLAY_TREE_MANIFEST = {
    "sha256": "91e8b5660cb228e78f48fe931bc28dd68f92d751475e9d76b93590246a255bb0",
    "regular_file_count": 2750,
    "logical_total_bytes": 51_873_110,
    "manifest_bytes": 418_598,
}
TRANSFORMERS_TREE_MANIFEST = {
    "sha256": "a33471c896d571395e22d4d4f1fa58f6b4fee7c0b66f281fceabaab1a804241a",
    "regular_file_count": 2740,
    "logical_total_bytes": 51_553_351,
    "manifest_bytes": 381_637,
}
HF_SNAPSHOT_TREE_MANIFEST = {
    "sha256": "5a2020450ee3804b0e3c5e8be0b1bf33eab679796706ca416e36517a10c3baf3",
    "regular_file_count": 12,
    "logical_total_bytes": 4_266_648_961,
    "manifest_bytes": 2_277,
}
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
_ACTIVE_MANIFEST_PATH: Path | None = None


def validate_monitor_watchdog_pair(
    watchdog_pair: tuple[float, float],
) -> tuple[float, float]:
    pair = (float(watchdog_pair[0]), float(watchdog_pair[1]))
    if pair not in APPROVED_MONITOR_WATCHDOG_PAIRS:
        raise ValueError(f"unapproved monitor watchdog pair: {pair}")
    return pair


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


COMMON_PARITY_CONFIGURATION_FIELDS = (
    "model",
    "revision",
    "prompt_sha256",
    "output_len",
    "seed",
    "frame_target",
    "video_count",
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


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        resolved_value = binding.get("resolved_path")
        if (
            not isinstance(resolved_value, str)
            or not Path(resolved_value).is_absolute()
        ):
            raise RuntimeError(f"runtime artifact {index} resolved path is invalid")
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
    vllm_native = sorted(
        resolved
        for resolved in claimed_by_resolved
        if Path(resolved).is_relative_to(source_root)
        and Path(resolved).suffix in {".so", ".pyd"}
    )
    if not vllm_native:
        raise RuntimeError("runtime artifact manifest lacks vLLM native extensions")
    discovered_vllm_native = sorted(
        str(path.resolve(strict=True))
        for pattern in ("*.so", "*.pyd")
        for path in source_root.joinpath("vllm").glob(pattern)
        if path.is_file()
    )
    if vllm_native != discovered_vllm_native:
        raise RuntimeError("runtime artifact manifest omits a vLLM native extension")
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
    resolved_vllm_paths = [Path(path).resolve(strict=True) for path in vllm_paths]
    vllm_package_parents = {path.parent for path in resolved_vllm_paths}
    if len(vllm_package_parents) != 1:
        raise RuntimeError(f"{label} vLLM native artifacts span package roots")
    vllm_package = next(iter(vllm_package_parents))
    discovered_vllm = sorted(
        str(path.resolve(strict=True))
        for pattern in ("*.so", "*.pyd")
        for path in vllm_package.glob(pattern)
        if path.is_file()
    )
    if sorted(pynv_paths) != discovered_pynv or sorted(vllm_paths) != discovered_vllm:
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


def tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_artifact_binding(
    binding: Mapping[str, Any], *, expected_path: Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    path = Path(str(binding.get("path", ""))).resolve()
    if path != expected_path.resolve() or not path.is_file():
        raise RuntimeError(f"{label} path mismatch: {path} != {expected_path}")
    actual_sha256 = sha256_file(path)
    if binding.get("sha256") != expected_sha256 or actual_sha256 != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual_sha256}


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


def validate_source_evidence(
    source: Mapping[str, Any], *, variant: str
) -> dict[str, Any]:
    if (
        source.get("commit") != COMMITS[variant]
        or source.get("tree") != TREES[variant]
        or source.get("status") != ""
        or source.get("source_harness_exists") is not False
        or source.get("source_harness_sha256") is not None
        or source.get("ignored_python_bytecode_or_cache_paths") != []
    ):
        raise RuntimeError(f"source evidence mismatch for {variant}")
    return {"commit": COMMITS[variant], "tree": TREES[variant], "pristine": True}


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


def validate_preflight_summary(
    path: Path,
    *,
    python: Path,
    harness_sha256: str,
    source_root: Path,
    corpus: Path,
    transformers_root: Path,
    current_artifacts: Mapping[str, Path],
    current_runtime_manifests: Mapping[str, Any],
    conflicting_controller_roots: Sequence[Path],
) -> dict[str, Any]:
    import torch

    path = path.resolve()
    python = python.resolve(strict=False)
    if path.name != "pilot-summary.json" or not path.is_file():
        raise RuntimeError("preflight summary path/filename mismatch")
    preflight_root = path.parent
    encoded = path.read_bytes()
    preflight = json.loads(encoded)
    if (
        preflight.get("status") != "passed"
        or preflight.get("schema") != "pynv-endpoint-persistent-preflight-v1"
        or preflight.get("harness_sha256") != harness_sha256
        or preflight.get("evidence_namespace")
        != {
            "root": str(preflight_root),
            "summary": str(path),
            "fresh_at_collection_start": True,
            "cross_namespace_sidecars_forbidden": True,
        }
    ):
        raise RuntimeError("endpoint preflight summary identity/status mismatch")
    expected_artifact_hashes = {
        "driver": sha256_file(Path(__file__).resolve()),
        "harness": CAMPAIGN_HARNESS_SHA256,
        "monitor": GPU_MONITOR_SHA256,
        "idle_gate": IDLE_GATE_SHA256,
        "guard_helper": GUARD_HELPER_SHA256,
        "runtime_manifest_tool": RUNTIME_TREE_MANIFEST_TOOL_SHA256,
        "runtime_manifest_test": RUNTIME_TREE_MANIFEST_TEST_SHA256,
        "pilot_runner": PREFLIGHT_RUNNER_SHA256,
        "pixel_preflight": PIXEL_PREFLIGHT_SHA256,
    }
    if set(current_artifacts) != set(expected_artifact_hashes):
        raise RuntimeError("current preflight artifact map is incomplete")
    artifact_audits: dict[str, dict[str, Any]] = {}
    stored_artifacts = preflight.get("artifacts")
    if not isinstance(stored_artifacts, Mapping):
        raise RuntimeError("preflight artifact graph is missing")
    for name in (
        "driver",
        "harness",
        "monitor",
        "idle_gate",
        "guard_helper",
        "runtime_manifest_tool",
        "runtime_manifest_test",
    ):
        artifact_audits[name] = validate_artifact_binding(
            stored_artifacts[name],
            expected_path=current_artifacts[name],
            expected_sha256=expected_artifact_hashes[name],
            label=f"preflight {name}",
        )
    artifact_audits["pilot_runner"] = validate_artifact_binding(
        preflight["pilot_runner"],
        expected_path=current_artifacts["pilot_runner"],
        expected_sha256=PREFLIGHT_RUNNER_SHA256,
        label="preflight runner",
    )
    artifact_audits["pixel_preflight"] = validate_artifact_binding(
        preflight["pixel_preflight_artifact"],
        expected_path=current_artifacts["pixel_preflight"],
        expected_sha256=PIXEL_PREFLIGHT_SHA256,
        label="pixel preflight",
    )
    if (
        preflight.get("runner_sha256") != expected_artifact_hashes["driver"]
        or preflight.get("harness_sha256") != CAMPAIGN_HARNESS_SHA256
        or preflight.get("runtime_manifests") != current_runtime_manifests
    ):
        raise RuntimeError("preflight driver/harness/runtime-manifest binding mismatch")
    top_checkpoints = preflight.get("runtime_manifest_checkpoints")
    if not isinstance(top_checkpoints, list) or len(top_checkpoints) != 2:
        raise RuntimeError("preflight top-level runtime checkpoints are incomplete")
    runtime_checkpoint_audits = [
        validate_runtime_manifest_checkpoint(
            top_checkpoints[index],
            expected_label=label,
            expected_manifests=current_runtime_manifests,
        )
        for index, label in enumerate(("preflight_start", "preflight_end"))
    ]

    expected_configuration = {
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
    if preflight.get("configuration") != expected_configuration:
        raise RuntimeError("endpoint preflight configuration mismatch")
    ingress_audit = validate_idle_gate_evidence(
        preflight.get("ingress_idle_gate", {}),
        expected_seconds=INGRESS_IDLE_SECONDS,
        expected_timeout=INGRESS_IDLE_TIMEOUT_SECONDS,
        conflicting_controller_roots=conflicting_controller_roots,
    )
    ingress_path = Path(ingress_audit["report_path"])
    if ingress_path != preflight_root / "preflight-ingress-idle-gate.json":
        raise RuntimeError("preflight ingress idle-gate namespace mismatch")
    ingress_sample_audit = ingress_audit["sample_log_audit"]

    pixel_binding = preflight["pixel_preflight"]
    pixel_path = Path(pixel_binding["result"]).resolve()
    if not pixel_path.is_file() or pixel_binding.get("result_sha256") != sha256_file(
        pixel_path
    ):
        raise RuntimeError("pixel preflight result binding mismatch")
    pixel_attempts = preflight.get("pixel_attempts")
    if not isinstance(pixel_attempts, list) or not pixel_attempts:
        raise RuntimeError("pixel preflight attempt evidence is missing")
    if [int(attempt.get("attempt", -1)) for attempt in pixel_attempts] != list(
        range(1, len(pixel_attempts) + 1)
    ):
        raise RuntimeError("pixel preflight attempt sequence is not contiguous")
    winning_pixel_attempts = [
        attempt
        for attempt in pixel_attempts
        if int(attempt["attempt"]) == int(pixel_binding["attempt"])
    ]
    if (
        len(winning_pixel_attempts) != 1
        or pixel_attempts[-1] is not winning_pixel_attempts[0]
    ):
        raise RuntimeError("pixel preflight does not have one terminal winning attempt")
    winning_pixel_attempt = winning_pixel_attempts[0]
    pixel_attempt_audits: list[dict[str, Any]] = []
    for attempt in pixel_attempts:
        attempt_number = int(attempt["attempt"])
        stem = f"pixel-parity-a{attempt_number:02d}"
        validate_runtime_manifest_checkpoint(
            attempt["runtime_manifest_before"],
            expected_label=f"{stem}:before_attempt",
            expected_manifests=current_runtime_manifests,
        )
        validate_runtime_manifest_checkpoint(
            attempt["runtime_manifest_after"],
            expected_label=f"{stem}:after_attempt",
            expected_manifests=current_runtime_manifests,
        )
        validate_source_evidence(attempt["source_after_attempt"], variant="pr-head")
        live_before = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_before", {}),
            label=f"{stem} before",
        )
        live_after = revalidate_live_runtime_artifact_manifest_binding(
            attempt.get("live_runtime_artifacts_after", {}),
            label=f"{stem} after",
        )
        if live_before != live_after:
            raise RuntimeError(f"pixel attempt live runtime artifacts changed: {stem}")
        idle_audit = validate_idle_gate_evidence(
            attempt["idle_evidence"],
            expected_seconds=CELL_IDLE_SECONDS,
            expected_timeout=CELL_IDLE_TIMEOUT_SECONDS,
            conflicting_controller_roots=conflicting_controller_roots,
        )
        result_path = Path(str(attempt["result"])).resolve()
        monitor_path = Path(str(attempt["monitor"])).resolve()
        log_path = Path(str(attempt["log"])).resolve()
        expected_result_path = preflight_root / f"{stem}.json"
        expected_monitor_path = preflight_root / f"{stem}-gpu-monitor.json"
        expected_log_path = preflight_root / f"{stem}.log"
        expected_idle_path = preflight_root / f"{stem}-idle-gate.json"
        expected_pixel_command = build_pixel_preflight_command(
            python=python,
            pixel_preflight=current_artifacts["pixel_preflight"],
            source_root=source_root,
            transformers_root=transformers_root,
            video=corpus / "traffic1080-00.mp4",
            result_path=expected_result_path,
        )
        expected_wrapper_command = build_monitored_command(
            python=python,
            monitor=current_artifacts["monitor"],
            output=expected_monitor_path,
            child_command=expected_pixel_command,
            watchdog_pair=(1200.0, 120.0),
            conflicting_controller_roots=conflicting_controller_roots,
        )
        if (
            result_path != expected_result_path
            or monitor_path != expected_monitor_path
            or log_path != expected_log_path
            or Path(str(attempt["idle_evidence"].get("report_path", ""))).resolve()
            != expected_idle_path
            or attempt.get("command") != expected_pixel_command
            or not monitor_path.is_file()
            or not log_path.is_file()
            or attempt.get("monitor_sha256") != sha256_file(monitor_path)
            or attempt.get("log_sha256") != sha256_file(log_path)
            or (
                result_path.is_file()
                and attempt.get("result_sha256") != sha256_file(result_path)
            )
        ):
            raise RuntimeError(f"pixel attempt artifact binding mismatch: {stem}")
        monitor = json.loads(monitor_path.read_text())
        monitor_sample_audit = validate_jsonl_binding(
            monitor,
            expected_path=monitor_path.with_name(monitor_path.stem + ".samples.jsonl"),
            expected_suffix="-gpu-monitor.samples.jsonl",
        )
        if monitor_sample_audit != attempt.get("monitor_sample_audit"):
            raise RuntimeError(f"pixel attempt monitor sample binding mismatch: {stem}")
        is_winner = attempt is winning_pixel_attempt
        if is_winner:
            monitor, monitor_sample_audit = validate_monitor_evidence(
                monitor_path,
                expected_command=expected_pixel_command,
                watchdog_pair=(1200.0, 120.0),
                conflicting_controller_roots=conflicting_controller_roots,
            )
            if (
                attempt.get("returncode") != 0
                or attempt.get("contaminated") is not False
            ):
                raise RuntimeError("winning pixel attempt status mismatch")
        else:
            contamination_audit = validate_contamination_retry_evidence(
                wrapper_returncode=attempt.get("returncode"),
                report_path=monitor_path,
                expected_wrapper_command=expected_wrapper_command,
                expected_child_command=expected_pixel_command,
                watchdog_pair=(1200.0, 120.0),
                conflicting_controller_roots=conflicting_controller_roots,
            )
            if attempt.get("contamination_retry_evidence") != contamination_audit:
                raise RuntimeError(
                    f"non-winning pixel contamination audit mismatch: {stem}"
                )
        pixel_attempt_audits.append(
            {
                "attempt": attempt_number,
                "winner": is_winner,
                "result_sha256": attempt.get("result_sha256"),
                "monitor_sha256": sha256_file(monitor_path),
                "log_sha256": sha256_file(log_path),
                "idle_gate": idle_audit,
                "monitor_sample_log": monitor_sample_audit,
                "live_runtime_artifacts": live_before,
            }
        )
    if (
        Path(str(winning_pixel_attempt["result"])).resolve() != pixel_path
        or winning_pixel_attempt.get("result_sha256")
        != pixel_binding.get("result_sha256")
        or Path(str(pixel_binding.get("monitor", ""))).resolve()
        != Path(str(winning_pixel_attempt["monitor"])).resolve()
        or Path(str(pixel_binding.get("log", ""))).resolve()
        != Path(str(winning_pixel_attempt["log"])).resolve()
    ):
        raise RuntimeError("winning pixel attempt/result binding mismatch")
    pixel_result = json.loads(pixel_path.read_text())
    if (
        pixel_result.get("schema") != "pynv-endpoint-pixel-parity-v2"
        or pixel_result.get("status") != "passed"
        or pixel_result.get("commits") != COMMITS
        or set(pixel_result.get("variants", {})) != set(COMMITS)
    ):
        raise RuntimeError("pixel preflight result identity mismatch")

    tensors: dict[str, dict[str, Any]] = {}
    pixel_worker_artifacts: dict[str, dict[str, Any]] = {}
    for variant, commit in COMMITS.items():
        variant_result = pixel_result["variants"][variant]
        worker_path = preflight_root / f"{pixel_path.stem}-{variant}-worker.json"
        worker_binding = variant_result.get("worker_result_artifact", {})
        if (
            variant_result.get("commit") != commit
            or variant_result.get("source", {}).get("tree") != TREES[variant]
            or variant_result.get("backend_kwargs") != variant_backend_kwargs(variant)
            or variant_result["processor"].get("configured_max_pixels_per_frame")
            != PIXEL_BUDGET[0] * PIXEL_BUDGET[1]
            or variant_result["processor"].get("configured_max_pixels_total")
            != TOTAL_MAX_PIXELS
            or variant_result["processor"].get("processed_width") != PIXEL_BUDGET[0]
            or variant_result["processor"].get("processed_height") != PIXEL_BUDGET[1]
            or Path(str(worker_binding.get("path", ""))).resolve() != worker_path
            or not worker_path.is_file()
            or worker_binding.get("bytes") != worker_path.stat().st_size
            or worker_binding.get("sha256") != sha256_file(worker_path)
        ):
            raise RuntimeError(f"pixel preflight {variant} configuration mismatch")
        artifact = variant_result["processor"]["tensor_artifact"]
        artifact_path = Path(artifact["path"]).resolve()
        expected_tensor_path = worker_path.with_suffix(".tensors.pt")
        if (
            artifact_path != expected_tensor_path
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError(f"pixel preflight {variant} tensor binding mismatch")
        tensors[variant] = torch.load(
            artifact_path, map_location="cpu", weights_only=True
        )
        for key in (
            "raw_processor_pixels",
            "model_visible_pixels",
            "video_grid_thw",
            "output_prompt_token_ids",
            "placeholder_is_embed",
        ):
            if key not in tensors[variant]:
                raise RuntimeError(f"pixel preflight tensor lacks {variant}/{key}")
        pixel_worker_artifacts[variant] = {
            "worker_result": dict(worker_binding),
            "tensor": dict(artifact),
        }

    if pixel_binding.get("worker_artifacts") != pixel_worker_artifacts:
        raise RuntimeError("pixel preflight worker artifact graph mismatch")

    baseline_result = pixel_result["variants"]["upstream"]
    final_result = pixel_result["variants"]["pr-head"]
    baseline_tensors = tensors["upstream"]
    final_tensors = tensors["pr-head"]
    exact_checks = {
        "canonical_thwc": (
            baseline_result["canonical_thwc"]["sha256"]
            == final_result["canonical_thwc"]["sha256"]
        ),
        "sampled_frame_indices": (
            baseline_result["metadata"]["frames_indices"]
            == final_result["metadata"]["frames_indices"]
        ),
        "source_frame_count": (
            baseline_result["metadata"]["total_num_frames"]
            == final_result["metadata"]["total_num_frames"]
        ),
        "raw_processor_pixels": torch.equal(
            baseline_tensors["raw_processor_pixels"],
            final_tensors["raw_processor_pixels"],
        ),
        "video_grid_thw": torch.equal(
            baseline_tensors["video_grid_thw"], final_tensors["video_grid_thw"]
        ),
        "output_prompt_token_ids": torch.equal(
            baseline_tensors["output_prompt_token_ids"],
            final_tensors["output_prompt_token_ids"],
        ),
        "placeholder_is_embed": torch.equal(
            baseline_tensors["placeholder_is_embed"],
            final_tensors["placeholder_is_embed"],
        ),
    }
    if not all(exact_checks.values()):
        raise RuntimeError(f"pixel preflight exact parity mismatch: {exact_checks}")
    grid_values = baseline_tensors["video_grid_thw"].tolist()
    if grid_values != [[16, 36, 64]]:
        raise RuntimeError(
            "pixel preflight did not realize the configured 1024x576/frame budget: "
            f"{grid_values}"
        )
    baseline_model = baseline_tensors["model_visible_pixels"]
    final_model = final_tensors["model_visible_pixels"]
    if (
        baseline_model.dtype != torch.bfloat16
        or final_model.dtype != torch.bfloat16
        or baseline_model.shape != final_model.shape
        or not torch.allclose(baseline_model, final_model, rtol=0.0, atol=2**-15)
    ):
        raise RuntimeError("pixel preflight model-visible BF16 parity mismatch")
    difference = (baseline_model.float() - final_model.float()).abs()

    pilots = preflight.get("pilots", [])
    if {(pilot["variant"], pilot["commit"]) for pilot in pilots} != set(
        COMMITS.items()
    ):
        raise RuntimeError("endpoint pilot variants/commits mismatch")
    pilot_results: dict[str, dict[str, Any]] = {}
    pilot_bindings = []
    pilot_attempts = preflight.get("pilot_attempts")
    if not isinstance(pilot_attempts, list):
        raise RuntimeError("endpoint pilot attempt evidence is missing")
    pilot_attempt_audits: list[dict[str, Any]] = []
    for pilot in pilots:
        variant = pilot["variant"]
        variant_attempts = [
            attempt for attempt in pilot_attempts if attempt.get("variant") == variant
        ]
        if [int(attempt.get("attempt", -1)) for attempt in variant_attempts] != list(
            range(1, len(variant_attempts) + 1)
        ):
            raise RuntimeError(f"endpoint pilot {variant} attempts are not contiguous")
        winning_attempts = [
            attempt
            for attempt in variant_attempts
            if int(attempt["attempt"]) == int(pilot["attempt"])
        ]
        if (
            len(winning_attempts) != 1
            or variant_attempts[-1] is not winning_attempts[0]
        ):
            raise RuntimeError(
                f"endpoint pilot {variant} winner is not terminal/unique"
            )
        winning_attempt = winning_attempts[0]
        winning_monitor: dict[str, Any] | None = None
        for attempt in variant_attempts:
            attempt_number = int(attempt["attempt"])
            stem = f"pilot-{variant}-c1-8-32-a{attempt_number:02d}"
            validate_runtime_manifest_checkpoint(
                attempt["runtime_manifest_before"],
                expected_label=f"{stem}:before_attempt",
                expected_manifests=current_runtime_manifests,
            )
            validate_runtime_manifest_checkpoint(
                attempt["runtime_manifest_after"],
                expected_label=f"{stem}:after_attempt",
                expected_manifests=current_runtime_manifests,
            )
            validate_source_evidence(attempt["source_after_attempt"], variant=variant)
            live_before = revalidate_live_runtime_artifact_manifest_binding(
                attempt.get("live_runtime_artifacts_before", {}),
                label=f"{stem} before",
            )
            live_after = revalidate_live_runtime_artifact_manifest_binding(
                attempt.get("live_runtime_artifacts_after", {}),
                label=f"{stem} after",
            )
            if live_before != live_after:
                raise RuntimeError(
                    f"endpoint pilot live runtime artifacts changed: {stem}"
                )
            idle_audit = validate_idle_gate_evidence(
                attempt["idle_evidence"],
                expected_seconds=CELL_IDLE_SECONDS,
                expected_timeout=CELL_IDLE_TIMEOUT_SECONDS,
                conflicting_controller_roots=conflicting_controller_roots,
            )
            attempt_result_path = Path(str(attempt["result"])).resolve()
            server_log_path = Path(str(attempt["server_log"])).resolve()
            monitor_path = Path(str(attempt["monitor"])).resolve()
            log_path = Path(str(attempt["log"])).resolve()
            expected_result_path = preflight_root / f"{stem}.json"
            expected_server_log_path = preflight_root / f"{stem}.server.log"
            expected_monitor_path = preflight_root / f"{stem}-gpu-monitor.json"
            expected_log_path = preflight_root / f"{stem}.log"
            expected_idle_path = preflight_root / f"{stem}-idle-gate.json"
            expected_pilot_command = build_harness_command(
                argparse.Namespace(
                    python=python,
                    harness=current_artifacts["harness"],
                    root=source_root,
                    transformers_root=transformers_root,
                    corpus=corpus,
                    port=18600,
                ),
                variant=variant,
                result_path=expected_result_path,
                videos=[corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)],
                concurrencies=[1, 8, 32],
                warmup_requests={1: 8, 8: 8, 32: 32},
                measured_requests={1: 8, 8: 8, 32: 32},
            )
            expected_wrapper_command = build_monitored_command(
                python=python,
                monitor=current_artifacts["monitor"],
                output=expected_monitor_path,
                child_command=expected_pilot_command,
                watchdog_pair=(3600.0, 120.0),
                conflicting_controller_roots=conflicting_controller_roots,
            )
            if (
                attempt_result_path != expected_result_path
                or server_log_path != expected_server_log_path
                or monitor_path != expected_monitor_path
                or log_path != expected_log_path
                or Path(str(attempt["idle_evidence"].get("report_path", ""))).resolve()
                != expected_idle_path
                or attempt.get("command") != expected_pilot_command
                or not monitor_path.is_file()
                or not log_path.is_file()
                or attempt.get("monitor_sha256") != sha256_file(monitor_path)
                or attempt.get("log_sha256") != sha256_file(log_path)
                or (
                    attempt_result_path.is_file()
                    and attempt.get("result_sha256") != sha256_file(attempt_result_path)
                )
                or (
                    server_log_path.is_file()
                    and attempt.get("server_log_sha256") != sha256_file(server_log_path)
                )
            ):
                raise RuntimeError(f"endpoint pilot artifact mismatch: {stem}")
            monitor = json.loads(monitor_path.read_text())
            monitor_sample_audit = validate_jsonl_binding(
                monitor,
                expected_path=monitor_path.with_name(
                    monitor_path.stem + ".samples.jsonl"
                ),
                expected_suffix="-gpu-monitor.samples.jsonl",
            )
            if monitor_sample_audit != attempt.get("monitor_sample_audit"):
                raise RuntimeError(f"endpoint pilot monitor sample mismatch: {stem}")
            is_winner = attempt is winning_attempt
            if is_winner:
                winning_monitor, monitor_sample_audit = validate_monitor_evidence(
                    monitor_path,
                    expected_command=expected_pilot_command,
                    watchdog_pair=(3600.0, 120.0),
                    conflicting_controller_roots=conflicting_controller_roots,
                )
                if (
                    attempt.get("returncode") != 0
                    or attempt.get("contaminated") is not False
                ):
                    raise RuntimeError(
                        f"winning endpoint pilot status mismatch: {stem}"
                    )
            else:
                contamination_audit = validate_contamination_retry_evidence(
                    wrapper_returncode=attempt.get("returncode"),
                    report_path=monitor_path,
                    expected_wrapper_command=expected_wrapper_command,
                    expected_child_command=expected_pilot_command,
                    watchdog_pair=(3600.0, 120.0),
                    conflicting_controller_roots=conflicting_controller_roots,
                )
                if attempt.get("contamination_retry_evidence") != contamination_audit:
                    raise RuntimeError(
                        f"non-winning endpoint pilot contamination audit mismatch: {stem}"
                    )
            pilot_attempt_audits.append(
                {
                    "variant": variant,
                    "attempt": attempt_number,
                    "winner": is_winner,
                    "result_sha256": attempt.get("result_sha256"),
                    "server_log_sha256": attempt.get("server_log_sha256"),
                    "monitor_sha256": sha256_file(monitor_path),
                    "log_sha256": sha256_file(log_path),
                    "idle_gate": idle_audit,
                    "monitor_sample_log": monitor_sample_audit,
                    "live_runtime_artifacts": live_before,
                }
            )
        pilot_path = Path(pilot["result"]).resolve()
        actual_sha256 = sha256_file(pilot_path)
        if (
            winning_monitor is None
            or Path(str(winning_attempt["result"])).resolve() != pilot_path
            or pilot.get("result_sha256") != actual_sha256
            or winning_attempt.get("result_sha256") != actual_sha256
            or Path(str(winning_attempt["monitor"])).resolve()
            != Path(str(pilot["monitor"])).resolve()
            or Path(str(winning_attempt["server_log"])).resolve()
            != Path(str(pilot["server_log"])).resolve()
        ):
            raise RuntimeError(f"endpoint pilot {variant} result binding mismatch")
        result = json.loads(pilot_path.read_text())
        if (
            result.get("status") != "passed"
            or result.get("schema")
            != "vllm-qwen3-vl-video-e2e-throughput-v3-persistent-http"
            or result["provenance"]["source"]["commit"] != COMMITS[variant]
            or result["provenance"]["source"]["tree"] != TREES[variant]
            or result["configuration"]["backend_kwargs"]
            != variant_backend_kwargs(variant)
            or result["configuration"]["extra_server_argv"]
            != variant_server_argv(variant)
        ):
            raise RuntimeError(f"endpoint pilot {variant} identity mismatch")
        server_log_path = Path(str(pilot["server_log"])).resolve()
        validated_result = validate_result(
            result,
            winning_monitor,
            commit=COMMITS[variant],
            variant=variant,
            concurrency_order=[1, 8, 32],
            harness=current_artifacts["harness"],
            harness_sha256=harness_sha256,
            expected_monitor_command=expected_pilot_command,
            corpus=corpus,
            transformers_root=transformers_root,
            source_root=source_root,
            server_log_path=server_log_path,
            warmup_requests={1: 8, 8: 8, 32: 32},
            measured_requests={1: 8, 8: 8, 32: 32},
            result_variant_label=variant,
        )
        if pilot.get("validated_result") != validated_result:
            raise RuntimeError(f"endpoint pilot {variant} validated metrics changed")
        if (
            validated_result["runtime_hardware_fingerprint"].get(
                "live_runtime_artifact_manifest"
            )
            != winning_attempt["live_runtime_artifacts_before"]
        ):
            raise RuntimeError(
                f"endpoint pilot {variant} live runtime artifact/result mismatch"
            )
        coverage = monitor_coverage_audit(result, winning_monitor)
        if not coverage["passed"] or pilot.get("monitor_coverage_audit") != coverage:
            raise RuntimeError(f"endpoint pilot {variant} monitor coverage mismatch")
        if pilot.get("server_log_sha256") != sha256_file(
            server_log_path
        ) or winning_attempt.get("server_log_sha256") != sha256_file(server_log_path):
            raise RuntimeError(f"endpoint pilot {variant} full server-log mismatch")
        pilot_results[variant] = result
        pilot_bindings.append(
            {
                "variant": variant,
                "path": str(pilot_path),
                "sha256": actual_sha256,
                "monitor_sha256": sha256_file(Path(str(pilot["monitor"]))),
                "server_log_sha256": sha256_file(server_log_path),
                "runtime_hardware_fingerprint": validated_result[
                    "runtime_hardware_fingerprint"
                ],
            }
        )

    baseline_pilot = pilot_results["upstream"]
    final_pilot = pilot_results["pr-head"]
    pilot_prompt_pairs = 0
    pilot_c1_completion_pairs = 0
    for concurrency in (1, 8, 32):
        blocks = []
        for result in (baseline_pilot, final_pilot):
            blocks.append(
                next(
                    block
                    for block in result["concurrency_blocks"]
                    if int(block["concurrency"]) == concurrency
                )
            )
        for phase in ("warmup", "measured"):
            records = [
                {
                    int(record["request_index"]): record
                    for record in block[phase]["records"]
                }
                for block in blocks
            ]
            if records[0].keys() != records[1].keys():
                raise RuntimeError("endpoint pilot request index mismatch")
            for request_index in sorted(records[0]):
                left = records[0][request_index]
                right = records[1][request_index]
                identity_fields = (
                    "request_index",
                    "video_index",
                    "video_sha256",
                    "request_payload_sha256",
                    "status",
                )
                if any(
                    left.get(field) != right.get(field) for field in identity_fields
                ):
                    raise RuntimeError("endpoint pilot request identity mismatch")
                if (
                    left["response"]["prompt_token_ids"]
                    != right["response"]["prompt_token_ids"]
                ):
                    raise RuntimeError("endpoint pilot prompt token mismatch")
                pilot_prompt_pairs += 1
                if concurrency == 1:
                    if (
                        left["response"]["completion_token_ids"]
                        != right["response"]["completion_token_ids"]
                    ):
                        raise RuntimeError(
                            "endpoint pilot C1 completion token mismatch"
                        )
                    pilot_c1_completion_pairs += 1

    if len(pilot_attempt_audits) != len(pilot_attempts):
        raise RuntimeError("preflight contains unreferenced endpoint pilot attempts")
    terminal_source = validate_source_evidence(
        preflight.get("terminal_source_revalidation", {}), variant="pr-head"
    )
    terminal_live_runtime_artifacts = revalidate_live_runtime_artifact_manifest_binding(
        preflight.get("terminal_live_runtime_artifact_revalidation", {}),
        label="preflight terminal",
    )
    if terminal_live_runtime_artifacts != pilot_attempts[-1].get(
        "live_runtime_artifacts_after"
    ):
        raise RuntimeError("preflight terminal live runtime artifacts mismatch")
    fingerprint_contract = preflight.get("runtime_hardware_fingerprint_contract")
    pilot_fingerprints = {
        binding["variant"]: binding["runtime_hardware_fingerprint"]
        for binding in pilot_bindings
    }
    if (
        not isinstance(fingerprint_contract, Mapping)
        or fingerprint_contract.get("status") != "passed"
        or fingerprint_contract.get("schema") != "pynv-runtime-hardware-fingerprint-v1"
        or set(fingerprint_contract.get("variants", {})) != set(COMMITS)
        or any(
            fingerprint_contract["variants"].get(variant) != fingerprint["sha256"]
            for variant, fingerprint in pilot_fingerprints.items()
        )
        or len({item["sha256"] for item in pilot_fingerprints.values()}) != 1
    ):
        raise RuntimeError("preflight runtime/hardware fingerprint contract mismatch")
    reference_fingerprint = pilot_fingerprints["upstream"]
    if (
        fingerprint_contract.get("sha256") != reference_fingerprint["sha256"]
        or fingerprint_contract.get("canonical") != reference_fingerprint["canonical"]
    ):
        raise RuntimeError("preflight runtime/hardware fingerprint content mismatch")

    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "pixel_result": {"path": str(pixel_path), "sha256": sha256_file(pixel_path)},
        "pixel_exact_checks": exact_checks,
        "model_visible": {
            "shape": list(baseline_model.shape),
            "dtype": str(baseline_model.dtype),
            "rtol": 0.0,
            "atol": 2**-15,
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "allclose": True,
            "baseline_sha256": tensor_sha256(baseline_model),
            "final_sha256": tensor_sha256(final_model),
        },
        "pilots": pilot_bindings,
        "artifacts": artifact_audits,
        "runtime_manifests": current_runtime_manifests,
        "runtime_manifest_checkpoints": runtime_checkpoint_audits,
        "pixel_attempts": pixel_attempt_audits,
        "pilot_attempts": pilot_attempt_audits,
        "terminal_source_revalidation": terminal_source,
        "runtime_hardware_fingerprint_contract": fingerprint_contract,
        "pilot_prompt_pair_count": pilot_prompt_pairs,
        "pilot_c1_completion_pair_count": pilot_c1_completion_pairs,
        "critical_parity_recomputed_from_raw_artifacts": True,
        "ingress_idle_gate": {
            "report": str(ingress_path),
            "report_sha256": sha256_file(ingress_path),
            "sample_log_audit": ingress_sample_audit,
        },
    }


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
        started_ns = measured.get("started_monotonic_ns")
        finished_ns = measured.get("finished_monotonic_ns")
        if (
            not isinstance(started_ns, int)
            or not isinstance(finished_ns, int)
            or finished_ns <= started_ns
        ):
            raise RuntimeError(
                f"c{concurrency} measured monotonic boundaries are invalid"
            )
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
                "maximum_allowed_gap_seconds": (MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS),
                "passed": bool(in_window)
                and maximum_gap <= MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS,
            }
        )
    return {
        "policy": (
            "Every measured window must contain monitor samples and all boundary/"
            "adjacent gaps must be <=1 second."
        ),
        "blocks": blocks,
        "passed": all(block["passed"] for block in blocks),
    }


def variant_backend_kwargs(variant: str) -> dict[str, Any]:
    if variant == "upstream":
        return {"hw_decoders": 2}
    if variant == "pr-head":
        return {"hw_decoders": 2, "output_layout": "tchw"}
    raise ValueError(f"unknown variant: {variant}")


def variant_server_argv(variant: str) -> list[str]:
    if variant == "upstream":
        return ["--no-mm-device-do-normalize"]
    if variant == "pr-head":
        return ["--mm-device-do-normalize"]
    raise ValueError(f"unknown variant: {variant}")


def build_harness_command(
    args: argparse.Namespace,
    *,
    variant: str,
    result_path: Path,
    videos: Sequence[Path],
    concurrencies: Sequence[int],
    warmup_requests: Mapping[int, int] = WARMUP_REQUESTS,
    measured_requests: Mapping[int, int] = MEASURED_REQUESTS,
) -> list[str]:
    command = [
        str(args.python),
        str(args.harness),
        "--source-root",
        str(args.root),
        "--python",
        str(args.python),
        "--pythonpath-extra",
        str(args.transformers_root),
        "--variant",
        variant,
        "--allowed-local-media-path",
        str(args.corpus),
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
        str(args.port),
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


def build_pixel_preflight_command(
    *,
    python: Path,
    pixel_preflight: Path,
    source_root: Path,
    transformers_root: Path,
    video: Path,
    result_path: Path,
) -> list[str]:
    return [
        str(python),
        str(pixel_preflight),
        "--root",
        str(source_root),
        "--python",
        str(python),
        "--transformers-root",
        str(transformers_root),
        "--video",
        str(video),
        "--output",
        str(result_path),
    ]


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
        str(path.relative_to(root))
        for path in (root / "vllm").rglob("*")
        if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
    )
    if bytecode_paths:
        raise RuntimeError(
            "source vllm tree contains ignored Python bytecode/cache paths: "
            f"{bytecode_paths[:20]}"
        )
    source_harness = root / "benchmarks/multimodal/benchmark_pynvvideocodec_e2e.py"
    source_harness_exists = source_harness.is_file()
    expected_source_harness_exists = False
    if source_harness_exists != expected_source_harness_exists:
        raise RuntimeError(
            "source harness presence mismatch: "
            f"{source_harness_exists} != {expected_source_harness_exists}"
        )
    return {
        "commit": actual_commit,
        "tree": actual_tree,
        "status": status,
        "source_harness_path": str(source_harness),
        "source_harness_exists": source_harness_exists,
        "source_harness_sha256": None,
        "ignored_python_bytecode_or_cache_paths": bytecode_paths,
    }


def post_attempt_integrity_checks(
    *,
    root: Path,
    commit: str,
    variant: str,
    runtime_manifests: Mapping[str, Any],
    runtime_label: str,
    runtime_validation_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    try:
        checks["source"] = validate_source(root, commit, variant=variant)
    except BaseException as error:
        errors.append(f"source:{type(error).__name__}")
    try:
        checks["runtime_manifests"] = runtime_manifest_checkpoint(
            expected=runtime_manifests,
            label=runtime_label,
            validation_kwargs=runtime_validation_kwargs,
        )
    except BaseException as error:
        errors.append(f"runtime_manifests:{type(error).__name__}")
    if errors:
        raise RuntimeError(
            "post-attempt source/runtime integrity validation failed: "
            + ",".join(errors)
        )
    return checks


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
    vllm_compiled_artifacts = sorted(
        (
            {
                "basename": Path(str(item["resolved_path"])).name,
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in live_manifest["artifacts"]
            if Path(item["resolved_path"]).is_relative_to(source_root)
            and Path(item["resolved_path"]).suffix in {".so", ".pyd"}
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


def measured_window_vram(
    monitor: Mapping[str, Any], block: Mapping[str, Any]
) -> dict[str, Any]:
    measured = block["measured"]
    started_ns = measured.get("started_monotonic_ns")
    finished_ns = measured.get("finished_monotonic_ns")
    if (
        not isinstance(started_ns, int)
        or not isinstance(finished_ns, int)
        or finished_ns <= started_ns
    ):
        raise RuntimeError("measured monotonic boundaries are invalid")
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
    peak_non_mps_compute_memory_mib = 0
    for sample in samples:
        non_mps_memory_mib = 0
        for app in sample.get("compute_apps", []):
            name = str(app["process_name"])
            used_memory_mib = int(app["used_memory_mib"])
            peak_compute_process_memory_mib_by_name[name] = max(
                peak_compute_process_memory_mib_by_name.get(name, 0),
                used_memory_mib,
            )
            if "nvidia-cuda-mps-server" not in name:
                non_mps_memory_mib += used_memory_mib
        peak_non_mps_compute_memory_mib = max(
            peak_non_mps_compute_memory_mib, non_mps_memory_mib
        )
    return {
        "measured_started_at": measured["started_at"],
        "measured_finished_at": measured["finished_at"],
        "measured_started_monotonic_ns": started_ns,
        "measured_finished_monotonic_ns": finished_ns,
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
        "peak_compute_process_memory_mib_by_name": (
            peak_compute_process_memory_mib_by_name
        ),
    }


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
    corpus: Path,
    transformers_root: Path,
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
    if monitor.get("command") != expected_monitor_command:
        raise RuntimeError("GPU monitor child command mismatch")

    provenance = result["provenance"]
    hardware = provenance["hardware"]
    if hardware.get("cuda_visible_devices") != "0":
        raise RuntimeError("harness CUDA_VISIBLE_DEVICES is not pinned to 0")
    gpu_lines = str(hardware.get("nvidia_smi_output", "")).splitlines()
    if len(gpu_lines) != 1:
        raise RuntimeError(f"expected one physical GPU provenance row: {gpu_lines}")
    gpu_fields = [field.strip() for field in gpu_lines[0].split(",")]
    if len(gpu_fields) < 3 or gpu_fields[0] != "0":
        raise RuntimeError(f"invalid harness GPU provenance row: {gpu_fields}")
    monitor_device = monitor.get("device", {})
    if (
        monitor_device.get("index") != 0
        or monitor_device.get("name") != gpu_fields[1]
        or monitor_device.get("uuid") != gpu_fields[2]
    ):
        raise RuntimeError(
            f"monitor/harness physical GPU identity mismatch: "
            f"{monitor_device} vs {gpu_fields[:3]}"
        )
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
        "variant": (
            result_variant_label if result_variant_label is not None else variant
        ),
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
        "video_kwargs_for_metric_derivation_unavailable_reason": None,
        "server_mm_processor_kwargs": {"max_pixels": TOTAL_MAX_PIXELS},
        "server_limit_mm_per_prompt": {"image": 0, "video": 1},
        "request_timeout_seconds": 1200.0,
        "startup_timeout_seconds": 600.0,
        "shutdown_timeout_seconds": 60.0,
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
                f"c{concurrency} persistent pool summary mismatch: "
                f"{persistent_pool}"
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
                f"c{concurrency} block aggregate differs from measured aggregate"
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

    server_log_binding = validate_full_server_log_binding(result, server_log_path)
    runtime_fingerprint = canonical_runtime_fingerprint(result)
    return {
        "blocks": block_summaries,
        **server_log_binding,
        "whole_run_peak_total_gpu_memory_used_mib": monitor["peak_memory_used_mib"],
        "monitor_sample_count": monitor["sample_count"],
        "runtime_hardware_fingerprint": runtime_fingerprint,
    }


def strict_expected_response_pair_count() -> int:
    return 6 * sum(
        WARMUP_REQUESTS[concurrency] + MEASURED_REQUESTS[concurrency]
        for concurrency in MEASURED_REQUESTS
    )


def strict_token_status(mismatch_counts: Mapping[str, int]) -> str:
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
    return (
        "failed_input_parity"
        if input_mismatches
        else "completion_or_text_mismatch" if generation_mismatches else "passed_exact"
    )


def strict_token_audit(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_rep_variant = {(int(cell["rep"]), str(cell["variant"])): cell for cell in cells}
    mismatches: list[dict[str, Any]] = []
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
    sources: list[dict[str, Any]] = []
    compared_response_pair_count = 0
    unique_responses: dict[str, set[str]] = {variant: set() for variant in COMMITS}

    def mismatch(kind: str, **details: Any) -> None:
        mismatch_counts[kind] += 1
        if len(mismatches) < 1000:
            mismatches.append({"kind": kind, **details})

    for rep in range(1, 7):
        loaded: dict[str, dict[str, Any]] = {}
        for variant in COMMITS:
            cell = by_rep_variant[(rep, variant)]
            path = Path(str(cell["output"])).resolve()
            encoded = path.read_bytes()
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
            loaded[variant] = json.loads(encoded)
            token_evidence_revalidation = []
            for block in loaded[variant].get("concurrency_blocks", []):
                concurrency = int(block["concurrency"])
                for phase in ("warmup", "measured"):
                    token_evidence_revalidation.append(
                        {
                            "concurrency": concurrency,
                            "phase": phase,
                            "audit": validate_batch_token_evidence(
                                block[phase],
                                context=f"strict audit rep {rep} {variant} "
                                f"c{concurrency} {phase}",
                            ),
                        }
                    )
            actual_treatment = {
                "backend_kwargs": loaded[variant]["configuration"].get(
                    "backend_kwargs"
                ),
                "extra_server_argv": loaded[variant]["configuration"].get(
                    "extra_server_argv"
                ),
            }
            expected_treatment = {
                "backend_kwargs": variant_backend_kwargs(variant),
                "extra_server_argv": variant_server_argv(variant),
            }
            if actual_treatment != expected_treatment:
                mismatch(
                    "treatment_configuration",
                    rep=rep,
                    variant=variant,
                    actual=actual_treatment,
                    expected=expected_treatment,
                )
            sources.append(
                {
                    "rep": rep,
                    "variant": variant,
                    "commit": COMMITS[variant],
                    "tree": TREES[variant],
                    "path": str(path),
                    "bytes": len(encoded),
                    "sha256": actual_result_sha256,
                    "accepted_attempt_recorded_sha256": recorded_result_sha256,
                    "actual_treatment": actual_treatment,
                    "expected_treatment": expected_treatment,
                    "token_evidence_revalidation": token_evidence_revalidation,
                }
            )
        baseline = loaded["upstream"]
        final = loaded["pr-head"]
        baseline_common = {
            field: baseline["configuration"].get(field)
            for field in COMMON_PARITY_CONFIGURATION_FIELDS
        }
        final_common = {
            field: final["configuration"].get(field)
            for field in COMMON_PARITY_CONFIGURATION_FIELDS
        }
        if baseline_common != final_common:
            mismatch(
                "common_configuration",
                rep=rep,
                baseline_sha256=sha256_json(baseline_common),
                final_sha256=sha256_json(final_common),
                differing_fields=[
                    field
                    for field in COMMON_PARITY_CONFIGURATION_FIELDS
                    if baseline_common[field] != final_common[field]
                ],
            )
        baseline_blocks = {
            int(block["concurrency"]): block for block in baseline["concurrency_blocks"]
        }
        final_blocks = {
            int(block["concurrency"]): block for block in final["concurrency_blocks"]
        }
        for concurrency in sorted(MEASURED_REQUESTS):
            for phase in ("warmup", "measured"):
                baseline_records = {
                    int(record["request_index"]): record
                    for record in baseline_blocks[concurrency][phase]["records"]
                }
                final_records = {
                    int(record["request_index"]): record
                    for record in final_blocks[concurrency][phase]["records"]
                }
                expected_request_count = (
                    WARMUP_REQUESTS[concurrency]
                    if phase == "warmup"
                    else MEASURED_REQUESTS[concurrency]
                )
                expected_request_indices = set(range(expected_request_count))
                if (
                    set(baseline_records) != expected_request_indices
                    or set(final_records) != expected_request_indices
                ):
                    mismatch(
                        "request_identity",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        reason="request_index_contract",
                        expected_count=expected_request_count,
                        baseline_count=len(baseline_records),
                        final_count=len(final_records),
                    )
                if baseline_records.keys() != final_records.keys():
                    mismatch(
                        "request_identity",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        reason="request_index_set",
                        baseline=sorted(baseline_records),
                        final=sorted(final_records),
                    )
                for request_index in sorted(
                    baseline_records.keys() & final_records.keys()
                ):
                    baseline_record = baseline_records[request_index]
                    final_record = final_records[request_index]
                    compared_response_pair_count += 1
                    identity_fields = (
                        "request_index",
                        "video_index",
                        "video_sha256",
                        "request_payload_sha256",
                        "status",
                    )
                    identity_differences = [
                        field
                        for field in identity_fields
                        if baseline_record.get(field) != final_record.get(field)
                    ]
                    if identity_differences:
                        mismatch(
                            "request_identity",
                            rep=rep,
                            concurrency=concurrency,
                            phase=phase,
                            request_index=request_index,
                            differing_fields=identity_differences,
                        )
                    baseline_response = baseline_record["response"]
                    final_response = final_record["response"]
                    for variant, response in (
                        ("upstream", baseline_response),
                        ("pr-head", final_response),
                    ):
                        unique_responses[variant].add(
                            sha256_json(
                                {
                                    "prompt": response["prompt_token_ids"],
                                    "completion": response["completion_token_ids"],
                                    "text_sha256": response["text_sha256"],
                                    "reasoning_content_sha256": response[
                                        "reasoning_content_sha256"
                                    ],
                                    "finish_reason": response["finish_reason"],
                                    "stop_reason": response["stop_reason"],
                                }
                            )
                        )
                    for kind, field in (
                        ("prompt_token_ids", "prompt_token_ids"),
                        ("completion_token_ids", "completion_token_ids"),
                        ("text_sha256", "text_sha256"),
                        ("reasoning_content_sha256", "reasoning_content_sha256"),
                        ("finish_reason", "finish_reason"),
                        ("stop_reason", "stop_reason"),
                    ):
                        baseline_value = baseline_response.get(field)
                        final_value = final_response.get(field)
                        if baseline_value == final_value:
                            continue
                        details: dict[str, Any] = {
                            "rep": rep,
                            "concurrency": concurrency,
                            "phase": phase,
                            "request_index": request_index,
                            "video_index": baseline_record.get("video_index"),
                            "baseline_sha256": sha256_json(baseline_value),
                            "final_sha256": sha256_json(final_value),
                        }
                        if kind in {"prompt_token_ids", "completion_token_ids"}:
                            first_difference = next(
                                (
                                    index
                                    for index, (left, right) in enumerate(
                                        zip(baseline_value, final_value)
                                    )
                                    if left != right
                                ),
                                min(len(baseline_value), len(final_value)),
                            )
                            details.update(
                                {
                                    "baseline_count": len(baseline_value),
                                    "final_count": len(final_value),
                                    "first_difference_index": first_difference,
                                    "baseline_token_at_first_difference": (
                                        baseline_value[first_difference]
                                        if first_difference < len(baseline_value)
                                        else None
                                    ),
                                    "final_token_at_first_difference": (
                                        final_value[first_difference]
                                        if first_difference < len(final_value)
                                        else None
                                    ),
                                }
                            )
                        mismatch(kind, **details)
    status = strict_token_status(mismatch_counts)
    first_generation_divergence = next(
        (
            item
            for item in mismatches
            if item["kind"]
            in {
                "completion_token_ids",
                "text_sha256",
                "reasoning_content_sha256",
                "finish_reason",
                "stop_reason",
            }
        ),
        None,
    )
    expected_response_pair_count = strict_expected_response_pair_count()
    if compared_response_pair_count != expected_response_pair_count:
        raise RuntimeError(
            "strict token audit response count mismatch: "
            f"{compared_response_pair_count} != {expected_response_pair_count}"
        )
    return {
        "schema": "pynv-endpoint-strict-token-parity-v1",
        "status": status,
        "timing_cell_acceptance_independent": True,
        "match_seeking_retry_count": 0,
        "comparison": (
            "same repetition/concurrency/request/video; full prompt and completion "
            "token ID arrays plus text SHA-256"
        ),
        "common_configuration_fields": list(COMMON_PARITY_CONFIGURATION_FIELDS),
        "endpoint_treatment_expected_difference": {
            variant: {
                "backend_kwargs": variant_backend_kwargs(variant),
                "extra_server_argv": variant_server_argv(variant),
            }
            for variant in COMMITS
        },
        "sources": sources,
        "expected_response_pair_count": expected_response_pair_count,
        "compared_response_pair_count": compared_response_pair_count,
        "expected_individual_response_count": expected_response_pair_count * 2,
        "compared_individual_response_count": compared_response_pair_count * 2,
        "unique_response_count_by_variant": {
            variant: len(values) for variant, values in unique_responses.items()
        },
        "mismatch_counts": mismatch_counts,
        "total_mismatch_count": sum(mismatch_counts.values()),
        "mismatches": mismatches,
        "mismatches_truncated": sum(mismatch_counts.values()) > len(mismatches),
        "first_generation_divergence": first_generation_divergence,
        "untimed_top20_logprob_diagnostic_required": (
            first_generation_divergence is not None
        ),
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
                if len(values) != 6:
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
        "strict_token_audit": {
            "status": "not_run",
            "note": "Token equivalence is intentionally audited separately.",
        },
        "model": MODEL,
        "revision": REVISION,
        "protocol": (
            "non-streaming persistent HTTP/1.1 chat/completions; one C-sized "
            "pool per block spans warmup and measurement; E2E latency only; no TTFT"
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
        "repetitions": 6,
        "aggregates": aggregates,
        "paired_endpoint_comparison": paired_endpoint_summary(cells),
    }


def paired_endpoint_summary(
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
    for concurrency in MEASURED_REQUESTS:
        per_metric: dict[str, Any] = {}
        for metric, getter in metric_getters.items():
            pairs = []
            for rep in range(1, 7):
                values: dict[str, float] = {}
                for variant in COMMITS:
                    cell = cells_by_rep_variant[(rep, variant)]
                    block = next(
                        item
                        for item in cell["validated_metrics"]["blocks"]
                        if int(item["concurrency"]) == concurrency
                    )
                    values[variant] = getter(block)
                baseline = values["upstream"]
                final = values["pr-head"]
                ratio = final / baseline
                pairs.append(
                    {
                        "rep": rep,
                        "baseline": baseline,
                        "final": final,
                        "final_minus_baseline": final - baseline,
                        "final_over_baseline": ratio,
                        "final_percent_delta": (ratio - 1.0) * 100.0,
                    }
                )
            baselines = [pair["baseline"] for pair in pairs]
            finals = [pair["final"] for pair in pairs]
            deltas = [pair["final_minus_baseline"] for pair in pairs]
            percent_deltas = [pair["final_percent_delta"] for pair in pairs]
            log_ratios = [math.log(pair["final_over_baseline"]) for pair in pairs]
            baseline_mean = statistics.fmean(baselines)
            final_mean = statistics.fmean(finals)
            log_ratio_mean = statistics.fmean(log_ratios)
            log_ratio_standard_error = statistics.stdev(log_ratios) / math.sqrt(
                len(log_ratios)
            )
            t_critical_95_df5 = 2.5705818366147395
            per_metric[metric] = {
                "pairs": pairs,
                "baseline_mean": baseline_mean,
                "baseline_median": statistics.median(baselines),
                "baseline_sample_stdev": statistics.stdev(baselines),
                "final_mean": final_mean,
                "final_median": statistics.median(finals),
                "final_sample_stdev": statistics.stdev(finals),
                "ratio_of_means": final_mean / baseline_mean,
                "ratio_of_means_percent_delta": ((final_mean / baseline_mean) - 1.0)
                * 100.0,
                "paired_difference_mean": statistics.fmean(deltas),
                "paired_difference_sample_stdev": statistics.stdev(deltas),
                "paired_percent_delta_mean": statistics.fmean(percent_deltas),
                "paired_percent_delta_median": statistics.median(percent_deltas),
                "paired_percent_delta_sample_stdev": statistics.stdev(percent_deltas),
                "paired_geomean_final_over_baseline": math.exp(log_ratio_mean),
                "paired_geomean_ratio_95_percent_t_ci": {
                    "degrees_of_freedom": 5,
                    "t_critical": t_critical_95_df5,
                    "low": math.exp(
                        log_ratio_mean - t_critical_95_df5 * log_ratio_standard_error
                    ),
                    "high": math.exp(
                        log_ratio_mean + t_critical_95_df5 * log_ratio_standard_error
                    ),
                },
            }
        comparisons[str(concurrency)] = per_metric
    return {
        "baseline": "upstream",
        "candidate": "pr-head",
        "pairing": "same repetition and concurrency",
        "repetitions": 6,
        "order_balance": (
            "UH/HU/UH/HU/UH/HU; each endpoint is first in three of six "
            "repetitions, and each concurrency occupies each block position twice."
        ),
        "by_concurrency": comparisons,
    }


def run_main() -> None:
    global _ACTIVE_MANIFEST_PATH
    _ACTIVE_MANIFEST_PATH = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--idle-gate", type=Path, required=True)
    parser.add_argument("--guard-helper", type=Path, required=True)
    parser.add_argument("--runtime-manifest-tool", type=Path, required=True)
    parser.add_argument("--runtime-manifest-test", type=Path, required=True)
    parser.add_argument(
        "--transformers-overlay-manifest-jsonl", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-overlay-manifest-summary", type=Path, required=True
    )
    parser.add_argument("--transformers-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--transformers-manifest-summary", type=Path, required=True)
    parser.add_argument("--hf-snapshot-root", type=Path, required=True)
    parser.add_argument("--hf-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--hf-manifest-summary", type=Path, required=True)
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
    args.corpus = args.corpus.resolve()
    args.results = args.results.resolve()
    args.harness = args.harness.resolve()
    args.monitor = args.monitor.resolve()
    args.idle_gate = args.idle_gate.resolve()
    args.guard_helper = args.guard_helper.resolve()
    args.runtime_manifest_tool = args.runtime_manifest_tool.resolve()
    args.runtime_manifest_test = args.runtime_manifest_test.resolve()
    args.transformers_overlay_manifest_jsonl = (
        args.transformers_overlay_manifest_jsonl.resolve()
    )
    args.transformers_overlay_manifest_summary = (
        args.transformers_overlay_manifest_summary.resolve()
    )
    args.transformers_manifest_jsonl = args.transformers_manifest_jsonl.resolve()
    args.transformers_manifest_summary = args.transformers_manifest_summary.resolve()
    args.hf_snapshot_root = args.hf_snapshot_root.resolve()
    args.hf_manifest_jsonl = args.hf_manifest_jsonl.resolve()
    args.hf_manifest_summary = args.hf_manifest_summary.resolve()
    args.preflight_summary = args.preflight_summary.resolve()
    args.conflicting_controller_root = sorted(
        {path.resolve(strict=False) for path in args.conflicting_controller_root},
        key=str,
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
        raise RuntimeError("guard helper hash mismatch")
    if sha256_file(args.runtime_manifest_tool) != RUNTIME_TREE_MANIFEST_TOOL_SHA256:
        raise RuntimeError("runtime tree manifest tool hash mismatch")
    if sha256_file(args.runtime_manifest_test) != RUNTIME_TREE_MANIFEST_TEST_SHA256:
        raise RuntimeError("runtime tree manifest test hash mismatch")
    if (
        args.monitor.parent != args.guard_helper.parent
        or args.idle_gate.parent != args.guard_helper.parent
    ):
        raise RuntimeError("guard scripts/helper must share one immutable directory")
    runtime_tree_manifests = validate_all_runtime_tree_manifests(args)
    runtime_manifest_validation_kwargs = {
        "python": args.python,
        "tool": args.runtime_manifest_tool,
        "transformers_root": args.transformers_root,
        "transformers_overlay_jsonl": args.transformers_overlay_manifest_jsonl,
        "transformers_overlay_summary": args.transformers_overlay_manifest_summary,
        "transformers_package_jsonl": args.transformers_manifest_jsonl,
        "transformers_package_summary": args.transformers_manifest_summary,
        "hf_snapshot_root": args.hf_snapshot_root,
        "hf_jsonl": args.hf_manifest_jsonl,
        "hf_summary": args.hf_manifest_summary,
    }
    videos = [args.corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    if not all(video.is_file() for video in videos):
        raise FileNotFoundError("eight-video corpus is incomplete")
    for video in videos:
        if video.stat().st_size != VIDEO_BYTES or sha256_file(video) != VIDEO_SHA256:
            raise RuntimeError(f"video corpus mismatch: {video}")
    support_root = args.harness.parent
    preflight_audit = validate_preflight_summary(
        args.preflight_summary,
        python=args.python,
        harness_sha256=CAMPAIGN_HARNESS_SHA256,
        source_root=args.root,
        corpus=args.corpus,
        transformers_root=args.transformers_root,
        current_artifacts={
            "driver": Path(__file__).resolve(),
            "harness": args.harness,
            "monitor": args.monitor,
            "idle_gate": args.idle_gate,
            "guard_helper": args.guard_helper,
            "runtime_manifest_tool": args.runtime_manifest_tool,
            "runtime_manifest_test": args.runtime_manifest_test,
            "pilot_runner": support_root / "run_pynv_endpoint_persistent_preflight.py",
            "pixel_preflight": support_root / "preflight_pynv_endpoint_pixel_parity.py",
        },
        current_runtime_manifests=runtime_tree_manifests,
        conflicting_controller_roots=args.conflicting_controller_root,
    )

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
        "runtime_tree_manifest_tool": {
            "path": str(args.runtime_manifest_tool),
            "sha256": sha256_file(args.runtime_manifest_tool),
        },
        "runtime_tree_manifest_test": {
            "path": str(args.runtime_manifest_test),
            "sha256": sha256_file(args.runtime_manifest_test),
        },
        "runtime_tree_manifests": runtime_tree_manifests,
        "preflight_summary": {
            "path": str(args.preflight_summary),
            "sha256": sha256_file(args.preflight_summary),
            "audit": preflight_audit,
        },
    }
    # All expensive read-only input, byte-tree, corpus, and preflight validation
    # above completes before the publication evidence namespace is created.
    args.results.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "status": "collection_running",
        "started_utc": datetime.now(UTC).isoformat(),
        "source_root": str(args.root),
        "results_root": str(args.results),
        "publication_clean": {
            "fresh_results_root_required": True,
            "prior_cell_reuse_count": 0,
            "historical_campaign_dependencies": [],
        },
        "commits": COMMITS,
        "schedule": SCHEDULE,
        "artifacts": artifacts,
        "configuration": {
            "model": MODEL,
            "revision": REVISION,
            "protocol": (
                "non-streaming persistent HTTP/1.1 chat/completions; exact C-sized "
                "pool per block; E2E latency only; no TTFT"
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
                "post-collection only; token differences do not change collection status"
            ),
            "cell_watchdog_seconds": 3600,
            "cell_watchdog_grace_seconds": 120,
            "per_attempt_idle_seconds": CELL_IDLE_SECONDS,
            "per_attempt_idle_timeout_seconds": CELL_IDLE_TIMEOUT_SECONDS,
            "guard_refinement_only": {
                "process_matching": (
                    "exact argv entrypoint/module and /proc comm; no broad command/"
                    "cwd substring matching"
                ),
                "telemetry": "direct NVML plus append-only JSONL",
                "monitor_coverage_maximum_gap_seconds": (
                    MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS
                ),
            },
        },
        "cells": [],
    }
    manifest_path = args.results / "matrix-manifest.json"
    write_json(manifest_path, manifest)
    _ACTIVE_MANIFEST_PATH = manifest_path

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    environment.update(hf_cache_environment(args.hf_snapshot_root))
    ingress_output = args.results / "ingress-idle-gate.json"
    ingress_sample_log = args.results / "ingress-idle-gate.samples.jsonl"
    if ingress_output.exists() or ingress_sample_log.exists():
        raise FileExistsError("refusing to overwrite ingress idle-gate evidence")
    ingress_completed = run(
        build_idle_gate_command(
            python=args.python,
            idle_gate=args.idle_gate,
            output=ingress_output,
            seconds=INGRESS_IDLE_SECONDS,
            timeout=INGRESS_IDLE_TIMEOUT_SECONDS,
            conflicting_controller_roots=args.conflicting_controller_root,
        ),
        env=environment,
        capture_output=True,
    )
    if ingress_completed.returncode:
        raise RuntimeError(ingress_completed.stderr or ingress_completed.stdout)
    ingress_report = json.loads(ingress_output.read_text())
    if (
        ingress_report.get("passed") is not True
        or ingress_report.get("configuration", {}).get("required_idle_seconds")
        != INGRESS_IDLE_SECONDS
        or ingress_report.get("configuration", {}).get("timeout_seconds")
        != INGRESS_IDLE_TIMEOUT_SECONDS
        or ingress_report.get("configuration", {}).get("idle_max_load_1m_per_cpu")
        != 0.25
        or ingress_report.get("configuration", {}).get("conflicting_controller_roots")
        != [str(path) for path in args.conflicting_controller_root]
        or ingress_report.get("guard_helper", {}).get("sha256") != GUARD_HELPER_SHA256
    ):
        raise RuntimeError("ingress idle-gate evidence/configuration mismatch")
    ingress_sample_audit = validate_jsonl_binding(
        ingress_report,
        expected_path=ingress_sample_log,
        expected_suffix="-idle-gate.samples.jsonl",
    )
    ingress_evidence_audit = validate_idle_gate_evidence(
        {
            "report_path": str(ingress_output),
            "report": ingress_report,
            "report_sha256": sha256_file(ingress_output),
            "sample_log_audit": ingress_sample_audit,
        },
        expected_seconds=INGRESS_IDLE_SECONDS,
        expected_timeout=INGRESS_IDLE_TIMEOUT_SECONDS,
        conflicting_controller_roots=args.conflicting_controller_root,
    )
    manifest["ingress_idle_gate"] = {
        "report": str(ingress_output),
        "report_sha256": sha256_file(ingress_output),
        "sample_log_audit": ingress_sample_audit,
        "evidence_audit": ingress_evidence_audit,
        "policy": (
            "a terminal 20-minute continuously clean structural CPU/direct-NVML "
            "interval precedes all publication cells"
        ),
    }
    write_json(manifest_path, manifest)
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
                before_attempt_runtime_manifests = runtime_manifest_checkpoint(
                    expected=runtime_tree_manifests,
                    label=f"{stem}:before_attempt",
                    validation_kwargs=runtime_manifest_validation_kwargs,
                )
                live_runtime_artifacts_before = capture_live_runtime_artifact_manifest(
                    harness=args.harness,
                    python=args.python,
                    source_root=args.root,
                    pythonpath_extras=[args.transformers_root],
                    environment=environment,
                )
                print(f"IDLE_GATE {stem}", flush=True)
                idle = run(
                    build_idle_gate_command(
                        python=args.python,
                        idle_gate=args.idle_gate,
                        output=idle_output,
                        seconds=args.idle_seconds,
                        timeout=args.idle_timeout,
                        conflicting_controller_roots=args.conflicting_controller_root,
                    ),
                    env=environment,
                    capture_output=True,
                )
                if idle.returncode:
                    raise RuntimeError(idle.stderr or idle.stdout)
                idle_report = json.loads(idle_output.read_text())
                if idle_report.get("passed") is not True:
                    raise RuntimeError(f"idle gate did not pass: {idle_output}")
                if (
                    idle_report.get("guard_helper", {}).get("sha256")
                    != GUARD_HELPER_SHA256
                    or idle_report.get("configuration", {}).get(
                        "idle_max_load_1m_per_cpu"
                    )
                    != 0.25
                    or idle_report.get("configuration", {}).get("device_index") != 0
                    or idle_report.get("configuration", {}).get("required_idle_seconds")
                    != CELL_IDLE_SECONDS
                    or idle_report.get("configuration", {}).get("timeout_seconds")
                    != CELL_IDLE_TIMEOUT_SECONDS
                    or idle_report.get("configuration", {}).get(
                        "conflicting_controller_roots"
                    )
                    != [str(path) for path in args.conflicting_controller_root]
                ):
                    raise RuntimeError("idle gate helper provenance mismatch")
                idle_sample_audit = validate_jsonl_binding(
                    idle_report,
                    expected_path=idle_sample_log,
                    expected_suffix="-idle-gate.samples.jsonl",
                )
                idle_evidence_audit = validate_idle_gate_evidence(
                    {
                        "report_path": str(idle_output),
                        "report": idle_report,
                        "report_sha256": sha256_file(idle_output),
                        "sample_log_audit": idle_sample_audit,
                    },
                    expected_seconds=CELL_IDLE_SECONDS,
                    expected_timeout=CELL_IDLE_TIMEOUT_SECONDS,
                    conflicting_controller_roots=args.conflicting_controller_root,
                )
                harness_command = build_harness_command(
                    args,
                    variant=variant,
                    result_path=result_path,
                    videos=videos,
                    concurrencies=concurrencies,
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
                    "idle_gate_sha256": sha256_file(idle_output),
                    "idle_gate_sample_log_audit": idle_sample_audit,
                    "idle_gate_evidence_audit": idle_evidence_audit,
                    "runtime_manifests_before": before_attempt_runtime_manifests,
                    "live_runtime_artifacts_before": live_runtime_artifacts_before,
                    "command": command,
                    "log": str(log_path),
                    "monitor": str(monitor_path),
                    "output": str(result_path),
                    "server_log": str(server_log_path),
                    "started_utc": datetime.now(UTC).isoformat(),
                }
                cell["attempts"].append(attempt_record)
                write_json(manifest_path, manifest)
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
                attempt_record["finished_utc"] = datetime.now(UTC).isoformat()
                attempt_record["returncode"] = completed.returncode
                post_attempt_integrity = post_attempt_integrity_checks(
                    root=args.root,
                    commit=commit,
                    variant=variant,
                    runtime_manifests=runtime_tree_manifests,
                    runtime_label=f"{stem}:after_attempt",
                    runtime_validation_kwargs=runtime_manifest_validation_kwargs,
                )
                attempt_record["source_after_attempt"] = post_attempt_integrity[
                    "source"
                ]
                attempt_record["runtime_manifests_after"] = post_attempt_integrity[
                    "runtime_manifests"
                ]
                live_runtime_artifacts_after = capture_live_runtime_artifact_manifest(
                    harness=args.harness,
                    python=args.python,
                    source_root=args.root,
                    pythonpath_extras=[args.transformers_root],
                    environment=environment,
                )
                attempt_record["live_runtime_artifacts_after"] = (
                    live_runtime_artifacts_after
                )
                if live_runtime_artifacts_after != live_runtime_artifacts_before:
                    raise RuntimeError(
                        f"live runtime artifacts changed during attempt: {stem}"
                    )
                monitor_result = (
                    json.loads(monitor_path.read_text())
                    if monitor_path.is_file()
                    else {}
                )
                monitor_sample_audit = (
                    validate_jsonl_binding(
                        monitor_result,
                        expected_path=monitor_sample_log,
                        expected_suffix="-gpu-monitor.samples.jsonl",
                    )
                    if monitor_result
                    else None
                )
                attempt_record["contaminated"] = monitor_result.get("contaminated")
                attempt_record["timed_out"] = monitor_result.get("timed_out")
                attempt_record["monitor_sample_log_audit"] = monitor_sample_audit
                attempt_record["result_sha256"] = (
                    sha256_file(result_path) if result_path.is_file() else None
                )
                attempt_record["server_log_sha256"] = (
                    sha256_file(server_log_path) if server_log_path.is_file() else None
                )
                attempt_record["monitor_sha256"] = (
                    sha256_file(monitor_path) if monitor_path.is_file() else None
                )
                attempt_record["log_sha256"] = sha256_file(log_path)
                write_json(manifest_path, manifest)
                if monitor_result.get("timed_out"):
                    raise RuntimeError(f"cell watchdog expired for {stem}")
                if (
                    completed.returncode == 99
                    or monitor_result.get("contaminated") is True
                ):
                    contamination_retry_evidence = (
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
                    attempt_record["contamination_retry_evidence"] = (
                        contamination_retry_evidence
                    )
                    print(f"CONTAMINATED {stem}; retrying", flush=True)
                    write_json(manifest_path, manifest)
                    continue
                if completed.returncode:
                    tail = log_path.read_text(errors="replace")[-16000:]
                    raise RuntimeError(f"cell failed {stem}:\n{tail}")
                monitor_result, accepted_monitor_sample_audit = (
                    validate_monitor_evidence(
                        monitor_path,
                        expected_command=harness_command,
                        watchdog_pair=TIMING_MONITOR_WATCHDOG_PAIR,
                        conflicting_controller_roots=args.conflicting_controller_root,
                    )
                )
                if accepted_monitor_sample_audit != monitor_sample_audit:
                    raise RuntimeError(f"accepted monitor JSONL audit changed: {stem}")
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
                    corpus=args.corpus,
                    transformers_root=args.transformers_root,
                    source_root=args.root,
                    server_log_path=server_log_path,
                )
                expected_runtime_fingerprint_sha256 = preflight_audit[
                    "runtime_hardware_fingerprint_contract"
                ]["sha256"]
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
                if (
                    validated_metrics["runtime_hardware_fingerprint"].get(
                        "live_runtime_artifact_manifest"
                    )
                    != live_runtime_artifacts_before
                ):
                    raise RuntimeError(
                        f"result live runtime artifacts differ from attempt: {stem}"
                    )
                coverage = monitor_coverage_audit(result, monitor_result)
                if not coverage["passed"]:
                    attempt_record["accepted"] = False
                    attempt_record["validation_status"] = "rejected_monitor_coverage"
                    attempt_record["monitor_coverage_audit"] = coverage
                    write_json(manifest_path, manifest)
                    raise RuntimeError(
                        f"monitor coverage failed for {stem}: {coverage['blocks']}"
                    )
                attempt_record["accepted"] = True
                attempt_record["validation_status"] = "accepted"
                attempt_record["monitor_coverage_audit"] = coverage
                cell.update(
                    {
                        "status": "passed",
                        "winning_attempt": attempt,
                        "output": str(result_path),
                        "monitor": str(monitor_path),
                        "server_log": str(server_log_path),
                        "validated_metrics": validated_metrics,
                        "monitor_coverage_audit": coverage,
                        "source_after_winning_attempt": attempt_record[
                            "source_after_attempt"
                        ],
                    }
                )
                print(f"PASS {stem}", flush=True)
                write_json(manifest_path, manifest)
                break
            else:
                raise RuntimeError(
                    f"too many contaminated attempts for {variant} rep {rep}"
                )

    last_rep, last_concurrencies, last_variants = SCHEDULE[-1]
    del last_rep, last_concurrencies
    last_variant = last_variants[-1]
    terminal_integrity = post_attempt_integrity_checks(
        root=args.root,
        commit=COMMITS[last_variant],
        variant=last_variant,
        runtime_manifests=runtime_tree_manifests,
        runtime_label="matrix_end",
        runtime_validation_kwargs=runtime_manifest_validation_kwargs,
    )
    manifest["terminal_source_revalidation"] = terminal_integrity["source"]
    manifest["final_runtime_tree_manifest_revalidation"] = terminal_integrity[
        "runtime_manifests"
    ]
    terminal_live_runtime_artifacts = capture_live_runtime_artifact_manifest(
        harness=args.harness,
        python=args.python,
        source_root=args.root,
        pythonpath_extras=[args.transformers_root],
        environment=environment,
    )
    if terminal_live_runtime_artifacts != live_runtime_artifacts_after:
        raise RuntimeError("live runtime artifacts changed at campaign end")
    manifest["terminal_live_runtime_artifact_revalidation"] = (
        terminal_live_runtime_artifacts
    )
    write_json(manifest_path, manifest)
    token_parity_path = args.results / "token-parity.json"
    if token_parity_path.exists():
        raise FileExistsError(f"refusing to overwrite token audit: {token_parity_path}")
    token_parity = strict_token_audit(manifest["cells"])
    write_json(token_parity_path, token_parity)
    token_parity_binding = {
        "path": str(token_parity_path),
        "bytes": token_parity_path.stat().st_size,
        "sha256": sha256_file(token_parity_path),
        "status": token_parity["status"],
        "mismatch_counts": token_parity["mismatch_counts"],
        "untimed_top20_logprob_diagnostic_required": token_parity[
            "untimed_top20_logprob_diagnostic_required"
        ],
    }

    summary = collection_summary(manifest["cells"])
    summary["strict_token_audit"] = token_parity_binding
    publication_status = {
        "passed_exact": "passed_exact",
        "failed_input_parity": "invalid_input_parity",
        "completion_or_text_mismatch": "timing_passed_completion_mismatch",
    }[token_parity["status"]]
    summary["status"] = publication_status
    summary["guard_refinement_provenance"] = {
        "publication_clean_restart": True,
        "prior_cell_reuse_count": 0,
        "refined_runner_sha256": sha256_file(Path(__file__).resolve()),
        "guard_helper_sha256": GUARD_HELPER_SHA256,
        "idle_gate_sha256": IDLE_GATE_SHA256,
        "gpu_monitor_sha256": GPU_MONITOR_SHA256,
        "monitor_coverage_maximum_gap_seconds": (MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS),
    }
    summary_path = args.results / "collection-summary.json"
    write_json(summary_path, summary)
    manifest["status"] = publication_status
    manifest["finished_utc"] = datetime.now(UTC).isoformat()
    manifest["collection_summary"] = {
        "path": str(summary_path),
        "bytes": summary_path.stat().st_size,
        "sha256": sha256_file(summary_path),
    }
    manifest["strict_token_audit"] = token_parity_binding
    write_json(manifest_path, manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if publication_status == "invalid_input_parity":
        raise SystemExit(2)


def sanitized_failure_category(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "controller_interrupted"
    if isinstance(error, FileExistsError):
        return "evidence_collision"
    if isinstance(error, TimeoutError):
        return "bounded_timeout"
    if isinstance(error, (ValueError, KeyError, TypeError, AssertionError)):
        return "contract_validation_failure"
    if isinstance(error, RuntimeError):
        return "runtime_validation_or_workload_failure"
    if isinstance(error, SystemExit):
        return "explicit_process_exit"
    return "unexpected_controller_failure"


def record_terminal_collection_failure(error: BaseException) -> None:
    path = _ACTIVE_MANIFEST_PATH
    if path is None or not path.is_file():
        return
    manifest = json.loads(path.read_text())
    if manifest.get("status") not in {"collection_running", "running"}:
        return
    active_cell = next(
        (
            {
                "rep": cell.get("rep"),
                "position": cell.get("position"),
                "variant": cell.get("variant"),
                "attempt_count": len(cell.get("attempts", [])),
            }
            for cell in reversed(manifest.get("cells", []))
            if cell.get("status") == "running"
        ),
        None,
    )
    manifest.update(
        {
            "status": "collection_failed",
            "finished_utc": datetime.now(UTC).isoformat(),
            "failure": {
                "category": sanitized_failure_category(error),
                "exception_type": type(error).__name__,
                "active_cell": active_cell,
                "message_omitted_from_manifest": True,
                "results_must_not_be_reused": True,
            },
        }
    )
    write_json(path, manifest)


def main() -> None:
    try:
        run_main()
    except BaseException as error:
        try:
            record_terminal_collection_failure(error)
        except BaseException:
            # Preserve the initiating error; a corrupt/unwritable evidence namespace
            # is itself fail-closed and must never be reused.
            pass
        raise


if __name__ == "__main__":
    main()
