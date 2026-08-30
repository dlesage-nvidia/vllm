# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark Qwen3-VL video captioning through a fresh ``vllm serve``.

The benchmark intentionally keeps the client dependency-free so the same
script can drive multiple vLLM source trees. It launches one server, warms it
up, and then measures one or more concurrency levels in command-line order.
All requests use deterministic sampling and return their exact token IDs.

Example::

    .venv/bin/python benchmarks/multimodal/benchmark_pynvvideocodec_e2e.py \
      --variant pynvvideocodec-tchw \
      --video /data/clip-0.mp4 --video /data/clip-1.mp4 \
      --backend pynvvideocodec \
      --backend-kwargs '{"hw_decoders":4,"output_layout":"tchw"}' \
      --concurrency 1 --concurrency 4 \
      --warmup-requests 4 --requests 20 \
      --output /results/pynvvideocodec-tchw.json

Use ``--backend opencv`` for a CPU decode baseline. ``--backend default``
omits the codec override and exercises the source tree's default codec.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import http.client
import json
import math
import os
import platform
import queue
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_PROMPT = "Describe this video concisely and factually."
DEFAULT_VIDEO_PIXEL_BUDGET = (1024, 576)
SERVED_MODEL_NAME = "qwen3-vl-video-throughput"
SERVER_LOG_TAIL_BYTES = 4 * 1024 * 1024
CLIENT_ABORT_DRAIN_TIMEOUT_SECONDS = 5.0
SERVER_HTTP_KEEP_ALIVE_TIMEOUT_SECONDS = 3600

PERFORMANCE_PARITY_CONFIGURATION_FIELDS = (
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
ENDPOINT_TREATMENT_CONFIGURATION_FIELDS = (
    "backend_kwargs",
    "server_media_io_kwargs",
    "video_kwargs_for_metric_derivation",
    "extra_server_argv",
)

CONTROLLED_SERVER_OPTIONS = frozenset(
    {
        "--allowed-local-media-path",
        "--api-server-count",
        "--config",
        "--dtype",
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "--host",
        "--kv-cache-memory-bytes",
        "--limit-mm-per-prompt",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--media-io-kwargs",
        "--mm-ipc-gpu-memory-gb",
        "--mm-processor-cache-gb",
        "--mm-processor-kwargs",
        "--model",
        "--no-enable-prefix-caching",
        "--port",
        "--revision",
        "--seed",
        "--served-model-name",
        "--tensor-parallel-size",
    }
)
PERFORMANCE_ENV_PREFIXES = (
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "FLASHINFER_",
    "HF_",
    "HUGGINGFACE_",
    "MKL_",
    "NCCL_",
    "NUMEXPR_",
    "NVIDIA_",
    "OMP_",
    "OPENBLAS_",
    "PYTORCH_",
    "RAY_",
    "TOKENIZERS_",
    "TORCH_",
    "TRANSFORMERS_",
    "TRITON_",
    "VECLIB_",
    "VLLM_",
)
PERFORMANCE_ENV_NAMES = frozenset(
    {
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
    }
)
SECRET_ENV_NAME_PARTS = frozenset(
    {
        "AUTH",
        "COOKIE",
        "CREDENTIAL",
        "CREDENTIALS",
        "PASS",
        "PASSWD",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    }
)


class TerminationRequested(RuntimeError):
    """Raised in the main thread so the server can be cleaned up on signals."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"received {signal.Signals(signum).name}")


class HttpRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        body: str | None,
        transport: Mapping[str, Any] | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.transport = None if transport is None else dict(transport)
        super().__init__(message)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def dimensions(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must have the form WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must have the form WIDTHxHEIGHT") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return width, height


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return parsed


def port_number(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def json_object(value: str) -> dict[str, Any]:
    source = value
    if value.startswith("@"):
        path = Path(value[1:]).expanduser()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as error:
            raise argparse.ArgumentTypeError(
                f"could not read JSON file {path}: {error}"
            ) from error
    try:
        parsed = json.loads(source)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must decode to a JSON object")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_root,
        help=f"vLLM checkout to serve (default: {default_root})",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable (default: SOURCE_ROOT/.venv/bin/python)",
    )
    parser.add_argument(
        "--pythonpath-extra",
        type=Path,
        action="append",
        default=[],
        help=(
            "Extra import directory; repeat to place entries after SOURCE_ROOT "
            "and before inherited PYTHONPATH"
        ),
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Stable label for this source/backend variant",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        required=True,
        help="Local encoded video; repeat to cycle paths by request index",
    )
    parser.add_argument(
        "--allowed-local-media-path",
        type=Path,
        help="Server media root (default: common parent of all --video paths)",
    )
    parser.add_argument(
        "--backend",
        default="opencv",
        help=(
            "Video codec backend, e.g. opencv or pynvvideocodec; use "
            "'default' to omit the codec override (default: opencv)"
        ),
    )
    parser.add_argument(
        "--backend-kwargs",
        type=json_object,
        default={},
        metavar="JSON|@FILE",
        help="JSON object merged into the server's video media-I/O kwargs",
    )
    parser.add_argument(
        "--media-io-kwargs",
        type=json_object,
        default={},
        metavar="JSON|@FILE",
        help=(
            "Full server media-I/O JSON, recursively merged last; accepts "
            "literal JSON or @FILE"
        ),
    )
    parser.add_argument(
        "--request-media-io-kwargs",
        type=json_object,
        default={},
        metavar="JSON|@FILE",
        help="Optional request-level media-I/O JSON sent with every request",
    )
    parser.add_argument(
        "--frames",
        type=positive_int,
        default=32,
        help="Exact Qwen3-VL sampling target (default: 32)",
    )
    parser.add_argument(
        "--video-pixel-budget",
        type=dimensions,
        default=DEFAULT_VIDEO_PIXEL_BUDGET,
        metavar="WIDTHxHEIGHT",
        help=(
            "Per-sampled-frame pixel budget used to derive Qwen3-VL's total "
            "max_pixels value (default: 1024x576)"
        ),
    )
    parser.add_argument(
        "--warmup-requests",
        type=positive_int,
        default=2,
        help=(
            "Positive minimum warmup requests before each concurrency block; "
            "raised to cover every video and persistent client slot (default: 2)"
        ),
    )
    parser.add_argument(
        "--requests",
        type=positive_int,
        default=20,
        help="Measured requests per concurrency block (default: 20)",
    )
    parser.add_argument(
        "--requests-by-concurrency",
        type=json_object,
        default={},
        metavar="JSON|@FILE",
        help=(
            "Optional concurrency-to-measured-request-count mapping, e.g. "
            '\'{"8":64,"16":128,"32":256}\'; unspecified levels use '
            "--requests"
        ),
    )
    parser.add_argument(
        "--warmup-requests-by-concurrency",
        type=json_object,
        default={},
        metavar="JSON|@FILE",
        help=(
            "Optional concurrency-to-warmup-request-count mapping; unspecified "
            "levels use --warmup-requests"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        action="append",
        help=(
            "Maximum in-flight requests; repeat to benchmark levels in CLI "
            "occurrence order (default: 1)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parity-reference",
        type=Path,
        help=(
            "Passed baseline result JSON whose measured prompt/completion token "
            "IDs must exactly match this run"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing an existing output JSON",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--output-len",
        "--osl",
        dest="output_len",
        type=positive_int,
        default=32,
        help="Exact generated token count per request (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=port_number, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=positive_int, default=1)
    parser.add_argument("--max-model-len", type=positive_int, default=32768)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=positive_int,
        default=9216,
        help=(
            "Scheduler and encoder-cache token budget; the default exactly "
            "covers a 32-frame 1024x576 Qwen3-VL video (default: 9216)"
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=positive_int,
        help="Server sequence capacity (default: maximum --concurrency)",
    )
    parser.add_argument(
        "--mm-ipc-gpu-memory-gb",
        type=positive_float,
        default=2.0,
        help="Frontend multimodal GPU memory budget in GiB (default: 2)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=positive_float,
        help="Optional vLLM GPU memory utilization override",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=positive_int,
        help="Optional fixed KV-cache allocation in bytes",
    )
    parser.add_argument("--startup-timeout", type=positive_float, default=600.0)
    parser.add_argument("--request-timeout", type=positive_float, default=600.0)
    parser.add_argument("--shutdown-timeout", type=positive_float, default=60.0)
    parser.add_argument(
        "--settle-seconds",
        type=nonnegative_float,
        default=0.0,
        help="Unmeasured delay after each warmup batch (default: 0)",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help=(
            "One additional vllm serve argv token; repeat once per token, "
            "using --server-arg=--flag for flag-shaped values"
        ),
    )
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def environment_name_is_secret(name: str) -> bool:
    parts = set(name.upper().split("_"))
    return bool(parts & SECRET_ENV_NAME_PARTS) or any(
        marker in name.upper() for marker in ("ACCESS_KEY", "API_KEY", "PRIVATE_KEY")
    )


def performance_environment_provenance(
    env: Mapping[str, str],
) -> dict[str, str | dict[str, Any]]:
    selected = {
        name: value
        for name, value in env.items()
        if name in PERFORMANCE_ENV_NAMES or name.startswith(PERFORMANCE_ENV_PREFIXES)
    }
    return {
        name: (
            {"redacted": True, "set": True, "value_length": len(value)}
            if environment_name_is_secret(name)
            else value
        )
        for name, value in sorted(selected.items())
    }


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    text: bool = True,
) -> str | bytes:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=text,
        timeout=timeout,
    )
    if result.returncode:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout
        raise RuntimeError(f"{shlex.join(command)} failed: {detail}")
    if text:
        assert isinstance(result.stdout, str)
        return result.stdout.strip()
    assert isinstance(result.stdout, bytes)
    return result.stdout


def git_output(source_root: Path, *arguments: str) -> str:
    output = run_command(["git", "-C", str(source_root), *arguments])
    assert isinstance(output, str)
    return output


def source_provenance(source_root: Path) -> dict[str, Any]:
    top_level = Path(git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != source_root:
        raise ValueError(
            f"--source-root must be the repository root ({top_level}), got "
            f"{source_root}"
        )

    diff = run_command(
        ["git", "-C", str(source_root), "diff", "--binary", "--full-index", "HEAD"],
        text=False,
    )
    assert isinstance(diff, bytes)
    untracked_output = run_command(
        [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        text=False,
    )
    assert isinstance(untracked_output, bytes)
    untracked: list[dict[str, Any]] = []
    for raw_path in untracked_output.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or not path.is_file():
            continue
        untracked.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "root": str(source_root),
        "commit": git_output(source_root, "rev-parse", "HEAD^{commit}"),
        "tree": git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "branch": git_output(source_root, "branch", "--show-current"),
        "commit_subject": git_output(source_root, "show", "-s", "--format=%s"),
        "commit_time": git_output(source_root, "show", "-s", "--format=%cI"),
        "status_porcelain_v2": git_output(
            source_root, "status", "--porcelain=v2", "--branch", "--untracked-files=all"
        ),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff),
        "untracked_files": untracked,
        "untracked_manifest_sha256": sha256_json(untracked),
    }


def server_environment(
    source_root: Path, python: Path, pythonpath_extras: Sequence[Path]
) -> dict[str, str]:
    env = dict(os.environ)
    old_pythonpath = env.get("PYTHONPATH")
    env.update(
        {
            "PATH": f"{python.parent}{os.pathsep}{env.get('PATH', '')}",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            # A block starts measurement only after every warmup request has
            # completed.  With heterogeneous video latency, an early slot can
            # otherwise exceed vLLM's five-second default while waiting for the
            # slowest slot and be closed before the measured wave starts.
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": str(SERVER_HTTP_KEEP_ALIVE_TIMEOUT_SECONDS),
        }
    )
    pythonpath_entries = [str(source_root), *(str(path) for path in pythonpath_extras)]
    if old_pythonpath:
        pythonpath_entries.extend(
            entry for entry in old_pythonpath.split(os.pathsep) if entry
        )
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def python_provenance(
    python: Path, source_root: Path, env: Mapping[str, str]
) -> dict[str, Any]:
    probe = r"""
import importlib.metadata
import importlib.util
import json
import platform
import sys
import numpy
import torch
import torch._C
import vllm

try:
    import numpy._core._multiarray_umath as numpy_native_core
except ImportError:
    import numpy.core._multiarray_umath as numpy_native_core

try:
    from torch.utils.cpp_extension import CUDA_HOME
except Exception:
    CUDA_HOME = None

packages = {}
for name in (
    "vllm", "torch", "transformers", "PyNvVideoCodec", "opencv-python",
    "av", "torchcodec", "numpy",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
module_origins = {}
for name in (
    "vllm", "torch", "transformers", "PyNvVideoCodec", "cv2", "av",
    "torchcodec", "numpy",
):
    spec = importlib.util.find_spec(name)
    module_origins[name] = spec.origin if spec is not None else None
print(json.dumps({
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "module_origins": module_origins,
    "native_module_origins": {
        "torch._C": torch._C.__file__,
        "numpy._core._multiarray_umath": numpy_native_core.__file__,
    },
    "packages": packages,
    "python_version": platform.python_version(),
    "sys_path": sys.path,
    "sys_version": sys.version,
    "torch_runtime": {
        "torch_version": str(torch.__version__),
        "compiled_cuda_version": torch.version.cuda,
        "cuda_home": CUDA_HOME,
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
    },
    "vllm_file": vllm.__file__,
    "vllm_version": vllm.__version__,
}))
"""
    output = run_command(
        [str(python), "-c", probe], cwd=source_root, env=env, timeout=120.0
    )
    assert isinstance(output, str)
    result = json.loads(output)
    imported_vllm = Path(result["vllm_file"]).resolve()
    if not imported_vllm.is_relative_to(source_root):
        raise RuntimeError(
            f"Python imported vLLM from {imported_vllm}, not {source_root}"
        )
    artifact_candidates = {python, imported_vllm}
    module_origins = result.get("module_origins", {})
    if not isinstance(module_origins, Mapping):
        raise RuntimeError("Python provenance returned invalid module origins")
    for origin in module_origins.values():
        if isinstance(origin, str):
            artifact_candidates.add(Path(origin))
    native_module_origins = result.get("native_module_origins", {})
    if (
        not isinstance(native_module_origins, Mapping)
        or set(native_module_origins) != {"torch._C", "numpy._core._multiarray_umath"}
        or any(not isinstance(origin, str) for origin in native_module_origins.values())
    ):
        raise RuntimeError("Python provenance returned invalid native module origins")
    artifact_candidates.update(
        Path(origin) for origin in native_module_origins.values()
    )
    for pattern in ("*.so", "*.pyd"):
        artifact_candidates.update((source_root / "vllm").rglob(pattern))
    pynvvideocodec_origin = module_origins.get("PyNvVideoCodec")
    if isinstance(pynvvideocodec_origin, str):
        package_dir = Path(pynvvideocodec_origin).parent
        artifact_candidates.update(package_dir.glob("*.so*"))

    artifacts_by_resolved_path: dict[str, dict[str, Any]] = {}
    for candidate in sorted(artifact_candidates, key=str):
        try:
            if not candidate.is_file():
                continue
            identity = file_identity(candidate)
        except OSError:
            continue
        artifacts_by_resolved_path.setdefault(identity["resolved_path"], identity)
    result["runtime_artifacts"] = [
        artifacts_by_resolved_path[path] for path in sorted(artifacts_by_resolved_path)
    ]
    torch_runtime = result.get("torch_runtime")
    if not isinstance(torch_runtime, dict):
        raise RuntimeError("Python provenance returned invalid torch runtime")
    cuda_home = torch_runtime.get("cuda_home")
    nvcc = Path(cuda_home) / "bin" / "nvcc" if isinstance(cuda_home, str) else None
    if nvcc is not None and nvcc.is_file():
        torch_runtime["nvcc"] = {
            **file_identity(nvcc),
            "version_output": run_command([str(nvcc), "--version"], timeout=30.0),
        }
    else:
        torch_runtime["nvcc"] = None
    return result


def hardware_provenance() -> dict[str, Any]:
    query = (
        "index,name,uuid,driver_version,memory.total,compute_cap,pci.bus_id,"
        "pstate,clocks.sm,clocks.mem"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = run_command(command)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as error:
        gpu = f"unavailable: {type(error).__name__}: {error}"
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "logical_cpus": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_query": query,
        "nvidia_smi_output": gpu,
    }


def _parse_fraction(value: object) -> float | None:
    if not isinstance(value, str) or value in {"", "N/A", "0/0"}:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            parsed = float(numerator) / float(denominator)
        else:
            parsed = float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_metadata_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_metadata_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def probe_video(video: Path, python: Path, env: Mapping[str, str]) -> dict[str, Any]:
    """Collect dimensions needed for MPix/s without decoding benchmark frames."""

    attempts: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "width": None,
        "height": None,
        "frame_count": None,
        "frames_per_second": None,
        "duration_seconds": None,
        "field_sources": {},
    }
    ffprobe_command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(video),
    ]
    try:
        output = run_command(ffprobe_command, timeout=60.0)
        assert isinstance(output, str)
        raw = json.loads(output)
        streams = raw.get("streams")
        stream = streams[0] if isinstance(streams, list) and streams else {}
        if not isinstance(stream, dict):
            stream = {}
        format_record = raw.get("format")
        if not isinstance(format_record, dict):
            format_record = {}
        metadata.update(
            {
                "width": _positive_metadata_int(stream.get("width")),
                "height": _positive_metadata_int(stream.get("height")),
                "frame_count": _positive_metadata_int(stream.get("nb_frames")),
                "frames_per_second": _parse_fraction(
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                ),
                "duration_seconds": _positive_metadata_float(
                    stream.get("duration") or format_record.get("duration")
                ),
            }
        )
        for field, value in metadata.items():
            if field not in {"field_sources", "probe_attempts"} and value is not None:
                metadata["field_sources"][field] = "ffprobe"
        attempts.append({"method": "ffprobe", "status": "passed", "raw": raw})
    except (
        FileNotFoundError,
        RuntimeError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as error:
        attempts.append(
            {
                "method": "ffprobe",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    missing = [
        field
        for field in ("width", "height", "frame_count", "frames_per_second")
        if metadata[field] is None
    ]
    if missing:
        opencv_probe = r"""
import cv2
import json
import sys

capture = cv2.VideoCapture(sys.argv[1])
try:
    if not capture.isOpened():
        raise RuntimeError("cv2.VideoCapture could not open the video")
    print(json.dumps({
        "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "frame_count": capture.get(cv2.CAP_PROP_FRAME_COUNT),
        "frames_per_second": capture.get(cv2.CAP_PROP_FPS),
    }))
finally:
    capture.release()
"""
        try:
            output = run_command(
                [str(python), "-c", opencv_probe, str(video)],
                env=env,
                timeout=60.0,
            )
            assert isinstance(output, str)
            raw = json.loads(output)
            for field in missing:
                parser = (
                    _positive_metadata_int
                    if field in {"width", "height", "frame_count"}
                    else _positive_metadata_float
                )
                value = parser(raw.get(field))
                if value is not None:
                    metadata[field] = value
                    metadata["field_sources"][field] = "opencv-metadata"
            attempts.append(
                {"method": "opencv-metadata", "status": "passed", "raw": raw}
            )
        except (
            FileNotFoundError,
            RuntimeError,
            subprocess.TimeoutExpired,
            json.JSONDecodeError,
        ) as error:
            attempts.append(
                {
                    "method": "opencv-metadata",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    if metadata["duration_seconds"] is None:
        frame_count = metadata["frame_count"]
        fps = metadata["frames_per_second"]
        if isinstance(frame_count, int) and isinstance(fps, (int, float)):
            metadata["duration_seconds"] = frame_count / fps
            metadata["field_sources"]["duration_seconds"] = "frame-count/fps"
    metadata["probe_attempts"] = attempts
    metadata["mpix_inputs_derivable"] = all(
        metadata[field] is not None for field in ("width", "height", "frame_count")
    )
    return metadata


def video_provenance(
    index: int, video: Path, python: Path, env: Mapping[str, str]
) -> dict[str, Any]:
    stat = video.stat()
    return {
        "video_index": index,
        "path": str(video),
        "file_uri": video.as_uri(),
        "bytes": stat.st_size,
        "sha256": sha256_file(video),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "probe": probe_video(video, python, env),
    }


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        old_value = merged.get(key)
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(old_value, value)
        else:
            merged[key] = value
    return merged


def resolved_media_io_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    video_kwargs: dict[str, Any] = {
        "video_backend": "qwen3_vl",
        "min_frames": args.frames,
        "max_frames": args.frames,
    }
    if args.backend.lower() not in {"default", "auto", "none"}:
        video_kwargs["backend"] = args.backend
    video_kwargs = deep_merge(video_kwargs, args.backend_kwargs)
    resolved = deep_merge({"video": video_kwargs}, args.media_io_kwargs)
    if not isinstance(resolved.get("video"), dict):
        raise ValueError("resolved media-I/O key 'video' must be a JSON object")
    return resolved


def video_kwargs_for_metric_derivation(
    static_media_io_kwargs: Mapping[str, Any],
    request_media_io_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    static_video = static_media_io_kwargs.get("video", {})
    request_video = request_media_io_kwargs.get("video", {})
    if not isinstance(static_video, Mapping) or not isinstance(request_video, Mapping):
        raise ValueError("media-I/O key 'video' must be a JSON object")
    if request_media_io_kwargs:
        return None, (
            "request-level media-I/O kwargs use source-specific merge semantics; "
            "the harness does not claim an effective video configuration"
        )
    return dict(static_video), None


def expected_video_work(
    video_record: Mapping[str, Any], video_kwargs: Mapping[str, Any] | None
) -> dict[str, Any]:
    probe = video_record["probe"]
    assert isinstance(probe, Mapping)
    width = probe.get("width")
    height = probe.get("height")
    total_frames = probe.get("frame_count")
    original_fps = probe.get("frames_per_second")
    min_frames = video_kwargs.get("min_frames", 4) if video_kwargs is not None else None
    max_frames = (
        video_kwargs.get("max_frames", 768) if video_kwargs is not None else None
    )
    target_fps = video_kwargs.get("fps", 2) if video_kwargs is not None else None

    reason: str | None = None
    sampled_frames: int | None = None
    if video_kwargs is None:
        reason = (
            "unavailable because request-level media-I/O kwargs require "
            "source-specific merge semantics"
        )
    elif not all(isinstance(value, int) and value > 0 for value in (width, height)):
        reason = "source width/height unavailable"
    elif not isinstance(total_frames, int) or total_frames <= 0:
        reason = "source frame count unavailable"
    elif not all(
        isinstance(value, (int, float)) and value > 0
        for value in (min_frames, max_frames)
    ):
        reason = "min_frames/max_frames are not positive numbers"
    elif min_frames > max_frames:
        reason = "min_frames exceeds max_frames"
    elif min_frames == max_frames:
        sampled_frames = min(int(min_frames), total_frames)
        reason = "Qwen3-VL equal min/max frame clamp"
    elif not isinstance(original_fps, (int, float)) or original_fps <= 0:
        reason = "source fps unavailable for Qwen3-VL sampling calculation"
    elif not isinstance(target_fps, (int, float)) or target_fps <= 0:
        reason = "target fps is not positive"
    else:
        sampled_frames = int(total_frames / original_fps * target_fps)
        sampled_frames = min(
            max(sampled_frames, int(min_frames)), int(max_frames), total_frames
        )
        reason = "Qwen3-VL fps sampling calculation"

    megapixels = None
    if sampled_frames is not None:
        assert isinstance(width, int) and isinstance(height, int)
        megapixels = sampled_frames * width * height / 1_000_000
    return {
        "source_width": width,
        "source_height": height,
        "source_frame_count": total_frames,
        "sampled_frames": sampled_frames,
        "sampled_source_megapixels_estimate": megapixels,
        "derivation": reason,
    }


def common_media_root(videos: Sequence[Path]) -> Path:
    common = Path(os.path.commonpath([str(video.parent) for video in videos]))
    if common == Path(common.anchor):
        raise ValueError(
            "videos do not share a safe non-root parent; pass "
            "--allowed-local-media-path explicitly after placing them under "
            "one dedicated directory"
        )
    return common


def validate_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, list[Path], Path, Path]:
    if os.name != "posix":
        raise RuntimeError("this server process-group harness requires POSIX")
    source_root = args.source_root.expanduser().resolve(strict=True)
    python_arg = args.python or (source_root / ".venv/bin/python")
    python = python_arg.expanduser().absolute()
    videos = [video.expanduser().resolve(strict=True) for video in args.video]
    output = args.output.expanduser().absolute()
    server_log_path = output.with_name(output.stem + ".server.log")
    if not source_root.is_dir() or not (source_root / "vllm").is_dir():
        raise ValueError(f"not a vLLM source root: {source_root}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"not an executable Python: {python}")
    pythonpath_extras = [
        path.expanduser().resolve(strict=True) for path in args.pythonpath_extra
    ]
    if any(not path.is_dir() for path in pythonpath_extras):
        raise ValueError("every --pythonpath-extra value must be a directory")
    args.pythonpath_extra = pythonpath_extras
    if any(not video.is_file() for video in videos):
        raise ValueError("every --video value must be a regular file")
    if output in videos:
        raise ValueError("--output must not overwrite a --video file")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output exists; pass --overwrite to replace it: {output}"
        )
    if server_log_path.exists():
        raise FileExistsError(
            "refusing to overwrite append-only server log sidecar: "
            f"{server_log_path}"
        )
    args.server_log_path = server_log_path

    if args.allowed_local_media_path is None:
        allowed_media_root = common_media_root(videos)
    else:
        allowed_media_root = args.allowed_local_media_path.expanduser().resolve(
            strict=True
        )
    if not allowed_media_root.is_dir():
        raise ValueError(
            f"--allowed-local-media-path is not a directory: {allowed_media_root}"
        )
    if allowed_media_root == Path(allowed_media_root.anchor):
        raise ValueError("refusing to expose a filesystem root as local media")
    for video in videos:
        if not video.is_relative_to(allowed_media_root):
            raise ValueError(
                f"video {video} is outside media root {allowed_media_root}"
            )
    return source_root, python, videos, output, allowed_media_root


def ensure_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"127.0.0.1:{port} is not available") from error


def validate_extra_server_args(arguments: Sequence[str]) -> None:
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        option = argument.partition("=")[0].replace("_", "-")
        if option in CONTROLLED_SERVER_OPTIONS:
            raise ValueError(
                f"--server-arg may not override harness-controlled option {option}; "
                "use the corresponding benchmark option instead"
            )


def server_command(
    args: argparse.Namespace,
    python: Path,
    allowed_media_root: Path,
    media_io_kwargs: Mapping[str, Any],
    max_num_seqs: int,
) -> list[str]:
    mm_processor_kwargs = video_mm_processor_kwargs(args)
    limit_mm_per_prompt = video_limit_mm_per_prompt(args)
    command = [
        str(python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--revision",
        args.revision,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(max_num_seqs),
        "--api-server-count",
        "1",
        "--limit-mm-per-prompt",
        json.dumps(limit_mm_per_prompt, separators=(",", ":"), sort_keys=True),
        "--allowed-local-media-path",
        str(allowed_media_root),
        "--media-io-kwargs",
        json.dumps(media_io_kwargs, separators=(",", ":"), sort_keys=True),
        "--mm-processor-kwargs",
        json.dumps(mm_processor_kwargs, separators=(",", ":"), sort_keys=True),
        "--mm-processor-cache-gb",
        "0",
        "--no-enable-prefix-caching",
        "--mm-ipc-gpu-memory-gb",
        str(args.mm_ipc_gpu_memory_gb),
    ]
    if args.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.kv_cache_memory_bytes is not None:
        command.extend(["--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes)])
    command.extend(args.server_arg)
    return command


def video_mm_processor_kwargs(args: argparse.Namespace) -> dict[str, int]:
    width, height = args.video_pixel_budget
    return {"max_pixels": width * height * args.frames}


def video_limit_mm_per_prompt(_args: argparse.Namespace) -> dict[str, int]:
    return {"image": 0, "video": 1}


def health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed loopback URL
            f"http://127.0.0.1:{port}/health", timeout=1.0
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_for_health(
    process: subprocess.Popen[bytes], port: int, timeout: float
) -> float:
    started = time.perf_counter()
    deadline = started + timeout
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "vllm serve exited before becoming healthy with code "
                f"{process.returncode}"
            )
        if health_ok(port):
            return time.perf_counter() - started
        time.sleep(0.5)
    raise TimeoutError(f"vllm serve was not healthy after {timeout:.1f} seconds")


def chat_request(
    args: argparse.Namespace,
    video: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video["file_uri"]},
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ],
        "max_completion_tokens": args.output_len,
        "ignore_eos": True,
        "n": 1,
        "seed": args.seed,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_token_ids": True,
    }
    if args.request_media_io_kwargs:
        payload["media_io_kwargs"] = args.request_media_io_kwargs
    return payload


class PersistentConnectionSlot:
    def __init__(self, slot_id: int, port: int, timeout: float) -> None:
        self.slot_id = slot_id
        self.port = port
        self.timeout = timeout
        self.connection: http.client.HTTPConnection | None = None
        self.generation = 0
        self.request_ordinal = 0
        self.warmed_generation: int | None = None
        self.open_count = 0
        self.reuse_count = 0
        self.close_count = 0
        self.close_reasons: Counter[str] = Counter()

    def _close(self, reason: str) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
            self.close_count += 1
            self.close_reasons[reason] += 1

    def close(self) -> None:
        self._close("pool_close")

    def abort(self) -> None:
        if self.connection is not None and self.connection.sock is not None:
            with contextlib.suppress(OSError):
                self.connection.sock.shutdown(socket.SHUT_RDWR)
        self._close("pool_abort")

    def post_chat(
        self,
        payload: Mapping[str, Any],
        *,
        phase: str,
        seeded_first_wave: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        body = canonical_json_bytes(payload)
        reused = self.connection is not None and self.connection.sock is not None
        if not reused:
            if self.connection is not None:
                self._close("stale_socket_before_request")
            self.connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=self.timeout
            )
            self.generation += 1
            self.request_ordinal = 0
            self.open_count += 1
        else:
            self.reuse_count += 1
            self.connection.sock.settimeout(self.timeout)
        self.request_ordinal += 1
        transport: dict[str, Any] = {
            "pool_slot_id": self.slot_id,
            "phase": phase,
            "seeded_first_wave": seeded_first_wave,
            "connection_generation": self.generation,
            "request_ordinal_on_generation": self.request_ordinal,
            "connection_reused": reused,
            "prewarmed_for_measurement": (
                phase == "measured" and self.warmed_generation == self.generation
            ),
            "request_connection_header": "keep-alive",
            "response_http_version": None,
            "response_connection_header": None,
            "response_will_close": None,
            "response_persistent": None,
        }
        try:
            self.connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Authorization": "Bearer EMPTY",
                    "Connection": "keep-alive",
                    "Content-Type": "application/json",
                    "User-Agent": "vllm-qwen3-vl-e2e-persistent-benchmark/1",
                },
            )
            response = self.connection.getresponse()
            response_body = response.read()
            status = response.status
            transport.update(
                {
                    "response_http_version": response.version,
                    "response_connection_header": response.getheader("Connection"),
                    "response_will_close": bool(response.will_close),
                    "response_persistent": bool(
                        not response.will_close and self.connection.sock is not None
                    ),
                }
            )
        except (OSError, http.client.HTTPException) as error:
            self._close("request_exception")
            raise HttpRequestError(
                f"chat request failed without retry: {error}",
                status=None,
                body=None,
                transport=transport,
            ) from error
        if response.will_close:
            self._close("response_will_close")
        if status != 200:
            detail = response_body.decode(errors="replace")
            raise HttpRequestError(
                f"chat request returned unexpected HTTP {status}: {detail}",
                status=status,
                body=detail,
                transport=transport,
            )
        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as error:
            detail = response_body.decode(errors="replace")
            raise HttpRequestError(
                f"chat response was not JSON: {detail}",
                status=status,
                body=detail,
                transport=transport,
            ) from error
        if not isinstance(decoded, dict):
            raise HttpRequestError(
                "chat response JSON was not an object",
                status=status,
                body=None,
                transport=transport,
            )
        return decoded, transport

    def snapshot(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "current_generation": self.generation,
            "warmed_generation": self.warmed_generation,
            "request_ordinal_on_current_generation": self.request_ordinal,
            "open_count": self.open_count,
            "reuse_count": self.reuse_count,
            "close_count": self.close_count,
            "close_reasons": dict(sorted(self.close_reasons.items())),
            "currently_open": (
                self.connection is not None and self.connection.sock is not None
            ),
        }


class PersistentHttpClientPool:
    def __init__(self, *, size: int, port: int, timeout: float) -> None:
        if size <= 0:
            raise ValueError("persistent HTTP pool size must be positive")
        self.size = size
        self.abort_requested = threading.Event()
        self.slots = [
            PersistentConnectionSlot(slot_id, port, timeout) for slot_id in range(size)
        ]
        self.available: queue.Queue[PersistentConnectionSlot] = queue.Queue(
            maxsize=size
        )
        for slot in self.slots:
            self.available.put_nowait(slot)
        self.closed = False
        self.phase_audits: dict[str, dict[str, Any]] = {}
        self.measured_start_counts: dict[str, int] | None = None

    def prepare_batch(self, *, phase: str, count: int) -> list[int | None]:
        if self.closed:
            raise RuntimeError("persistent HTTP pool is closed")
        if count < self.size:
            raise RuntimeError(
                f"{phase} must contain at least one request per pool slot"
            )
        if self.available.qsize() != self.size:
            raise RuntimeError("pool does not own all slots at batch boundary")
        drained = [self.available.get_nowait() for _ in range(self.size)]
        if sorted(slot.slot_id for slot in drained) != list(range(self.size)):
            raise RuntimeError("pool slot inventory mismatch")
        if phase == "measured":
            self.measured_start_counts = self.counts()
        return [*range(self.size), *([None] * (count - self.size))]

    @contextlib.contextmanager
    def lease(self, seeded_slot_id: int | None) -> Iterator[PersistentConnectionSlot]:
        slot = (
            self.slots[seeded_slot_id]
            if seeded_slot_id is not None
            else self.available.get()
        )
        try:
            yield slot
        finally:
            self.available.put(slot)

    def post_chat(
        self,
        payload: Mapping[str, Any],
        *,
        phase: str,
        seeded_slot_id: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.abort_requested.is_set() or self.closed:
            raise RuntimeError("persistent HTTP pool has been aborted or closed")
        with self.lease(seeded_slot_id) as slot:
            if self.abort_requested.is_set() or self.closed:
                raise RuntimeError("persistent HTTP pool has been aborted or closed")
            return slot.post_chat(
                payload,
                phase=phase,
                seeded_first_wave=seeded_slot_id is not None,
            )

    def counts(self) -> dict[str, int]:
        return {
            "open_count": sum(slot.open_count for slot in self.slots),
            "reuse_count": sum(slot.reuse_count for slot in self.slots),
            "close_count": sum(slot.close_count for slot in self.slots),
        }

    def audit_phase(
        self, phase: str, records: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        transports = [
            (
                record.get("transport")
                if record["status"] == "passed"
                else record.get("error", {}).get("transport")
            )
            for record in records
        ]
        complete = [item for item in transports if isinstance(item, Mapping)]
        used_slots = sorted({int(item["pool_slot_id"]) for item in complete})
        reasons: list[str] = []
        if used_slots != list(range(self.size)):
            reasons.append("not_all_pool_slots_used")
        if len(complete) != len(records):
            reasons.append("missing_transport_metadata")
        seeded_mapping = {
            int(record["request_index"]): int(transport["pool_slot_id"])
            for record, transport in zip(records, transports, strict=True)
            if isinstance(transport, Mapping)
            and transport.get("seeded_first_wave") is True
        }
        if seeded_mapping != {slot_id: slot_id for slot_id in range(self.size)}:
            reasons.append("seeded_first_wave_mapping_mismatch")
        if any(record["status"] != "passed" for record in records):
            reasons.append("request_failure")
        if any(item.get("response_http_version") != 11 for item in complete):
            reasons.append("response_not_http_1_1")
        if any(item.get("response_persistent") is not True for item in complete):
            reasons.append("response_not_persistent")
        if phase == "warmup":
            for slot in self.slots:
                slot_records = [
                    item
                    for item in complete
                    if int(item["pool_slot_id"]) == slot.slot_id
                ]
                if not slot_records or any(
                    int(item["connection_generation"]) != slot.generation
                    for item in slot_records
                ):
                    reasons.append(f"slot_{slot.slot_id}_warmup_generation_mismatch")
                else:
                    slot.warmed_generation = slot.generation
        elif phase == "measured":
            if any(item.get("connection_reused") is not True for item in complete):
                reasons.append("measured_connection_not_reused")
            if any(
                item.get("prewarmed_for_measurement") is not True for item in complete
            ):
                reasons.append("measured_connection_not_prewarmed")
            if self.measured_start_counts is None:
                reasons.append("missing_measured_start_counts")
            else:
                current = self.counts()
                if current["open_count"] != self.measured_start_counts["open_count"]:
                    reasons.append("connection_opened_during_measurement")
                if current["close_count"] != self.measured_start_counts["close_count"]:
                    reasons.append("connection_closed_during_measurement")
        else:
            reasons.append("unknown_phase")
        audit = {
            "status": "passed" if not reasons else "failed",
            "phase": phase,
            "pool_size": self.size,
            "request_count": len(records),
            "used_slot_ids": used_slots,
            "seeded_first_wave_request_to_slot": seeded_mapping,
            "reasons": reasons,
            "counts_at_phase_end": self.counts(),
            "slot_snapshots_at_phase_end": [slot.snapshot() for slot in self.slots],
        }
        self.phase_audits[phase] = audit
        return audit

    def close(self) -> None:
        if self.closed:
            return
        if self.available.qsize() != self.size:
            raise RuntimeError("cannot close pool while a slot is leased")
        for slot in self.slots:
            slot.close()
        self.closed = True

    def abort(self) -> None:
        if self.closed:
            return
        self.abort_requested.set()
        self.closed = True
        errors: list[BaseException] = []
        for slot in self.slots:
            try:
                slot.abort()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                f"{len(errors)} persistent HTTP slot abort(s) failed; first: "
                f"{type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]

    def snapshot(self) -> dict[str, Any]:
        return {
            "implementation": "stdlib http.client.HTTPConnection HTTP/1.1",
            "pool_size": self.size,
            "connection_scope": "one pool per concurrency block",
            "phase_scope": "same slots span warmup, settle, and measured phases",
            "request_streaming": False,
            "request_retry_count": 0,
            "counts": self.counts(),
            "closed": self.closed,
            "slots": [slot.snapshot() for slot in self.slots],
            "phase_audits": self.phase_audits,
        }

    def __enter__(self) -> "PersistentHttpClientPool":
        return self

    def __exit__(
        self, unused_type: Any, unused_value: Any, unused_traceback: Any
    ) -> None:
        self.close()


def attach_cleanup_note(original_error: BaseException, note: str) -> None:
    """Attach cleanup evidence without requiring BaseException.add_note (3.11+)."""

    cleanup_notes = getattr(original_error, "cleanup_notes", None)
    if not isinstance(cleanup_notes, list):
        cleanup_notes = []
        setattr(original_error, "cleanup_notes", cleanup_notes)
    cleanup_notes.append(note)
    add_note = getattr(original_error, "add_note", None)
    if callable(add_note):
        add_note(note)


def abort_pool_preserving_exception(
    client_pool: PersistentHttpClientPool, original_error: BaseException
) -> None:
    try:
        client_pool.abort()
    except BaseException as cleanup_error:
        attach_cleanup_note(
            original_error,
            "persistent HTTP pool abort cleanup failed without replacing the "
            f"original exception: {type(cleanup_error).__name__}: {cleanup_error}",
        )


def response_record(raw_response: Mapping[str, Any], output_len: int) -> dict[str, Any]:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"expected one chat choice, got {choices!r}")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("chat choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("chat choice has no message object")
    prompt_token_ids = raw_response.get("prompt_token_ids")
    completion_token_ids = choice.get("token_ids")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token, int) for token in prompt_token_ids
    ):
        raise RuntimeError("return_token_ids did not provide prompt token IDs")
    if not isinstance(completion_token_ids, list) or not all(
        isinstance(token, int) for token in completion_token_ids
    ):
        raise RuntimeError("return_token_ids did not provide completion token IDs")
    if len(completion_token_ids) != output_len:
        raise RuntimeError(
            f"expected {output_len} completion token IDs, got "
            f"{len(completion_token_ids)}"
        )

    text = message.get("content")
    reasoning_content = message.get("reasoning")
    usage = raw_response.get("usage")
    if isinstance(usage, Mapping):
        usage_completion_tokens = usage.get("completion_tokens")
        if isinstance(usage_completion_tokens, int) and usage_completion_tokens != len(
            completion_token_ids
        ):
            raise RuntimeError(
                "usage.completion_tokens disagrees with returned completion "
                f"token IDs: {usage_completion_tokens} != "
                f"{len(completion_token_ids)}"
            )
    return {
        "id": raw_response.get("id"),
        "model": raw_response.get("model"),
        "prompt_token_count": len(prompt_token_ids),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": sha256_json(prompt_token_ids),
        "completion_token_count": len(completion_token_ids),
        "completion_token_ids": completion_token_ids,
        "completion_token_ids_sha256": sha256_json(completion_token_ids),
        "prompt_and_completion_token_ids_sha256": sha256_json(
            {
                "prompt": prompt_token_ids,
                "completion": completion_token_ids,
            }
        ),
        "text": text,
        "text_sha256": sha256_json(text),
        "reasoning_content": reasoning_content,
        "reasoning_content_sha256": sha256_json(reasoning_content),
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "usage": usage,
        "server_metrics": raw_response.get("metrics"),
        "raw_response_sha256": sha256_json(raw_response),
        "raw_response": dict(raw_response),
    }


def request_specifications(
    args: argparse.Namespace,
    videos: Sequence[Mapping[str, Any]],
    video_works: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    block_index: int,
    concurrency: int,
    count: int,
    first_global_request_index: int,
) -> list[dict[str, Any]]:
    specifications = []
    for request_index in range(count):
        video_index = request_index % len(videos)
        video = videos[video_index]
        payload = chat_request(args, video)
        specifications.append(
            {
                "phase": phase,
                "block_index": block_index,
                "concurrency": concurrency,
                "request_index": request_index,
                "global_request_index": first_global_request_index + request_index,
                "video_index": video_index,
                "video_path": video["path"],
                "video_file_uri": video["file_uri"],
                "video_sha256": video["sha256"],
                "video_work": dict(video_works[video_index]),
                "request_payload_sha256": sha256_json(payload),
                "payload": payload,
            }
        )
    return specifications


def effective_warmup_request_count(
    requested: int, concurrency: int, video_count: int
) -> int:
    if requested <= 0:
        raise ValueError(
            "persistent HTTP requires a positive warmup count so every slot is "
            "established before measurement"
        )
    return max(requested, concurrency, video_count)


def resolve_request_counts_by_concurrency(
    overrides: Mapping[str, object],
    concurrencies: Sequence[int],
    default: int,
    *,
    allow_zero: bool,
    option: str,
) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for key, value in overrides.items():
        try:
            concurrency = int(key)
        except ValueError as error:
            raise ValueError(f"{option} key must be an integer: {key!r}") from error
        if str(concurrency) != key or concurrency not in concurrencies:
            raise ValueError(
                f"{option} contains unknown concurrency {key!r}; "
                f"expected one of {[str(item) for item in concurrencies]}"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{option}[{key!r}] must be an integer")
        if value < 0 or (value == 0 and not allow_zero):
            requirement = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{option}[{key!r}] must be {requirement}")
        normalized[concurrency] = value
    return {
        concurrency: normalized.get(concurrency, default)
        for concurrency in concurrencies
    }


def execute_request(
    args: argparse.Namespace,
    specification: Mapping[str, Any],
    start_gate: threading.Event,
    batch_clock: Mapping[str, int],
    client_pool: PersistentHttpClientPool,
    seeded_slot_id: int | None,
) -> dict[str, Any]:
    start_gate.wait()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    record = {
        key: specification[key]
        for key in (
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
        )
    }
    try:
        raw_response, transport = client_pool.post_chat(
            specification["payload"],
            phase=str(specification["phase"]),
            seeded_slot_id=seeded_slot_id,
        )
        record.update(
            {
                "status": "passed",
                "http_status": 200,
                "transport": transport,
                "response": response_record(raw_response, args.output_len),
            }
        )
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "http_status": getattr(error, "status", None),
                    "http_body": getattr(error, "body", None),
                    "transport": getattr(error, "transport", None),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finished_ns = time.monotonic_ns()
    record.update(
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "started_monotonic_ns": started_ns,
            "finished_monotonic_ns": finished_ns,
            "start_offset_seconds": (started_ns - batch_clock["started_monotonic_ns"])
            / 1e9,
            "finish_offset_seconds": (finished_ns - batch_clock["started_monotonic_ns"])
            / 1e9,
            "latency_seconds": (finished_ns - started_ns) / 1e9,
            "latency_ms": (finished_ns - started_ns) / 1e6,
        }
    )
    return record


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary_ms(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "population_stdev": statistics.pstdev(values),
        "percentile_method": "linear interpolation at (n - 1) * fraction",
    }


def peak_in_flight_requests(records: Sequence[Mapping[str, Any]]) -> int:
    events: list[tuple[float, int]] = []
    for record in records:
        events.append((float(record["start_offset_seconds"]), 1))
        events.append((float(record["finish_offset_seconds"]), -1))
    in_flight = 0
    peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        in_flight += delta
        peak = max(peak, in_flight)
    return peak


def batch_aggregate(
    records: Sequence[Mapping[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    passed = [record for record in records if record["status"] == "passed"]
    failed = [record for record in records if record["status"] != "passed"]
    latencies_ms = [float(record["latency_ms"]) for record in passed]
    prompt_tokens = sum(
        int(record["response"]["prompt_token_count"]) for record in passed
    )
    generated_tokens = sum(
        int(record["response"]["completion_token_count"]) for record in passed
    )
    sampled_source_megapixel_estimates: list[float] = []
    video_work_missing: list[dict[str, Any]] = []
    ordered_fingerprints = []
    for record in passed:
        work = record["video_work"]
        megapixels = work.get("sampled_source_megapixels_estimate")
        if isinstance(megapixels, (int, float)):
            sampled_source_megapixel_estimates.append(float(megapixels))
        else:
            video_work_missing.append(
                {
                    "request_index": record["request_index"],
                    "video_index": record["video_index"],
                    "video_path": record["video_path"],
                    "reason": work.get("derivation"),
                }
            )
        response = record["response"]
        ordered_fingerprints.append(
            {
                "request_index": record["request_index"],
                "video_index": record["video_index"],
                "video_path": record["video_path"],
                "prompt_token_ids_sha256": response["prompt_token_ids_sha256"],
                "completion_token_ids_sha256": response["completion_token_ids_sha256"],
                "prompt_and_completion_token_ids_sha256": response[
                    "prompt_and_completion_token_ids_sha256"
                ],
            }
        )

    completion_hash_counts = Counter(
        fingerprint["completion_token_ids_sha256"]
        for fingerprint in ordered_fingerprints
    )
    request_throughput = len(passed) / elapsed_seconds if elapsed_seconds else None
    attempted_throughput = len(records) / elapsed_seconds if elapsed_seconds else None
    generated_throughput = (
        generated_tokens / elapsed_seconds if elapsed_seconds else None
    )
    prompt_throughput = prompt_tokens / elapsed_seconds if elapsed_seconds else None
    all_token_throughput = (
        (prompt_tokens + generated_tokens) / elapsed_seconds
        if elapsed_seconds
        else None
    )
    mpix_throughput = None
    if elapsed_seconds and passed and not video_work_missing:
        mpix_throughput = sum(sampled_source_megapixel_estimates) / elapsed_seconds

    return {
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
            sum(sampled_source_megapixel_estimates)
            if passed and not video_work_missing
            else None
        ),
        "sampled_source_megapixels_estimate_per_second": mpix_throughput,
        "video_megapixel_estimate_method": (
            "sum(estimated Qwen3-VL sampled frames * externally probed encoded "
            "source width * encoded source height) / measured client wall time; "
            "this is not a count of frames actually decoded by the codec"
        ),
        "video_megapixel_estimate_unavailable": video_work_missing,
        "latency_ms": latency_summary_ms(latencies_ms),
        "achieved_mean_in_flight_requests": (
            sum(float(record["latency_seconds"]) for record in passed) / elapsed_seconds
            if elapsed_seconds
            else None
        ),
        "achieved_peak_in_flight_requests": peak_in_flight_requests(records),
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


def load_json_object_file(path: Path) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        decoded = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read result JSON {path}: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"result JSON is not an object: {path}")
    return decoded, hashlib.sha256(encoded).hexdigest()


def performance_configuration_fingerprint(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        field: configuration.get(field)
        for field in PERFORMANCE_PARITY_CONFIGURATION_FIELDS
    }
    return {
        "fields": list(PERFORMANCE_PARITY_CONFIGURATION_FIELDS),
        "values": values,
        "sha256": sha256_json(values),
    }


def result_token_parity(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
    reference_path: Path,
    reference_sha256: str,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    mismatch_count = 0
    current_fingerprint: dict[str, Any] | None = None
    reference_fingerprint: dict[str, Any] | None = None

    def mismatch(kind: str, **details: Any) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 100:
            mismatches.append({"kind": kind, **details})

    if reference.get("status") != "passed":
        mismatch("reference_status", reference_status=reference.get("status"))

    current_configuration = current.get("configuration")
    reference_configuration = reference.get("configuration")
    if not isinstance(current_configuration, Mapping) or not isinstance(
        reference_configuration, Mapping
    ):
        mismatch("configuration_missing")
    else:
        current_fingerprint = performance_configuration_fingerprint(
            current_configuration
        )
        reference_fingerprint = performance_configuration_fingerprint(
            reference_configuration
        )
        for field in PERFORMANCE_PARITY_CONFIGURATION_FIELDS:
            current_value = current_configuration.get(field)
            reference_value = reference_configuration.get(field)
            if current_value != reference_value:
                mismatch(
                    "configuration",
                    field=field,
                    current=current_value,
                    reference=reference_value,
                )
        if current_fingerprint["sha256"] != reference_fingerprint["sha256"]:
            mismatch(
                "performance_configuration_fingerprint",
                current=current_fingerprint,
                reference=reference_fingerprint,
            )

    current_videos = current.get("videos")
    reference_videos = reference.get("videos")
    current_video_hashes = (
        [video.get("sha256") for video in current_videos]
        if isinstance(current_videos, list)
        and all(isinstance(video, Mapping) for video in current_videos)
        else None
    )
    reference_video_hashes = (
        [video.get("sha256") for video in reference_videos]
        if isinstance(reference_videos, list)
        and all(isinstance(video, Mapping) for video in reference_videos)
        else None
    )
    if current_video_hashes != reference_video_hashes:
        mismatch(
            "video_manifest",
            current=current_video_hashes,
            reference=reference_video_hashes,
        )

    current_blocks = current.get("concurrency_blocks")
    reference_blocks = reference.get("concurrency_blocks")
    if not isinstance(current_blocks, list) or not isinstance(reference_blocks, list):
        mismatch("concurrency_blocks_missing")
        current_blocks = []
        reference_blocks = []
    if len(current_blocks) != len(reference_blocks):
        mismatch(
            "concurrency_block_count",
            current=len(current_blocks),
            reference=len(reference_blocks),
        )

    for block_index, (current_block, reference_block) in enumerate(
        zip(current_blocks, reference_blocks)
    ):
        if not isinstance(current_block, Mapping) or not isinstance(
            reference_block, Mapping
        ):
            mismatch("concurrency_block_invalid", block_index=block_index)
            continue
        if current_block.get("concurrency") != reference_block.get("concurrency"):
            mismatch(
                "concurrency",
                block_index=block_index,
                current=current_block.get("concurrency"),
                reference=reference_block.get("concurrency"),
            )
        current_measured = current_block.get("measured")
        reference_measured = reference_block.get("measured")
        current_records = (
            current_measured.get("records")
            if isinstance(current_measured, Mapping)
            else None
        )
        reference_records = (
            reference_measured.get("records")
            if isinstance(reference_measured, Mapping)
            else None
        )
        if not isinstance(current_records, list) or not isinstance(
            reference_records, list
        ):
            mismatch("measured_records_missing", block_index=block_index)
            continue
        if len(current_records) != len(reference_records):
            mismatch(
                "measured_record_count",
                block_index=block_index,
                current=len(current_records),
                reference=len(reference_records),
            )
        for request_index, (current_record, reference_record) in enumerate(
            zip(current_records, reference_records)
        ):
            if not isinstance(current_record, Mapping) or not isinstance(
                reference_record, Mapping
            ):
                mismatch(
                    "measured_record_invalid",
                    block_index=block_index,
                    request_index=request_index,
                )
                continue
            for field in ("request_index", "video_index", "video_sha256", "status"):
                current_value = current_record.get(field)
                reference_value = reference_record.get(field)
                if current_value != reference_value:
                    mismatch(
                        "request_field",
                        block_index=block_index,
                        request_index=request_index,
                        field=field,
                        current=current_value,
                        reference=reference_value,
                    )
            current_response = current_record.get("response")
            reference_response = reference_record.get("response")
            if not isinstance(current_response, Mapping) or not isinstance(
                reference_response, Mapping
            ):
                mismatch(
                    "response_missing",
                    block_index=block_index,
                    request_index=request_index,
                )
                continue
            for field in (
                "prompt_token_count",
                "prompt_token_ids_sha256",
                "completion_token_count",
                "completion_token_ids_sha256",
            ):
                current_value = current_response.get(field)
                reference_value = reference_response.get(field)
                if current_value != reference_value:
                    mismatch(
                        "response_token_parity",
                        block_index=block_index,
                        request_index=request_index,
                        field=field,
                        current=current_value,
                        reference=reference_value,
                    )

    return {
        "status": "passed" if mismatch_count == 0 else "failed",
        "reference_path": str(reference_path),
        "reference_sha256": reference_sha256,
        "comparison": "exact prompt and completion token IDs by block/request",
        "current_performance_configuration_fingerprint": current_fingerprint,
        "reference_performance_configuration_fingerprint": reference_fingerprint,
        "endpoint_treatment_configuration": {
            "fields": list(ENDPOINT_TREATMENT_CONFIGURATION_FIELDS),
            "current": (
                {
                    field: current_configuration.get(field)
                    for field in ENDPOINT_TREATMENT_CONFIGURATION_FIELDS
                }
                if isinstance(current_configuration, Mapping)
                else None
            ),
            "reference": (
                {
                    field: reference_configuration.get(field)
                    for field in ENDPOINT_TREATMENT_CONFIGURATION_FIELDS
                }
                if isinstance(reference_configuration, Mapping)
                else None
            ),
            "note": (
                "Treatment fields are intentionally endpoint-specific and must "
                "be validated against the campaign's exact expected-difference map."
            ),
        },
        "mismatch_count": mismatch_count,
        "mismatches": mismatches,
        "mismatches_truncated": mismatch_count > len(mismatches),
    }


def execute_batch(
    args: argparse.Namespace,
    specifications: Sequence[Mapping[str, Any]],
    *,
    concurrency: int,
    client_pool: PersistentHttpClientPool,
) -> dict[str, Any]:
    if client_pool.size != concurrency:
        raise RuntimeError(
            f"persistent HTTP pool size {client_pool.size} does not match "
            f"requested concurrency {concurrency}"
        )
    if not specifications:
        return {
            "status": "skipped",
            "requested_concurrency": concurrency,
            "effective_client_workers": 0,
            "started_at": None,
            "finished_at": None,
            "started_monotonic_ns": None,
            "finished_monotonic_ns": None,
            "measured_window_seconds": 0.0,
            "records": [],
            "aggregate": None,
        }

    workers = min(concurrency, len(specifications))
    phase = str(specifications[0]["phase"])
    if any(str(specification["phase"]) != phase for specification in specifications):
        raise RuntimeError("one request batch cannot mix phases")
    seeded_slots = client_pool.prepare_batch(phase=phase, count=len(specifications))
    start_gate = threading.Event()
    batch_clock: dict[str, int] = {"started_monotonic_ns": 0}
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    try:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="vllm-e2e-client"
        )
        for spec, seeded_slot in zip(specifications, seeded_slots, strict=True):
            futures.append(
                executor.submit(
                    execute_request,
                    args,
                    spec,
                    start_gate,
                    batch_clock,
                    client_pool,
                    seeded_slot,
                )
            )
        started_at = utc_now()
        batch_clock["started_monotonic_ns"] = time.monotonic_ns()
        start_gate.set()
        records = [future.result() for future in futures]
        finished_ns = time.monotonic_ns()
        finished_at = utc_now()
    except BaseException as original_error:
        # A signal or submit failure can occur before the start gate opens. Release
        # any partially submitted workers, then abort sockets so they cannot block.
        for future in futures:
            future.cancel()
        abort_pool_preserving_exception(client_pool, original_error)
        start_gate.set()
        deadline = time.monotonic() + CLIENT_ABORT_DRAIN_TIMEOUT_SECONDS
        while (
            any(not future.done() for future in futures) and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        lingering = sum(not future.done() for future in futures)
        if lingering:
            attach_cleanup_note(
                original_error,
                f"{lingering} client future(s) remained after bounded "
                f"{CLIENT_ABORT_DRAIN_TIMEOUT_SECONDS:.1f}s abort drain; outer "
                "cell watchdog remains authoritative",
            )
        raise
    else:
        assert executor is not None
        executor.shutdown(wait=True)
    elapsed_seconds = (finished_ns - batch_clock["started_monotonic_ns"]) / 1e9
    records.sort(key=lambda record: int(record["request_index"]))
    aggregate = batch_aggregate(records, elapsed_seconds)
    transport_audit = client_pool.audit_phase(phase, records)
    aggregate["persistent_transport_audit"] = transport_audit
    if transport_audit["status"] != "passed":
        aggregate["status"] = "failed"
    return {
        "status": aggregate["status"],
        "requested_concurrency": concurrency,
        "effective_client_workers": workers,
        "started_at": started_at,
        "finished_at": finished_at,
        "started_monotonic_ns": batch_clock["started_monotonic_ns"],
        "finished_monotonic_ns": finished_ns,
        "measured_window_seconds": elapsed_seconds,
        "records": records,
        "aggregate": aggregate,
    }


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(
    process: subprocess.Popen[bytes], pgid: int, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not process_group_exists(pgid):
            return True
        time.sleep(0.2)
    process.poll()
    return not process_group_exists(pgid)


def stop_server(
    process: subprocess.Popen[bytes], pgid: int, graceful_timeout: float
) -> dict[str, Any]:
    if pgid <= 1 or pgid == os.getpgrp():
        raise RuntimeError(f"refusing to signal unsafe process group {pgid}")
    actions: list[dict[str, Any]] = []
    started = time.perf_counter()
    process.poll()
    for name, signum, timeout in (
        ("SIGINT", signal.SIGINT, graceful_timeout),
        ("SIGTERM", signal.SIGTERM, 15.0),
        ("SIGKILL", signal.SIGKILL, 10.0),
    ):
        if not process_group_exists(pgid):
            break
        action_started = time.perf_counter()
        try:
            os.killpg(pgid, signum)
            signal_sent = True
        except ProcessLookupError:
            signal_sent = False
        exited = wait_for_process_group_exit(process, pgid, timeout)
        actions.append(
            {
                "signal": name,
                "signal_sent": signal_sent,
                "wait_seconds": time.perf_counter() - action_started,
                "process_group_exited": exited,
            }
        )
        if exited:
            break

    try:
        return_code = process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        return_code = process.poll()
    group_exited = not process_group_exists(pgid)
    return {
        "process_group": pgid,
        "actions": actions,
        "elapsed_seconds": time.perf_counter() - started,
        "leader_return_code": return_code,
        "process_group_exited": group_exited,
        "cleanup_escalated": any(
            action["signal"] != "SIGINT" and action["signal_sent"] for action in actions
        ),
    }


@contextlib.contextmanager
def termination_handlers() -> Iterator[None]:
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    old_handlers: dict[signal.Signals, Any] = {}

    def request_termination(signum: int, _frame: Any) -> None:
        raise TerminationRequested(signum)

    try:
        for signum in signals:
            old_handlers[signum] = signal.signal(signum, request_termination)
        yield
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def server_log_record(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with path.open("rb") as file:
        if size > SERVER_LOG_TAIL_BYTES:
            file.seek(-SERVER_LOG_TAIL_BYTES, os.SEEK_END)
        tail = file.read().decode(errors="replace")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": sha256_file(path),
        "storage": "append-only full server-log sidecar",
        "tail": tail,
        "tail_truncated": size > SERVER_LOG_TAIL_BYTES,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, allow_nan=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()


def run(args: argparse.Namespace) -> int:
    started_at = utc_now()
    run_started = time.perf_counter()
    source_root, python, video_paths, output, allowed_media_root = validate_paths(args)
    args.source_root = source_root
    args.python = python
    args.video = video_paths
    args.output = output
    parity_reference_path: Path | None = None
    parity_reference: dict[str, Any] | None = None
    parity_reference_sha256: str | None = None
    if args.parity_reference is not None:
        parity_reference_path = args.parity_reference.expanduser().resolve(strict=True)
        if not parity_reference_path.is_file():
            raise ValueError(
                f"--parity-reference is not a regular file: {parity_reference_path}"
            )
        if parity_reference_path == output:
            raise ValueError("--parity-reference and --output must be different files")
        parity_reference, parity_reference_sha256 = load_json_object_file(
            parity_reference_path
        )
        args.parity_reference = parity_reference_path
    if not args.variant.strip():
        raise ValueError("--variant must not be empty")
    if not args.backend.strip():
        raise ValueError("--backend must not be empty")
    concurrencies = args.concurrency or [1]
    measured_request_counts = resolve_request_counts_by_concurrency(
        args.requests_by_concurrency,
        concurrencies,
        args.requests,
        allow_zero=False,
        option="--requests-by-concurrency",
    )
    warmup_request_counts = resolve_request_counts_by_concurrency(
        args.warmup_requests_by_concurrency,
        concurrencies,
        args.warmup_requests,
        allow_zero=False,
        option="--warmup-requests-by-concurrency",
    )
    max_num_seqs = args.max_num_seqs or max(concurrencies)
    if max_num_seqs < max(concurrencies):
        raise ValueError(
            f"--max-num-seqs={max_num_seqs} is below maximum concurrency "
            f"{max(concurrencies)}"
        )
    validate_extra_server_args(args.server_arg)
    ensure_port_available(args.port)

    media_io_kwargs = resolved_media_io_kwargs(args)
    metric_video_kwargs, metric_video_kwargs_unavailable_reason = (
        video_kwargs_for_metric_derivation(
            media_io_kwargs, args.request_media_io_kwargs
        )
    )
    env = server_environment(source_root, python, args.pythonpath_extra)
    videos = [
        video_provenance(index, video, python, env)
        for index, video in enumerate(video_paths)
    ]
    video_works = [expected_video_work(video, metric_video_kwargs) for video in videos]
    command = server_command(
        args,
        python,
        allowed_media_root,
        media_io_kwargs,
        max_num_seqs,
    )
    script = Path(__file__).resolve()
    warmup_requests_by_concurrency = [
        {
            "concurrency": concurrency,
            "requested": warmup_request_counts[concurrency],
            "effective": effective_warmup_request_count(
                warmup_request_counts[concurrency], concurrency, len(videos)
            ),
        }
        for concurrency in concurrencies
    ]
    measured_requests_by_concurrency = [
        {
            "concurrency": concurrency,
            "requests": measured_request_counts[concurrency],
        }
        for concurrency in concurrencies
    ]
    pixel_budget_width, pixel_budget_height = args.video_pixel_budget
    mm_processor_kwargs = video_mm_processor_kwargs(args)
    limit_mm_per_prompt = video_limit_mm_per_prompt(args)
    result: dict[str, Any] = {
        "schema": "vllm-qwen3-vl-video-e2e-throughput-v3-persistent-http",
        "status": "running",
        "started_at": started_at,
        "configuration": {
            "variant": args.variant,
            "model": args.model,
            "revision": args.revision,
            "prompt": args.prompt,
            "prompt_sha256": sha256_json(args.prompt),
            "output_len": args.output_len,
            "seed": args.seed,
            "video_cycle_policy": (
                "video_index = phase-local request_index modulo video count; "
                "reset for each warmup and measured batch"
            ),
            "video_count": len(videos),
            "backend_argument": args.backend,
            "backend_kwargs": args.backend_kwargs,
            "server_media_io_kwargs": media_io_kwargs,
            "request_media_io_kwargs": args.request_media_io_kwargs,
            "video_kwargs_for_metric_derivation": metric_video_kwargs,
            "video_kwargs_for_metric_derivation_unavailable_reason": (
                metric_video_kwargs_unavailable_reason
            ),
            "frame_target": args.frames,
            "video_pixel_budget": {
                "reference_width": pixel_budget_width,
                "reference_height": pixel_budget_height,
                "max_pixels_per_sampled_frame": (
                    pixel_budget_width * pixel_budget_height
                ),
                "sampled_frames": args.frames,
                "max_pixels_total": mm_processor_kwargs["max_pixels"],
                "resize_policy": (
                    "Qwen3-VL preserves source aspect ratio and rounds to its "
                    "spatial factor within this total pixel ceiling"
                ),
            },
            "server_mm_processor_kwargs": mm_processor_kwargs,
            "server_limit_mm_per_prompt": limit_mm_per_prompt,
            "warmup_requests_by_concurrency": warmup_requests_by_concurrency,
            "warmup_policy": (
                "positive warmup required; max(requested, concurrency, video_count) "
                "so every video and persistent client slot is covered"
            ),
            "measured_requests_per_concurrency": measured_requests_by_concurrency,
            "concurrency_order": concurrencies,
            "concurrency_block_state_policy": (
                "one server per run; each block is warmed independently, while "
                "process and operating-system state persists across blocks"
            ),
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
                "server_keep_alive_timeout_seconds": (
                    SERVER_HTTP_KEEP_ALIVE_TIMEOUT_SECONDS
                ),
                "measured_connection_requirement": (
                    "same successful warmup generation, reused and persistent"
                ),
            },
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "mm_ipc_gpu_memory_gb": args.mm_ipc_gpu_memory_gb,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "mm_processor_cache_gb": 0,
            "prefix_caching": False,
            "allowed_local_media_path": str(allowed_media_root),
            "port": args.port,
            "startup_timeout_seconds": args.startup_timeout,
            "request_timeout_seconds": args.request_timeout,
            "shutdown_timeout_seconds": args.shutdown_timeout,
            "settle_seconds": args.settle_seconds,
            "extra_server_argv": args.server_arg,
            "pythonpath_extra": [str(path) for path in args.pythonpath_extra],
            "parity_reference": (
                str(parity_reference_path)
                if parity_reference_path is not None
                else None
            ),
        },
        "provenance": {
            "source": source_provenance(source_root),
            "python": python_provenance(python, source_root, env),
            "hardware": hardware_provenance(),
            "harness": {
                "path": str(script),
                "sha256": sha256_file(script),
                "argv": sys.argv,
                "working_directory": str(Path.cwd()),
            },
        },
        "videos": videos,
        "request_payloads_by_video": [
            {
                "video_index": video["video_index"],
                "video_path": video["path"],
                "payload": (payload := chat_request(args, video)),
                "payload_sha256": sha256_json(payload),
            }
            for video in videos
        ],
        "server": {
            "command": command,
            "performance_environment": performance_environment_provenance(env),
        },
        "concurrency_blocks": [],
    }
    atomic_json(output, result)

    process: subprocess.Popen[bytes] | None = None
    pgid: int | None = None
    log_file = None
    failure: BaseException | None = None
    timed_request_failures = 0
    print(f"Launching: {shlex.join(command)}", flush=True)
    with contextlib.nullcontext():
        log_path = args.server_log_path
        try:
            with termination_handlers():
                log_file = log_path.open("xb")
                launch_started = time.perf_counter()
                process = subprocess.Popen(
                    command,
                    cwd=source_root,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                pgid = process.pid
                actual_pgid = os.getpgid(process.pid)
                if actual_pgid != pgid:
                    raise RuntimeError(
                        "vllm serve did not become its own process-group leader: "
                        f"pid={process.pid}, pgid={actual_pgid}"
                    )
                result["server"].update(
                    {
                        "pid": process.pid,
                        "process_group": pgid,
                        "launched_at": utc_now(),
                    }
                )
                startup_seconds = wait_for_health(
                    process, args.port, args.startup_timeout
                )
                result["server"].update(
                    {
                        "startup_seconds": startup_seconds,
                        "launch_to_health_seconds": time.perf_counter()
                        - launch_started,
                        "healthy_at": utc_now(),
                    }
                )
                atomic_json(output, result)

                global_request_index = 0
                for block_index, concurrency in enumerate(concurrencies):
                    if process.poll() is not None:
                        raise RuntimeError(
                            "vllm serve exited before concurrency block "
                            f"{block_index} with code {process.returncode}"
                        )
                    block: dict[str, Any] = {
                        "block_index": block_index,
                        "concurrency": concurrency,
                        "requested_warmup_requests": warmup_request_counts[concurrency],
                        "requested_measured_requests": measured_request_counts[
                            concurrency
                        ],
                        "effective_warmup_requests": (
                            warmup_requests_by_concurrency[block_index]["effective"]
                        ),
                        "status": "running",
                        "started_at": utc_now(),
                    }
                    result["concurrency_blocks"].append(block)
                    client_pool = PersistentHttpClientPool(
                        size=concurrency,
                        port=args.port,
                        timeout=args.request_timeout,
                    )
                    try:
                        warmup_specs = request_specifications(
                            args,
                            videos,
                            video_works,
                            phase="warmup",
                            block_index=block_index,
                            concurrency=concurrency,
                            count=block["effective_warmup_requests"],
                            first_global_request_index=global_request_index,
                        )
                        global_request_index += len(warmup_specs)
                        warmup = execute_batch(
                            args,
                            warmup_specs,
                            concurrency=concurrency,
                            client_pool=client_pool,
                        )
                        warmup["excluded_from_reported_throughput"] = True
                        block["warmup"] = warmup
                        atomic_json(output, result)
                        if warmup["status"] == "failed":
                            failed_count = warmup["aggregate"]["failed_requests"]
                            transport_reasons = warmup["aggregate"][
                                "persistent_transport_audit"
                            ]["reasons"]
                            raise RuntimeError(
                                f"{failed_count} warmup request(s) failed in "
                                f"concurrency block {block_index}; persistent "
                                f"transport reasons={transport_reasons}"
                            )

                        if args.settle_seconds:
                            settle_started = time.perf_counter()
                            time.sleep(args.settle_seconds)
                            block["actual_settle_seconds"] = (
                                time.perf_counter() - settle_started
                            )

                        measured_specs = request_specifications(
                            args,
                            videos,
                            video_works,
                            phase="measured",
                            block_index=block_index,
                            concurrency=concurrency,
                            count=measured_request_counts[concurrency],
                            first_global_request_index=global_request_index,
                        )
                        global_request_index += len(measured_specs)
                        measured = execute_batch(
                            args,
                            measured_specs,
                            concurrency=concurrency,
                            client_pool=client_pool,
                        )
                        block["measured"] = measured
                        block["aggregate"] = measured["aggregate"]
                        block["status"] = measured["status"]
                        block["finished_at"] = utc_now()
                        timed_request_failures += int(
                            measured["aggregate"]["failed_requests"]
                        )
                    except BaseException as original_error:
                        abort_pool_preserving_exception(client_pool, original_error)
                        raise
                    finally:
                        if not client_pool.closed:
                            client_pool.close()
                        block["persistent_http_pool"] = client_pool.snapshot()
                    atomic_json(output, result)
                    if block["status"] != "passed":
                        transport_reasons = block["aggregate"][
                            "persistent_transport_audit"
                        ]["reasons"]
                        raise RuntimeError(
                            f"measured persistent transport failed in concurrency "
                            f"block {block_index}: {transport_reasons}"
                        )
                    if process.poll() is not None:
                        raise RuntimeError(
                            "vllm serve exited during concurrency block "
                            f"{block_index} with code {process.returncode}"
                        )

                if timed_request_failures:
                    raise RuntimeError(
                        f"{timed_request_failures} measured request(s) failed"
                    )
        except BaseException as error:
            failure = error
            result["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            if isinstance(error, TerminationRequested):
                result["error"]["signal"] = signal.Signals(error.signum).name
        finally:
            if process is not None and pgid is not None:
                try:
                    shutdown = stop_server(process, pgid, args.shutdown_timeout)
                    result["server"]["shutdown"] = shutdown
                    if not shutdown["process_group_exited"]:
                        cleanup_error = RuntimeError(
                            "vllm serve process group did not exit after shutdown"
                        )
                        if failure is None:
                            failure = cleanup_error
                            result["error"] = {
                                "type": type(cleanup_error).__name__,
                                "message": str(cleanup_error),
                            }
                    elif shutdown["cleanup_escalated"]:
                        result.setdefault("warnings", []).append(
                            "server shutdown required SIGTERM or SIGKILL escalation"
                        )
                except BaseException as cleanup_error:
                    result["server"]["shutdown_error"] = {
                        "type": type(cleanup_error).__name__,
                        "message": str(cleanup_error),
                        "traceback": traceback.format_exc(),
                    }
                    if failure is None:
                        failure = cleanup_error
                        result["error"] = result["server"]["shutdown_error"]
            if log_file is not None:
                log_file.flush()
                log_file.close()
            if log_path.exists():
                result["server"]["log"] = server_log_record(log_path)

    if parity_reference is not None:
        assert parity_reference_path is not None
        assert parity_reference_sha256 is not None
        if failure is None:
            parity = result_token_parity(
                result,
                parity_reference,
                parity_reference_path,
                parity_reference_sha256,
            )
            result["token_parity"] = parity
            if parity["status"] != "passed":
                parity_error = RuntimeError(
                    "prompt/completion token parity failed against reference: "
                    f"{parity['mismatch_count']} mismatch(es)"
                )
                failure = parity_error
                result["error"] = {
                    "type": type(parity_error).__name__,
                    "message": str(parity_error),
                }
        else:
            result["token_parity"] = {
                "status": "skipped",
                "reference_path": str(parity_reference_path),
                "reference_sha256": parity_reference_sha256,
                "reason": "benchmark failed before parity could be evaluated",
            }

    result["status"] = "passed" if failure is None else "failed"
    result["finished_at"] = utc_now()
    result["total_harness_wall_seconds"] = time.perf_counter() - run_started
    atomic_json(output, result)
    print(f"Wrote {result['status']} result to {output}", flush=True)
    if failure is not None:
        print(f"{type(failure).__name__}: {failure}", file=sys.stderr)
        if isinstance(failure, TerminationRequested):
            return 128 + failure.signum
        return 1
    for block in result["concurrency_blocks"]:
        aggregate = block["aggregate"]
        latency = aggregate["latency_ms"]
        print(
            f"concurrency={block['concurrency']}: "
            f"{aggregate['request_throughput_per_second']:.4f} req/s, "
            f"{aggregate['generated_token_throughput_per_second']:.2f} tok/s, "
            f"p50={latency['p50']:.2f} ms, p95={latency['p95']:.2f} ms",
            flush=True,
        )
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (KeyboardInterrupt, Exception) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
