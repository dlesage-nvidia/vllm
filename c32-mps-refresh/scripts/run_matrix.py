# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run the PR #1 C32 base/head matrix with CUDA MPS off and on."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
BASE = "bc8abf31fef015339473f6071eda0de0305dd9b2"
HEAD = "fc52204ce7e0203456ceca030b90283dde28232a"
COMMITS = {"base": BASE, "pr": HEAD}
TREES = {
    "base": "09423356278c6c4bd871ccda98499474fad78bdd",
    "pr": "ae2af5c1d60f346efbb8a2375f46663b95835802",
}
TRANSFORMERS_MANIFEST = {
    "sha256": "39591d428561f5a29479229b49643bfe2ebcf433b7b3c086f5064da5fef2f259",
    "file_count": 2566,
    "logical_bytes": 47_133_316,
}
TRAFFIC_SHA256 = "b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d"
TRAFFIC_BYTES = 13_267_543
MONITOR_SHA256 = "db2ac87f4a1c21974ca89c15bcd28807433621dd1fc8e79a03ca6b30c3209ad2"
WARMUP_REQUESTS = 96
MEASURED_REQUESTS = 256
CONCURRENCY = 32
FRAMES = 32
OUTPUT_LENGTH = 32

# Every repetition contains all four cells. Each mode is first three times, and
# base/head order is balanced 3:3 within each mode.
SCHEDULE = [
    [("base", "off"), ("pr", "off"), ("base", "on"), ("pr", "on")],
    [("pr", "on"), ("base", "on"), ("pr", "off"), ("base", "off")],
    [("pr", "off"), ("base", "off"), ("base", "on"), ("pr", "on")],
    [("pr", "on"), ("base", "on"), ("base", "off"), ("pr", "off")],
    [("base", "off"), ("pr", "off"), ("pr", "on"), ("base", "on")],
    [("base", "on"), ("pr", "on"), ("pr", "off"), ("base", "off")],
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3": {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "harness": "benchmark_qwen3_e2e_persistent.py",
        "harness_sha256": "08bac47c2c8f2143f0717800fc40d5aad66de7cb896f124d0eb0a5c8148518de",
        "max_pixels": 18_874_368,
        "max_num_batched_tokens": 9216,
        "video_backend": "qwen3_vl",
        "backend_kwargs": {
            "base": {"hw_decoders": 2},
            "pr": {"hw_decoders": 2, "output_layout": "tchw"},
        },
        "server_argv": {
            "base": ["--no-mm-device-do-normalize"],
            "pr": ["--mm-device-do-normalize"],
        },
    },
    "qwen25": {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3",
        "harness": "benchmark_qwen25_e2e_persistent.py",
        "harness_sha256": "8154f437b896c5c8a7436049147b86d671cb5adfe3a14cbf11e6eb6442db6ff8",
        "max_pixels": 589_824,
        "max_num_batched_tokens": 12_288,
        "video_backend": "qwen2_vl",
        "backend_kwargs": {
            variant: {
                "hw_decoders": 2,
                "max_frames": 32,
                "min_frames": 32,
                "video_backend": "qwen2_vl",
            }
            for variant in ("base", "pr")
        },
        "server_argv": {
            "base": ["--mm-device-do-normalize"],
            "pr": ["--mm-device-do-normalize"],
        },
    },
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def command(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        input=input_text,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {argv}\n{result.stdout}")
    return result


def git(root: Path, *argv: str) -> str:
    return command(["git", "-C", str(root), *argv]).stdout.strip()


def source_record(root: Path, variant: str | None = None) -> dict[str, Any]:
    record = {
        "root": str(root),
        "commit": git(root, "rev-parse", "HEAD"),
        "tree": git(root, "rev-parse", "HEAD^{tree}"),
        "status": git(root, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if record["status"]:
        raise RuntimeError(f"vLLM source checkout is dirty:\n{record['status']}")
    if variant is not None and (
        record["commit"] != COMMITS[variant] or record["tree"] != TREES[variant]
    ):
        raise RuntimeError(f"vLLM source does not match {variant}: {record}")
    return record


def validate_transformers(root: Path) -> dict[str, Any]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".lock"
    )
    digest = hashlib.sha256()
    logical_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        logical_bytes += len(data)
    record = {
        "root": str(root),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "excluded": ["*/__pycache__/*", "*.pyc", "*/.lock"],
    }
    if {key: record[key] for key in TRANSFORMERS_MANIFEST} != TRANSFORMERS_MANIFEST:
        raise RuntimeError(f"Transformers package export mismatch: {record}")
    return record


class Lease:
    """The only machine-availability authority used by this runner."""

    def __init__(self, holder: str, seconds: int, renew_seconds: int, purpose: str):
        self.holder = holder
        self.seconds = seconds
        self.renew_seconds = renew_seconds
        self.purpose = purpose
        self.events: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _call(self, action: str, *extra: str) -> subprocess.CompletedProcess[str]:
        result = command(
            ["/usr/local/bin/gpulock", action, self.holder, *extra], check=False
        )
        self.events.append(
            {
                "action": action,
                "utc": now(),
                "returncode": result.returncode,
                "output": result.stdout.strip(),
            }
        )
        return result

    def acquire(self) -> None:
        result = self._call("acquire", str(self.seconds), self.purpose)
        if result.returncode != 0:
            raise RuntimeError("gpulock acquire failed")
        self.thread = threading.Thread(target=self._renew, daemon=True)
        self.thread.start()

    def _renew(self) -> None:
        while not self.stop_event.wait(self.renew_seconds):
            if self._call("renew", str(self.seconds)).returncode != 0:
                self.lost_event.set()
                return

    def check(self) -> None:
        if self.lost_event.is_set():
            raise RuntimeError("gpulock renewal failed; benchmark aborted")

    def release(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        if self._call("release").returncode != 0:
            raise RuntimeError("gpulock release failed")


def prepare_corpus(source: Path, corpus: Path) -> list[Path]:
    if source.stat().st_size != TRAFFIC_BYTES or sha256_file(source) != TRAFFIC_SHA256:
        raise RuntimeError("traffic video does not match the frozen PR #1 input")
    corpus.mkdir(parents=True, exist_ok=True)
    videos = [corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    expected = (source.stat().st_dev, source.stat().st_ino)
    for video in videos:
        if not video.exists():
            os.link(source, video)
        if (video.stat().st_dev, video.stat().st_ino) != expected:
            raise RuntimeError(f"video is not a hard link to the frozen input: {video}")
    return videos


def hf_snapshot(cache: Path, spec: Mapping[str, Any]) -> Path:
    repository = "models--" + str(spec["model"]).replace("/", "--")
    snapshot = cache / repository / "snapshots" / str(spec["revision"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"offline model snapshot is missing: {snapshot}")
    return snapshot.resolve()


def support_paths(root: Path, spec: Mapping[str, Any]) -> tuple[Path, Path]:
    harness = root / "support" / str(spec["harness"])
    monitor = root / "support" / "run_with_gpu_monitor_refined.py"
    for path, expected in (
        (harness, str(spec["harness_sha256"])),
        (monitor, MONITOR_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen support file mismatch: {path}")
    return harness.resolve(), monitor.resolve()


def build_harness_command(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    harness: Path,
    videos: Sequence[Path],
    *,
    rep: int,
    variant: str,
    mode: str,
    output: Path,
) -> list[str]:
    argv = [
        str(args.python),
        str(harness),
        "--source-root",
        str(args.source_root),
        "--python",
        str(args.python),
        "--pythonpath-extra",
        str(args.transformers_root),
        "--variant",
        f"{args.model}-r{rep:02d}-{variant}-mps-{mode}",
        "--model",
        str(spec["model"]),
        "--revision",
        str(spec["revision"]),
        "--allowed-local-media-path",
        str(args.corpus),
        "--backend",
        "pynvvideocodec",
        "--backend-kwargs",
        json.dumps(spec["backend_kwargs"][variant], separators=(",", ":"), sort_keys=True),
        "--frames",
        str(FRAMES),
        "--video-pixel-budget",
        "1024x576",
        "--warmup-requests",
        "1",
        "--warmup-requests-by-concurrency",
        json.dumps({str(CONCURRENCY): WARMUP_REQUESTS}, separators=(",", ":")),
        "--requests",
        "1",
        "--requests-by-concurrency",
        json.dumps({str(CONCURRENCY): MEASURED_REQUESTS}, separators=(",", ":")),
        "--concurrency",
        str(CONCURRENCY),
        "--output-len",
        str(OUTPUT_LENGTH),
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        str(spec["max_num_batched_tokens"]),
        "--max-num-seqs",
        "32",
        "--mm-ipc-gpu-memory-gb",
        "2",
        "--kv-cache-memory-bytes",
        "42949672960",
        "--settle-seconds",
        "1",
        "--request-timeout",
        "1200",
        "--startup-timeout",
        "600",
        "--shutdown-timeout",
        "60",
        "--port",
        str(args.port),
        "--output",
        str(output),
    ]
    for server_arg in spec["server_argv"][variant]:
        argv.append(f"--server-arg={server_arg}")
    for video in videos:
        argv.extend(["--video", str(video)])
    return argv


def control(env: Mapping[str, str], text: str) -> subprocess.CompletedProcess[str]:
    return command(
        ["nvidia-cuda-mps-control"],
        env=env,
        input_text=text + "\n",
        timeout=5,
        check=False,
    )


def numeric_lines(text: str) -> list[int]:
    return [int(line) for line in text.splitlines() if re.fullmatch(r"\s*[0-9]+\s*", line)]


def start_mps(env: Mapping[str, str]) -> dict[str, Any]:
    before = control(env, "get_server_list")
    if before.returncode == 0:
        raise RuntimeError("private MPS pipe unexpectedly has a control daemon")
    started = command(["nvidia-cuda-mps-control", "-d"], env=env, timeout=10, check=False)
    if started.returncode:
        raise RuntimeError(f"CUDA MPS daemon failed to start: {started.stdout}")
    deadline = time.monotonic() + 10
    while True:
        probe = control(env, "get_server_list")
        if probe.returncode == 0:
            percentage = control(env, "get_default_active_thread_percentage")
            values = re.findall(r"[0-9]+(?:\.[0-9]+)?", percentage.stdout)
            if percentage.returncode or not values or float(values[-1]) != 100.0:
                raise RuntimeError(
                    f"CUDA MPS default active-thread percentage is not 100: {percentage.stdout}"
                )
            return {
                "started_utc": now(),
                "start_returncode": started.returncode,
                "start_output": started.stdout.strip(),
                "initial_server_list": probe.stdout.strip(),
                "default_active_thread_percentage": float(values[-1]),
                "default_active_thread_query": percentage.stdout.strip(),
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(f"CUDA MPS control daemon did not answer: {probe.stdout}")
        time.sleep(0.2)


def stop_mps(env: Mapping[str, str], lifecycle: dict[str, Any]) -> None:
    stopped = control(env, "quit")
    lifecycle.update(
        {
            "stopped_utc": now(),
            "stop_returncode": stopped.returncode,
            "stop_output": stopped.stdout.strip(),
        }
    )
    if stopped.returncode:
        raise RuntimeError("CUDA MPS daemon refused quit")
    deadline = time.monotonic() + 10
    while control(env, "get_server_list").returncode == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError("CUDA MPS daemon remained reachable after quit")
        time.sleep(0.2)


def observe_mps_clients(env: Mapping[str, str]) -> dict[str, Any] | None:
    servers = control(env, "get_server_list")
    if servers.returncode:
        raise RuntimeError("CUDA MPS daemon disappeared during the cell")
    for server_pid in numeric_lines(servers.stdout):
        status = control(env, f"get_server_status {server_pid}")
        clients = control(env, f"get_client_list {server_pid}")
        client_pids = numeric_lines(clients.stdout) if clients.returncode == 0 else []
        if client_pids and status.returncode == 0 and status.stdout.strip() == "ACTIVE":
            return {
                "observed_utc": now(),
                "server_pid": server_pid,
                "server_status": status.stdout.strip(),
                "client_pids": client_pids,
            }
    return None


MPS_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
    "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING",
    "CUDA_MPS_SM_PARTITION",
)


def cell_environment(
    args: argparse.Namespace, spec: Mapping[str, Any], mode: str, mps_root: Path | None
) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for name in MPS_ENV_VARS:
        env.pop(name, None)
    env.update(
        {
            "HF_HUB_CACHE": str(args.hf_hub_cache),
            "HUGGINGFACE_HUB_CACHE": str(args.hf_hub_cache),
            "HF_HOME": str(args.hf_hub_cache.parent),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "3600",
        }
    )
    if mode == "on":
        if mps_root is None:
            raise ValueError("MPS-on cell needs a private root")
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(mps_root / "pipe")
        env["CUDA_MPS_LOG_DIRECTORY"] = str(mps_root / "log")
    else:
        env["CUDA_MPS_PIPE_DIRECTORY"] = ""
    return env


def daemon_environment(client_env: Mapping[str, str], gpu_uuid: str) -> dict[str, str]:
    env = dict(client_env)
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return env


def clear_known_stale_mps() -> dict[str, Any]:
    env = dict(os.environ)
    for name in MPS_ENV_VARS:
        env.pop(name, None)
    env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-pipe"
    probe = control(env, "get_server_list")
    record: dict[str, Any] = {
        "pipe_directory": "/tmp/nvidia-pipe",
        "probe_returncode": probe.returncode,
        "probe_output": probe.stdout.strip(),
        "quit_sent": False,
    }
    if probe.returncode:
        return record
    stopped = control(env, "quit")
    record.update(
        {"quit_sent": True, "quit_returncode": stopped.returncode, "quit_output": stopped.stdout.strip()}
    )
    if stopped.returncode:
        raise RuntimeError(f"known stale MPS daemon refused quit: {stopped.stdout}")
    deadline = time.monotonic() + 10
    while control(env, "get_server_list").returncode == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError("known stale MPS daemon remained reachable after quit")
        time.sleep(0.2)
    record["stopped_utc"] = now()
    return record


def validate_monitor(report: Mapping[str, Any], harness_command: Sequence[str]) -> None:
    expected = {
        "schema": "pynv-passive-aggregate-gpu-monitor-v1",
        "status": "passed",
        "returncode": 0,
        "timed_out": False,
        "termination_signal": None,
        "lifecycle_errors": [],
        "telemetry_events": [],
        "cleanup": None,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"GPU monitor {key} mismatch: {report.get(key)!r}")
    if report.get("command") != list(harness_command) or report.get("sample_count", 0) < 1:
        raise RuntimeError("GPU monitor did not cover the exact benchmark command")


def validate_result(
    result: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    rep: int,
    variant: str,
    mode: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if result.get("status") != "passed":
        raise RuntimeError("benchmark result did not pass")
    source = result.get("provenance", {}).get("source", {})
    if (
        source.get("commit") != COMMITS[variant]
        or source.get("tree") != TREES[variant]
        or source.get("tracked_diff_bytes") != 0
        or source.get("untracked_files") != []
    ):
        raise RuntimeError(f"result source provenance mismatch: {source}")
    config = result.get("configuration", {})
    expected = {
        "variant": f"{args_model(spec)}-r{rep:02d}-{variant}-mps-{mode}",
        "model": spec["model"],
        "revision": spec["revision"],
        "frame_target": FRAMES,
        "output_len": OUTPUT_LENGTH,
        "video_count": 8,
        "backend_argument": "pynvvideocodec",
        "backend_kwargs": spec["backend_kwargs"][variant],
        "concurrency_order": [CONCURRENCY],
        "max_model_len": 32768,
        "max_num_batched_tokens": spec["max_num_batched_tokens"],
        "max_num_seqs": 32,
        "mm_ipc_gpu_memory_gb": 2.0,
        "kv_cache_memory_bytes": 42_949_672_960,
        "mm_processor_cache_gb": 0,
        "prefix_caching": False,
        "extra_server_argv": spec["server_argv"][variant],
        "server_mm_processor_kwargs": {"max_pixels": spec["max_pixels"]},
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"result configuration {key} mismatch: {config.get(key)!r}")
    protocol = config.get("client_http_protocol", {})
    if protocol.get("request_retries") != 0 or protocol.get("streaming") is not False:
        raise RuntimeError("request transport policy changed")
    if config.get("warmup_requests_by_concurrency") != [
        {"concurrency": CONCURRENCY, "effective": WARMUP_REQUESTS, "requested": WARMUP_REQUESTS}
    ]:
        raise RuntimeError("warmup configuration changed")
    if config.get("measured_requests_per_concurrency") != [
        {"concurrency": CONCURRENCY, "requests": MEASURED_REQUESTS}
    ]:
        raise RuntimeError("measured request configuration changed")
    media = config.get("server_media_io_kwargs", {}).get("video", {})
    if (
        media.get("backend") != "pynvvideocodec"
        or media.get("hw_decoders") != 2
        or media.get("min_frames") != FRAMES
        or media.get("max_frames") != FRAMES
        or media.get("video_backend") != spec["video_backend"]
        or media.get("output_layout")
        != spec["backend_kwargs"][variant].get("output_layout")
    ):
        raise RuntimeError(f"resolved media configuration changed: {media}")
    blocks = result.get("concurrency_blocks", [])
    if len(blocks) != 1 or blocks[0].get("concurrency") != CONCURRENCY:
        raise RuntimeError("result is not C32-only")
    block = blocks[0]
    if (
        block.get("status") != "passed"
        or block.get("requested_warmup_requests") != WARMUP_REQUESTS
        or block.get("effective_warmup_requests") != WARMUP_REQUESTS
        or block.get("requested_measured_requests") != MEASURED_REQUESTS
    ):
        raise RuntimeError("C32 block configuration or status mismatch")
    signatures: dict[str, list[dict[str, Any]]] = {}
    for phase, count in (("warmup", WARMUP_REQUESTS), ("measured", MEASURED_REQUESTS)):
        batch = block.get(phase, {})
        aggregate = batch.get("aggregate", {})
        records = batch.get("records", [])
        if (
            len(records) != count
            or aggregate.get("attempted_requests") != count
            or aggregate.get("successful_requests") != count
            or aggregate.get("failed_requests") != 0
            or aggregate.get("achieved_peak_in_flight_requests") != CONCURRENCY
        ):
            raise RuntimeError(f"{phase} request accounting mismatch")
        signatures[phase] = []
        for record in records:
            response = record.get("response", {})
            if (
                record.get("status") != "passed"
                or record.get("http_status") != 200
                or response.get("completion_token_count") != OUTPUT_LENGTH
            ):
                raise RuntimeError(f"failed or short {phase} response")
            signatures[phase].append(
                {
                    "request_index": record.get("request_index"),
                    "video_index": record.get("video_index"),
                    "video_sha256": record.get("video_sha256"),
                    "request_payload_sha256": record.get("request_payload_sha256"),
                    "prompt_token_count": response.get("prompt_token_count"),
                    "prompt_token_ids_sha256": response.get("prompt_token_ids_sha256"),
                    "completion_token_count": response.get("completion_token_count"),
                    "completion_token_ids_sha256": response.get("completion_token_ids_sha256"),
                    "finish_reason": response.get("finish_reason"),
                    "stop_reason": response.get("stop_reason"),
                }
            )
    metrics = {
        "request_throughput_per_second": block["measured"]["aggregate"][
            "request_throughput_per_second"
        ],
        "measured_window_seconds": block["measured"]["aggregate"][
            "measured_window_seconds"
        ],
        "prompt_tokens": block["measured"]["aggregate"]["prompt_tokens"],
        "generated_tokens": block["measured"]["aggregate"]["generated_tokens"],
    }
    return metrics, signatures


def args_model(spec: Mapping[str, Any]) -> str:
    return "qwen25" if "2.5" in str(spec["model"]) else "qwen3"


def gzip_result(path: Path) -> tuple[Path, str, str]:
    raw_sha = sha256_file(path)
    target = path.with_suffix(path.suffix + ".gz")
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with path.open("rb") as source, temporary.open("xb") as raw_target:
        with gzip.GzipFile(fileobj=raw_target, mode="wb", compresslevel=1, mtime=0) as stream:
            shutil.copyfileobj(source, stream, length=1024 * 1024)
    with gzip.open(temporary, "rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != raw_sha:
        temporary.unlink()
        raise RuntimeError("gzip payload verification failed")
    os.replace(temporary, target)
    path.unlink()
    return target, raw_sha, sha256_file(target)


def parity_pair(
    left: Mapping[str, list[dict[str, Any]]],
    right: Mapping[str, list[dict[str, Any]]],
    *,
    label: str,
) -> dict[str, Any]:
    input_fields = (
        "request_index",
        "video_index",
        "video_sha256",
        "request_payload_sha256",
        "prompt_token_count",
        "prompt_token_ids_sha256",
    )
    output_fields = (
        "completion_token_count",
        "completion_token_ids_sha256",
        "finish_reason",
        "stop_reason",
    )
    audit: dict[str, Any] = {
        "label": label,
        "requests": 0,
        "input_mismatches": 0,
        "output_mismatches": 0,
        "output_mismatch_examples": [],
    }
    for phase in ("warmup", "measured"):
        left_records = {record["request_index"]: record for record in left[phase]}
        right_records = {record["request_index"]: record for record in right[phase]}
        if left_records.keys() != right_records.keys():
            raise RuntimeError(f"{label} {phase} request-index set mismatch")
        for request_index in sorted(left_records):
            audit["requests"] += 1
            a, b = left_records[request_index], right_records[request_index]
            if any(a[field] != b[field] for field in input_fields):
                audit["input_mismatches"] += 1
            if any(a[field] != b[field] for field in output_fields):
                audit["output_mismatches"] += 1
                if len(audit["output_mismatch_examples"]) < 8:
                    audit["output_mismatch_examples"].append(
                        {"phase": phase, "request_index": request_index}
                    )
    if audit["input_mismatches"]:
        raise RuntimeError(f"input parity failed: {audit}")
    audit["status"] = (
        "passed_exact" if audit["output_mismatches"] == 0 else "passed_input_only"
    )
    return audit


def interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "sample_stdev": statistics.stdev(values),
    }


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = {
        (int(cell["rep"]), str(cell["variant"]), str(cell["mps"])): float(
            cell["metrics"]["request_throughput_per_second"]
        )
        for cell in cells
    }
    comparisons: dict[str, Any] = {}
    definitions = {
        "pr_vs_base_mps_off": ("base", "off", "pr", "off"),
        "pr_vs_base_mps_on": ("base", "on", "pr", "on"),
        "mps_on_vs_off_base": ("base", "off", "base", "on"),
        "mps_on_vs_off_pr": ("pr", "off", "pr", "on"),
    }
    for label, (left_variant, left_mode, right_variant, right_mode) in definitions.items():
        left = [rates[(rep, left_variant, left_mode)] for rep in range(1, 7)]
        right = [rates[(rep, right_variant, right_mode)] for rep in range(1, 7)]
        changes = [(candidate / baseline - 1.0) * 100 for baseline, candidate in zip(left, right)]
        comparisons[label] = {
            "left_request_throughput_per_second": interval(left),
            "right_request_throughput_per_second": interval(right),
            "paired_change_percent": interval(changes),
            "pairs": [
                {"rep": rep, "left": a, "right": b, "change_percent": change}
                for rep, (a, b, change) in enumerate(zip(left, right, changes), 1)
            ],
        }
    return {"comparisons": comparisons}


def write_summary(results: Path, summary: Mapping[str, Any]) -> None:
    write_json(results / "summary.json", summary)
    with (results / "paired-throughput.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "rep", "left_req_s", "right_req_s", "change_percent"])
        for label, comparison in summary["comparisons"].items():
            for pair in comparison["pairs"]:
                writer.writerow([label, pair["rep"], pair["left"], pair["right"], pair["change_percent"]])
    lines = [
        "| Comparison | Left req/s median | Right req/s median | Paired change median [min, max] |",
        "|---|---:|---:|---:|",
    ]
    for label, comparison in summary["comparisons"].items():
        left = comparison["left_request_throughput_per_second"]
        right = comparison["right_request_throughput_per_second"]
        change = comparison["paired_change_percent"]
        lines.append(
            f"| {label} | {left['median']:.4f} | {right['median']:.4f} | "
            f"{change['median']:+.2f}% [{change['min']:+.2f}%, {change['max']:+.2f}%] |"
        )
    (results / "summary-table.md").write_text("\n".join(lines) + "\n")


def run_cell(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    harness: Path,
    monitor: Path,
    videos: Sequence[Path],
    lease: Lease,
    manifest: dict[str, Any],
    manifest_path: Path,
    gpu_uuid: str,
    *,
    rep: int,
    position: int,
    variant: str,
    mode: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    command(["git", "-C", str(args.source_root), "checkout", "--quiet", "--detach", COMMITS[variant]])
    source = source_record(args.source_root, variant)
    stem = f"r{rep:02d}-p{position:02d}-{variant}-mps-{mode}-c32"
    result_path = args.results / f"{stem}.json"
    monitor_path = args.results / f"{stem}-gpu-monitor.json"
    log_path = args.results / f"{stem}.log"
    mps_root = (
        Path(tempfile.mkdtemp(prefix="vllm-pynv-mps.", dir="/tmp"))
        if mode == "on"
        else None
    )
    if mps_root is not None:
        (mps_root / "pipe").mkdir(mode=0o700)
        (mps_root / "log").mkdir(mode=0o700)
    env = cell_environment(args, spec, mode, mps_root)
    lifecycle: dict[str, Any] = {
        "mode": mode,
        "pipe_directory": env["CUDA_MPS_PIPE_DIRECTORY"],
        "log_directory": env.get("CUDA_MPS_LOG_DIRECTORY"),
        "active_thread_percentage": env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
        "live_client_proof": None,
    }
    mps_env: Mapping[str, str] | None = None
    if mode == "on":
        mps_env = daemon_environment(env, gpu_uuid)
        lifecycle["daemon_cuda_visible_devices"] = gpu_uuid
        lifecycle["daemon_start_attempted"] = False
    else:
        lifecycle["daemon_start_attempted"] = False
    harness_command = build_harness_command(
        args,
        spec,
        harness,
        videos,
        rep=rep,
        variant=variant,
        mode=mode,
        output=result_path,
    )
    monitored_command = [
        str(args.python),
        str(monitor),
        "--output",
        str(monitor_path),
        "--device-index",
        "0",
        "--timeout-seconds",
        "3600",
        "--timeout-grace-seconds",
        "120",
        "--",
        *harness_command,
    ]
    cell: dict[str, Any] = {
        "rep": rep,
        "position": position,
        "variant": variant,
        "mps": mode,
        "status": "running",
        "started_utc": now(),
        "source": source,
        "command": monitored_command,
        "result": str(result_path),
        "monitor": str(monitor_path),
        "log": str(log_path),
        "mps_lifecycle": lifecycle,
    }
    manifest["cells"].append(cell)
    write_json(manifest_path, manifest)
    process: subprocess.Popen[Any] | None = None
    try:
        if mode == "on" and mps_env is not None:
            lifecycle["daemon_start_attempted"] = True
            lifecycle.update(start_mps(mps_env))
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                monitored_command,
                cwd=args.source_root,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while process.poll() is None:
                try:
                    lease.check()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=180)
                    raise
                if mode == "on" and lifecycle["live_client_proof"] is None:
                    proof = observe_mps_clients(mps_env or env)
                    if proof is not None:
                        lifecycle["live_client_proof"] = proof
                        write_json(manifest_path, manifest)
                time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"cell failed ({process.returncode}):\n{log_path.read_text(errors='replace')[-16000:]}")
        if mode == "on" and lifecycle["live_client_proof"] is None:
            raise RuntimeError("CUDA MPS cell completed without live-client proof")
        result = json.loads(result_path.read_text())
        monitor_report = json.loads(monitor_path.read_text())
        validate_monitor(monitor_report, harness_command)
        metrics, signatures = validate_result(
            result, spec, rep=rep, variant=variant, mode=mode
        )
        compressed, raw_sha, compressed_sha = gzip_result(result_path)
        cell.update(
            {
                "status": "passed",
                "finished_utc": now(),
                "result": str(compressed),
                "result_uncompressed_sha256": raw_sha,
                "result_sha256": compressed_sha,
                "monitor_sha256": sha256_file(monitor_path),
                "monitor_samples_sha256": sha256_file(
                    monitor_path.with_name(monitor_path.stem + ".samples.jsonl")
                ),
                "log_sha256": sha256_file(log_path),
                "metrics": metrics,
            }
        )
        return cell, signatures
    except BaseException as error:
        cell.update({"status": "failed", "finished_utc": now(), "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=180)
        if mode == "on" and mps_root is not None and mps_env is not None:
            cleanup_error = None
            try:
                stop_mps(mps_env, lifecycle)
            except BaseException as error:
                cleanup_error = error
                lifecycle["cleanup_error"] = f"{type(error).__name__}: {error}"
            archive = args.results / "mps-logs" / stem
            archive.mkdir(parents=True)
            for source in mps_root.rglob("*"):
                if source.is_file():
                    target = archive / source.relative_to(mps_root)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            lifecycle["archive"] = str(archive)
            if mps_root.parent != Path("/tmp") or not mps_root.name.startswith(
                "vllm-pynv-mps."
            ):
                raise RuntimeError(f"refusing to remove unexpected MPS path: {mps_root}")
            shutil.rmtree(mps_root)
            lifecycle["temporary_root_removed"] = True
        write_json(manifest_path, manifest)
        if (
            mode == "on"
            and cleanup_error is not None
            and process is not None
            and process.returncode == 0
        ):
            raise cleanup_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--traffic-video", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--holder", required=True)
    parser.add_argument("--gpu-label", required=True)
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--lease-seconds", type=int, default=21600)
    parser.add_argument("--renew-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    spec = MODEL_SPECS[args.model]
    args.source_root = args.source_root.resolve(strict=True)
    # Keep the venv launcher path intact. Resolving its symlink to the base
    # interpreter drops the venv's site-packages (including pynvml and vLLM).
    args.python = args.python.expanduser().absolute()
    args.transformers_root = args.transformers_root.resolve(strict=True)
    args.hf_hub_cache = args.hf_hub_cache.resolve(strict=True)
    args.traffic_video = args.traffic_video.resolve(strict=True)
    args.corpus = args.corpus.absolute()
    args.results = args.results.absolute()
    if args.lease_seconds <= args.renew_seconds or args.renew_seconds < 60:
        raise ValueError("lease must exceed a renewal interval of at least 60 seconds")
    if not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"Python is not executable: {args.python}")
    if shutil.which("nvidia-cuda-mps-control") is None:
        raise FileNotFoundError("nvidia-cuda-mps-control")
    harness, monitor = support_paths(root, spec)
    source_start = source_record(args.source_root)
    for variant in COMMITS:
        if git(args.source_root, "cat-file", "-t", COMMITS[variant]) != "commit":
            raise RuntimeError(f"missing vLLM commit: {COMMITS[variant]}")
        if git(args.source_root, "rev-parse", f"{COMMITS[variant]}^{{tree}}") != TREES[variant]:
            raise RuntimeError(f"vLLM tree mismatch for {variant}")
    transformers = validate_transformers(args.transformers_root)
    snapshot = hf_snapshot(args.hf_hub_cache, spec)
    if args.traffic_video.stat().st_size != TRAFFIC_BYTES or sha256_file(args.traffic_video) != TRAFFIC_SHA256:
        raise RuntimeError("traffic video does not match the frozen input")
    if args.results.exists():
        raise FileExistsError(f"result directory already exists: {args.results}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "source": source_start,
                    "transformers": transformers,
                    "hf_snapshot": str(snapshot),
                    "schedule": SCHEDULE,
                    "pixel_budget_per_frame": {"width": 1024, "height": 576, "pixels": 589824},
                    "server_max_pixels": spec["max_pixels"],
                    "warmup_requests": WARMUP_REQUESTS,
                    "measured_requests": MEASURED_REQUESTS,
                },
                indent=2,
            )
        )
        return 0

    lease = Lease(
        args.holder,
        args.lease_seconds,
        args.renew_seconds,
        f"PR1 {args.model} C32 base/head MPS off/on on {args.gpu_label}",
    )
    bootstrap_ok = False
    lease.acquire()
    try:
        uuid_result = command(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            timeout=10,
        )
        gpu_uuids = [line.strip() for line in uuid_result.stdout.splitlines() if line.strip()]
        if len(gpu_uuids) != 1 or not gpu_uuids[0].startswith("GPU-"):
            raise RuntimeError(f"expected exactly one GPU UUID: {gpu_uuids}")
        gpu_uuid = gpu_uuids[0]
        stale_mps_cleanup = clear_known_stale_mps()
        args.results.mkdir(parents=True)
        videos = prepare_corpus(args.traffic_video, args.corpus)
        runner_sha = sha256_file(Path(__file__).resolve())
        harness_sha = sha256_file(harness)
        monitor_sha = sha256_file(monitor)
        bootstrap_ok = True
    finally:
        if not bootstrap_ok:
            lease.release()
    manifest_path = args.results / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": "pynv-pr1-c32-mps-matrix-v1",
        "status": "running",
        "started_utc": now(),
        "model_key": args.model,
        "model": spec["model"],
        "revision": spec["revision"],
        "gpu_label": args.gpu_label,
        "gpu_uuid": gpu_uuid,
        "known_stale_mps_cleanup": stale_mps_cleanup,
        "commits": COMMITS,
        "trees": TREES,
        "transformers": transformers,
        "hf_snapshot": str(snapshot),
        "schedule": SCHEDULE,
        "configuration": {
            "concurrency": CONCURRENCY,
            "sampled_frames": FRAMES,
            "pixel_budget_per_frame": {"width": 1024, "height": 576, "pixels": 589824},
            "server_max_pixels": spec["max_pixels"],
            "warmup_requests": WARMUP_REQUESTS,
            "measured_requests": MEASURED_REQUESTS,
            "output_length": OUTPUT_LENGTH,
            "mps_default_active_thread_percentage_required": 100,
            "mps_daemon_lifetime": "one fresh private daemon per MPS-on cell",
            "availability_authority": "gpulock exit codes only",
        },
        "support": {
            "runner": {"path": str(Path(__file__).resolve()), "sha256": runner_sha},
            "harness": {"path": str(harness), "sha256": harness_sha},
            "monitor": {"path": str(monitor), "sha256": monitor_sha},
            "traffic_video": {"path": str(args.traffic_video), "sha256": TRAFFIC_SHA256},
        },
        "cells": [],
    }
    try:
        write_json(manifest_path, manifest)
    except BaseException:
        lease.release()
        raise
    signatures: dict[tuple[int, str, str], dict[str, list[dict[str, Any]]]] = {}
    pending_error: BaseException | None = None
    try:
        for rep, cells in enumerate(SCHEDULE, 1):
            for position, (variant, mode) in enumerate(cells, 1):
                lease.check()
                cell, cell_signatures = run_cell(
                    args,
                    spec,
                    harness,
                    monitor,
                    videos,
                    lease,
                    manifest,
                    manifest_path,
                    gpu_uuid,
                    rep=rep,
                    position=position,
                    variant=variant,
                    mode=mode,
                )
                signatures[(rep, variant, mode)] = cell_signatures
                print(
                    f"PASS r{rep:02d} {variant} mps={mode} "
                    f"{cell['metrics']['request_throughput_per_second']:.6f} req/s",
                    flush=True,
                )
        parity = []
        for rep in range(1, 7):
            for mode in ("off", "on"):
                parity.append(
                    parity_pair(
                        signatures[(rep, "base", mode)],
                        signatures[(rep, "pr", mode)],
                        label=f"r{rep:02d}-pr-vs-base-mps-{mode}",
                    )
                )
            for variant in ("base", "pr"):
                parity.append(
                    parity_pair(
                        signatures[(rep, variant, "off")],
                        signatures[(rep, variant, "on")],
                        label=f"r{rep:02d}-mps-on-vs-off-{variant}",
                    )
                )
        write_json(args.results / "parity-audit.json", parity)
        summary = summarize(manifest["cells"])
        summary.update(
            {
                "schema": "pynv-pr1-c32-mps-summary-v1",
                "model": spec["model"],
                "pixel_budget_per_sampled_frame": {
                    "width": 1024,
                    "height": 576,
                    "pixels": 589824,
                },
                "parity_status": (
                    "passed_exact"
                    if all(item["status"] == "passed_exact" for item in parity)
                    else "passed_input_only"
                ),
            }
        )
        write_summary(args.results, summary)
        manifest.update({"status": "passed", "finished_utc": now(), "summary": summary})
    except BaseException as error:
        pending_error = error
        manifest.update({"status": "failed", "finished_utc": now(), "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            command(
                ["git", "-C", str(args.source_root), "checkout", "--quiet", "--detach", HEAD],
                check=False,
            )
            manifest["source_terminal"] = source_record(args.source_root, "pr")
            manifest["transformers_terminal"] = validate_transformers(
                args.transformers_root
            )
        except BaseException as error:
            if pending_error is None:
                pending_error = error
                manifest.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        try:
            lease.release()
        except BaseException as error:
            if pending_error is None:
                pending_error = error
                manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        manifest["lease_events"] = lease.events
        write_json(manifest_path, manifest)
    if pending_error is not None:
        raise pending_error
    print((args.results / "summary-table.md").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
