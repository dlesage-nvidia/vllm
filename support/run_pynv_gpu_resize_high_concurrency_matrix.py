# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run the PR #1 C8/C16/C32 publication matrix against GPU resize."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
COMMITS = {
    "pr1": "fc52204ce7e0203456ceca030b90283dde28232a",
    "gpu-resize": "3a64f5325f8e27581461c983902e23c52d906989",
}
TREES = {
    "pr1": "ae2af5c1d60f346efbb8a2375f46663b95835802",
    "gpu-resize": "a4ff4b35a27b1a59fb42628acf8db403cd1ec44f",
}
BACKEND_KWARGS = {
    "pr1": {"hw_decoders": 2, "output_layout": "tchw"},
    "gpu-resize": {
        "gpu_resize": True,
        "hw_decoders": 2,
        "output_layout": "tchw",
    },
}
SCHEDULE = [
    (1, [8, 16, 32], ["pr1", "gpu-resize"]),
    (2, [32, 16, 8], ["gpu-resize", "pr1"]),
    (3, [16, 32, 8], ["pr1", "gpu-resize"]),
    (4, [8, 32, 16], ["gpu-resize", "pr1"]),
    (5, [32, 8, 16], ["pr1", "gpu-resize"]),
    (6, [16, 8, 32], ["gpu-resize", "pr1"]),
]
WARMUP_REQUESTS = {8: 24, 16: 48, 32: 96}
MEASURED_REQUESTS = {8: 64, 16: 128, 32: 256}
FRAMES = 32
OUTPUT_LENGTH = 32
EXPECTED_FRAMES_PER_CELL = (
    sum(WARMUP_REQUESTS[c] + MEASURED_REQUESTS[c] for c in WARMUP_REQUESTS) * FRAMES
)
TRAFFIC_SHA256 = "b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d"
TRAFFIC_BYTES = 13_267_543
HARNESS_SHA256 = "08bac47c2c8f2143f0717800fc40d5aad66de7cb896f124d0eb0a5c8148518de"
MONITOR_SHA256 = "db2ac87f4a1c21974ca89c15bcd28807433621dd1fc8e79a03ca6b30c3209ad2"
SERVER_NORMALIZE_ARGV = ["--mm-device-do-normalize"]

COMMON_CONFIGURATION_FIELDS = (
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
    "extra_server_argv",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdout: Any = subprocess.PIPE,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        check=check,
        text=True,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )


def git(root: Path, *arguments: str) -> str:
    return run(["git", *arguments], cwd=root).stdout.strip()


def source_record(root: Path, expected_variant: str | None = None) -> dict[str, Any]:
    commit = git(root, "rev-parse", "HEAD")
    tree = git(root, "show", "-s", "--format=%T", "HEAD")
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"source checkout is dirty:\n{status}")
    if expected_variant is not None and (
        commit != COMMITS[expected_variant] or tree != TREES[expected_variant]
    ):
        raise RuntimeError(
            f"source mismatch for {expected_variant}: commit={commit}, tree={tree}"
        )
    return {"root": str(root), "commit": commit, "tree": tree, "status": status}


def validate_support_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected_sha256}")


def prepare_corpus(source: Path, corpus: Path) -> list[Path]:
    if source.stat().st_size != TRAFFIC_BYTES or sha256_file(source) != TRAFFIC_SHA256:
        raise RuntimeError("traffic source does not match the frozen PR #1 input")
    corpus.mkdir(parents=True, exist_ok=True)
    videos = [corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    source_stat = source.stat()
    for video in videos:
        if not video.exists():
            os.link(source, video)
        stat = video.stat()
        if (
            stat.st_dev != source_stat.st_dev
            or stat.st_ino != source_stat.st_ino
            or stat.st_size != TRAFFIC_BYTES
        ):
            raise RuntimeError(f"corpus path is not the expected hardlink: {video}")
    return videos


def build_harness_command(
    args: argparse.Namespace,
    *,
    variant: str,
    rep: int,
    concurrencies: Sequence[int],
    result_path: Path,
    videos: Sequence[Path],
) -> list[str]:
    label = f"{variant}-r{rep:02d}"
    command = [
        str(args.python),
        str(args.harness),
        "--source-root",
        str(args.source_root),
        "--python",
        str(args.python),
        "--pythonpath-extra",
        str(args.transformers_root),
        "--variant",
        label,
        "--allowed-local-media-path",
        str(args.corpus),
        "--backend",
        "pynvvideocodec",
        "--backend-kwargs",
        json.dumps(BACKEND_KWARGS[variant], separators=(",", ":"), sort_keys=True),
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
        json.dumps(WARMUP_REQUESTS, separators=(",", ":"), sort_keys=True),
        "--requests",
        "1",
        "--requests-by-concurrency",
        json.dumps(MEASURED_REQUESTS, separators=(",", ":"), sort_keys=True),
        "--output-len",
        str(OUTPUT_LENGTH),
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "9216",
        "--max-num-seqs",
        "32",
        "--mm-ipc-gpu-memory-gb",
        "2",
        "--kv-cache-memory-bytes",
        "42949672960",
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
        "--server-arg=--mm-device-do-normalize",
    ]
    for video in videos:
        command.extend(["--video", str(video)])
    for concurrency in concurrencies:
        command.extend(["--concurrency", str(concurrency)])
    return command


def build_monitored_command(
    args: argparse.Namespace, harness_command: Sequence[str], monitor_path: Path
) -> list[str]:
    return [
        str(args.python),
        str(args.monitor),
        "--output",
        str(monitor_path),
        "--timeout-seconds",
        "3600",
        "--timeout-grace-seconds",
        "120",
        "--",
        *harness_command,
    ]


def block_map(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    blocks = result.get("concurrency_blocks")
    if not isinstance(blocks, list):
        raise RuntimeError("result has no concurrency blocks")
    return {int(block["concurrency"]): block for block in blocks}


def validate_result(
    result: Mapping[str, Any],
    monitor: Mapping[str, Any],
    *,
    variant: str,
    rep: int,
    concurrencies: Sequence[int],
    server_log: Path,
) -> dict[str, Any]:
    if result.get("status") != "passed":
        raise RuntimeError(f"harness result did not pass for {variant} r{rep}")
    provenance = result.get("provenance", {}).get("source", {})
    if (
        provenance.get("commit") != COMMITS[variant]
        or provenance.get("tree") != TREES[variant]
        or provenance.get("tracked_diff_bytes") != 0
        or provenance.get("untracked_files") != []
    ):
        raise RuntimeError(f"result source provenance mismatch for {variant} r{rep}")
    configuration = result.get("configuration", {})
    expected_configuration = {
        "variant": f"{variant}-r{rep:02d}",
        "model": MODEL,
        "revision": REVISION,
        "frame_target": FRAMES,
        "output_len": OUTPUT_LENGTH,
        "video_count": 8,
        "backend_argument": "pynvvideocodec",
        "backend_kwargs": BACKEND_KWARGS[variant],
        "concurrency_order": list(concurrencies),
        "max_model_len": 32768,
        "max_num_batched_tokens": 9216,
        "max_num_seqs": 32,
        "mm_ipc_gpu_memory_gb": 2.0,
        "kv_cache_memory_bytes": 42_949_672_960,
        "mm_processor_cache_gb": 0,
        "prefix_caching": False,
        "extra_server_argv": SERVER_NORMALIZE_ARGV,
    }
    for field, expected in expected_configuration.items():
        if configuration.get(field) != expected:
            raise RuntimeError(
                f"configuration mismatch for {variant} r{rep} {field}: "
                f"{configuration.get(field)!r} != {expected!r}"
            )
    if result.get("server", {}).get("returncode") not in (0, None):
        raise RuntimeError(f"server return code mismatch for {variant} r{rep}")
    if (
        monitor.get("schema") != "pynv-passive-aggregate-gpu-monitor-v1"
        or monitor.get("status") != "passed"
        or monitor.get("returncode") != 0
        or monitor.get("timed_out") is not False
        or monitor.get("termination_signal") is not None
        or monitor.get("lifecycle_errors") != []
        or monitor.get("telemetry_events") != []
        or monitor.get("cleanup") is not None
    ):
        raise RuntimeError(f"monitor evidence is not clean for {variant} r{rep}")

    blocks = block_map(result)
    if set(blocks) != set(MEASURED_REQUESTS):
        raise RuntimeError(f"concurrency coverage mismatch for {variant} r{rep}")
    block_summary: dict[str, Any] = {}
    for concurrency in MEASURED_REQUESTS:
        block = blocks[concurrency]
        if block.get("status") != "passed":
            raise RuntimeError(f"failed c{concurrency} block for {variant} r{rep}")
        for phase, expected_count in (
            ("warmup", WARMUP_REQUESTS[concurrency]),
            ("measured", MEASURED_REQUESTS[concurrency]),
        ):
            batch = block.get(phase, {})
            aggregate = batch.get("aggregate", {})
            if (
                len(batch.get("records", [])) != expected_count
                or aggregate.get("attempted_requests") != expected_count
                or aggregate.get("successful_requests") != expected_count
                or aggregate.get("failed_requests") != 0
            ):
                raise RuntimeError(
                    "request-count mismatch for "
                    f"{variant} r{rep} c{concurrency} {phase}"
                )
        aggregate = block["aggregate"]
        latency = aggregate["latency_ms"]
        block_summary[str(concurrency)] = {
            "request_throughput_per_second": aggregate["request_throughput_per_second"],
            "generated_token_throughput_per_second": aggregate[
                "generated_token_throughput_per_second"
            ],
            "e2e_latency_p50_ms": latency["p50"],
            "e2e_latency_p95_ms": latency["p95"],
        }

    log_text = server_log.read_text(errors="replace")
    outcome_matches = re.findall(r"GPU resize outcomes: (\{[^\n]+\})", log_text)
    if variant == "gpu-resize":
        if "gpu_resize is using CV-CUDA HQResize" not in log_text:
            raise RuntimeError(f"CV-CUDA HQResize was not selected for r{rep}")
        if len(outcome_matches) != 1:
            raise RuntimeError(f"missing unique GPU-resize counters for r{rep}")
        outcomes = ast.literal_eval(outcome_matches[0])
        expected = {
            "gpu_resized": EXPECTED_FRAMES_PER_CELL,
            "resize_cvcuda": EXPECTED_FRAMES_PER_CELL,
        }
        if outcomes != expected:
            raise RuntimeError(
                f"GPU-resize counters mismatch: {outcomes} != {expected}"
            )
    elif outcome_matches or "gpu_resize is using CV-CUDA HQResize" in log_text:
        raise RuntimeError(f"PR #1 control unexpectedly used GPU resize in r{rep}")
    return {
        "blocks": block_summary,
        "peak_gpu_memory_used_mib": monitor.get("peak_memory_used_mib"),
        "gpu_monitor_sample_count": monitor.get("sample_count"),
        "gpu_resize_outcomes": (
            ast.literal_eval(outcome_matches[0]) if outcome_matches else None
        ),
    }


def strict_pair_audit(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, rep: int
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    counts = {
        "common_configuration": 0,
        "request_identity": 0,
        "prompt_token_ids": 0,
        "completion_token_ids": 0,
        "text": 0,
    }

    def mismatch(kind: str, **details: Any) -> None:
        counts[kind] += 1
        if len(mismatches) < 100:
            mismatches.append({"kind": kind, **details})

    baseline_configuration = baseline["configuration"]
    candidate_configuration = candidate["configuration"]
    for field in COMMON_CONFIGURATION_FIELDS:
        if baseline_configuration.get(field) != candidate_configuration.get(field):
            mismatch("common_configuration", field=field)

    baseline_blocks = block_map(baseline)
    candidate_blocks = block_map(candidate)
    compared = 0
    for concurrency in sorted(MEASURED_REQUESTS):
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
                mismatch(
                    "request_identity",
                    rep=rep,
                    concurrency=concurrency,
                    phase=phase,
                    reason="request_index_set",
                )
            for request_index in sorted(baseline_records.keys() & candidate_records):
                left = baseline_records[request_index]
                right = candidate_records[request_index]
                compared += 1
                identity_fields = (
                    "request_index",
                    "video_index",
                    "video_sha256",
                    "request_payload_sha256",
                    "status",
                )
                differing = [
                    field
                    for field in identity_fields
                    if left.get(field) != right.get(field)
                ]
                if differing:
                    mismatch(
                        "request_identity",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        request_index=request_index,
                        differing_fields=differing,
                    )
                left_response = left.get("response", {})
                right_response = right.get("response", {})
                if left_response.get("prompt_token_ids") != right_response.get(
                    "prompt_token_ids"
                ):
                    mismatch(
                        "prompt_token_ids",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        request_index=request_index,
                    )
                if left_response.get("completion_token_ids") != right_response.get(
                    "completion_token_ids"
                ):
                    mismatch(
                        "completion_token_ids",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        request_index=request_index,
                    )
                text_fields = (
                    "text_sha256",
                    "reasoning_content_sha256",
                    "finish_reason",
                    "stop_reason",
                )
                if any(
                    left_response.get(field) != right_response.get(field)
                    for field in text_fields
                ):
                    mismatch(
                        "text",
                        rep=rep,
                        concurrency=concurrency,
                        phase=phase,
                        request_index=request_index,
                    )
    input_mismatches = sum(
        counts[key]
        for key in ("common_configuration", "request_identity", "prompt_token_ids")
    )
    output_mismatches = counts["completion_token_ids"] + counts["text"]
    status = (
        "failed_input_parity"
        if input_mismatches
        else "completion_or_text_mismatch"
        if output_mismatches
        else "passed_exact"
    )
    return {
        "rep": rep,
        "status": status,
        "compared_response_pairs": compared,
        "mismatch_counts": counts,
        "mismatches": mismatches,
        "mismatches_truncated": sum(counts.values()) > len(mismatches),
    }


def interval(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(int(cell["rep"]), str(cell["variant"])): cell for cell in cells}
    comparisons: dict[str, Any] = {}
    rows: list[list[object]] = []
    for concurrency in sorted(MEASURED_REQUESTS):
        baseline = [
            float(
                by_key[(rep, "pr1")]["metrics"]["blocks"][str(concurrency)][
                    "request_throughput_per_second"
                ]
            )
            for rep in range(1, 7)
        ]
        candidate = [
            float(
                by_key[(rep, "gpu-resize")]["metrics"]["blocks"][str(concurrency)][
                    "request_throughput_per_second"
                ]
            )
            for rep in range(1, 7)
        ]
        paired_changes = [
            (right / left - 1.0) * 100.0 for left, right in zip(baseline, candidate)
        ]
        comparisons[str(concurrency)] = {
            "pr1_request_throughput_per_second": interval(baseline),
            "gpu_resize_request_throughput_per_second": interval(candidate),
            "paired_percent_change": interval(paired_changes),
            "paired_geomean_speedup": math.exp(
                statistics.fmean(
                    math.log(right / left) for left, right in zip(baseline, candidate)
                )
            ),
            "pairs": [
                {
                    "rep": rep,
                    "pr1_request_throughput_per_second": left,
                    "gpu_resize_request_throughput_per_second": right,
                    "gpu_resize_over_pr1": right / left,
                    "percent_change": (right / left - 1.0) * 100.0,
                }
                for rep, (left, right) in enumerate(zip(baseline, candidate), 1)
            ],
        }
        rows.append(
            [
                concurrency,
                *baseline,
                *candidate,
                *paired_changes,
            ]
        )
    return {
        "schema": "pynv-gpu-resize-pr1-paired-summary-v1",
        "status": "passed",
        "baseline": "literal PR #1 head",
        "candidate": "stacked GPU-resize head",
        "repetitions": 6,
        "comparisons": comparisons,
        "csv_rows": rows,
    }


def write_summary_files(
    results: Path, summary: Mapping[str, Any], token_audits: Sequence[Mapping[str, Any]]
) -> None:
    write_json(results / "summary.json", summary)
    write_json(
        results / "token-parity.json",
        {
            "status": (
                "failed_input_parity"
                if any(item["status"] == "failed_input_parity" for item in token_audits)
                else "completion_or_text_mismatch"
                if any(
                    item["status"] == "completion_or_text_mismatch"
                    for item in token_audits
                )
                else "passed_exact"
            ),
            "repetitions": list(token_audits),
        },
    )
    header = [
        "concurrency",
        *[f"pr1_r{rep}" for rep in range(1, 7)],
        *[f"gpu_resize_r{rep}" for rep in range(1, 7)],
        *[f"paired_percent_change_r{rep}" for rep in range(1, 7)],
    ]
    with (results / "paired-throughput.csv").open(
        "x", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(summary["csv_rows"])
    lines = [
        "| Concurrency | PR #1 req/s, median [min, max] | "
        "GPU resize req/s, median [min, max] | "
        "Paired change, median [min, max] |",
        "|---:|---:|---:|---:|",
    ]
    for concurrency, comparison in summary["comparisons"].items():
        baseline = comparison["pr1_request_throughput_per_second"]
        candidate = comparison["gpu_resize_request_throughput_per_second"]
        change = comparison["paired_percent_change"]
        lines.append(
            f"| {concurrency} | {baseline['median']:.4f} "
            f"[{baseline['min']:.4f}, {baseline['max']:.4f}] | "
            f"{candidate['median']:.4f} "
            f"[{candidate['min']:.4f}, {candidate['max']:.4f}] | "
            f"{change['median']:+.2f}% [{change['min']:+.2f}%, "
            f"{change['max']:+.2f}%] |"
        )
    (results / "summary-table.md").write_text("\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--traffic-video", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gpu-label", required=True)
    parser.add_argument("--port", type=int, default=18700)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.source_root = args.source_root.resolve(strict=True)
    args.python = args.python.expanduser().absolute()
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"not an executable Python: {args.python}")
    args.transformers_root = args.transformers_root.resolve(strict=True)
    args.harness = args.harness.resolve(strict=True)
    args.monitor = args.monitor.resolve(strict=True)
    args.traffic_video = args.traffic_video.resolve(strict=True)
    args.corpus = args.corpus.absolute()
    args.results = args.results.absolute()
    validate_support_file(args.harness, HARNESS_SHA256, "frozen PR #1 harness")
    validate_support_file(args.monitor, MONITOR_SHA256, "frozen GPU monitor")
    source_start = source_record(args.source_root)
    for variant, commit in COMMITS.items():
        if git(args.source_root, "cat-file", "-t", commit) != "commit":
            raise RuntimeError(f"missing {variant} commit: {commit}")
        if git(args.source_root, "show", "-s", "--format=%T", commit) != TREES[variant]:
            raise RuntimeError(f"tree mismatch for {variant}")

    if args.dry_run:
        videos = [args.corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
        commands = []
        for rep, concurrencies, variants in SCHEDULE:
            for variant in variants:
                result_path = args.results / f"r{rep:02d}-{variant}.json"
                harness_command = build_harness_command(
                    args,
                    variant=variant,
                    rep=rep,
                    concurrencies=concurrencies,
                    result_path=result_path,
                    videos=videos,
                )
                commands.append(
                    {
                        "rep": rep,
                        "variant": variant,
                        "concurrencies": concurrencies,
                        "command": build_monitored_command(
                            args,
                            harness_command,
                            result_path.with_name(
                                result_path.stem + "-gpu-monitor.json"
                            ),
                        ),
                    }
                )
        print(json.dumps({"source": source_start, "commands": commands}, indent=2))
        return 0

    lock_holder = os.environ.get("PYNV_GPU_LOCK_HOLDER")
    lock_status = run(["/usr/local/bin/gpulock", "status"], check=False).stdout.strip()
    if not lock_holder or f"holder={lock_holder}" not in lock_status:
        raise RuntimeError("runner must execute under its declared gpulock lease")
    if args.results.exists():
        raise FileExistsError(f"refusing to reuse result directory: {args.results}")
    args.results.mkdir(parents=True)
    videos = prepare_corpus(args.traffic_video, args.corpus)
    gpu_identity = run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    manifest: dict[str, Any] = {
        "schema": "pynv-gpu-resize-pr1-matrix-v1",
        "status": "running",
        "started_utc": utc_now(),
        "gpu_label": args.gpu_label,
        "gpu_identity": gpu_identity,
        "lock_status": lock_status,
        "source_start": source_start,
        "commits": COMMITS,
        "trees": TREES,
        "schedule": SCHEDULE,
        "warmup_requests": WARMUP_REQUESTS,
        "measured_requests": MEASURED_REQUESTS,
        "support": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "harness": {"path": str(args.harness), "sha256": HARNESS_SHA256},
            "monitor": {"path": str(args.monitor), "sha256": MONITOR_SHA256},
            "traffic_video": {
                "path": str(args.traffic_video),
                "bytes": TRAFFIC_BYTES,
                "sha256": TRAFFIC_SHA256,
                "hardlinks": [str(path) for path in videos],
            },
        },
        "cells": [],
    }
    manifest_path = args.results / "manifest.json"
    write_json(manifest_path, manifest)
    results_by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        for rep, concurrencies, variants in SCHEDULE:
            for variant in variants:
                commit = COMMITS[variant]
                run(
                    [
                        "git",
                        "-c",
                        "advice.detachedHead=false",
                        "checkout",
                        "--detach",
                        commit,
                    ],
                    cwd=args.source_root,
                )
                source_before = source_record(args.source_root, variant)
                stem = f"r{rep:02d}-{variant}-c" + "-".join(map(str, concurrencies))
                result_path = args.results / f"{stem}.json"
                monitor_path = args.results / f"{stem}-gpu-monitor.json"
                log_path = args.results / f"{stem}.log"
                harness_command = build_harness_command(
                    args,
                    variant=variant,
                    rep=rep,
                    concurrencies=concurrencies,
                    result_path=result_path,
                    videos=videos,
                )
                command = build_monitored_command(args, harness_command, monitor_path)
                cell: dict[str, Any] = {
                    "rep": rep,
                    "variant": variant,
                    "concurrencies": concurrencies,
                    "source_before": source_before,
                    "command": command,
                    "result": str(result_path),
                    "monitor": str(monitor_path),
                    "log": str(log_path),
                    "started_utc": utc_now(),
                    "status": "running",
                }
                manifest["cells"].append(cell)
                write_json(manifest_path, manifest)
                print(f"RUN {stem} commit={commit}", flush=True)
                with log_path.open("x", encoding="utf-8") as log:
                    completed = run(
                        command,
                        cwd=args.source_root,
                        env=environment,
                        stdout=log,
                        check=False,
                    )
                cell["finished_utc"] = utc_now()
                cell["returncode"] = completed.returncode
                if completed.returncode:
                    tail = log_path.read_text(errors="replace")[-16000:]
                    raise RuntimeError(f"cell failed {stem}:\n{tail}")
                result = json.loads(result_path.read_text())
                monitor = json.loads(monitor_path.read_text())
                server_log = result_path.with_name(result_path.stem + ".server.log")
                metrics = validate_result(
                    result,
                    monitor,
                    variant=variant,
                    rep=rep,
                    concurrencies=concurrencies,
                    server_log=server_log,
                )
                cell.update(
                    {
                        "status": "passed",
                        "source_after": source_record(args.source_root, variant),
                        "result_sha256": sha256_file(result_path),
                        "monitor_sha256": sha256_file(monitor_path),
                        "server_log": str(server_log),
                        "server_log_sha256": sha256_file(server_log),
                        "log_sha256": sha256_file(log_path),
                        "metrics": metrics,
                    }
                )
                results_by_pair[(rep, variant)] = result
                write_json(manifest_path, manifest)
                print(f"PASS {stem}", flush=True)

        token_audits = [
            strict_pair_audit(
                results_by_pair[(rep, "pr1")],
                results_by_pair[(rep, "gpu-resize")],
                rep=rep,
            )
            for rep in range(1, 7)
        ]
        summary = summarize(manifest["cells"])
        summary.update(
            {
                "gpu_label": args.gpu_label,
                "gpu_identity": gpu_identity,
                "token_parity_status": (
                    "failed_input_parity"
                    if any(
                        audit["status"] == "failed_input_parity"
                        for audit in token_audits
                    )
                    else "completion_or_text_mismatch"
                    if any(
                        audit["status"] == "completion_or_text_mismatch"
                        for audit in token_audits
                    )
                    else "passed_exact"
                ),
            }
        )
        del summary["csv_rows"]
        csv_summary = summarize(manifest["cells"])
        write_summary_files(args.results, csv_summary, token_audits)
        # Replace the JSON summary with the public form that excludes the CSV helper.
        write_json(args.results / "summary.json", summary)
        manifest.update(
            {
                "status": "passed",
                "finished_utc": utc_now(),
                "summary": summary,
                "token_audits": token_audits,
            }
        )
    finally:
        run(
            [
                "git",
                "-c",
                "advice.detachedHead=false",
                "checkout",
                "--detach",
                COMMITS["gpu-resize"],
            ],
            cwd=args.source_root,
            check=False,
        )
        manifest["source_terminal"] = source_record(args.source_root, "gpu-resize")
        write_json(manifest_path, manifest)
    print((args.results / "summary-table.md").read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
