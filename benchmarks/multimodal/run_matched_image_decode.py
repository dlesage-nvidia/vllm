# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Orchestrate matched fresh-process Pillow and nvImageCodec decode runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESOLUTIONS = {
    "1080p": ("corpus-1080p", "1920x1080", 1920, 1080),
    "4k": ("corpus-4k", "3840x2160", 3840, 2160),
}
BACKENDS = ("pillow", "nvimagecodec")


def parse_positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or min(values) <= 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "values must be unique positive comma-separated integers"
        )
    return values


def git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def source_identity(source_root: Path, expected_commit: str) -> dict[str, str]:
    source_root = source_root.resolve()
    expected = git_output(source_root, "rev-parse", f"{expected_commit}^{{commit}}")
    actual = git_output(source_root, "rev-parse", "HEAD^{commit}")
    if actual != expected:
        raise RuntimeError(f"{source_root}: expected {expected}, found {actual}")
    status = git_output(source_root, "status", "--porcelain=v1", "--untracked-files=no")
    if status:
        raise RuntimeError(f"{source_root}: tracked worktree is dirty\n{status}")
    return {
        "root": str(source_root),
        "commit": actual,
        "tree": git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "tracked_git_status": "clean",
    }


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_schedule(
    repetitions: int, concurrencies: tuple[int, ...]
) -> list[dict[str, object]]:
    base = [
        (resolution, concurrency)
        for resolution in RESOLUTIONS
        for concurrency in concurrencies
    ]
    schedule: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        rotation = (repetition - 1) % len(base)
        cells = base[rotation:] + base[:rotation]
        if repetition % 2 == 0:
            cells.reverse()
        for pair_position, (resolution, concurrency) in enumerate(cells, start=1):
            backend_order = list(BACKENDS)
            if (repetition + pair_position) % 2:
                backend_order.reverse()
            for backend_position, backend in enumerate(backend_order, start=1):
                schedule.append(
                    {
                        "repetition": repetition,
                        "resolution": resolution,
                        "concurrency": concurrency,
                        "pair_position": pair_position,
                        "backend_position": backend_position,
                        "backend": backend,
                    }
                )
    return schedule


def validate_child(
    result: dict[str, Any],
    item: dict[str, object],
    *,
    source: dict[str, str],
    harness_sha256: str,
    decoders: int,
    batch_size: int,
    pipeline_depth: int,
    warmup: float,
    window: float,
    telemetry_interval: float,
) -> None:
    backend = str(item["backend"])
    resolution = str(item["resolution"])
    concurrency = int(item["concurrency"])
    if result.get("schema") != "vllm-matched-image-decode-cell-v1":
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: bad schema")
    if result["source"]["commit"] != source["commit"]:
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: source mismatch")
    if result["harness"]["sha256"] != harness_sha256:
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: harness mismatch")

    expected_width = RESOLUTIONS[resolution][2]
    expected_height = RESOLUTIONS[resolution][3]
    corpus = result["corpus"]
    if (corpus["width"], corpus["height"]) != (
        expected_width,
        expected_height,
    ):
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: wrong corpus")
    configuration = result["configuration"]
    expected_configuration = {
        "backend": backend,
        "concurrency": concurrency,
        "warmup_seconds": warmup,
        "window_seconds": window,
        "telemetry_interval_seconds": telemetry_interval,
    }
    for key, expected in expected_configuration.items():
        if configuration[key] != expected:
            raise RuntimeError(
                f"{backend}/{resolution}/c{concurrency}: configuration {key} mismatch"
            )
    if backend == "nvimagecodec":
        for key, expected in (
            ("decoders", decoders),
            ("batch_size", batch_size),
            ("pipeline_depth", pipeline_depth),
        ):
            if configuration[key] != expected:
                raise RuntimeError(
                    f"{backend}/{resolution}/c{concurrency}: {key} mismatch"
                )

    failed_validation = [
        name for name, passed in result["validation"].items() if passed is not True
    ]
    if failed_validation:
        raise RuntimeError(
            f"{backend}/{resolution}/c{concurrency}: validation failed: "
            + ", ".join(failed_validation)
        )
    point = result["point"]
    if point["fallbacks"] or point["accounting"]["accounting_gap"]:
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: accounting")
    if point["nvjpg_utilization"]["samples"] <= 0:
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: no NVJPG data")
    if point["device_memory"]["used_bytes_peak"] <= 0:
        raise RuntimeError(f"{backend}/{resolution}/c{concurrency}: no VRAM data")
    if backend == "nvimagecodec":
        accounting = point["accounting"]
        for key in (
            "service_accounting_gap",
            "native_accounting_gap",
        ):
            if accounting[key]:
                raise RuntimeError(
                    f"{backend}/{resolution}/c{concurrency}: nonzero {key}"
                )
        if accounting["gpu_pool"]["outstanding_after_bytes"]:
            raise RuntimeError(
                f"{backend}/{resolution}/c{concurrency}: leaked GPU lease"
            )
        native_widths = {
            int(width): int(jobs)
            for width, jobs in accounting["native_width_histogram"].items()
        }
        if native_widths.get(batch_size, 0) <= 0:
            raise RuntimeError(
                f"{backend}/{resolution}/c{concurrency}: no full native batch"
            )
        service_widths = {
            int(width): int(jobs)
            for width, jobs in accounting["service_width_histogram"].items()
        }
        if not any(
            width > batch_size and jobs > 0 for width, jobs in service_widths.items()
        ):
            raise RuntimeError(
                f"{backend}/{resolution}/c{concurrency}: "
                "ring never claimed more than one batch"
            )


def aggregate_results(
    records: list[dict[str, Any]], repetitions: int
) -> dict[str, object]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["resolution"],
            int(record["concurrency"]),
            record["backend"],
        )
        grouped[key].append(record["result"]["point"])

    groups = []
    for (resolution, concurrency, backend), points in sorted(grouped.items()):
        if len(points) != repetitions:
            raise RuntimeError(
                f"{backend}/{resolution}/c{concurrency}: expected {repetitions} "
                f"points, found {len(points)}"
            )
        groups.append(
            {
                "resolution": resolution,
                "concurrency": concurrency,
                "backend": backend,
                "images_per_second": numeric_summary(
                    [float(point["images_per_second"]) for point in points]
                ),
                "gpixels_per_second": numeric_summary(
                    [float(point["gpixels_per_second"]) for point in points]
                ),
                "latency_p50_ms": numeric_summary(
                    [float(point["request_latency_ms"]["p50"]) for point in points]
                ),
                "server_worker_cpu_cores": numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["server_worker_tree"])
                        for point in points
                    ]
                ),
                "mps_cpu_cores": numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["mps_server"])
                        for point in points
                    ]
                ),
                "server_worker_plus_mps_cpu_cores": numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["server_worker_plus_mps"])
                        for point in points
                    ]
                ),
                "nvjpg_mean_percent": numeric_summary(
                    [float(point["nvjpg_utilization"]["mean"]) for point in points]
                ),
                "nvjpg_p95_percent": numeric_summary(
                    [float(point["nvjpg_utilization"]["p95"]) for point in points]
                ),
                "nvjpg_nonzero_percent": numeric_summary(
                    [
                        float(point["nvjpg_utilization"]["nonzero_percent"])
                        for point in points
                    ]
                ),
                "device_memory_used_peak_bytes": numeric_summary(
                    [
                        float(point["device_memory"]["used_bytes_peak"])
                        for point in points
                    ]
                ),
                "device_memory_peak_delta_bytes": numeric_summary(
                    [
                        float(
                            point["device_memory"][
                                "peak_minus_first_timed_sample_bytes"
                            ]
                        )
                        for point in points
                    ]
                ),
                "images": sum(int(point["images"]) for point in points),
                "fallbacks": sum(int(point["fallbacks"]) for point in points),
            }
        )

    paired = []
    identity: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    for record in records:
        key = (
            int(record["repetition"]),
            record["resolution"],
            int(record["concurrency"]),
            record["backend"],
        )
        identity[key] = record["result"]["point"]
    dimensions = sorted(
        {(record["resolution"], int(record["concurrency"])) for record in records}
    )
    for resolution, concurrency in dimensions:
        rows = []
        for repetition in range(1, repetitions + 1):
            pillow = identity[(repetition, resolution, concurrency, "pillow")]
            nvimagecodec = identity[
                (repetition, resolution, concurrency, "nvimagecodec")
            ]
            rows.append(
                {
                    "repetition": repetition,
                    "throughput_speedup_percent": 100
                    * (
                        float(nvimagecodec["images_per_second"])
                        / float(pillow["images_per_second"])
                        - 1
                    ),
                    "throughput_ratio": float(nvimagecodec["images_per_second"])
                    / float(pillow["images_per_second"]),
                    "cpu_core_delta": float(
                        nvimagecodec["cpu"]["average_cores"]["server_worker_plus_mps"]
                    )
                    - float(pillow["cpu"]["average_cores"]["server_worker_plus_mps"]),
                    "nvjpg_mean_point_delta": float(
                        nvimagecodec["nvjpg_utilization"]["mean"]
                    )
                    - float(pillow["nvjpg_utilization"]["mean"]),
                    "peak_vram_byte_delta": int(
                        nvimagecodec["device_memory"]["used_bytes_peak"]
                    )
                    - int(pillow["device_memory"]["used_bytes_peak"]),
                }
            )
        paired.append(
            {
                "resolution": resolution,
                "concurrency": concurrency,
                "per_repetition": rows,
                "throughput_speedup_percent": numeric_summary(
                    [float(row["throughput_speedup_percent"]) for row in rows]
                ),
                "throughput_ratio": numeric_summary(
                    [float(row["throughput_ratio"]) for row in rows]
                ),
                "cpu_core_delta": numeric_summary(
                    [float(row["cpu_core_delta"]) for row in rows]
                ),
                "nvjpg_mean_point_delta": numeric_summary(
                    [float(row["nvjpg_mean_point_delta"]) for row in rows]
                ),
                "peak_vram_byte_delta": numeric_summary(
                    [float(row["peak_vram_byte_delta"]) for row in rows]
                ),
            }
        )
    return {
        "schema": "vllm-matched-image-decode-aggregate-v1",
        "groups": groups,
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--pillow-root", type=Path, required=True)
    parser.add_argument("--nvimagecodec-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--pillow-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--nvimagecodec-python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument(
        "--cell-harness",
        type=Path,
        default=Path(__file__).with_name("benchmark_image_decode_cell.py"),
    )
    parser.add_argument("--pillow-commit", default="d125b540")
    parser.add_argument("--nvimagecodec-commit", default="0b71ce65")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrencies", type=parse_positive_csv, default=(128, 256))
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--inter-cell-delay", type=float, default=2)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--decoders", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--coalesce-timeout-ms", type=float, default=0.25)
    parser.add_argument("--gpu-pool-gb", type=float, default=8)
    parser.add_argument("--media-loading-threads", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions != 3:
        parser.error("matched publication runs require exactly R3")
    if (
        min(
            args.decoders,
            args.batch_size,
            args.pipeline_depth,
            args.media_loading_threads,
            args.max_image_pixels,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if (
        min(
            args.warmup,
            args.window,
            args.telemetry_interval,
            args.gpu_pool_gb,
        )
        <= 0
        or args.inter_cell_delay < 0
    ):
        parser.error("timings and GPU pool size must be positive")
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(args.output_dir)

    harness = args.cell_harness.resolve()
    if not harness.is_file():
        raise FileNotFoundError(harness)
    harness_sha256 = hashlib.sha256(harness.read_bytes()).hexdigest()
    roots = {
        "pillow": args.pillow_root.resolve(),
        "nvimagecodec": args.nvimagecodec_root.resolve(),
    }
    commits = {
        "pillow": args.pillow_commit,
        "nvimagecodec": args.nvimagecodec_commit,
    }
    # Preserve an explicitly selected virtualenv launcher. Resolving its
    # ``bin/python`` symlink would invoke the base interpreter outside the venv.
    pythons = {
        "pillow": Path(os.path.abspath(args.pillow_python.expanduser())),
        "nvimagecodec": Path(os.path.abspath(args.nvimagecodec_python.expanduser())),
    }
    for backend, executable in pythons.items():
        if not executable.is_file():
            raise FileNotFoundError(f"{backend} Python: {executable}")
    sources = {
        backend: source_identity(roots[backend], commits[backend])
        for backend in BACKENDS
    }
    if sources["pillow"]["commit"] == sources["nvimagecodec"]["commit"]:
        raise RuntimeError("Pillow and nvImageCodec source commits are identical")
    for corpus_dir, _resolution_text, _width, _height in RESOLUTIONS.values():
        path = args.media_root / corpus_dir
        if not path.is_dir():
            raise FileNotFoundError(path)

    schedule = build_schedule(args.repetitions, args.concurrencies)
    commands: list[list[str]] = []
    for item in schedule:
        backend = str(item["backend"])
        resolution = str(item["resolution"])
        corpus_dir, resolution_text, _width, _height = RESOLUTIONS[resolution]
        commands.append(
            [
                str(pythons[backend]),
                str(harness),
                str((args.media_root / corpus_dir).resolve()),
                "--backend",
                backend,
                "--source-root",
                str(roots[backend]),
                "--expected-commit",
                commits[backend],
                "--resolution",
                resolution_text,
                "--output",
                "<CELL_OUTPUT>",
                "--concurrency",
                str(item["concurrency"]),
                "--warmup",
                str(args.warmup),
                "--window",
                str(args.window),
                "--telemetry-interval",
                str(args.telemetry_interval),
                "--device-index",
                str(args.device_index),
                "--decoders",
                str(args.decoders),
                "--batch-size",
                str(args.batch_size),
                "--pipeline-depth",
                str(args.pipeline_depth),
                "--coalesce-timeout-ms",
                str(args.coalesce_timeout_ms),
                "--gpu-pool-gb",
                str(args.gpu_pool_gb),
            ]
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "sources": sources,
                    "harness": {
                        "path": str(harness),
                        "sha256": harness_sha256,
                    },
                    "cells": len(schedule),
                    "schedule": schedule,
                    "commands": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    args.output_dir.mkdir(parents=True)
    orchestrator = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema": "vllm-matched-image-decode-manifest-v1",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "configuration": {
            "repetitions": args.repetitions,
            "concurrencies": list(args.concurrencies),
            "resolutions": list(RESOLUTIONS),
            "warmup_seconds": args.warmup,
            "window_seconds": args.window,
            "telemetry_interval_seconds": args.telemetry_interval,
            "inter_cell_delay_seconds": args.inter_cell_delay,
            "decoders": args.decoders,
            "batch_size": args.batch_size,
            "pipeline_depth": args.pipeline_depth,
            "coalesce_timeout_ms": args.coalesce_timeout_ms,
            "gpu_pool_gb": args.gpu_pool_gb,
            "media_loading_threads": args.media_loading_threads,
            "max_image_pixels": args.max_image_pixels,
            "process_isolation": "one fresh child process per cell",
            "pairing": "adjacent Pillow/nvImageCodec pairs with alternating order",
        },
        "orchestrator": {
            "path": str(orchestrator),
            "sha256": hashlib.sha256(orchestrator.read_bytes()).hexdigest(),
            "argv": sys.argv,
        },
        "cell_harness": {"path": str(harness), "sha256": harness_sha256},
        "schedule": schedule,
        "results": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    try:
        for item, command_template in zip(schedule, commands):
            name = (
                f"rep{int(item['repetition']):02d}-{item['resolution']}-"
                f"c{item['concurrency']}-{item['backend']}-"
                f"pair{int(item['pair_position']):02d}-"
                f"pos{item['backend_position']}"
            )
            artifact = args.output_dir / f"{name}.json"
            log_path = args.output_dir / f"{name}.log"
            command = [
                str(artifact) if value == "<CELL_OUTPUT>" else value
                for value in command_template
            ]
            backend = str(item["backend"])
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHONPATH": str(roots[backend]),
                    "PYTHONNOUSERSITE": "1",
                    "VLLM_MAX_IMAGE_PIXELS": str(args.max_image_pixels),
                    "VLLM_MEDIA_LOADING_THREAD_COUNT": str(args.media_loading_threads),
                }
            )
            environment.pop("VLLM_IMAGE_LOADER_BACKEND", None)
            print(f"{name}: starting fresh process", flush=True)
            with log_path.open("w") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=roots[backend],
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(f"{name}: {line}", end="", flush=True)
                    log_stream.write(line)
                    log_stream.flush()
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f"{name} exited {returncode}; see {log_path}")
            result = json.loads(artifact.read_text())
            validate_child(
                result,
                item,
                source=sources[backend],
                harness_sha256=harness_sha256,
                decoders=args.decoders,
                batch_size=args.batch_size,
                pipeline_depth=args.pipeline_depth,
                warmup=args.warmup,
                window=args.window,
                telemetry_interval=args.telemetry_interval,
            )
            if result["process"]["pid"] != process.pid:
                raise RuntimeError(f"{name}: child PID provenance mismatch")
            record = dict(item)
            record.update(
                {
                    "name": name,
                    "artifact": str(artifact),
                    "log": str(log_path),
                    "command": command,
                    "process": result["process"],
                    "result": result,
                }
            )
            manifest["results"].append(record)
            atomic_json(manifest_path, manifest)
            point = result["point"]
            print(
                f"{name}: {point['images_per_second']:.3f} img/s, "
                f"NVJPG {point['nvjpg_utilization']['mean']:.2f}% mean/"
                f"{point['nvjpg_utilization']['p95']:.2f}% p95",
                flush=True,
            )
            if args.inter_cell_delay:
                time.sleep(args.inter_cell_delay)

        expected_cells = (
            args.repetitions * len(args.concurrencies) * len(RESOLUTIONS) * 2
        )
        if len(manifest["results"]) != expected_cells:
            raise RuntimeError(
                f"expected {expected_cells} cells, found {len(manifest['results'])}"
            )
        identities = {
            (record["process"]["pid"], record["process"]["create_time"])
            for record in manifest["results"]
        }
        if len(identities) != expected_cells:
            raise RuntimeError("fresh-process identity was reused")
        device_uuids = {
            record["result"]["device"]["uuid"] for record in manifest["results"]
        }
        if len(device_uuids) != 1:
            raise RuntimeError(f"cells used multiple devices: {device_uuids}")
        for resolution in RESOLUTIONS:
            corpus_digests = {
                record["result"]["corpus"]["sha256"]
                for record in manifest["results"]
                if record["resolution"] == resolution
            }
            if len(corpus_digests) != 1:
                raise RuntimeError(
                    f"{resolution}: cells used different corpora: {corpus_digests}"
                )

        aggregate = aggregate_results(manifest["results"], args.repetitions)
        aggregate["validation"] = {
            "cells": len(manifest["results"]),
            "expected_cells": expected_cells,
            "fresh_process_identities": len(identities),
            "device_uuid": next(iter(device_uuids)),
            "all_cell_validations_passed": True,
            "same_corpus_per_resolution": True,
        }
        atomic_json(args.output_dir / "aggregate.json", aggregate)
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, manifest)

        group_index = {
            (group["resolution"], group["concurrency"], group["backend"]): group
            for group in aggregate["groups"]
        }
        paired_index = {
            (pair["resolution"], pair["concurrency"]): pair
            for pair in aggregate["paired"]
        }
        print(
            "resolution concurrency pillow_img_s nvimagecodec_img_s speedup_pct "
            "nvjpg_mean nvjpg_p95 nvjpg_nonzero peak_vram_bytes",
            flush=True,
        )
        for resolution in RESOLUTIONS:
            for concurrency in args.concurrencies:
                pillow = group_index[(resolution, concurrency, "pillow")]
                nvimagecodec = group_index[(resolution, concurrency, "nvimagecodec")]
                paired = paired_index[(resolution, concurrency)]
                print(
                    f"{resolution} c{concurrency} "
                    f"{pillow['images_per_second']['mean']:.3f} "
                    f"{nvimagecodec['images_per_second']['mean']:.3f} "
                    f"{paired['throughput_speedup_percent']['mean']:.2f} "
                    f"{nvimagecodec['nvjpg_mean_percent']['mean']:.2f} "
                    f"{nvimagecodec['nvjpg_p95_percent']['mean']:.2f} "
                    f"{nvimagecodec['nvjpg_nonzero_percent']['mean']:.2f} "
                    f"{nvimagecodec['device_memory_used_peak_bytes']['mean']:.0f}",
                    flush=True,
                )
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
