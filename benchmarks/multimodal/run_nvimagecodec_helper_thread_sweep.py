# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run a benchmark-only nvImageCodec decoder/helper-thread sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import benchmark_host_cpu_topology as cpu_topology
import run_matched_image_decode as matched

RESOLUTIONS = {
    "1080p": ("corpus-1080p", "1920x1080"),
    "4k": ("corpus-4k", "3840x2160"),
}
FULL_CONFIGURATIONS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 2),
    (2, 4),
)
MATRICES = {
    "budget-2": ((1, 2), (2, 1)),
    "primary": ((1, 2), (1, 4), (2, 1)),
    "fixed-budget": ((1, 2), (2, 1), (1, 4), (2, 2)),
    "full": FULL_CONFIGURATIONS,
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_schedule(
    repetitions: int, configurations: tuple[tuple[int, int], ...]
) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        resolution_order = list(RESOLUTIONS)
        if repetition % 2 == 0:
            resolution_order.reverse()
        rotation = (repetition - 1) % len(configurations)
        config_order = configurations[rotation:] + configurations[:rotation]
        for resolution_position, resolution in enumerate(resolution_order, start=1):
            ordered = list(config_order)
            if (repetition + resolution_position) % 2:
                ordered.reverse()
            for config_position, (decoders, helper_threads) in enumerate(
                ordered, start=1
            ):
                schedule.append(
                    {
                        "repetition": repetition,
                        "resolution": resolution,
                        "resolution_position": resolution_position,
                        "config_position": config_position,
                        "decoders": decoders,
                        "helper_threads": helper_threads,
                        "nominal_helper_thread_budget": (decoders * helper_threads),
                        "configuration": f"d{decoders}-t{helper_threads}",
                    }
                )
    return schedule


def validate_injection(
    result: dict[str, Any],
    *,
    wrapper_sha256: str,
    base_cell_sha256: str,
    topology_helper_sha256: str,
    helper_threads: int,
    decoders: int,
    topology_sha256: str,
) -> None:
    injection = result["benchmark_injection"]
    expected = {
        "max_num_cpu_threads": helper_threads,
        "num_cuda_streams": 1,
        "cpu_fallback_decoder_unchanged": True,
        "source_method_sha256": injection["expected_source_method_sha256"],
    }
    for key, expected_value in expected.items():
        if injection[key] != expected_value:
            raise RuntimeError(f"benchmark injection {key} mismatch")
    if injection["wrapper"]["sha256"] != wrapper_sha256:
        raise RuntimeError("benchmark injection wrapper hash mismatch")
    if injection["base_cell"]["sha256"] != base_cell_sha256:
        raise RuntimeError("benchmark base-cell hash mismatch")
    if injection["cpu_topology_helper"]["sha256"] != topology_helper_sha256:
        raise RuntimeError("CPU-topology helper hash mismatch")
    configuration = result["configuration"]
    if configuration["nvimagecodec_max_num_cpu_threads"] != helper_threads:
        raise RuntimeError("annotated helper-thread count mismatch")
    if configuration["nominal_decoder_helper_thread_budget"] != (
        decoders * helper_threads
    ):
        raise RuntimeError("nominal helper-thread budget mismatch")
    if result["cpu_topology"]["static_sha256"] != topology_sha256:
        raise RuntimeError("host CPU topology changed between cells")


def aggregate_results(
    records: list[dict[str, Any]], repetitions: int
) -> dict[str, object]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["resolution"],
            int(record["decoders"]),
            int(record["helper_threads"]),
        )
        grouped[key].append(record["result"]["point"])

    groups = []
    for (resolution, decoders, helper_threads), points in sorted(grouped.items()):
        if len(points) != repetitions:
            raise RuntimeError(
                f"{resolution}/d{decoders}/t{helper_threads}: expected "
                f"{repetitions} points, found {len(points)}"
            )
        groups.append(
            {
                "resolution": resolution,
                "decoders": decoders,
                "helper_threads": helper_threads,
                "nominal_helper_thread_budget": decoders * helper_threads,
                "images_per_second": matched.numeric_summary(
                    [float(point["images_per_second"]) for point in points]
                ),
                "gpixels_per_second": matched.numeric_summary(
                    [float(point["gpixels_per_second"]) for point in points]
                ),
                "latency_p50_ms": matched.numeric_summary(
                    [float(point["request_latency_ms"]["p50"]) for point in points]
                ),
                "server_worker_cpu_cores": matched.numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["server_worker_tree"])
                        for point in points
                    ]
                ),
                "mps_cpu_cores": matched.numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["mps_server"])
                        for point in points
                    ]
                ),
                "server_worker_plus_mps_cpu_cores": matched.numeric_summary(
                    [
                        float(point["cpu"]["average_cores"]["server_worker_plus_mps"])
                        for point in points
                    ]
                ),
                "nvjpg_mean_percent": matched.numeric_summary(
                    [float(point["nvjpg_utilization"]["mean"]) for point in points]
                ),
                "nvjpg_p95_percent": matched.numeric_summary(
                    [float(point["nvjpg_utilization"]["p95"]) for point in points]
                ),
                "nvjpg_nonzero_percent": matched.numeric_summary(
                    [
                        float(point["nvjpg_utilization"]["nonzero_percent"])
                        for point in points
                    ]
                ),
                "device_memory_used_peak_bytes": matched.numeric_summary(
                    [
                        float(point["device_memory"]["used_bytes_peak"])
                        for point in points
                    ]
                ),
                "images": sum(int(point["images"]) for point in points),
                "fallbacks": sum(int(point["fallbacks"]) for point in points),
            }
        )

    points_by_identity = {
        (
            int(record["repetition"]),
            record["resolution"],
            int(record["decoders"]),
            int(record["helper_threads"]),
        ): record["result"]["point"]
        for record in records
    }
    available = {
        (int(record["decoders"]), int(record["helper_threads"])) for record in records
    }
    comparisons = []
    comparison_specs = (
        ("fixed-budget-2-d1t2-vs-d2t1", (1, 2), (2, 1)),
        ("d1t4-vs-current-d2t1", (1, 4), (2, 1)),
        ("fixed-budget-4-d1t4-vs-d2t2", (1, 4), (2, 2)),
        ("d1-helper-2-vs-1", (1, 2), (1, 1)),
        ("d1-helper-4-vs-1", (1, 4), (1, 1)),
        ("d2-helper-2-vs-1", (2, 2), (2, 1)),
        ("d2-helper-4-vs-1", (2, 4), (2, 1)),
    )
    for name, numerator, denominator in comparison_specs:
        if numerator not in available or denominator not in available:
            continue
        for resolution in RESOLUTIONS:
            rows = []
            for repetition in range(1, repetitions + 1):
                numerator_point = points_by_identity[
                    (repetition, resolution, *numerator)
                ]
                denominator_point = points_by_identity[
                    (repetition, resolution, *denominator)
                ]
                rows.append(
                    {
                        "repetition": repetition,
                        "throughput_percent": 100
                        * (
                            float(numerator_point["images_per_second"])
                            / float(denominator_point["images_per_second"])
                            - 1
                        ),
                        "cpu_core_delta": float(
                            numerator_point["cpu"]["average_cores"][
                                "server_worker_plus_mps"
                            ]
                        )
                        - float(
                            denominator_point["cpu"]["average_cores"][
                                "server_worker_plus_mps"
                            ]
                        ),
                        "nvjpg_mean_point_delta": float(
                            numerator_point["nvjpg_utilization"]["mean"]
                        )
                        - float(denominator_point["nvjpg_utilization"]["mean"]),
                        "peak_vram_byte_delta": int(
                            numerator_point["device_memory"]["used_bytes_peak"]
                        )
                        - int(denominator_point["device_memory"]["used_bytes_peak"]),
                    }
                )
            comparisons.append(
                {
                    "name": name,
                    "resolution": resolution,
                    "numerator": {
                        "decoders": numerator[0],
                        "helper_threads": numerator[1],
                        "nominal_helper_thread_budget": (numerator[0] * numerator[1]),
                    },
                    "denominator": {
                        "decoders": denominator[0],
                        "helper_threads": denominator[1],
                        "nominal_helper_thread_budget": (
                            denominator[0] * denominator[1]
                        ),
                    },
                    "per_repetition": rows,
                    "throughput_percent": matched.numeric_summary(
                        [float(row["throughput_percent"]) for row in rows]
                    ),
                    "cpu_core_delta": matched.numeric_summary(
                        [float(row["cpu_core_delta"]) for row in rows]
                    ),
                    "nvjpg_mean_point_delta": matched.numeric_summary(
                        [float(row["nvjpg_mean_point_delta"]) for row in rows]
                    ),
                    "peak_vram_byte_delta": matched.numeric_summary(
                        [float(row["peak_vram_byte_delta"]) for row in rows]
                    ),
                }
            )
    return {
        "schema": "vllm-nvimagecodec-helper-thread-sweep-aggregate-v1",
        "groups": groups,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cell-wrapper",
        type=Path,
        default=Path(__file__).with_name(
            "benchmark_nvimagecodec_helper_thread_cell.py"
        ),
    )
    parser.add_argument(
        "--base-cell",
        type=Path,
        default=Path(__file__).with_name("benchmark_image_decode_cell.py"),
    )
    parser.add_argument(
        "--topology-helper",
        type=Path,
        default=Path(__file__).with_name("benchmark_host_cpu_topology.py"),
    )
    parser.add_argument("--source-commit", default="0b71ce65")
    parser.add_argument("--matrix", choices=tuple(MATRICES), default="full")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--inter-cell-delay", type=float, default=2)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--coalesce-timeout-ms", type=float, default=0.25)
    parser.add_argument("--gpu-pool-gb", type=float, default=8)
    parser.add_argument("--media-loading-threads", type=int, default=8)
    parser.add_argument("--max-image-pixels", type=int, default=50_000_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions != 3:
        parser.error("helper-thread sweeps require exactly R3")
    if args.concurrency != 256:
        parser.error("helper-thread sweeps require fixed c256")
    if (
        min(
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

    source_root = args.source_root.resolve()
    source = matched.source_identity(source_root, args.source_commit)
    python = Path(os.path.abspath(args.python.expanduser()))
    paths = {
        "cell_wrapper": args.cell_wrapper.resolve(),
        "base_cell": args.base_cell.resolve(),
        "topology_helper": args.topology_helper.resolve(),
    }
    if not python.is_file():
        raise FileNotFoundError(python)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    for corpus_dir, _resolution in RESOLUTIONS.values():
        if not (args.media_root / corpus_dir).is_dir():
            raise FileNotFoundError(args.media_root / corpus_dir)

    configurations = MATRICES[args.matrix]
    schedule = build_schedule(args.repetitions, configurations)
    topology_before = cpu_topology.capture_cpu_topology()
    commands = []
    for item in schedule:
        corpus_dir, resolution = RESOLUTIONS[str(item["resolution"])]
        commands.append(
            [
                str(python),
                str(paths["cell_wrapper"]),
                str((args.media_root / corpus_dir).resolve()),
                "--helper-threads",
                str(item["helper_threads"]),
                "--backend",
                "nvimagecodec",
                "--source-root",
                str(source_root),
                "--expected-commit",
                args.source_commit,
                "--resolution",
                resolution,
                "--output",
                "<CELL_OUTPUT>",
                "--concurrency",
                str(args.concurrency),
                "--warmup",
                str(args.warmup),
                "--window",
                str(args.window),
                "--telemetry-interval",
                str(args.telemetry_interval),
                "--device-index",
                str(args.device_index),
                "--decoders",
                str(item["decoders"]),
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
                    "source": source,
                    "python": str(python),
                    "matrix": args.matrix,
                    "configurations": configurations,
                    "cells": len(schedule),
                    "host_cpu_topology": topology_before,
                    "paths": {name: str(path) for name, path in paths.items()},
                    "sha256": hashes,
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
        "schema": "vllm-nvimagecodec-helper-thread-sweep-manifest-v1",
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "source": source,
        "python": str(python),
        "host_cpu_topology_before": topology_before,
        "configuration": {
            "matrix": args.matrix,
            "configurations": [
                {
                    "decoders": decoders,
                    "helper_threads": helper_threads,
                    "nominal_helper_thread_budget": decoders * helper_threads,
                }
                for decoders, helper_threads in configurations
            ],
            "repetitions": args.repetitions,
            "concurrency": args.concurrency,
            "resolutions": list(RESOLUTIONS),
            "warmup_seconds": args.warmup,
            "window_seconds": args.window,
            "telemetry_interval_seconds": args.telemetry_interval,
            "inter_cell_delay_seconds": args.inter_cell_delay,
            "batch_size": args.batch_size,
            "pipeline_depth": args.pipeline_depth,
            "coalesce_timeout_ms": args.coalesce_timeout_ms,
            "gpu_pool_gb": args.gpu_pool_gb,
            "process_isolation": "one fresh process per cell",
            "product_configuration_modified": False,
        },
        "orchestrator": {
            "path": str(orchestrator),
            "sha256": hashlib.sha256(orchestrator.read_bytes()).hexdigest(),
            "argv": sys.argv,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "schedule": schedule,
        "results": [],
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "PYTHONNOUSERSITE": "1",
            "VLLM_MAX_IMAGE_PIXELS": str(args.max_image_pixels),
            "VLLM_MEDIA_LOADING_THREAD_COUNT": str(args.media_loading_threads),
        }
    )
    environment.pop("VLLM_IMAGE_LOADER_BACKEND", None)
    try:
        for item, command_template in zip(schedule, commands):
            name = (
                f"rep{int(item['repetition']):02d}-{item['resolution']}-c256-"
                f"d{item['decoders']}-t{item['helper_threads']}-"
                f"pos{int(item['config_position']):02d}"
            )
            artifact = args.output_dir / f"{name}.json"
            log_path = args.output_dir / f"{name}.log"
            command = [
                str(artifact) if value == "<CELL_OUTPUT>" else value
                for value in command_template
            ]
            print(f"{name}: starting fresh process", flush=True)
            with log_path.open("w") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=source_root,
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
            matched.validate_child(
                result,
                {
                    "backend": "nvimagecodec",
                    "resolution": item["resolution"],
                    "concurrency": args.concurrency,
                },
                source=source,
                harness_sha256=hashes["base_cell"],
                decoders=int(item["decoders"]),
                batch_size=args.batch_size,
                pipeline_depth=args.pipeline_depth,
                warmup=args.warmup,
                window=args.window,
                telemetry_interval=args.telemetry_interval,
            )
            validate_injection(
                result,
                wrapper_sha256=hashes["cell_wrapper"],
                base_cell_sha256=hashes["base_cell"],
                topology_helper_sha256=hashes["topology_helper"],
                helper_threads=int(item["helper_threads"]),
                decoders=int(item["decoders"]),
                topology_sha256=str(topology_before["static_sha256"]),
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
                f"CPU={point['cpu']['average_cores']['server_worker_plus_mps']:.3f}, "
                f"NVJPG={point['nvjpg_utilization']['mean']:.2f}%",
                flush=True,
            )
            if args.inter_cell_delay:
                time.sleep(args.inter_cell_delay)

        expected_cells = args.repetitions * len(RESOLUTIONS) * len(configurations)
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
        topology_after = cpu_topology.capture_cpu_topology()
        if topology_after["static_sha256"] != topology_before["static_sha256"]:
            raise RuntimeError("host CPU topology changed during the sweep")

        aggregate = aggregate_results(manifest["results"], args.repetitions)
        aggregate["validation"] = {
            "cells": len(manifest["results"]),
            "expected_cells": expected_cells,
            "fresh_process_identities": len(identities),
            "device_uuid": next(iter(device_uuids)),
            "fixed_host_cpu_topology": True,
            "host_cpu_topology_sha256": topology_before["static_sha256"],
            "all_accounting_and_correctness_checks_passed": True,
        }
        atomic_json(args.output_dir / "aggregate.json", aggregate)
        manifest["host_cpu_topology_after"] = topology_after
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, manifest)

        print(
            "resolution decoders helper_threads helper_budget images_per_second "
            "cpu_cores nvjpg_mean nvjpg_p95 peak_vram_bytes",
            flush=True,
        )
        for group in aggregate["groups"]:
            print(
                f"{group['resolution']} {group['decoders']} "
                f"{group['helper_threads']} "
                f"{group['nominal_helper_thread_budget']} "
                f"{group['images_per_second']['mean']:.3f} "
                f"{group['server_worker_plus_mps_cpu_cores']['mean']:.3f} "
                f"{group['nvjpg_mean_percent']['mean']:.2f} "
                f"{group['nvjpg_p95_percent']['mean']:.2f} "
                f"{group['device_memory_used_peak_bytes']['mean']:.0f}",
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
