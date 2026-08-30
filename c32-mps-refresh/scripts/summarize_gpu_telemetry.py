# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Summarize GPU/NVDEC telemetry over each measured request window."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }


def values_at(samples: list[dict[str, Any]], *keys: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value: Any = sample
        for key in keys:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def resolve_result(path: Path) -> Path:
    if path.exists():
        return path
    compressed = Path(f"{path}.gz")
    if compressed.exists():
        return compressed
    raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    manifest_path = args.result_dir / "manifest.json"
    manifest = load_json(manifest_path)
    cells: list[dict[str, Any]] = []

    for cell in manifest["cells"]:
        if cell.get("status") != "passed":
            continue

        result_path = resolve_result(Path(cell["result"]))
        result = load_json(result_path)
        block = result["concurrency_blocks"][0]
        measured = block["measured"]
        started = parse_time(measured["started_at"])
        finished = parse_time(measured["finished_at"])

        monitor_path = Path(cell["monitor"])
        monitor = load_json(monitor_path)
        monitor_configuration = monitor["configuration"]
        if monitor.get("status") != "passed":
            raise RuntimeError(f"monitor did not pass: {monitor_path}")
        if monitor_configuration.get("cpu_queries") is not False:
            raise RuntimeError(f"unexpected CPU telemetry: {monitor_path}")
        if monitor_configuration.get("process_queries") is not False:
            raise RuntimeError(f"unexpected process telemetry: {monitor_path}")
        if monitor_configuration.get("sample_interval_seconds") != 0.2:
            raise RuntimeError(f"unexpected sample interval: {monitor_path}")
        if monitor["device"]["uuid"] != manifest["gpu_uuid"]:
            raise RuntimeError(f"GPU UUID mismatch: {monitor_path}")
        measured_samples = [
            sample
            for sample in monitor["samples"]
            if started <= parse_time(sample["utc"]) <= finished
        ]
        gaps = [
            value
            for value in values_at(measured_samples, "sample_gap_seconds")
            if value >= 0
        ]
        sample_errors = [
            sample["sample_error"]
            for sample in measured_samples
            if sample.get("sample_error") is not None
        ]
        if not measured_samples:
            raise RuntimeError(f"no measured-window samples: {monitor_path}")
        if sample_errors:
            raise RuntimeError(f"sample errors in measured window: {monitor_path}")

        metrics = {
            "gpu_utilization_percent": stats(
                values_at(measured_samples, "utilization_percent")
            ),
            "memory_controller_utilization_percent": stats(
                values_at(measured_samples, "memory_utilization_percent")
            ),
            "nvdec_utilization_percent": stats(
                values_at(
                    measured_samples,
                    "engine_utilization",
                    "decoder",
                    "utilization_percent",
                )
            ),
            "memory_used_mib": stats(values_at(measured_samples, "memory_used_mib")),
        }
        cells.append(
            {
                "rep": cell["rep"],
                "position": cell["position"],
                "variant": cell["variant"],
                "mps": cell["mps"],
                "request_throughput_per_second": cell["metrics"][
                    "request_throughput_per_second"
                ],
                "measured_started_utc": measured["started_at"],
                "measured_finished_utc": measured["finished_at"],
                "measured_window_seconds": measured["measured_window_seconds"],
                "monitor_sample_interval_seconds": monitor["configuration"][
                    "sample_interval_seconds"
                ],
                "monitor_sample_count_total": monitor["sample_count"],
                "monitor_sample_count_measured": len(measured_samples),
                "monitor_max_sample_gap_seconds_measured": max(gaps, default=None),
                "monitor_sample_errors_measured": sample_errors,
                "metrics": metrics,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[f"{cell['variant']}_mps_{cell['mps']}"].append(cell)

    group_summary: dict[str, Any] = {}
    for name, group_cells in sorted(grouped.items()):
        group_summary[name] = {
            "cell_count": len(group_cells),
            "request_throughput_per_second": stats(
                [cell["request_throughput_per_second"] for cell in group_cells]
            ),
            "per_cell_means": {
                metric: stats(
                    [
                        cell["metrics"][metric]["mean"]
                        for cell in group_cells
                        if cell["metrics"][metric]["mean"] is not None
                    ]
                )
                for metric in (
                    "gpu_utilization_percent",
                    "memory_controller_utilization_percent",
                    "nvdec_utilization_percent",
                    "memory_used_mib",
                )
            },
        }

    payload = {
        "schema": "vllm-pr1-c32-measured-window-gpu-telemetry-summary-v1",
        "source_manifest": manifest_path.name,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "gpu_label": manifest["gpu_label"],
        "gpu_uuid": manifest["gpu_uuid"],
        "model_key": manifest["model_key"],
        "configuration": {
            "cpu_utilization_captured": False,
            "gpu_and_nvdec_sample_interval_seconds": 0.2,
            "window": "measured requests only",
        },
        "groups": group_summary,
        "cells": sorted(cells, key=lambda cell: (cell["rep"], cell["position"])),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
