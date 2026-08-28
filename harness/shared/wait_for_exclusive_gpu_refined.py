# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wait for continuously idle GPU telemetry without broad process matching."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynv_gpu_guard as guard

UTC = timezone.utc

EXPECTED_HELPER_SHA256 = (
    "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
)


def utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, UTC).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.2)
    parser.add_argument("--maximum-sample-gap-seconds", type=float, default=1.0)
    parser.add_argument("--idle-memory-ceiling-mib", type=int, default=1024)
    parser.add_argument("--idle-max-load-1m-per-cpu", type=float, default=0.25)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--conflicting-controller-root",
        action="append",
        default=[],
        help="Absolute cwd in which exact './screen.sh' is a conflicting controller",
    )
    args = parser.parse_args()
    controller_roots = guard.normalize_conflicting_controller_roots(
        args.conflicting_controller_root
    )
    args.output = args.output.resolve()
    sample_log = args.output.with_name(args.output.stem + ".samples.jsonl")
    if args.seconds <= 0 or args.timeout <= args.seconds:
        raise ValueError("timeout must be greater than a positive idle interval")
    if args.sample_interval_seconds != 0.2:
        raise ValueError("sample interval must remain exactly 0.2 seconds")
    if args.maximum_sample_gap_seconds != 1.0:
        raise ValueError("maximum sample gap must remain exactly 1.0 second")
    if (
        args.idle_memory_ceiling_mib != 1024
        or args.idle_max_load_1m_per_cpu != 0.25
        or args.device_index != 0
    ):
        raise ValueError(
            "idle memory/load/device must remain exactly 1024 MiB/0.25/device 0"
        )
    preexisting = [path for path in (args.output, sample_log) if path.exists()]
    if preexisting:
        raise FileExistsError(
            "refusing to overwrite idle-gate evidence: "
            + ", ".join(str(path) for path in preexisting)
        )

    script = Path(__file__).resolve()
    helper = Path(guard.__file__).resolve()
    helper_sha256 = guard.sha256_file(helper)
    if helper_sha256 != EXPECTED_HELPER_SHA256:
        raise RuntimeError(
            f"guard helper hash mismatch: {helper_sha256} != {EXPECTED_HELPER_SHA256}"
        )
    lineage = guard.ancestor_identities(os.getpid())
    started_monotonic_ns = time.monotonic_ns()
    started_time_ns = time.time_ns()
    stop_signal: int | None = None

    def request_stop(signum: int, unused_frame: Any) -> None:
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, request_stop) for signum in handled_signals
    }
    report: dict[str, Any] = {
        "status": "running",
        "passed": False,
        "started_utc": utc_from_ns(started_time_ns),
        "started_time_ns": started_time_ns,
        "process": {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "effective_uid": os.geteuid(),
            "argv": sys.argv,
            "executable": sys.executable,
            "script_path": str(script),
            "script_sha256": guard.sha256_file(script),
        },
        "guard_helper": {
            "path": str(helper),
            "sha256": helper_sha256,
        },
        "excluded_process_lineage_identities": [
            {"pid": pid, "start_time_ticks": start}
            for pid, start in sorted(lineage.items())
        ],
        "configuration": {
            "device_index": args.device_index,
            "required_idle_seconds": args.seconds,
            "timeout_seconds": args.timeout,
            "sample_interval_seconds": args.sample_interval_seconds,
            "maximum_sample_gap_seconds": args.maximum_sample_gap_seconds,
            "idle_memory_ceiling_mib": args.idle_memory_ceiling_mib,
            "idle_max_load_1m_per_cpu": args.idle_max_load_1m_per_cpu,
            "conflicting_controller_roots": controller_roots,
            "telemetry": "direct NVML; no external commands or pgrep",
            "cpu_conflict_policy": (
                "exact argv entrypoint/module or /proc comm classification only; "
                "cwd and joined-command substrings are never match keys"
            ),
        },
        "sample_log": {"path": str(sample_log), "format": "JSON Lines"},
        "events": [],
        "observer_events": [],
        "handled_signals": [signal.Signals(item).name for item in handled_signals],
    }
    sample_log.open("x").close()
    guard.write_json_atomic(args.output, report)

    probe: guard.ProbeWorker | None = None
    sample_count = 0
    dirty_sample_count = 0
    observer_sample_count = 0
    observer_identities: set[tuple[int, int, str]] = set()
    previous_monotonic_ns: int | None = None
    quiet_started_monotonic_ns: int | None = None
    quiet_started_time_ns: int | None = None
    quiet_start_index: int | None = None
    quiet_sample_count = 0
    quiet_maximum_gap = 0.0
    last_event_signature: str | None = None
    last_observer_signature: str | None = None
    try:
        probe = guard.ProbeWorker(
            args.device_index, conflicting_controller_roots=controller_roots
        )
        report["device"] = probe.device
        report["probe_worker"] = {
            **probe.worker_identity,
            "startup_timeout_seconds": guard.PROBE_STARTUP_TIMEOUT_SECONDS,
            "response_timeout_seconds": guard.PROBE_RESPONSE_TIMEOUT_SECONDS,
            "shutdown_timeout_seconds": guard.PROBE_SHUTDOWN_TIMEOUT_SECONDS,
        }
        guard.write_json_atomic(args.output, report)
        next_sample = time.monotonic()
        with sample_log.open("a") as stream:
            while True:
                if stop_signal is not None:
                    report["status"] = "interrupted"
                    report["termination_signal"] = signal.Signals(stop_signal).name
                    break
                loop_monotonic_ns = time.monotonic_ns()
                elapsed_seconds = (loop_monotonic_ns - started_monotonic_ns) / 1e9
                if elapsed_seconds > args.timeout:
                    report["status"] = "timed_out"
                    break
                try:
                    probe_result = probe.sample(excluded_identities=lineage)
                    gpu = probe_result["gpu"]
                    cpu = probe_result["cpu"]
                    probe_duration_seconds = probe_result["worker_duration_seconds"]
                    sample_error = None
                except Exception as error:
                    gpu = None
                    cpu = {
                        "host_load": None,
                        "conflicts": [],
                        "observers": [],
                        "owned_processes": [],
                        "errors": [],
                    }
                    probe_duration_seconds = None
                    sample_error = f"{type(error).__name__}: {error}"
                now_monotonic_ns = time.monotonic_ns()
                now_time_ns = time.time_ns()
                sample_gap = (
                    None
                    if previous_monotonic_ns is None
                    else (now_monotonic_ns - previous_monotonic_ns) / 1e9
                )
                reasons: list[str] = []
                if sample_error is not None:
                    reasons.append("sample_error")
                if cpu["errors"]:
                    reasons.append("process_inspection_error")
                if (
                    sample_gap is not None
                    and sample_gap > args.maximum_sample_gap_seconds
                ):
                    reasons.append("sampling_gap_exceeded")
                if gpu is not None:
                    reasons.extend(
                        guard.gpu_is_idle(
                            gpu, memory_ceiling_mib=args.idle_memory_ceiling_mib
                        )
                    )
                if cpu["conflicts"]:
                    reasons.append("conflicting_cpu_process")
                host_load = cpu.get("host_load")
                if (
                    host_load is not None
                    and float(host_load["load_1m_per_cpu"])
                    > args.idle_max_load_1m_per_cpu
                ):
                    reasons.append("host_load_1m_per_cpu_above_idle_ceiling")

                observers = cpu["observers"]
                compact_observers = [guard.compact_observer(item) for item in observers]
                if observers:
                    observer_sample_count += 1
                    for observer in observers:
                        observer_identities.add(
                            (
                                int(observer["pid"]),
                                int(observer["start_time_ticks"]),
                                str(observer["classification_reason"]),
                            )
                        )
                if reasons:
                    quiet_started_monotonic_ns = None
                    quiet_started_time_ns = None
                    quiet_start_index = None
                    quiet_sample_count = 0
                    quiet_maximum_gap = 0.0
                    quiet_elapsed = 0.0
                else:
                    if quiet_started_monotonic_ns is None:
                        quiet_started_monotonic_ns = now_monotonic_ns
                        quiet_started_time_ns = now_time_ns
                        quiet_start_index = sample_count
                        quiet_sample_count = 1
                    else:
                        quiet_sample_count += 1
                        if sample_gap is None:
                            raise AssertionError("noninitial quiet sample lacks a gap")
                        quiet_maximum_gap = max(quiet_maximum_gap, sample_gap)
                    quiet_elapsed = (
                        now_monotonic_ns - quiet_started_monotonic_ns
                    ) / 1e9
                sample = {
                    "sample_index": sample_count,
                    "utc": utc_from_ns(now_time_ns),
                    "time_ns": now_time_ns,
                    "monotonic_ns": now_monotonic_ns,
                    "sample_gap_seconds": sample_gap,
                    "gpu": gpu,
                    "cpu": guard.compact_cpu_evidence(cpu),
                    "sample_error": sample_error,
                    "probe_worker_duration_seconds": probe_duration_seconds,
                    "reset_reasons": reasons,
                    "quiet_elapsed_seconds": quiet_elapsed,
                }
                stream.write(
                    json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                )
                stream.flush()
                sample_count += 1
                if reasons:
                    dirty_sample_count += 1
                    signature = json.dumps(
                        {
                            "reasons": reasons,
                            "gpu": gpu,
                            "conflicts": cpu["conflicts"],
                            "errors": cpu["errors"],
                            "sample_error": sample_error,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if signature != last_event_signature:
                        report["events"].append(sample)
                    last_event_signature = signature
                else:
                    last_event_signature = None
                observer_signature = json.dumps(
                    compact_observers, separators=(",", ":"), sort_keys=True
                )
                if observers and observer_signature != last_observer_signature:
                    report["observer_events"].append(
                        {
                            "sample_index": sample["sample_index"],
                            "utc": sample["utc"],
                            "time_ns": sample["time_ns"],
                            "observers": compact_observers,
                            "gpu": gpu,
                        }
                    )
                last_observer_signature = observer_signature if observers else None
                previous_monotonic_ns = now_monotonic_ns

                if sample_error is not None:
                    report["status"] = "probe_error"
                    break

                if (
                    quiet_started_monotonic_ns is not None
                    and quiet_elapsed >= args.seconds
                ):
                    if (
                        quiet_started_time_ns is None
                        or quiet_start_index is None
                        or quiet_sample_count < 2
                    ):
                        raise AssertionError("invalid completed idle interval")
                    report.update(
                        {
                            "status": "passed",
                            "passed": True,
                            "finished_utc": sample["utc"],
                            "finished_time_ns": now_time_ns,
                            "elapsed_seconds": elapsed_seconds,
                            "quiet_interval": {
                                "started_utc": utc_from_ns(quiet_started_time_ns),
                                "started_time_ns": quiet_started_time_ns,
                                "finished_utc": sample["utc"],
                                "finished_time_ns": now_time_ns,
                                "duration_seconds": quiet_elapsed,
                                "sample_start_index": quiet_start_index,
                                "sample_end_index_inclusive": sample_count - 1,
                                "sample_count": quiet_sample_count,
                                "maximum_sample_gap_seconds": quiet_maximum_gap,
                                "all_samples_clean": True,
                            },
                        }
                    )
                    break
                if sample_count % 25 == 0:
                    report.update(
                        {
                            "last_checkpoint_utc": sample["utc"],
                            "last_checkpoint_time_ns": now_time_ns,
                            "elapsed_seconds": elapsed_seconds,
                            "current_quiet_elapsed_seconds": quiet_elapsed,
                            "sample_count": sample_count,
                            "dirty_sample_count": dirty_sample_count,
                            "observer_sample_count": observer_sample_count,
                            "observer_identity_count": len(observer_identities),
                            "event_count": len(report["events"]),
                            "observer_event_count": len(report["observer_events"]),
                        }
                    )
                    guard.write_json_atomic(args.output, report)
                next_sample += args.sample_interval_seconds
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_sample = time.monotonic()
    except Exception as error:
        report["status"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception as error:
                report["status"] = "error"
                report["passed"] = False
                report["shutdown_error"] = f"{type(error).__name__}: {error}"
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    if report["status"] != "passed":
        finished_time_ns = time.time_ns()
        report.update(
            {
                "finished_utc": utc_from_ns(finished_time_ns),
                "finished_time_ns": finished_time_ns,
                "elapsed_seconds": (time.monotonic_ns() - started_monotonic_ns) / 1e9,
            }
        )
    report.update(
        {
            "sample_count": sample_count,
            "dirty_sample_count": dirty_sample_count,
            "observer_sample_count": observer_sample_count,
            "observer_identity_count": len(observer_identities),
            "event_count": len(report["events"]),
            "observer_event_count": len(report["observer_events"]),
        }
    )
    report["sample_log"].update(
        {"bytes": sample_log.stat().st_size, "sha256": guard.sha256_file(sample_log)}
    )
    guard.write_json_atomic(args.output, report)
    if report["status"] != "passed":
        raise SystemExit(
            128 + stop_signal
            if report["status"] == "interrupted" and stop_signal is not None
            else 1
        )


if __name__ == "__main__":
    main()
