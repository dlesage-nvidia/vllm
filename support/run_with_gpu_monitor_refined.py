# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one benchmark cell with a watchdog and passive aggregate GPU telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pynvml

UTC = timezone.utc
SAMPLE_INTERVAL_SECONDS = 0.2
APPROVED_WATCHDOG_PAIRS = frozenset({(1200.0, 120.0), (3600.0, 120.0)})


def validate_watchdog_pair(
    timeout_seconds: float, grace_seconds: float
) -> tuple[float, float]:
    pair = (float(timeout_seconds), float(grace_seconds))
    if pair not in APPROVED_WATCHDOG_PAIRS:
        raise ValueError(f"watchdog pair is not approved: {pair}")
    return pair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, UTC).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def decode_nvml_string(value: object) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


class AggregateGpuSampler:
    """NVML aggregate counters only; this class never issues a process query."""

    def __init__(self, device_index: int) -> None:
        pynvml.nvmlInit()
        self.closed = False
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device = {
            "index": device_index,
            "uuid": decode_nvml_string(pynvml.nvmlDeviceGetUUID(self.handle)),
            "name": decode_nvml_string(pynvml.nvmlDeviceGetName(self.handle)),
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            pynvml.nvmlShutdown()

    def optional_scalar(
        self,
        function_name: str,
        *,
        argument_constant_name: str | None = None,
        divisor: float = 1.0,
    ) -> int | float | None:
        function = getattr(pynvml, function_name, None)
        if function is None:
            return None
        arguments: list[object] = [self.handle]
        if argument_constant_name is not None:
            constant = getattr(pynvml, argument_constant_name, None)
            if constant is None:
                return None
            arguments.append(constant)
        try:
            value = function(*arguments)
        except pynvml.NVMLError_NotSupported:
            return None
        return int(value) if divisor == 1.0 else float(value) / divisor

    def engine(self, function_name: str) -> dict[str, int] | None:
        function = getattr(pynvml, function_name, None)
        if function is None:
            return None
        try:
            utilization, sampling_period = function(self.handle)
        except pynvml.NVMLError_NotSupported:
            return None
        return {
            "utilization_percent": int(utilization),
            "sampling_period_us": int(sampling_period),
        }

    def sample(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        return {
            "memory_used_mib": int(memory.used / 1024**2),
            "memory_total_mib": int(memory.total / 1024**2),
            "utilization_gpu_percent": int(utilization.gpu),
            "utilization_memory_percent": int(utilization.memory),
            "engine_utilization": {
                "decoder": self.engine("nvmlDeviceGetDecoderUtilization"),
                "encoder": self.engine("nvmlDeviceGetEncoderUtilization"),
                "jpeg": self.engine("nvmlDeviceGetJpgUtilization"),
                "ofa": self.engine("nvmlDeviceGetOfaUtilization"),
            },
            "operating_telemetry": {
                "sm_clock_mhz": self.optional_scalar(
                    "nvmlDeviceGetClockInfo",
                    argument_constant_name="NVML_CLOCK_SM",
                ),
                "memory_clock_mhz": self.optional_scalar(
                    "nvmlDeviceGetClockInfo",
                    argument_constant_name="NVML_CLOCK_MEM",
                ),
                "temperature_c": self.optional_scalar(
                    "nvmlDeviceGetTemperature",
                    argument_constant_name="NVML_TEMPERATURE_GPU",
                ),
                "power_w": self.optional_scalar(
                    "nvmlDeviceGetPowerUsage", divisor=1000.0
                ),
                "performance_state": self.optional_scalar(
                    "nvmlDeviceGetPerformanceState"
                ),
            },
            "query_duration_seconds": (time.monotonic_ns() - started_ns) / 1e9,
        }


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_group_exit(process_group: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while process_group_exists(process_group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def terminate_process_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float, reason: str
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    process_group = process.pid
    for signum, timeout in (
        (signal.SIGINT, grace_seconds),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 5.0),
    ):
        if not process_group_exists(process_group):
            break
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            break
        actions.append({"signal": signal.Signals(signum).name, "signal_sent": True})
        if wait_for_group_exit(process_group, timeout):
            break
    try:
        returncode = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        returncode = None
    return {
        "reason": reason,
        "signal_actions": actions,
        "returncode": returncode,
        "process_group_alive": process_group_exists(process_group),
        "completed": returncode is not None and not process_group_exists(process_group),
    }


def collect_sample(
    sampler: AggregateGpuSampler | None,
    *,
    index: int,
    previous_monotonic_ns: int | None,
) -> dict[str, Any]:
    error = None
    gpu = None
    if sampler is not None:
        try:
            gpu = sampler.sample()
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    time_ns = time.time_ns()
    monotonic_ns = time.monotonic_ns()
    return {
        "sample_index": index,
        "utc": utc_from_ns(time_ns),
        "time_ns": time_ns,
        "monotonic_ns": monotonic_ns,
        "sample_gap_seconds": (
            None
            if previous_monotonic_ns is None
            else (monotonic_ns - previous_monotonic_ns) / 1e9
        ),
        "memory_used_mib": None if gpu is None else gpu["memory_used_mib"],
        "memory_total_mib": None if gpu is None else gpu["memory_total_mib"],
        "utilization_percent": (
            None if gpu is None else gpu["utilization_gpu_percent"]
        ),
        "memory_utilization_percent": (
            None if gpu is None else gpu["utilization_memory_percent"]
        ),
        "engine_utilization": None if gpu is None else gpu["engine_utilization"],
        "operating_telemetry": None if gpu is None else gpu["operating_telemetry"],
        "gpu_query_duration_seconds": (
            None if gpu is None else gpu["query_duration_seconds"]
        ),
        "sample_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--timeout-grace-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise ValueError("missing child command")
    validate_watchdog_pair(args.timeout_seconds, args.timeout_grace_seconds)
    args.output = args.output.resolve()
    sample_log = args.output.with_name(args.output.stem + ".samples.jsonl")
    if args.output.exists() or sample_log.exists():
        raise FileExistsError("refusing to overwrite monitor evidence")

    script = Path(__file__).resolve()
    started_time_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    report: dict[str, Any] = {
        "schema": "pynv-passive-aggregate-gpu-monitor-v1",
        "status": "running",
        "command": args.command,
        "started_utc": utc_from_ns(started_time_ns),
        "started_time_ns": started_time_ns,
        "process": {
            "pid": os.getpid(),
            "uid": os.getuid(),
            "effective_uid": os.geteuid(),
            "argv": sys.argv,
            "executable": sys.executable,
            "script_path": str(script),
            "script_sha256": sha256_file(script),
        },
        "configuration": {
            "device_index": args.device_index,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "telemetry_scope": "aggregate GPU counters and clocks only",
            "telemetry_authoritative": False,
            "exclusivity_authority": "gpulock lease outside this wrapper",
            "process_queries": False,
            "cpu_queries": False,
        },
        "timeout_seconds": args.timeout_seconds,
        "timeout_grace_seconds": args.timeout_grace_seconds,
        "telemetry_events": [],
        "lifecycle_errors": [],
        "samples": [],
        "sample_log": {"path": str(sample_log), "format": "JSON Lines"},
    }
    sample_log.open("x").close()
    write_json_atomic(args.output, report)

    termination_signal: int | None = None

    def request_termination(signum: int, unused_frame: object) -> None:
        nonlocal termination_signal
        if termination_signal is None:
            termination_signal = signum

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, request_termination) for signum in handled_signals
    }

    sampler: AggregateGpuSampler | None = None
    cleanup: dict[str, Any] | None = None
    timed_out = False
    process = subprocess.Popen(args.command, start_new_session=True)
    report["command_pid"] = process.pid
    try:
        try:
            sampler = AggregateGpuSampler(args.device_index)
            report["device"] = sampler.device
        except Exception as error:
            report["telemetry_events"].append(
                {
                    "phase": "initialization",
                    "error": f"{type(error).__name__}: {error}",
                    "non_authoritative": True,
                }
            )
        next_sample = time.monotonic()
        previous_monotonic_ns: int | None = None
        with sample_log.open("a") as stream:
            while process.poll() is None:
                if termination_signal is not None:
                    cleanup = terminate_process_group(
                        process,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="wrapper_signal",
                    )
                    break
                if (
                    time.monotonic_ns() - started_monotonic_ns
                ) / 1e9 > args.timeout_seconds:
                    timed_out = True
                    cleanup = terminate_process_group(
                        process,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="cell_watchdog_timeout",
                    )
                    break
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                sample = collect_sample(
                    sampler,
                    index=len(report["samples"]),
                    previous_monotonic_ns=previous_monotonic_ns,
                )
                report["samples"].append(sample)
                stream.write(
                    json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                )
                stream.flush()
                previous_monotonic_ns = sample["monotonic_ns"]
                if sample["sample_error"] is not None:
                    report["telemetry_events"].append(
                        {
                            "sample_index": sample["sample_index"],
                            "error": sample["sample_error"],
                            "non_authoritative": True,
                        }
                    )
                next_sample = time.monotonic() + SAMPLE_INTERVAL_SECONDS

            if cleanup is None:
                returncode = process.wait()
                if process_group_exists(process.pid):
                    cleanup = terminate_process_group(
                        process,
                        grace_seconds=0.5,
                        reason="process_group_survived_child_exit",
                    )
                    report["lifecycle_errors"].append(
                        "process_group_survived_child_exit"
                    )
            else:
                returncode = process.poll()

            for ordinal in range(2):
                sample = collect_sample(
                    sampler,
                    index=len(report["samples"]),
                    previous_monotonic_ns=previous_monotonic_ns,
                )
                sample["post_exit_telemetry"] = True
                sample["post_exit_ordinal"] = ordinal
                report["samples"].append(sample)
                stream.write(
                    json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                )
                stream.flush()
                previous_monotonic_ns = sample["monotonic_ns"]
                if ordinal == 0:
                    time.sleep(SAMPLE_INTERVAL_SECONDS)
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except Exception as error:
                report["telemetry_events"].append(
                    {
                        "phase": "shutdown",
                        "error": f"{type(error).__name__}: {error}",
                        "non_authoritative": True,
                    }
                )
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    finished_time_ns = time.time_ns()
    successful_lifecycle = not report["lifecycle_errors"] and (
        cleanup is None or cleanup.get("completed") is True
    )
    if termination_signal is not None:
        status = "interrupted"
    elif timed_out:
        status = "timed_out"
    elif not successful_lifecycle:
        status = "lifecycle_error"
    elif returncode == 0:
        status = "passed"
    else:
        status = "child_failed"
    report.update(
        {
            "status": status,
            "finished_utc": utc_from_ns(finished_time_ns),
            "finished_time_ns": finished_time_ns,
            "returncode": returncode,
            "timed_out": timed_out,
            "termination_signal": (
                None
                if termination_signal is None
                else signal.Signals(termination_signal).name
            ),
            "cleanup": cleanup,
            "sample_count": len(report["samples"]),
            "peak_memory_used_mib": max(
                (
                    sample["memory_used_mib"]
                    for sample in report["samples"]
                    if sample["memory_used_mib"] is not None
                ),
                default=None,
            ),
            "peak_utilization_percent": max(
                (
                    sample["utilization_percent"]
                    for sample in report["samples"]
                    if sample["utilization_percent"] is not None
                ),
                default=None,
            ),
        }
    )
    report["sample_log"].update(
        {"bytes": sample_log.stat().st_size, "sha256": sha256_file(sample_log)}
    )
    write_json_atomic(args.output, report)
    if termination_signal is not None:
        raise SystemExit(128 + termination_signal)
    if timed_out:
        raise SystemExit(124)
    if not successful_lifecycle:
        raise SystemExit(98)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
