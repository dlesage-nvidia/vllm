# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one benchmark cell with structural conflict detection and direct NVML."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pynv_gpu_guard as guard

UTC = timezone.utc

EXPECTED_HELPER_SHA256 = (
    "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
)
SAMPLE_INTERVAL_SECONDS = 0.2
MAXIMUM_SAMPLE_GAP_SECONDS = 1.0
IDLE_MEMORY_CEILING_MIB = 1024
PR_SET_CHILD_SUBREAPER = 36
APPROVED_WATCHDOG_PAIRS = frozenset({(1200.0, 120.0), (3600.0, 120.0)})


def validate_watchdog_pair(
    timeout_seconds: float, grace_seconds: float
) -> tuple[float, float]:
    pair = (float(timeout_seconds), float(grace_seconds))
    if pair not in APPROVED_WATCHDOG_PAIRS:
        approved = ", ".join(
            f"{timeout:g}/{grace:g}"
            for timeout, grace in sorted(APPROVED_WATCHDOG_PAIRS)
        )
        raise ValueError(
            f"watchdog pair {pair[0]:g}/{pair[1]:g} is not approved; {approved}"
        )
    return pair


def utc_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1e9, UTC).isoformat()


def enable_child_subreaper() -> dict[str, Any]:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return {
        "enabled": True,
        "prctl_option": "PR_SET_CHILD_SUBREAPER",
        "controller_pid": os.getpid(),
        "controller_start_time_ticks": guard.process_start_time_ticks(os.getpid()),
    }


def flatten_sample(
    *,
    index: int,
    time_ns: int,
    monotonic_ns: int,
    gap: float | None,
    gpu: Mapping[str, Any] | None,
    cpu: Mapping[str, Any],
    external_gpu_processes: list[dict[str, Any]],
    sample_error: str | None,
    probe_worker_duration_seconds: float | None,
    monitor_errors: list[str],
) -> dict[str, Any]:
    compute_apps = []
    graphics_apps = []
    mps_compute_apps = []
    mps_compute_process_query = None
    mps_daemon_present = None
    if gpu is not None:
        compute_apps = [
            {
                "pid": int(item["pid"]),
                "process_name": str(item["process_name"]),
                "used_memory_mib": (
                    -1
                    if item.get("used_memory_mib") is None
                    else int(item["used_memory_mib"])
                ),
            }
            for item in gpu["compute_processes"]
        ]
        graphics_apps = [
            {
                "pid": int(item["pid"]),
                "process_name": str(item["process_name"]),
                "used_memory_mib": (
                    -1
                    if item.get("used_memory_mib") is None
                    else int(item["used_memory_mib"])
                ),
            }
            for item in gpu["graphics_processes"]
        ]
        mps_compute_apps = [
            {
                "pid": int(item["pid"]),
                "process_name": str(item["process_name"]),
                "used_memory_mib": (
                    -1
                    if item.get("used_memory_mib") is None
                    else int(item["used_memory_mib"])
                ),
            }
            for item in gpu["mps_compute_processes"]
        ]
        mps_compute_process_query = gpu["mps_compute_process_query"]
        mps_daemon_present = bool(gpu["mps_daemon_present"])
    return {
        "sample_index": index,
        "utc": utc_from_ns(time_ns),
        "time_ns": time_ns,
        "monotonic_ns": monotonic_ns,
        "sample_gap_seconds": gap,
        "memory_used_mib": None if gpu is None else int(gpu["memory_used_mib"]),
        "utilization_percent": (
            None if gpu is None else int(gpu["utilization_gpu_percent"])
        ),
        "memory_utilization_percent": (
            None if gpu is None else int(gpu["utilization_memory_percent"])
        ),
        "compute_apps": compute_apps,
        "graphics_apps": graphics_apps,
        "mps_compute_apps": mps_compute_apps,
        "mps_compute_process_query": mps_compute_process_query,
        "mps_daemon_present": mps_daemon_present,
        "engine_utilization": (
            None
            if gpu is None
            else {name: gpu.get(name) for name in ("decoder", "encoder", "jpeg", "ofa")}
        ),
        "operating_telemetry": (
            None if gpu is None else gpu.get("operating_telemetry")
        ),
        "gpu_query_duration_seconds": (
            None if gpu is None else gpu["query_duration_seconds"]
        ),
        "cpu_conflicts": cpu["conflicts"],
        "host_load": cpu.get("host_load"),
        "cpu_observers": [guard.compact_observer(item) for item in cpu["observers"]],
        "owned_processes": guard.compact_cpu_evidence(cpu)["owned_processes"],
        "process_inspection_errors": cpu["errors"],
        "external_gpu_processes": external_gpu_processes,
        "sample_error": sample_error,
        "probe_worker_duration_seconds": probe_worker_duration_seconds,
        "monitor_errors": monitor_errors,
    }


def process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def update_owned_registry(
    registry: dict[tuple[int, int], dict[str, Any]],
    processes: Sequence[Mapping[str, Any]],
) -> None:
    for process in processes:
        identity = (int(process["pid"]), int(process["start_time_ticks"]))
        evidence = {
            "pid": identity[0],
            "start_time_ticks": identity[1],
            "parent_pid": int(process["parent_pid"]),
            "process_group": int(process["process_group"]),
            "comm": str(process["comm"]),
            "ownership_reason": str(process["ownership_reason"]),
        }
        prior = registry.get(identity)
        if prior is not None and int(prior["process_group"]) != int(
            evidence["process_group"]
        ):
            raise RuntimeError(
                f"owned PID {identity[0]} changed process group from "
                f"{prior['process_group']} to {evidence['process_group']}"
            )
        registry[identity] = evidence


def surviving_owned_processes(
    registry: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    survivors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for (pid, expected_start), evidence in sorted(registry.items()):
        try:
            current_start = guard.process_start_time_ticks(pid)
            current_group = guard.process_group_id(pid)
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, RuntimeError, ValueError) as error:
            errors.append(
                {
                    "pid": pid,
                    "operation": "verify_owned_survivor",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if current_start != expected_start:
            errors.append(
                {
                    "pid": pid,
                    "operation": "verify_owned_survivor",
                    "error": "PID identity changed; refusing to signal reused PID",
                }
            )
            continue
        recorded_group = int(evidence["process_group"])
        if current_group != recorded_group:
            errors.append(
                {
                    "pid": pid,
                    "operation": "verify_owned_survivor_group",
                    "error": (
                        f"process group changed from {recorded_group} to "
                        f"{current_group}; refusing to signal"
                    ),
                }
            )
            continue
        survivors.append(dict(evidence))
    return survivors, errors


def reap_adopted_owned_children(
    registry: Mapping[tuple[int, int], Mapping[str, Any]], *, root_pid: int | None
) -> list[dict[str, Any]]:
    reaped: list[dict[str, Any]] = []
    for pid, expected_start in sorted(registry):
        if root_pid is not None and pid == root_pid:
            continue
        try:
            waited_pid, wait_status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue
        if waited_pid == pid:
            reaped.append(
                {
                    "pid": pid,
                    "start_time_ticks": expected_start,
                    "wait_status": wait_status,
                }
            )
    return reaped


def capture_owned_tree(
    *,
    root_identity: tuple[int, int] | None,
    controller_identity: tuple[int, int],
    registry: dict[tuple[int, int], dict[str, Any]],
    excluded_identities: Mapping[int, int],
) -> list[dict[str, Any]]:
    scan = guard.scan_cpu_processes(
        excluded_identities=excluded_identities,
        allowed_subtree_identity=root_identity,
        allowed_process_group_identity=root_identity,
        allowed_identities={pid: start for pid, start in registry},
        allowed_subreaper_identity=controller_identity,
    )
    update_owned_registry(registry, scan["owned_processes"])
    return list(scan["errors"])


def current_process_group_members(
    process_group: int,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    members: list[dict[str, int]] = []
    errors: list[dict[str, Any]] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as error:
        return [], [
            {
                "pid": None,
                "operation": "list_proc_for_group_verification",
                "error": f"{type(error).__name__}: {error}",
            }
        ]
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            current_group = guard.process_group_id(pid)
            start_ticks = guard.process_start_time_ticks(pid)
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, RuntimeError, ValueError) as error:
            errors.append(
                {
                    "pid": pid,
                    "operation": "inspect_process_group_member",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if current_group == process_group:
            members.append({"pid": pid, "start_time_ticks": start_ticks})
    return sorted(members, key=lambda item: item["pid"]), errors


def signal_verified_owned_groups(
    survivors: Sequence[Mapping[str, Any]],
    owned_registry: Mapping[tuple[int, int], Mapping[str, Any]],
    signum: int,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    current_controller_group = os.getpgrp()
    by_group: dict[int, list[dict[str, int]]] = {}
    for process in survivors:
        group = int(process["process_group"])
        if group <= 1 or group == current_controller_group:
            raise RuntimeError(f"refusing to signal unsafe owned process group {group}")
        by_group.setdefault(group, []).append(
            {
                "pid": int(process["pid"]),
                "start_time_ticks": int(process["start_time_ticks"]),
            }
        )
    for group, identities in sorted(by_group.items()):
        current_members, inspection_errors = current_process_group_members(group)
        if inspection_errors:
            raise RuntimeError(
                f"could not verify all members of owned process group {group}: "
                f"{inspection_errors}"
            )
        unowned_members = [
            member
            for member in current_members
            if (member["pid"], member["start_time_ticks"]) not in owned_registry
        ]
        if unowned_members:
            raise RuntimeError(
                f"refusing to signal process group {group} with unowned members: "
                f"{unowned_members}"
            )
        try:
            os.killpg(group, signum)
            sent = True
        except ProcessLookupError:
            sent = False
        actions.append(
            {
                "process_group": group,
                "signal": signal.Signals(signum).name,
                "signal_sent": sent,
                "identity_verified_members": identities,
                "all_current_group_members": current_members,
            }
        )
    return actions


def terminate_owned_workload(
    process: subprocess.Popen[bytes] | None,
    *,
    root_identity: tuple[int, int] | None,
    controller_identity: tuple[int, int],
    owned_registry: dict[tuple[int, int], dict[str, Any]],
    excluded_identities: Mapping[int, int],
    grace_seconds: float,
    reason: str,
) -> dict[str, Any]:
    process_group = None if root_identity is None else root_identity[0]
    root_pid = None if process is None else process.pid
    cleanup: dict[str, Any] = {
        "reason": reason,
        "process_group": process_group,
        "grace_seconds": grace_seconds,
        "capture_errors": [],
        "identity_errors": [],
        "signal_actions": [],
        "reaped_adopted_children": [],
    }
    cleanup["capture_errors"].extend(
        capture_owned_tree(
            root_identity=root_identity,
            controller_identity=controller_identity,
            registry=owned_registry,
            excluded_identities=excluded_identities,
        )
    )
    survivors, identity_errors = surviving_owned_processes(owned_registry)
    cleanup["identity_errors"].extend(identity_errors)
    cleanup["owned_registry_before_signal"] = list(owned_registry.values())
    signaled_groups: set[int] = set()

    def signal_new_groups(signum: int) -> None:
        pending = [
            item
            for item in survivors
            if int(item["process_group"]) not in signaled_groups
        ]
        if not pending:
            return
        cleanup["signal_actions"].extend(
            signal_verified_owned_groups(pending, owned_registry, signum)
        )
        signaled_groups.update(int(item["process_group"]) for item in pending)

    signal_new_groups(signal.SIGINT)
    deadline = time.monotonic() + grace_seconds
    original_groups = {int(item["process_group"]) for item in owned_registry.values()}
    while time.monotonic() < deadline:
        if process is not None:
            process.poll()
        cleanup["reaped_adopted_children"].extend(
            reap_adopted_owned_children(owned_registry, root_pid=root_pid)
        )
        cleanup["capture_errors"].extend(
            capture_owned_tree(
                root_identity=root_identity,
                controller_identity=controller_identity,
                registry=owned_registry,
                excluded_identities=excluded_identities,
            )
        )
        original_groups.update(
            int(item["process_group"]) for item in owned_registry.values()
        )
        survivors, identity_errors = surviving_owned_processes(owned_registry)
        cleanup["identity_errors"].extend(identity_errors)
        signal_new_groups(signal.SIGINT)
        groups_alive = sorted(
            group for group in original_groups if process_group_exists(group)
        )
        if not survivors and not groups_alive:
            break
        time.sleep(0.1)
    survivors, identity_errors = surviving_owned_processes(owned_registry)
    cleanup["identity_errors"].extend(identity_errors)
    if survivors:
        cleanup["signal_actions"].extend(
            signal_verified_owned_groups(survivors, owned_registry, signal.SIGKILL)
        )
    final_deadline = time.monotonic() + 10.0
    while time.monotonic() < final_deadline:
        if process is not None:
            process.poll()
        cleanup["reaped_adopted_children"].extend(
            reap_adopted_owned_children(owned_registry, root_pid=root_pid)
        )
        survivors, identity_errors = surviving_owned_processes(owned_registry)
        cleanup["identity_errors"].extend(identity_errors)
        groups_alive = sorted(
            group for group in original_groups if process_group_exists(group)
        )
        if not survivors and not groups_alive:
            break
        time.sleep(0.1)
    if process is not None:
        process.poll()
    cleanup["reaped_adopted_children"].extend(
        reap_adopted_owned_children(owned_registry, root_pid=root_pid)
    )
    survivors, identity_errors = surviving_owned_processes(owned_registry)
    cleanup["identity_errors"].extend(identity_errors)
    groups_alive = sorted(
        group for group in original_groups if process_group_exists(group)
    )
    cleanup["returncode"] = None if process is None else process.poll()
    cleanup["surviving_owned_processes"] = survivors
    cleanup["surviving_process_groups"] = groups_alive
    cleanup["completed"] = (
        (process is None or process.poll() is not None)
        and not survivors
        and not groups_alive
        and not cleanup["capture_errors"]
        and not cleanup["identity_errors"]
    )
    return cleanup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--timeout-grace-seconds", type=float, default=120.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--conflicting-controller-root",
        action="append",
        default=[],
        help="Absolute cwd in which exact './screen.sh' is a conflicting controller",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    controller_roots = guard.normalize_conflicting_controller_roots(
        args.conflicting_controller_root
    )
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing command")
    validate_watchdog_pair(args.timeout_seconds, args.timeout_grace_seconds)
    if args.device_index != 0:
        raise ValueError("publication monitor must remain on physical device index 0")
    args.output = args.output.resolve()
    sample_log = args.output.with_name(args.output.stem + ".samples.jsonl")
    preexisting = [path for path in (args.output, sample_log) if path.exists()]
    if preexisting:
        raise FileExistsError(
            "refusing to overwrite monitor evidence: "
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
    started_time_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    subreaper = enable_child_subreaper()
    controller_identity = (
        int(subreaper["controller_pid"]),
        int(subreaper["controller_start_time_ticks"]),
    )
    termination_signal: int | None = None

    def request_termination(signum: int, unused_frame: Any) -> None:
        nonlocal termination_signal
        if termination_signal is None:
            termination_signal = signum

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers = {
        signum: signal.signal(signum, request_termination) for signum in handled_signals
    }
    report: dict[str, Any] = {
        "status": "running",
        "command": command,
        "started_utc": utc_from_ns(started_time_ns),
        "started_time_ns": started_time_ns,
        "controller_pid": os.getpid(),
        "process": {
            "uid": os.getuid(),
            "effective_uid": os.geteuid(),
            "argv": sys.argv,
            "executable": sys.executable,
            "script_path": str(script),
            "script_sha256": guard.sha256_file(script),
        },
        "guard_helper": {"path": str(helper), "sha256": helper_sha256},
        "configuration": {
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "maximum_sample_gap_seconds": MAXIMUM_SAMPLE_GAP_SECONDS,
            "initial_idle_memory_ceiling_mib": IDLE_MEMORY_CEILING_MIB,
            "device_index": args.device_index,
            "conflicting_controller_roots": controller_roots,
            "telemetry": "direct NVML; no external telemetry commands or pgrep",
            "workload_ownership": (
                "new process session/group plus PID/start_ticks ancestry"
            ),
        },
        "timeout_seconds": args.timeout_seconds,
        "timeout_grace_seconds": args.timeout_grace_seconds,
        "excluded_process_lineage_identities": [
            {"pid": pid, "start_time_ticks": start}
            for pid, start in sorted(lineage.items())
        ],
        "sample_log": {"path": str(sample_log), "format": "JSON Lines"},
        "foreign_events": [],
        "observer_events": [],
        "monitor_errors": [],
        "handled_signals": [signal.Signals(item).name for item in handled_signals],
        "child_subreaper": subreaper,
    }
    sample_log.open("x").close()
    guard.write_json_atomic(args.output, report)

    probe: guard.ProbeWorker | None = None
    process: subprocess.Popen[bytes] | None = None
    process_group: int | None = None
    workload_identity: tuple[int, int] | None = None
    owned_registry: dict[tuple[int, int], dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    timed_out = False
    contaminated = False
    fatal_monitor_error = False
    cleanup: dict[str, Any] | None = None
    observer_identities: set[tuple[int, int, str]] = set()
    previous_monotonic_ns: int | None = None
    monitor_excluded_identities = dict(lineage)
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
            "excluded_from_subreaper_workload_ownership": True,
        }
        probe_pid = int(probe.worker_identity["pid"])
        probe_start = int(probe.worker_identity["start_time_ticks"])
        prior_excluded_start = monitor_excluded_identities.get(probe_pid)
        if prior_excluded_start is not None and prior_excluded_start != probe_start:
            raise RuntimeError(
                "probe worker PID collides with a different excluded identity"
            )
        monitor_excluded_identities[probe_pid] = probe_start
        report["probe_worker_excluded_identity"] = {
            "pid": probe_pid,
            "start_time_ticks": probe_start,
        }
        initial_probe = probe.sample(excluded_identities=monitor_excluded_identities)
        initial_gpu = initial_probe["gpu"]
        initial_cpu = initial_probe["cpu"]
        initial_reasons = guard.gpu_is_idle(
            initial_gpu, memory_ceiling_mib=IDLE_MEMORY_CEILING_MIB
        )
        if initial_cpu["errors"]:
            initial_reasons.append("process_inspection_error")
        if initial_cpu["conflicts"]:
            initial_reasons.append("conflicting_cpu_process")
        initial_time_ns = time.time_ns()
        initial_monotonic_ns = time.monotonic_ns()
        initial_external_gpu = initial_probe["external_gpu_processes"]
        initial_sample = flatten_sample(
            index=0,
            time_ns=initial_time_ns,
            monotonic_ns=initial_monotonic_ns,
            gap=None,
            gpu=initial_gpu,
            cpu=initial_cpu,
            external_gpu_processes=initial_external_gpu,
            sample_error=None,
            probe_worker_duration_seconds=initial_probe["worker_duration_seconds"],
            monitor_errors=initial_reasons,
        )
        samples.append(initial_sample)
        with sample_log.open("a") as stream:
            stream.write(
                json.dumps(initial_sample, separators=(",", ":"), sort_keys=True) + "\n"
            )
            stream.flush()
        previous_monotonic_ns = initial_monotonic_ns
        if initial_reasons:
            contaminated = bool(initial_external_gpu or initial_cpu["conflicts"])
            fatal_monitor_error = not contaminated
            if contaminated:
                report["foreign_events"].append(
                    {
                        "sample_index": 0,
                        "utc": initial_sample["utc"],
                        "apps": [
                            *initial_external_gpu,
                            *(
                                {
                                    **item,
                                    "process_name": " ".join(item["argv"]),
                                    "used_memory_mib": -1,
                                }
                                for item in initial_cpu["conflicts"]
                            ),
                        ],
                        "reasons": initial_reasons,
                    }
                )
            else:
                report["monitor_errors"].append(
                    {"sample_index": 0, "reasons": initial_reasons}
                )
        elif termination_signal is not None:
            report["termination_signal"] = signal.Signals(termination_signal).name
            returncode = None
        else:
            process = subprocess.Popen(command, start_new_session=True)
            workload_identity = (
                process.pid,
                guard.process_start_time_ticks(process.pid),
            )
            process_group = os.getpgid(process.pid)
            owned_registry[workload_identity] = {
                "pid": workload_identity[0],
                "start_time_ticks": workload_identity[1],
                "parent_pid": os.getpid(),
                "process_group": process_group,
                "comm": Path(command[0]).name,
                "ownership_reason": "launched_workload_root",
            }
            if process_group != process.pid:
                raise RuntimeError(
                    f"workload process group mismatch: {process_group} != {process.pid}"
                )
            report["command_pid"] = process.pid
            report["workload_identity"] = {
                "pid": workload_identity[0],
                "start_time_ticks": workload_identity[1],
                "process_group": process_group,
            }
            next_sample = time.monotonic() + SAMPLE_INTERVAL_SECONDS
            last_observer_signature: str | None = None
            while process.poll() is None:
                if termination_signal is not None:
                    report["termination_signal"] = signal.Signals(
                        termination_signal
                    ).name
                    cleanup = terminate_owned_workload(
                        process,
                        root_identity=workload_identity,
                        controller_identity=controller_identity,
                        owned_registry=owned_registry,
                        excluded_identities=monitor_excluded_identities,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="monitor_signal_termination",
                    )
                    break
                elapsed = (time.monotonic_ns() - started_monotonic_ns) / 1e9
                if elapsed > args.timeout_seconds:
                    timed_out = True
                    cleanup = terminate_owned_workload(
                        process,
                        root_identity=workload_identity,
                        controller_identity=controller_identity,
                        owned_registry=owned_registry,
                        excluded_identities=monitor_excluded_identities,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="cell_watchdog_timeout",
                    )
                    break
                try:
                    probe_result = probe.sample(
                        excluded_identities=monitor_excluded_identities,
                        allowed_subtree_identity=workload_identity,
                        allowed_process_group_identity=workload_identity,
                        allowed_subreaper_identity=controller_identity,
                        allowed_identities={
                            pid: start for pid, start in owned_registry
                        },
                    )
                    gpu = probe_result["gpu"]
                    cpu = probe_result["cpu"]
                    external_gpu = probe_result["external_gpu_processes"]
                    probe_duration_seconds = probe_result["worker_duration_seconds"]
                    sample_error = None
                    update_owned_registry(owned_registry, cpu["owned_processes"])
                except Exception as error:
                    gpu = None
                    cpu = {
                        "conflicts": [],
                        "observers": [],
                        "owned_processes": [],
                        "errors": [],
                    }
                    external_gpu = []
                    probe_duration_seconds = None
                    sample_error = f"{type(error).__name__}: {error}"
                now_monotonic_ns = time.monotonic_ns()
                now_time_ns = time.time_ns()
                gap = (now_monotonic_ns - previous_monotonic_ns) / 1e9
                monitor_errors: list[str] = []
                if sample_error is not None:
                    monitor_errors.append("sample_error")
                if gap > MAXIMUM_SAMPLE_GAP_SECONDS:
                    monitor_errors.append("sampling_gap_exceeded")
                if cpu["errors"]:
                    monitor_errors.append("process_inspection_error")
                apps = [
                    *external_gpu,
                    *(
                        {
                            **item,
                            "process_name": " ".join(item["argv"]),
                            "used_memory_mib": -1,
                        }
                        for item in cpu["conflicts"]
                    ),
                ]
                sample = flatten_sample(
                    index=len(samples),
                    time_ns=now_time_ns,
                    monotonic_ns=now_monotonic_ns,
                    gap=gap,
                    gpu=gpu,
                    cpu=cpu,
                    external_gpu_processes=external_gpu,
                    sample_error=sample_error,
                    probe_worker_duration_seconds=probe_duration_seconds,
                    monitor_errors=monitor_errors,
                )
                samples.append(sample)
                with sample_log.open("a") as stream:
                    stream.write(
                        json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                    )
                    stream.flush()
                observers = cpu["observers"]
                compact_observers = [guard.compact_observer(item) for item in observers]
                for observer in observers:
                    observer_identities.add(
                        (
                            int(observer["pid"]),
                            int(observer["start_time_ticks"]),
                            str(observer["classification_reason"]),
                        )
                    )
                observer_signature = json.dumps(
                    compact_observers, separators=(",", ":"), sort_keys=True
                )
                if observers and observer_signature != last_observer_signature:
                    report["observer_events"].append(
                        {
                            "sample_index": sample["sample_index"],
                            "utc": sample["utc"],
                            "observers": compact_observers,
                            "gpu_processes_still_checked": True,
                        }
                    )
                last_observer_signature = observer_signature if observers else None
                if apps:
                    contaminated = True
                    report["foreign_events"].append(
                        {
                            "sample_index": sample["sample_index"],
                            "monotonic_ns": sample["monotonic_ns"],
                            "utc": sample["utc"],
                            "apps": apps,
                        }
                    )
                    cleanup = terminate_owned_workload(
                        process,
                        root_identity=workload_identity,
                        controller_identity=controller_identity,
                        owned_registry=owned_registry,
                        excluded_identities=monitor_excluded_identities,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="foreign_workload_detected",
                    )
                    break
                if monitor_errors:
                    fatal_monitor_error = True
                    report["monitor_errors"].append(
                        {
                            "sample_index": sample["sample_index"],
                            "utc": sample["utc"],
                            "reasons": monitor_errors,
                            "sample_error": sample_error,
                            "process_inspection_errors": cpu["errors"],
                        }
                    )
                    cleanup = terminate_owned_workload(
                        process,
                        root_identity=workload_identity,
                        controller_identity=controller_identity,
                        owned_registry=owned_registry,
                        excluded_identities=monitor_excluded_identities,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="monitor_failure",
                    )
                    break
                previous_monotonic_ns = now_monotonic_ns
                next_sample += SAMPLE_INTERVAL_SECONDS
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_sample = time.monotonic()

            if process.poll() is None and cleanup is None:
                cleanup = terminate_owned_workload(
                    process,
                    root_identity=workload_identity,
                    controller_identity=controller_identity,
                    owned_registry=owned_registry,
                    excluded_identities=monitor_excluded_identities,
                    grace_seconds=args.timeout_grace_seconds,
                    reason="unexpected_monitor_loop_exit",
                )
                fatal_monitor_error = True
            elif process.poll() is not None and cleanup is None:
                natural_capture_errors = capture_owned_tree(
                    root_identity=workload_identity,
                    controller_identity=controller_identity,
                    registry=owned_registry,
                    excluded_identities=monitor_excluded_identities,
                )
                natural_survivors, natural_identity_errors = surviving_owned_processes(
                    owned_registry
                )
                natural_groups = sorted(
                    {
                        int(item["process_group"])
                        for item in owned_registry.values()
                        if process_group_exists(int(item["process_group"]))
                    }
                )
                if (
                    natural_capture_errors
                    or natural_identity_errors
                    or natural_survivors
                    or natural_groups
                ):
                    fatal_monitor_error = True
                    report["monitor_errors"].append(
                        {
                            "reason": "owned_process_survived_natural_harness_exit",
                            "capture_errors": natural_capture_errors,
                            "identity_errors": natural_identity_errors,
                            "survivors": natural_survivors,
                            "process_groups_alive": natural_groups,
                        }
                    )
                    if natural_survivors:
                        cleanup = terminate_owned_workload(
                            process,
                            root_identity=workload_identity,
                            controller_identity=controller_identity,
                            owned_registry=owned_registry,
                            excluded_identities=monitor_excluded_identities,
                            grace_seconds=args.timeout_grace_seconds,
                            reason="orphan_after_natural_harness_exit",
                        )
            if cleanup is not None and cleanup.get("completed") is not True:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {
                        "reason": "owned_process_cleanup_incomplete",
                        "cleanup": cleanup,
                    }
                )
            returncode = process.poll()
            if returncode is None:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {"reason": "workload_process_failed_to_exit_after_cleanup"}
                )
            next_post_exit_sample = time.monotonic()
            for post_exit_ordinal in range(2):
                delay = next_post_exit_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                try:
                    post_probe = probe.sample(
                        excluded_identities=monitor_excluded_identities,
                        allowed_subtree_identity=workload_identity,
                        allowed_process_group_identity=workload_identity,
                        allowed_subreaper_identity=controller_identity,
                        allowed_identities={
                            pid: start for pid, start in owned_registry
                        },
                    )
                    post_gpu = post_probe["gpu"]
                    post_cpu = post_probe["cpu"]
                    update_owned_registry(owned_registry, post_cpu["owned_processes"])
                    post_external_gpu = post_probe["external_gpu_processes"]
                    post_probe_duration = post_probe["worker_duration_seconds"]
                    post_error = None
                except Exception as error:
                    post_gpu = None
                    post_cpu = {
                        "conflicts": [],
                        "observers": [],
                        "owned_processes": [],
                        "errors": [],
                    }
                    post_external_gpu = []
                    post_probe_duration = None
                    post_error = f"{type(error).__name__}: {error}"
                post_monotonic_ns = time.monotonic_ns()
                post_time_ns = time.time_ns()
                post_gap = (
                    None
                    if previous_monotonic_ns is None
                    else (post_monotonic_ns - previous_monotonic_ns) / 1e9
                )
                post_monitor_errors: list[str] = []
                if post_error is not None:
                    post_monitor_errors.append("post_exit_sample_error")
                if post_gap is not None and post_gap > MAXIMUM_SAMPLE_GAP_SECONDS:
                    post_monitor_errors.append("sampling_gap_exceeded")
                if post_cpu["errors"]:
                    post_monitor_errors.append("process_inspection_error")
                if post_cpu["owned_processes"]:
                    post_monitor_errors.append("owned_process_survived_post_exit")
                post_apps = [
                    *post_external_gpu,
                    *(
                        {
                            **item,
                            "process_name": " ".join(item["argv"]),
                            "used_memory_mib": -1,
                        }
                        for item in post_cpu["conflicts"]
                    ),
                ]
                post_sample = flatten_sample(
                    index=len(samples),
                    time_ns=post_time_ns,
                    monotonic_ns=post_monotonic_ns,
                    gap=post_gap,
                    gpu=post_gpu,
                    cpu=post_cpu,
                    external_gpu_processes=post_external_gpu,
                    sample_error=post_error,
                    probe_worker_duration_seconds=post_probe_duration,
                    monitor_errors=post_monitor_errors,
                )
                post_sample["post_exit_telemetry"] = True
                post_sample["post_exit_ordinal"] = post_exit_ordinal
                samples.append(post_sample)
                with sample_log.open("a") as stream:
                    stream.write(
                        json.dumps(post_sample, separators=(",", ":"), sort_keys=True)
                        + "\n"
                    )
                    stream.flush()
                previous_monotonic_ns = post_monotonic_ns
                report.setdefault("post_exit_sample_indices", []).append(
                    post_sample["sample_index"]
                )
                if post_apps:
                    contaminated = True
                    report["foreign_events"].append(
                        {
                            "sample_index": post_sample["sample_index"],
                            "monotonic_ns": post_sample["monotonic_ns"],
                            "utc": post_sample["utc"],
                            "apps": post_apps,
                            "phase": "post_exit",
                        }
                    )
                if post_monitor_errors:
                    fatal_monitor_error = True
                    report["monitor_errors"].append(
                        {
                            "reason": "post_exit_telemetry_failure",
                            "sample_index": post_sample["sample_index"],
                            "reasons": post_monitor_errors,
                            "sample_error": post_error,
                            "process_inspection_errors": post_cpu["errors"],
                        }
                    )
                if post_cpu["owned_processes"] and post_exit_ordinal == 0:
                    cleanup = terminate_owned_workload(
                        process,
                        root_identity=workload_identity,
                        controller_identity=controller_identity,
                        owned_registry=owned_registry,
                        excluded_identities=monitor_excluded_identities,
                        grace_seconds=args.timeout_grace_seconds,
                        reason="owned_process_found_by_post_exit_sample",
                    )
                next_post_exit_sample = time.monotonic() + SAMPLE_INTERVAL_SECONDS
        if process is None:
            returncode = None
    except Exception as error:
        fatal_monitor_error = True
        report["monitor_errors"].append(
            {
                "reason": "controller_exception",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        if process is not None:
            cleanup = terminate_owned_workload(
                process,
                root_identity=workload_identity,
                controller_identity=controller_identity,
                owned_registry=owned_registry,
                excluded_identities=monitor_excluded_identities,
                grace_seconds=args.timeout_grace_seconds,
                reason="controller_exception",
            )
        returncode = None if process is None else process.poll()
    finally:
        if process is not None:
            final_capture_errors = capture_owned_tree(
                root_identity=workload_identity,
                controller_identity=controller_identity,
                registry=owned_registry,
                excluded_identities=monitor_excluded_identities,
            )
            final_survivors, final_identity_errors = surviving_owned_processes(
                owned_registry
            )
            final_groups = sorted(
                {
                    int(item["process_group"])
                    for item in owned_registry.values()
                    if process_group_exists(int(item["process_group"]))
                }
            )
            report["post_popen_adopted_child_audit"] = {
                "ran": True,
                "root_identity_initialized": workload_identity is not None,
                "probe_worker_identity_excluded": report.get(
                    "probe_worker_excluded_identity"
                ),
                "capture_errors": final_capture_errors,
                "identity_errors": final_identity_errors,
                "survivors_before_cleanup": final_survivors,
                "process_groups_alive_before_cleanup": final_groups,
            }
            needs_final_cleanup = (
                process.poll() is None or bool(final_survivors) or bool(final_groups)
            )
            if final_capture_errors or final_identity_errors:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {
                        "reason": "post_popen_adopted_child_audit_failed",
                        "capture_errors": final_capture_errors,
                        "identity_errors": final_identity_errors,
                    }
                )
            if needs_final_cleanup:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {
                        "reason": "post_popen_owned_process_required_cleanup",
                        "root_identity_initialized": workload_identity is not None,
                        "survivors": final_survivors,
                        "process_groups_alive": final_groups,
                    }
                )
                cleanup = terminate_owned_workload(
                    process,
                    root_identity=workload_identity,
                    controller_identity=controller_identity,
                    owned_registry=owned_registry,
                    excluded_identities=monitor_excluded_identities,
                    grace_seconds=args.timeout_grace_seconds,
                    reason="post_popen_adopted_child_cleanup",
                )
            if cleanup is not None and cleanup.get("completed") is not True:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {
                        "reason": "post_popen_owned_process_cleanup_incomplete",
                        "cleanup": cleanup,
                    }
                )
            returncode = process.poll()
        if probe is not None:
            try:
                probe.close()
            except Exception as error:
                fatal_monitor_error = True
                report["monitor_errors"].append(
                    {
                        "reason": "nvml_shutdown_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    finished_time_ns = time.time_ns()
    report.update(
        {
            "status": (
                "interrupted"
                if termination_signal is not None
                else (
                    "timed_out"
                    if timed_out
                    else (
                        "contaminated"
                        if contaminated
                        else "monitor_error" if fatal_monitor_error else "passed"
                    )
                )
            ),
            "finished_utc": utc_from_ns(finished_time_ns),
            "finished_time_ns": finished_time_ns,
            "returncode": returncode,
            "timed_out": timed_out,
            "contaminated": contaminated,
            "timeout_cleanup": cleanup,
            "sample_count": len(samples),
            "observer_identity_count": len(observer_identities),
            "owned_process_registry": list(owned_registry.values()),
            "peak_memory_used_mib": max(
                (
                    int(sample["memory_used_mib"])
                    for sample in samples
                    if sample["memory_used_mib"] is not None
                ),
                default=None,
            ),
            "peak_utilization_percent": max(
                (
                    int(sample["utilization_percent"])
                    for sample in samples
                    if sample["utilization_percent"] is not None
                ),
                default=None,
            ),
            "samples": samples,
        }
    )
    report["sample_log"].update(
        {"bytes": sample_log.stat().st_size, "sha256": guard.sha256_file(sample_log)}
    )
    guard.write_json_atomic(args.output, report)
    if termination_signal is not None:
        raise SystemExit(128 + termination_signal)
    if timed_out:
        raise SystemExit(124)
    if contaminated:
        raise SystemExit(99)
    if fatal_monitor_error:
        raise SystemExit(98)
    raise SystemExit(0 if returncode is None else returncode)


if __name__ == "__main__":
    main()
