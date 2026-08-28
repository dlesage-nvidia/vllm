# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Structural process classification and direct-NVML telemetry for GPU guards."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import time
import traceback
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

import pynvml

MPS_PROCESS_BASENAME = "nvidia-cuda-mps-server"

# This is the SHA-256 of argv[2], not a substring pattern.  It identifies the
# recurring read-only status observer that previously tripped broad `pgrep -f`
# matching.  It is retained as telemetry and never excluded from GPU checks.
STATUS_OBSERVER_SHELL_PAYLOAD_SHA256 = (
    "69c34e3836164ad4b0e1151021c8f2e36a2294075bc94e1fd62326907339b143"
)

CONFLICTING_SCRIPT_BASENAMES = frozenset(
    {
        "benchmark_image_decode.py",
        "benchmark_pynvvideocodec_e2e.py",
        "benchmark_pynvvideocodec_e2e_high_concurrency.py",
        "benchmark_pynvvideocodec_e2e_persistent.py",
        "decode_rtx.py",
        "e2e_rtx.py",
        "e2e_smoke.py",
        "run_nvimagecodec_clean_e2e.py",
        "run_nvimagecodec_e2e.py",
        "run_nvimagecodec_latest_source_e2e.py",
        "run_pynv_endpoint_high_concurrency_matrix.py",
        "run_pynv_endpoint_high_concurrency_matrix_refined.py",
        "run_pynv_endpoint_high_concurrency_pilots.py",
        "sweep2.py",
    }
)
CONFLICTING_SCRIPT_PATTERNS = (
    re.compile(r"^run_nvimagecodec(?:_[A-Za-z0-9]+)*_e2e\.py$"),
    re.compile(
        r"^run_pynv(?:_[A-Za-z0-9]+)*_(?:matrix|pilot|pilots)"
        r"(?:_[A-Za-z0-9]+)*\.py$"
    ),
)
CONFLICTING_PYTHON_MODULES = frozenset(
    {
        "vllm.entrypoints.cli.main",
        "vllm.entrypoints.openai.api_server",
    }
)
CONFLICTING_PROCESS_NAMES = frozenset(
    {
        "VLLM::EngineCore",
        "VLLM::Worker",
    }
)
CONFLICTING_SHELL_CONTROLLER = {
    "executable_basename": "bash",
    "argv_tail": ["./screen.sh"],
}

PROBE_STARTUP_TIMEOUT_SECONDS = 5.0
PROBE_RESPONSE_TIMEOUT_SECONDS = 0.75
PROBE_SHUTDOWN_TIMEOUT_SECONDS = 1.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_stat(raw: str) -> tuple[int, int, int]:
    fields = raw.rsplit(") ", maxsplit=1)[1].split()
    return int(fields[1]), int(fields[2]), int(fields[19])


def process_start_time_ticks(pid: int) -> int:
    unused_parent, unused_group, start_time_ticks = _parse_stat(
        Path(f"/proc/{pid}/stat").read_text()
    )
    del unused_parent, unused_group
    return start_time_ticks


def process_parent_pid(pid: int) -> int:
    parent_pid, unused_group, unused_start = _parse_stat(
        Path(f"/proc/{pid}/stat").read_text()
    )
    del unused_group, unused_start
    return parent_pid


def process_group_id(pid: int) -> int:
    unused_parent, process_group, unused_start = _parse_stat(
        Path(f"/proc/{pid}/stat").read_text()
    )
    del unused_parent, unused_start
    return process_group


def ancestor_identities(pid: int) -> dict[int, int]:
    identities: dict[int, int] = {}
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        identities[pid] = process_start_time_ticks(pid)
        pid = process_parent_pid(pid)
    return identities


def identity_still_matches(pid: int, expected_start_time_ticks: int) -> bool:
    try:
        return process_start_time_ticks(pid) == expected_start_time_ticks
    except FileNotFoundError:
        return False


def is_descendant_of(
    pid: int, ancestor_pid: int, ancestor_start_time_ticks: int
) -> bool | None:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            current_start = process_start_time_ticks(pid)
        except FileNotFoundError:
            return None
        if pid == ancestor_pid:
            return current_start == ancestor_start_time_ticks
        try:
            pid = process_parent_pid(pid)
        except FileNotFoundError:
            return None
    return False


def is_descendant_of_any(
    pid: int, ancestor_identities: Mapping[int, int]
) -> bool | None:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            current_start = process_start_time_ticks(pid)
        except FileNotFoundError:
            return None
        expected_start = ancestor_identities.get(pid)
        if expected_start is not None:
            return current_start == expected_start
        try:
            pid = process_parent_pid(pid)
        except FileNotFoundError:
            return None
    return False


def _python_entrypoint(argv: list[str]) -> tuple[str | None, str | None]:
    index = 1
    while index < len(argv):
        token = argv[index]
        if token in {"-c", "--command"}:
            return "code", None
        if token in {"-m", "--module"}:
            return (
                ("module", argv[index + 1]) if index + 1 < len(argv) else (None, None)
            )
        if token == "--":
            return (
                ("script", argv[index + 1]) if index + 1 < len(argv) else (None, None)
            )
        if token in {"-W", "-X"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return "script", token
    return None, None


def _script_is_conflicting(script: str) -> bool:
    basename = Path(script).name
    return basename in CONFLICTING_SCRIPT_BASENAMES or any(
        pattern.fullmatch(basename) for pattern in CONFLICTING_SCRIPT_PATTERNS
    )


def _is_shell_controller_candidate(argv: list[str]) -> bool:
    return bool(
        argv
        and Path(argv[0]).name == CONFLICTING_SHELL_CONTROLLER["executable_basename"]
        and argv[1:] == CONFLICTING_SHELL_CONTROLLER["argv_tail"]
    )


def normalize_conflicting_controller_roots(values: Sequence[str | Path]) -> list[str]:
    roots: set[str] = set()
    for value in values:
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"conflicting controller root is not absolute: {path}")
        resolved = path.resolve(strict=False)
        if resolved == Path(resolved.anchor):
            raise ValueError(f"conflicting controller root is too broad: {resolved}")
        roots.add(str(resolved))
    return sorted(roots)


def classify_process(
    argv: list[str],
    comm: str,
    cwd: str | None = None,
    *,
    conflicting_controller_roots: Sequence[str | Path] = (),
) -> tuple[str | None, str | None]:
    """Return (class, reason), using argv tokens rather than joined substrings."""

    if not argv:
        return None, None
    executable = Path(argv[0]).name
    if executable == "tmux" or executable.startswith("tmux:"):
        return None, None
    if comm in CONFLICTING_PROCESS_NAMES or executable in CONFLICTING_PROCESS_NAMES:
        return "conflict", "exact_process_name"
    if executable == "vllm" and "serve" in argv[1:]:
        return "conflict", "exact_vllm_cli_serve"
    if _is_shell_controller_candidate(argv):
        roots = normalize_conflicting_controller_roots(conflicting_controller_roots)
        if cwd is not None and str(Path(cwd).resolve(strict=False)) in roots:
            return "conflict", "exact_shell_controller_argv_and_cwd"
        return None, None

    if executable.startswith("python") or executable in {"pypy", "pypy3"}:
        kind, entrypoint = _python_entrypoint(argv)
        if kind == "module" and entrypoint in CONFLICTING_PYTHON_MODULES:
            return "conflict", "exact_python_module"
        if (
            kind == "script"
            and entrypoint is not None
            and _script_is_conflicting(entrypoint)
        ):
            return "conflict", "exact_python_script_basename"
    elif _script_is_conflicting(argv[0]):
        return "conflict", "exact_executable_script_basename"

    if executable in {"bash", "sh", "dash"} and len(argv) >= 3 and argv[1] == "-c":
        if sha256_bytes(argv[2].encode()) == STATUS_OBSERVER_SHELL_PAYLOAD_SHA256:
            return "observer", "exact_status_observer_payload_sha256"
    return None, None


def status_observer_helper_is_exact(argv: list[str]) -> bool:
    executable = Path(argv[0]).name if argv else ""
    if executable == "date":
        return argv == ["date", "-u", "+%H:%M:%S"]
    if executable == "cut":
        return argv == ["cut", "-d ", "-f1-3", "/proc/loadavg"]
    if executable == "nvidia-smi":
        return argv == [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.jpeg,memory.used",
            "--format=csv,noheader",
        ]
    if executable == "tr":
        return argv == ["tr", "\n", " "]
    if executable == "pgrep":
        return argv == [
            "pgrep",
            "-c",
            "-f",
            (
                "run_pynv|wait_for_exclusive|benchmark_image_decode|"
                "run_nvimagecodec|api_serve[r]"
            ),
        ]
    return False


def compact_observer(process: Mapping[str, Any]) -> dict[str, Any]:
    argv = process.get("argv")
    helper_basename = (
        Path(str(argv[0])).name if isinstance(argv, list) and argv else None
    )
    return {
        "pid": int(process["pid"]),
        "start_time_ticks": int(process["start_time_ticks"]),
        "classification_reason": str(process["classification_reason"]),
        "helper_basename": helper_basename,
        "observer_parent_identity": process.get("observer_parent_identity"),
    }


def compact_cpu_evidence(cpu: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "host_load": cpu.get("host_load"),
        "conflicts": cpu["conflicts"],
        "observers": [compact_observer(item) for item in cpu["observers"]],
        "owned_processes": [
            {
                "pid": int(item["pid"]),
                "parent_pid": int(item["parent_pid"]),
                "process_group": int(item["process_group"]),
                "start_time_ticks": int(item["start_time_ticks"]),
                "comm": str(item["comm"]),
                "ownership_reason": str(item["ownership_reason"]),
            }
            for item in cpu.get("owned_processes", [])
        ],
        "errors": cpu["errors"],
    }


def host_load_snapshot() -> dict[str, Any]:
    load_1m, load_5m, load_15m = os.getloadavg()
    cpu_count = os.cpu_count()
    if not isinstance(cpu_count, int) or cpu_count <= 0:
        raise RuntimeError(f"invalid os.cpu_count(): {cpu_count!r}")
    fields = Path("/proc/loadavg").read_text().split()
    if len(fields) < 4 or "/" not in fields[3]:
        raise RuntimeError(f"invalid /proc/loadavg: {fields!r}")
    runnable, total = fields[3].split("/", 1)
    return {
        "cpu_count": cpu_count,
        "load_1m": float(load_1m),
        "load_5m": float(load_5m),
        "load_15m": float(load_15m),
        "load_1m_per_cpu": float(load_1m) / cpu_count,
        "runnable_processes": int(runnable),
        "total_processes": int(total),
    }


def _read_process(pid: int) -> dict[str, Any] | None:
    proc = Path(f"/proc/{pid}")
    try:
        first_stat = (proc / "stat").read_text()
        parent_pid, process_group, start_time_ticks = _parse_stat(first_stat)
        raw_command = (proc / "cmdline").read_bytes()
        comm = (proc / "comm").read_text().rstrip("\n")
        second_start_time_ticks = process_start_time_ticks(pid)
    except FileNotFoundError:
        return None
    if second_start_time_ticks != start_time_ticks:
        raise RuntimeError(f"PID {pid} changed identity during inspection")
    argv = [part.decode(errors="replace") for part in raw_command.split(b"\0") if part]
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "process_group": process_group,
        "start_time_ticks": start_time_ticks,
        "argv": argv,
        "argv0": argv[0] if argv else None,
        "comm": comm,
        "cwd": None,
        "cgroup": None,
    }


def enrich_relevant_process_metadata(process: dict[str, Any]) -> dict[str, Any] | None:
    pid = int(process["pid"])
    proc = Path(f"/proc/{pid}")
    try:
        process["cwd"] = str((proc / "cwd").resolve())
        process["cgroup"] = (proc / "cgroup").read_text().splitlines()
        current_start = process_start_time_ticks(pid)
    except FileNotFoundError:
        process["metadata_race"] = "process_exited"
        return None
    if current_start != int(process["start_time_ticks"]):
        raise RuntimeError(f"PID {pid} changed identity during metadata enrichment")
    return process


def classify_process_snapshot(
    inspected: list[dict[str, Any]],
    *,
    conflicting_controller_roots: Sequence[str | Path] = (),
) -> dict[str, list[dict[str, Any]]]:
    conflicts: list[dict[str, Any]] = []
    observers: list[dict[str, Any]] = []
    observer_shells: dict[int, dict[str, Any]] = {}
    for process in inspected:
        process_class, reason = classify_process(
            process["argv"],
            process["comm"],
            process.get("cwd"),
            conflicting_controller_roots=conflicting_controller_roots,
        )
        process["classification"] = process_class
        process["classification_reason"] = reason
        if reason == "exact_status_observer_payload_sha256":
            observer_shells[int(process["pid"])] = process

    by_pid = {int(process["pid"]): process for process in inspected}

    def observer_ancestor(process: Mapping[str, Any]) -> dict[str, Any] | None:
        parent_pid = int(process["parent_pid"])
        seen: set[int] = set()
        while parent_pid > 1 and parent_pid not in seen:
            seen.add(parent_pid)
            observer = observer_shells.get(parent_pid)
            if observer is not None:
                return observer
            parent = by_pid.get(parent_pid)
            if parent is None:
                return None
            parent_pid = int(parent["parent_pid"])
        return None

    for process in inspected:
        process_class = process["classification"]
        reason = process["classification_reason"]
        observer = observer_ancestor(process)
        if observer is not None:
            process["observer_parent_identity"] = {
                "pid": observer["pid"],
                "start_time_ticks": observer["start_time_ticks"],
            }
            if reason == "exact_status_observer_payload_sha256":
                pass
            elif not status_observer_helper_is_exact(process["argv"]):
                process_class = "conflict"
                reason = "unexpected_status_observer_descendant"
            elif process_class is None:
                process_class = "observer"
                reason = "exact_status_observer_helper_descendant"
        if process_class is None:
            continue
        process["classification"] = process_class
        process["classification_reason"] = reason
        (conflicts if process_class == "conflict" else observers).append(process)
    sort_key = lambda item: (int(item["pid"]), int(item["start_time_ticks"]))
    return {
        "conflicts": sorted(conflicts, key=sort_key),
        "observers": sorted(observers, key=sort_key),
    }


def scan_cpu_processes(
    *,
    excluded_identities: Mapping[int, int],
    allowed_subtree_identity: tuple[int, int] | None = None,
    allowed_process_group_identity: tuple[int, int] | None = None,
    allowed_identities: Mapping[int, int] | None = None,
    allowed_subreaper_identity: tuple[int, int] | None = None,
    conflicting_controller_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    try:
        host_load = host_load_snapshot()
    except (OSError, RuntimeError, ValueError) as error:
        host_load = None
        errors.append(
            {
                "pid": None,
                "operation": "read_host_load",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError as error:
        return {
            "host_load": host_load,
            "conflicts": [],
            "observers": [],
            "owned_processes": [],
            "errors": [
                *errors,
                {
                    "pid": None,
                    "operation": "list_proc",
                    "error": f"{type(error).__name__}: {error}",
                },
            ],
        }
    for proc in proc_entries:
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        expected_start = excluded_identities.get(pid)
        if expected_start is not None:
            try:
                if identity_still_matches(pid, expected_start):
                    continue
            except (PermissionError, OSError, RuntimeError, ValueError) as error:
                errors.append(
                    {
                        "pid": pid,
                        "operation": "verify_excluded_identity",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
        try:
            process = _read_process(pid)
        except (
            PermissionError,
            OSError,
            RuntimeError,
            UnicodeDecodeError,
            ValueError,
        ) as error:
            errors.append(
                {
                    "pid": pid,
                    "operation": "inspect_process",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if process is None or not process["argv"]:
            continue
        allowed_reason: str | None = None
        if allowed_identities:
            expected_allowed_start = allowed_identities.get(pid)
            if expected_allowed_start is not None:
                if int(process["start_time_ticks"]) == int(expected_allowed_start):
                    allowed_reason = "exact_owned_pid_start_identity"
                else:
                    errors.append(
                        {
                            "pid": pid,
                            "operation": "verify_owned_identity",
                            "error": "PID start_time_ticks did not match registry",
                        }
                    )
            if allowed_reason is None:
                try:
                    descendant = is_descendant_of_any(pid, allowed_identities)
                except (PermissionError, OSError, RuntimeError, ValueError) as error:
                    errors.append(
                        {
                            "pid": pid,
                            "operation": "verify_owned_registry_subtree",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    descendant = None
                if descendant is True:
                    allowed_reason = "owned_registry_descendant"
        if allowed_process_group_identity is not None:
            group_pid, group_leader_start = allowed_process_group_identity
            if int(process["process_group"]) == group_pid and identity_still_matches(
                group_pid, group_leader_start
            ):
                allowed_reason = allowed_reason or "owned_root_process_group"
        if allowed_reason is None and allowed_subtree_identity is not None:
            try:
                descendant = is_descendant_of(pid, *allowed_subtree_identity)
            except (PermissionError, OSError, RuntimeError, ValueError) as error:
                errors.append(
                    {
                        "pid": pid,
                        "operation": "verify_allowed_subtree",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if descendant is True:
                allowed_reason = "owned_root_descendant"
        if allowed_reason is None and allowed_subreaper_identity is not None:
            subreaper_pid, subreaper_start = allowed_subreaper_identity
            if int(process["parent_pid"]) == subreaper_pid and identity_still_matches(
                subreaper_pid, subreaper_start
            ):
                allowed_reason = "owned_subreaper_adopted_process"
        if allowed_reason is not None:
            process["ownership_reason"] = allowed_reason
            inspected.append(process)
            continue
        if _is_shell_controller_candidate(process["argv"]):
            try:
                enriched = enrich_relevant_process_metadata(process)
            except (PermissionError, OSError, RuntimeError, ValueError) as error:
                errors.append(
                    {
                        "pid": pid,
                        "operation": "enrich_shell_controller_candidate",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            if enriched is None:
                continue
        inspected.append(process)

    # A verified child-sub-reaper adopts an orphaned workload process directly.
    # Once that identity is owned, its descendants are workload-owned even when
    # they have generic command lines and have created separate sessions/groups.
    # Exact exclusions (notably the probe worker) were removed before this pass.
    owned_by_pid = {
        int(process["pid"]): int(process["start_time_ticks"])
        for process in inspected
        if process.get("ownership_reason") is not None
    }
    changed = True
    while changed:
        changed = False
        for process in inspected:
            if process.get("ownership_reason") is not None:
                continue
            parent_pid = int(process["parent_pid"])
            if parent_pid not in owned_by_pid:
                continue
            process["ownership_reason"] = "owned_process_descendant"
            owned_by_pid[int(process["pid"])] = int(process["start_time_ticks"])
            changed = True
    owned_processes = [
        process for process in inspected if process.get("ownership_reason") is not None
    ]
    classified = classify_process_snapshot(
        [process for process in inspected if process.get("ownership_reason") is None],
        conflicting_controller_roots=conflicting_controller_roots,
    )
    relevant_by_identity = {
        (int(process["pid"]), int(process["start_time_ticks"])): process
        for process in [*classified["conflicts"], *classified["observers"]]
    }
    for process in relevant_by_identity.values():
        if process.get("cwd") is not None:
            continue
        try:
            enrich_relevant_process_metadata(process)
        except (PermissionError, OSError, RuntimeError, ValueError) as error:
            errors.append(
                {
                    "pid": int(process["pid"]),
                    "operation": "enrich_relevant_process_metadata",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "host_load": host_load,
        "conflicts": classified["conflicts"],
        "observers": classified["observers"],
        "owned_processes": sorted(
            owned_processes,
            key=lambda item: (int(item["pid"]), int(item["start_time_ticks"])),
        ),
        "errors": sorted(
            errors,
            key=lambda item: (
                -1 if item["pid"] is None else int(item["pid"]),
                str(item["operation"]),
            ),
        ),
    }


def _decode_nvml_string(value: object) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


class NvmlSampler:
    """Direct NVML sampling; this class launches no subprocess commands."""

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.initialized = False
        pynvml.nvmlInit()
        self.initialized = True
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device = {
            "index": device_index,
            "uuid": _decode_nvml_string(pynvml.nvmlDeviceGetUUID(self.handle)),
            "name": _decode_nvml_string(pynvml.nvmlDeviceGetName(self.handle)),
        }

    def close(self) -> None:
        if self.initialized:
            pynvml.nvmlShutdown()
            self.initialized = False

    def _engine_utilization(self, function_name: str) -> dict[str, int] | None:
        getter = getattr(pynvml, function_name, None)
        if getter is None:
            return None
        try:
            utilization, sampling_period_us = getter(self.handle)
        except pynvml.NVMLError_NotSupported:
            return None
        return {
            "utilization_percent": int(utilization),
            "sampling_period_us": int(sampling_period_us),
        }

    def _optional_scalar(
        self,
        function_name: str,
        *,
        argument_constant_name: str | None = None,
        divisor: float = 1.0,
    ) -> dict[str, Any]:
        getter = getattr(pynvml, function_name, None)
        if getter is None:
            return {"status": "unavailable", "value": None}
        arguments: list[Any] = [self.handle]
        if argument_constant_name is not None:
            constant = getattr(pynvml, argument_constant_name, None)
            if constant is None:
                return {"status": "unavailable", "value": None}
            arguments.append(constant)
        try:
            value = getter(*arguments)
        except pynvml.NVMLError_NotSupported:
            return {"status": "not_supported", "value": None}
        except pynvml.NVMLError as error:
            raise RuntimeError(f"NVML {function_name} query failed: {error}") from error
        return {"status": "passed", "value": float(value) / divisor}

    def _running_processes(self, kind: str, getter: Any) -> list[dict[str, Any]]:
        try:
            running = getter(self.handle)
        except pynvml.NVMLError as error:
            raise RuntimeError(f"NVML {kind} process query failed: {error}") from error
        processes = []
        for item in running:
            pid = int(item.pid)
            try:
                name = _decode_nvml_string(pynvml.nvmlSystemGetProcessName(pid))
            except pynvml.NVMLError:
                name = "<unavailable>"
            raw_memory = getattr(item, "usedGpuMemory", None)
            used_memory_mib = (
                int(raw_memory / 1024**2)
                if isinstance(raw_memory, int) and 0 <= raw_memory < 2**63
                else None
            )
            processes.append(
                {
                    "pid": pid,
                    "process_name": name,
                    "used_memory_mib": used_memory_mib,
                    "kind": kind,
                }
            )
        return processes

    def _mps_running_processes(
        self, *, daemon_present: bool
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        attempts: list[dict[str, str]] = []
        for function_name in (
            "nvmlDeviceGetMPSComputeRunningProcesses_v3",
            "nvmlDeviceGetMPSComputeRunningProcesses_v2",
            "nvmlDeviceGetMPSComputeRunningProcesses",
        ):
            getter = getattr(pynvml, function_name, None)
            if getter is None:
                attempts.append({"function": function_name, "status": "unavailable"})
                continue
            try:
                processes = self._running_processes("mps_compute", getter)
            except RuntimeError as error:
                cause = error.__cause__
                not_supported_type = getattr(pynvml, "NVMLError_NotSupported", ())
                if not_supported_type and isinstance(cause, not_supported_type):
                    attempts.append(
                        {"function": function_name, "status": "not_supported"}
                    )
                    continue
                attempts.append(
                    {
                        "function": function_name,
                        "status": "error",
                        "error": f"{type(cause or error).__name__}: {cause or error}",
                    }
                )
                if daemon_present:
                    raise RuntimeError(
                        "MPS compute-client attribution failed while the MPS daemon "
                        f"is present: {attempts[-1]['error']}"
                    ) from error
                return (
                    {
                        "status": "query_error_without_mps_daemon",
                        "function": function_name,
                        "attempts": attempts,
                        "daemon_present": False,
                    },
                    [],
                )
            return (
                {
                    "status": "passed",
                    "function": function_name,
                    "attempts": attempts,
                    "daemon_present": daemon_present,
                },
                processes,
            )
        if daemon_present:
            raise RuntimeError(
                "PyNVML exposes no usable MPS compute-client process query while "
                f"the MPS daemon is present; attempts={attempts}"
            )
        return (
            {
                "status": "unavailable_without_mps_daemon",
                "function": None,
                "attempts": attempts,
                "daemon_present": False,
            },
            [],
        )

    def sample(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        compute = self._running_processes(
            "compute", pynvml.nvmlDeviceGetComputeRunningProcesses
        )
        graphics = self._running_processes(
            "graphics", pynvml.nvmlDeviceGetGraphicsRunningProcesses
        )
        mps_daemon_present = any(
            Path(str(process.get("process_name", ""))).name == MPS_PROCESS_BASENAME
            for process in compute
        )
        mps_query, mps_compute = self._mps_running_processes(
            daemon_present=mps_daemon_present
        )
        return {
            "memory_used_mib": int(memory.used / 1024**2),
            "memory_total_mib": int(memory.total / 1024**2),
            "utilization_gpu_percent": int(utilization.gpu),
            "utilization_memory_percent": int(utilization.memory),
            "decoder": self._engine_utilization("nvmlDeviceGetDecoderUtilization"),
            "encoder": self._engine_utilization("nvmlDeviceGetEncoderUtilization"),
            "jpeg": self._engine_utilization("nvmlDeviceGetJpgUtilization"),
            "ofa": self._engine_utilization("nvmlDeviceGetOfaUtilization"),
            "operating_telemetry": {
                "sm_clock_mhz": self._optional_scalar(
                    "nvmlDeviceGetClockInfo", argument_constant_name="NVML_CLOCK_SM"
                ),
                "memory_clock_mhz": self._optional_scalar(
                    "nvmlDeviceGetClockInfo", argument_constant_name="NVML_CLOCK_MEM"
                ),
                "temperature_c": self._optional_scalar(
                    "nvmlDeviceGetTemperature",
                    argument_constant_name="NVML_TEMPERATURE_GPU",
                ),
                "power_w": self._optional_scalar(
                    "nvmlDeviceGetPowerUsage", divisor=1000.0
                ),
                "performance_state": self._optional_scalar(
                    "nvmlDeviceGetPerformanceState"
                ),
            },
            "compute_processes": compute,
            "graphics_processes": graphics,
            "mps_compute_process_query": mps_query,
            "mps_compute_processes": mps_compute,
            "mps_daemon_present": mps_daemon_present,
            "query_duration_seconds": (time.monotonic_ns() - started_ns) / 1e9,
        }


def attributable_gpu_processes(gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[int, str]] = set()
    processes: list[dict[str, Any]] = []
    for key in (
        "compute_processes",
        "graphics_processes",
        "mps_compute_processes",
    ):
        for process in gpu.get(key, []):
            name = str(process.get("process_name", ""))
            if Path(name).name == MPS_PROCESS_BASENAME:
                continue
            identity = (int(process["pid"]), str(process["kind"]))
            if identity not in seen:
                seen.add(identity)
                processes.append(dict(process))
    return processes


def non_mps_gpu_processes(gpu: Mapping[str, Any]) -> list[dict[str, Any]]:
    return attributable_gpu_processes(gpu)


def gpu_process_matches_workload(
    process: Mapping[str, Any],
    workload_identity: tuple[int, int],
    allowed_identities: Mapping[int, int] | None = None,
) -> bool:
    pid = int(process["pid"])
    if allowed_identities is not None:
        expected_start = allowed_identities.get(pid)
        if expected_start is not None and identity_still_matches(pid, expected_start):
            return True
    group_pid, group_start = workload_identity
    if process_group_id(pid) == group_pid and identity_still_matches(
        group_pid, group_start
    ):
        return True
    return is_descendant_of(pid, group_pid, group_start) is True


def external_gpu_processes(
    gpu: Mapping[str, Any],
    workload_identity: tuple[int, int] | None = None,
    allowed_identities: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    processes = attributable_gpu_processes(gpu)
    if workload_identity is None:
        return processes
    return [
        process
        for process in processes
        if not gpu_process_matches_workload(
            process, workload_identity, allowed_identities
        )
    ]


def gpu_is_idle(gpu: Mapping[str, Any], *, memory_ceiling_mib: int) -> list[str]:
    reasons: list[str] = []
    if non_mps_gpu_processes(gpu):
        reasons.append("non_mps_gpu_process")
    if int(gpu["utilization_gpu_percent"]) > 0:
        reasons.append("gpu_utilization_nonzero")
    if int(gpu["utilization_memory_percent"]) > 0:
        reasons.append("memory_utilization_nonzero")
    for engine in ("decoder", "encoder", "jpeg", "ofa"):
        telemetry = gpu.get(engine)
        if telemetry is not None and int(telemetry["utilization_percent"]) > 0:
            reasons.append(f"{engine}_utilization_nonzero")
    if int(gpu["memory_used_mib"]) > memory_ceiling_mib:
        reasons.append("gpu_memory_above_idle_ceiling")
    return reasons


def _probe_worker_main(
    connection: Any, device_index: int, conflicting_controller_roots: Sequence[str]
) -> None:
    sampler: NvmlSampler | None = None
    try:
        sampler = NvmlSampler(device_index)
        worker_pid = multiprocessing.current_process().pid
        connection.send(
            {
                "kind": "ready",
                "pid": worker_pid,
                "start_time_ticks": process_start_time_ticks(worker_pid),
                "device": sampler.device,
            }
        )
        while True:
            request = connection.recv()
            if request.get("operation") == "close":
                return
            if request.get("operation") != "sample":
                raise RuntimeError(f"unknown probe operation: {request!r}")
            request_id = int(request["request_id"])
            started_ns = time.monotonic_ns()
            try:
                gpu = sampler.sample()
                cpu = scan_cpu_processes(
                    excluded_identities=request["excluded_identities"],
                    allowed_subtree_identity=request.get("allowed_subtree_identity"),
                    allowed_process_group_identity=request.get(
                        "allowed_process_group_identity"
                    ),
                    allowed_identities=request.get("allowed_identities"),
                    allowed_subreaper_identity=request.get(
                        "allowed_subreaper_identity"
                    ),
                    conflicting_controller_roots=conflicting_controller_roots,
                )
                workload_identity = request.get("allowed_process_group_identity")
                effective_allowed_identities = dict(
                    request.get("allowed_identities") or {}
                )
                effective_allowed_identities.update(
                    {
                        int(process["pid"]): int(process["start_time_ticks"])
                        for process in cpu["owned_processes"]
                    }
                )
                foreign_gpu_processes = external_gpu_processes(
                    gpu, workload_identity, effective_allowed_identities
                )
                response = {
                    "kind": "sample",
                    "request_id": request_id,
                    "ok": True,
                    "gpu": gpu,
                    "cpu": cpu,
                    "external_gpu_processes": foreign_gpu_processes,
                    "worker_duration_seconds": (time.monotonic_ns() - started_ns) / 1e9,
                }
            except BaseException as error:
                response = {
                    "kind": "sample",
                    "request_id": request_id,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "worker_duration_seconds": (time.monotonic_ns() - started_ns) / 1e9,
                }
            connection.send(response)
    except EOFError:
        return
    except BaseException as error:
        try:
            connection.send(
                {
                    "kind": "worker_error",
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        except BaseException:
            pass
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except BaseException:
                pass
        connection.close()


class ProbeWorker:
    """Persistent probe process with bounded startup, response, and teardown."""

    def __init__(
        self,
        device_index: int,
        *,
        conflicting_controller_roots: Sequence[str | Path] = (),
        _worker_target: Any = _probe_worker_main,
        _startup_timeout_seconds: float = PROBE_STARTUP_TIMEOUT_SECONDS,
        _response_timeout_seconds: float = PROBE_RESPONSE_TIMEOUT_SECONDS,
        _shutdown_timeout_seconds: float = PROBE_SHUTDOWN_TIMEOUT_SECONDS,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self.connection = parent_connection
        normalized_controller_roots = normalize_conflicting_controller_roots(
            conflicting_controller_roots
        )
        self.process = context.Process(
            target=_worker_target,
            args=(child_connection, device_index, normalized_controller_roots),
            name="pynv-refined-gpu-probe",
            daemon=True,
        )
        self.request_id = 0
        self.closed = False
        self.response_timeout_seconds = _response_timeout_seconds
        self.shutdown_timeout_seconds = _shutdown_timeout_seconds
        self.process.start()
        child_connection.close()
        try:
            ready = self._receive(_startup_timeout_seconds, "startup")
        except BaseException:
            self.abort()
            raise
        if ready.get("kind") != "ready":
            self.abort()
            raise RuntimeError(f"probe worker failed to initialize: {ready!r}")
        self.worker_identity = {
            "pid": int(ready["pid"]),
            "start_time_ticks": int(ready["start_time_ticks"]),
        }
        self.device = ready["device"]

    def _receive(self, timeout_seconds: float, stage: str) -> dict[str, Any]:
        if not self.connection.poll(timeout_seconds):
            raise TimeoutError(
                f"probe worker {stage} exceeded {timeout_seconds:.3f} seconds"
            )
        response = self.connection.recv()
        if not isinstance(response, dict):
            raise RuntimeError(f"invalid probe worker response: {response!r}")
        return response

    def sample(
        self,
        *,
        excluded_identities: Mapping[int, int],
        allowed_subtree_identity: tuple[int, int] | None = None,
        allowed_process_group_identity: tuple[int, int] | None = None,
        allowed_identities: Mapping[int, int] | None = None,
        allowed_subreaper_identity: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("probe worker is closed")
        self.request_id += 1
        request_id = self.request_id
        self.connection.send(
            {
                "operation": "sample",
                "request_id": request_id,
                "excluded_identities": dict(excluded_identities),
                "allowed_subtree_identity": allowed_subtree_identity,
                "allowed_process_group_identity": allowed_process_group_identity,
                "allowed_identities": dict(allowed_identities or {}),
                "allowed_subreaper_identity": allowed_subreaper_identity,
            }
        )
        try:
            response = self._receive(self.response_timeout_seconds, "sample")
        except BaseException:
            self.abort()
            raise
        if response.get("kind") != "sample" or response.get("request_id") != request_id:
            self.abort()
            raise RuntimeError(f"probe worker protocol mismatch: {response!r}")
        if not response.get("ok"):
            self.abort()
            raise RuntimeError(
                "probe worker sampling failed: "
                f"{response.get('error')}; {response.get('traceback')}"
            )
        return response

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(self.shutdown_timeout_seconds)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(self.shutdown_timeout_seconds)
        self.connection.close()

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.connection.send({"operation": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        self.process.join(self.shutdown_timeout_seconds)
        if self.process.is_alive():
            self.abort()
            return
        self.closed = True
        self.connection.close()

    def __enter__(self) -> "ProbeWorker":
        return self

    def __exit__(
        self, unused_type: Any, unused_value: Any, unused_traceback: Any
    ) -> None:
        self.close()
