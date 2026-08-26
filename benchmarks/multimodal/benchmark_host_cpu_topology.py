# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture the host CPU budget used by image decode benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import psutil


def _read_optional(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _lscpu_summary() -> dict[str, str]:
    output = subprocess.check_output(["lscpu", "--json"], text=True)
    payload = json.loads(output)
    return {
        str(item["field"]).rstrip(":"): str(item.get("data", ""))
        for item in payload["lscpu"]
    }


def _mps_affinities() -> list[list[int]]:
    affinities: set[tuple[int, ...]] = set()
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            identity = (
                f"{process.info['name']} {' '.join(process.info['cmdline'] or [])}"
            ).lower()
            if "nvidia-cuda-mps-server" in identity:
                affinities.add(tuple(sorted(process.cpu_affinity())))
        except (psutil.Error, OSError, TypeError):
            continue
    return [list(affinity) for affinity in sorted(affinities)]


def capture_cpu_topology() -> dict[str, object]:
    """Return static topology, process CPU budget, and a stable fingerprint."""
    lscpu = _lscpu_summary()
    numa_nodes = {
        path.name: _read_optional(path / "cpulist")
        for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
    }
    process_affinity = sorted(psutil.Process().cpu_affinity())
    sched_affinity = sorted(os.sched_getaffinity(0))
    static: dict[str, object] = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "process_cpu_affinity": process_affinity,
        "sched_cpu_affinity": sched_affinity,
        "sysfs_online_cpus": _read_optional(Path("/sys/devices/system/cpu/online")),
        "cgroup_cpuset_effective": _read_optional(
            Path("/sys/fs/cgroup/cpuset.cpus.effective")
        ),
        "cgroup_cpu_max": _read_optional(Path("/sys/fs/cgroup/cpu.max")),
        "numa_node_cpulists": numa_nodes,
        "lscpu": {
            key: lscpu.get(key)
            for key in (
                "Architecture",
                "CPU(s)",
                "On-line CPU(s) list",
                "Model name",
                "Thread(s) per core",
                "Core(s) per socket",
                "Socket(s)",
                "NUMA node(s)",
            )
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(static, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "static": static,
        "static_sha256": fingerprint,
        "mps_server_cpu_affinities": _mps_affinities(),
    }
