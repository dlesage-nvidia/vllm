# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import pynvml

    del pynvml
except ModuleNotFoundError:
    sys.modules["pynvml"] = types.ModuleType("pynvml")

import pynv_gpu_guard as guard

BUNDLE_ROOT = Path(__file__).resolve().parent
FIXTURE_CONTROLLER_ROOT = (BUNDLE_ROOT / "fixture-controller-root").resolve()
ORIGINAL_POPEN = subprocess.Popen
ORIGINAL_PROCESS_START_TIME_TICKS = guard.process_start_time_ticks

STATUS_PAYLOAD = """
    printf "%s | load=%s | gpu=%s\\n" "$(date -u +%H:%M:%S)" "$(cut -d" " -f1-3 /proc/loadavg)" "$(nvidia-smi --query-gpu=utilization.gpu,utilization.jpeg,memory.used --format=csv,noheader | tr "\\n" " ")"
    N=$(pgrep -c -f "run_pynv|wait_for_exclusive|benchmark_image_decode|run_nvimagecodec|api_serve[r]" 2>/dev/null || echo 0)
    echo "    foreign_procs=$N"
  """


def process(
    pid: int,
    parent_pid: int,
    argv: list[str],
    *,
    comm: str | None = None,
) -> dict[str, object]:
    return {
        "pid": pid,
        "parent_pid": parent_pid,
        "process_group": pid,
        "start_time_ticks": pid * 100,
        "argv": argv,
        "argv0": argv[0],
        "comm": comm or argv[0].rsplit("/", maxsplit=1)[-1],
        "cwd": str(FIXTURE_CONTROLLER_ROOT),
        "cgroup": ["0::/fixture"],
    }


def blocking_probe_worker(
    connection: object,
    unused_device_index: int,
    unused_conflicting_controller_roots: list[str],
) -> None:
    connection.send(
        {
            "kind": "ready",
            "pid": os.getpid(),
            "start_time_ticks": 123,
            "device": {"index": 0, "uuid": "fixture", "name": "fixture"},
        }
    )
    connection.recv()
    time.sleep(60)


class FixtureProbeWorker:
    def __init__(
        self, device_index: int, *, conflicting_controller_roots: object = ()
    ) -> None:
        self.conflicting_controller_roots = conflicting_controller_roots
        self.device = {"index": device_index, "uuid": "fixture-uuid", "name": "fixture"}
        self.worker_identity = {
            "pid": os.getpid(),
            "start_time_ticks": guard.process_start_time_ticks(os.getpid()),
        }

    def sample(self, **kwargs: object) -> dict[str, object]:
        cpu = guard.scan_cpu_processes(
            excluded_identities=kwargs["excluded_identities"],
            allowed_subtree_identity=kwargs.get("allowed_subtree_identity"),
            allowed_process_group_identity=kwargs.get("allowed_process_group_identity"),
            allowed_identities=kwargs.get("allowed_identities"),
            allowed_subreaper_identity=kwargs.get("allowed_subreaper_identity"),
            conflicting_controller_roots=self.conflicting_controller_roots,
        )
        gpu = {
            "memory_used_mib": 694,
            "memory_total_mib": 96 * 1024,
            "utilization_gpu_percent": 0,
            "utilization_memory_percent": 0,
            "decoder": {"utilization_percent": 0, "sampling_period_us": 1000},
            "encoder": {"utilization_percent": 0, "sampling_period_us": 1000},
            "jpeg": {"utilization_percent": 0, "sampling_period_us": 1000},
            "ofa": None,
            "compute_processes": [
                {
                    "pid": 999999,
                    "process_name": "/usr/bin/nvidia-cuda-mps-server",
                    "used_memory_mib": 52,
                    "kind": "compute",
                }
            ],
            "graphics_processes": [],
            "mps_compute_process_query": {
                "status": "passed",
                "function": "fixture_mps_query",
                "attempts": [],
                "daemon_present": True,
            },
            "mps_compute_processes": [],
            "mps_daemon_present": True,
            "query_duration_seconds": 0.001,
        }
        workload_identity = kwargs.get("allowed_process_group_identity")
        effective_owned = dict(kwargs.get("allowed_identities") or {})
        effective_owned.update(
            {
                int(process["pid"]): int(process["start_time_ticks"])
                for process in cpu["owned_processes"]
            }
        )
        external = guard.external_gpu_processes(gpu, workload_identity, effective_owned)
        return {
            "gpu": gpu,
            "cpu": cpu,
            "external_gpu_processes": external,
            "worker_duration_seconds": 0.001,
        }

    def close(self) -> None:
        return


class DistinctLiveFixtureProbeWorker(FixtureProbeWorker):
    def __init__(
        self, device_index: int, *, conflicting_controller_roots: object = ()
    ) -> None:
        super().__init__(
            device_index,
            conflicting_controller_roots=conflicting_controller_roots,
        )
        self.probe_process = ORIGINAL_POPEN(["/usr/bin/sleep", "60"])
        self.worker_identity = {
            "pid": self.probe_process.pid,
            "start_time_ticks": ORIGINAL_PROCESS_START_TIME_TICKS(
                self.probe_process.pid
            ),
        }

    def close(self) -> None:
        state_path = Path(os.environ["PYNV_DISTINCT_PROBE_STATE_FILE"])
        state_path.write_text(
            json.dumps(
                {
                    **self.worker_identity,
                    "alive_before_close": self.probe_process.poll() is None,
                },
                sort_keys=True,
            )
            + "\n"
        )
        if self.probe_process.poll() is None:
            self.probe_process.terminate()
        self.probe_process.wait(timeout=5.0)


def import_script(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, BUNDLE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def controller_fixture_entry() -> None:
    mode = sys.argv[1]
    output = Path(sys.argv[2]).resolve()
    guard.ProbeWorker = (
        DistinctLiveFixtureProbeWorker
        if mode == "monitor_identity_race_distinct_probe"
        else FixtureProbeWorker
    )
    if mode in {"monitor_identity_race", "monitor_identity_race_distinct_probe"}:
        original_popen = subprocess.Popen
        original_start_time_ticks = guard.process_start_time_ticks
        workload_pid: dict[str, int | bool | None] = {
            "pid": None,
            "failure_injected": False,
        }

        def recording_popen(*args: object, **kwargs: object):
            child = original_popen(*args, **kwargs)
            workload_pid["pid"] = child.pid
            return child

        def fail_workload_identity_once(pid: int) -> int:
            if pid == workload_pid["pid"] and workload_pid["failure_injected"] is False:
                pid_file = Path(os.environ["PYNV_IDENTITY_RACE_PID_FILE"])
                wait_for_nonempty_path(pid_file, timeout=5.0)
                workload_pid["failure_injected"] = True
                raise FileNotFoundError(
                    "fixture: workload leader exited before identity"
                )
            return original_start_time_ticks(pid)

        subprocess.Popen = recording_popen
        guard.process_start_time_ticks = fail_workload_identity_once
        mode = "monitor"
    if mode == "monitor":
        module = import_script("fixture_monitor", "run_with_gpu_monitor_refined.py")
        sys.argv = [
            str(BUNDLE_ROOT / "run_with_gpu_monitor_refined.py"),
            "--output",
            str(output),
            "--conflicting-controller-root",
            str(FIXTURE_CONTROLLER_ROOT),
            "--",
            *sys.argv[3:],
        ]
    elif mode == "gate":
        module = import_script("fixture_gate", "wait_for_exclusive_gpu_refined.py")
        sys.argv = [
            str(BUNDLE_ROOT / "wait_for_exclusive_gpu_refined.py"),
            "--seconds",
            "30",
            "--timeout",
            "60",
            "--output",
            str(output),
            "--conflicting-controller-root",
            str(FIXTURE_CONTROLLER_ROOT),
        ]
    else:
        raise ValueError(f"unknown fixture controller mode: {mode}")
    module.main()


def main() -> None:
    assert hashlib.sha256(STATUS_PAYLOAD.encode()).hexdigest() == (
        guard.STATUS_OBSERVER_SHELL_PAYLOAD_SHA256
    )
    assert guard.classify_process(["bash", "-c", STATUS_PAYLOAD], "bash") == (
        "observer",
        "exact_status_observer_payload_sha256",
    )
    assert guard.classify_process(
        [
            "pgrep",
            "-c",
            "-f",
            "run_pynv|wait_for_exclusive|run_nvimagecodec",
        ],
        "pgrep",
    ) == (None, None)
    assert guard.classify_process(
        ["python", "/fixture/controller/e2e_rtx.py"], "python"
    ) == ("conflict", "exact_python_script_basename")
    for tagged_controller in (
        "run_pynv_persistent_three_arm_high_concurrency_matrix_legacy.py",
        "run_pynv_persistent_three_arm_high_concurrency_pilots_legacy.py",
        "run_pynv_endpoint_high_concurrency_matrix_refined_legacy.py",
    ):
        assert guard.classify_process(
            ["python", f"/tmp/{tagged_controller}"], "python"
        ) == ("conflict", "exact_python_script_basename")
    assert guard.classify_process(
        ["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"],
        "python",
    ) == ("conflict", "exact_python_module")
    assert guard.classify_process(["VLLM::EngineCore"], "VLLM::EngineCore") == (
        "conflict",
        "exact_process_name",
    )
    assert guard.classify_process(
        ["python", "-c", "print('e2e_rtx.py run_nvimagecodec')"], "python"
    ) == (None, None)
    assert guard.classify_process(
        ["python", "/tmp/status.py", "e2e_rtx.py", "run_nvimagecodec"],
        "python",
    ) == (None, None)
    assert guard.classify_process(
        ["tmux", "new-session", "run_nvimagecodec_clean_e2e.py"], "tmux"
    ) == (None, None)
    assert guard.classify_process(
        ["bash", "-c", STATUS_PAYLOAD + "echo unexpected"], "bash"
    ) == (None, None)
    assert guard.classify_process(
        ["/bin/bash", "./screen.sh"],
        "bash",
        str(FIXTURE_CONTROLLER_ROOT),
        conflicting_controller_roots=[FIXTURE_CONTROLLER_ROOT],
    ) == ("conflict", "exact_shell_controller_argv_and_cwd")
    assert guard.classify_process(
        ["/bin/bash", "./screen.sh"],
        "bash",
        "/tmp/unrelated",
        conflicting_controller_roots=[FIXTURE_CONTROLLER_ROOT],
    ) == (None, None)

    observer_tree = [
        process(100, 1, ["bash", "-c", STATUS_PAYLOAD], comm="bash"),
        process(
            101,
            100,
            [
                "pgrep",
                "-c",
                "-f",
                (
                    "run_pynv|wait_for_exclusive|benchmark_image_decode|"
                    "run_nvimagecodec|api_serve[r]"
                ),
            ],
            comm="pgrep",
        ),
        process(
            102,
            100,
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.jpeg,memory.used",
                "--format=csv,noheader",
            ],
            comm="nvidia-smi",
        ),
    ]
    classified = guard.classify_process_snapshot(copy.deepcopy(observer_tree))
    assert not classified["conflicts"]
    assert {item["pid"] for item in classified["observers"]} == {100, 101, 102}
    assert all("start_time_ticks" in item for item in classified["observers"])

    altered_helper_tree = [
        *observer_tree[:1],
        process(
            104,
            100,
            ["pgrep", "-c", "-f", "run_nvimagecodec|unexpected"],
            comm="pgrep",
        ),
    ]
    classified = guard.classify_process_snapshot(copy.deepcopy(altered_helper_tree))
    assert [
        (item["pid"], item["classification_reason"]) for item in classified["conflicts"]
    ] == [(104, "unexpected_status_observer_descendant")]

    unexpected_tree = [
        *observer_tree,
        process(103, 100, ["python", "/tmp/unexpected.py"], comm="python"),
    ]
    classified = guard.classify_process_snapshot(copy.deepcopy(unexpected_tree))
    assert [
        (item["pid"], item["classification_reason"]) for item in classified["conflicts"]
    ] == [(103, "unexpected_status_observer_descendant")]

    real_tree = [
        process(200, 1, ["tmux", "new-session", "e2e_rtx.py"], comm="tmux"),
        process(
            201,
            200,
            ["/opt/venv/bin/python", "/fixture/controller/e2e_rtx.py"],
            comm="python",
        ),
    ]
    classified = guard.classify_process_snapshot(copy.deepcopy(real_tree))
    assert [item["pid"] for item in classified["conflicts"]] == [201]
    unrelated_in_benchmark_cwd = [
        process(300, 1, ["python", "/tmp/unrelated.py"], comm="python")
    ]
    classified = guard.classify_process_snapshot(
        copy.deepcopy(unrelated_in_benchmark_cwd)
    )
    assert classified == {"conflicts": [], "observers": []}

    gpu = {
        "memory_used_mib": 694,
        "utilization_gpu_percent": 0,
        "utilization_memory_percent": 0,
        "decoder": {"utilization_percent": 0},
        "encoder": {"utilization_percent": 0},
        "jpeg": {"utilization_percent": 0},
        "ofa": None,
        "compute_processes": [
            {
                "pid": 1,
                "process_name": "/usr/bin/nvidia-cuda-mps-server",
                "used_memory_mib": 52,
                "kind": "compute",
            }
        ],
        "graphics_processes": [],
    }
    assert guard.gpu_is_idle(gpu, memory_ceiling_mib=1024) == []
    gpu["compute_processes"].append(
        {
            "pid": 2,
            "process_name": "/tmp/not-nvidia-cuda-mps-server-wrapper",
            "used_memory_mib": 1,
            "kind": "compute",
        }
    )
    assert guard.gpu_is_idle(gpu, memory_ceiling_mib=1024) == ["non_mps_gpu_process"]
    gpu["compute_processes"].pop()
    gpu["decoder"] = {"utilization_percent": 1}
    assert guard.gpu_is_idle(gpu, memory_ceiling_mib=1024) == [
        "decoder_utilization_nonzero"
    ]
    lineage = guard.ancestor_identities(os.getpid())
    assert lineage[os.getpid()] == guard.process_start_time_ticks(os.getpid())
    probe = guard.ProbeWorker(
        0,
        _worker_target=blocking_probe_worker,
        _startup_timeout_seconds=1.0,
        _response_timeout_seconds=0.05,
        _shutdown_timeout_seconds=0.1,
    )
    started = time.monotonic()
    try:
        probe.sample(excluded_identities={})
    except TimeoutError:
        pass
    else:
        raise AssertionError("blocking probe did not time out")
    assert time.monotonic() - started < 1.0
    assert not probe.process.is_alive()
    print("refined GPU guard fixtures passed")


def test_refined_gpu_guard_fixtures() -> None:
    main()


class FakeNvmlError(Exception):
    pass


class FakeNvmlNotSupported(FakeNvmlError):
    pass


def fake_process(pid: int, memory_mib: int = 1) -> SimpleNamespace:
    return SimpleNamespace(pid=pid, usedGpuMemory=memory_mib * 1024**2)


def fake_nvml(**functions: object) -> SimpleNamespace:
    return SimpleNamespace(
        NVMLError=FakeNvmlError,
        NVMLError_NotSupported=FakeNvmlNotSupported,
        nvmlSystemGetProcessName=lambda pid: f"/fixture/process-{pid}".encode(),
        **functions,
    )


def bare_sampler() -> guard.NvmlSampler:
    sampler = object.__new__(guard.NvmlSampler)
    sampler.handle = object()
    return sampler


def test_host_load_and_optional_operating_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_load = guard.host_load_snapshot()
    assert host_load["cpu_count"] > 0
    assert host_load["load_1m_per_cpu"] == pytest.approx(
        host_load["load_1m"] / host_load["cpu_count"]
    )
    assert host_load["runnable_processes"] >= 0
    assert host_load["total_processes"] > 0

    monkeypatch.setattr(
        guard,
        "pynvml",
        fake_nvml(
            NVML_CLOCK_SM=1,
            nvmlDeviceGetClockInfo=lambda unused_handle, unused_clock: 2100,
            nvmlDeviceGetPowerUsage=lambda unused_handle: 275_500,
        ),
    )
    sampler = bare_sampler()
    assert sampler._optional_scalar(
        "nvmlDeviceGetClockInfo", argument_constant_name="NVML_CLOCK_SM"
    ) == {"status": "passed", "value": 2100.0}
    assert sampler._optional_scalar(
        "nvmlDeviceGetClockInfo", argument_constant_name="NVML_CLOCK_MEM"
    ) == {"status": "unavailable", "value": None}
    assert sampler._optional_scalar("nvmlDeviceGetPowerUsage", divisor=1000.0) == {
        "status": "passed",
        "value": 275.5,
    }

    def query_error(unused_handle: object) -> int:
        raise FakeNvmlError("fixture failure")

    monkeypatch.setattr(
        guard,
        "pynvml",
        fake_nvml(nvmlDeviceGetPowerUsage=query_error),
    )
    with pytest.raises(RuntimeError, match="PowerUsage query failed"):
        sampler._optional_scalar("nvmlDeviceGetPowerUsage")


def test_mps_query_selection_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported(unused_handle: object) -> list[object]:
        raise FakeNvmlNotSupported("fixture unsupported")

    monkeypatch.setattr(
        guard,
        "pynvml",
        fake_nvml(
            nvmlDeviceGetMPSComputeRunningProcesses_v3=unsupported,
            nvmlDeviceGetMPSComputeRunningProcesses_v2=lambda unused_handle: [
                fake_process(44, 3)
            ],
        ),
    )
    query, processes = bare_sampler()._mps_running_processes(daemon_present=True)
    assert query["status"] == "passed"
    assert query["function"] == "nvmlDeviceGetMPSComputeRunningProcesses_v2"
    assert query["attempts"] == [
        {
            "function": "nvmlDeviceGetMPSComputeRunningProcesses_v3",
            "status": "not_supported",
        }
    ]
    assert [(item["pid"], item["kind"]) for item in processes] == [(44, "mps_compute")]

    monkeypatch.setattr(
        guard,
        "pynvml",
        fake_nvml(
            nvmlDeviceGetMPSComputeRunningProcesses_v3=unsupported,
            nvmlDeviceGetMPSComputeRunningProcesses_v2=unsupported,
            nvmlDeviceGetMPSComputeRunningProcesses=unsupported,
        ),
    )
    with pytest.raises(RuntimeError, match="no usable MPS compute-client"):
        bare_sampler()._mps_running_processes(daemon_present=True)
    unavailable, clients = bare_sampler()._mps_running_processes(daemon_present=False)
    assert unavailable["status"] == "unavailable_without_mps_daemon"
    assert clients == []

    def query_error(unused_handle: object) -> list[object]:
        raise FakeNvmlError("fixture query failure")

    monkeypatch.setattr(
        guard,
        "pynvml",
        fake_nvml(nvmlDeviceGetMPSComputeRunningProcesses_v3=query_error),
    )
    with pytest.raises(RuntimeError, match="attribution failed"):
        bare_sampler()._mps_running_processes(daemon_present=True)
    error_without_daemon, clients = bare_sampler()._mps_running_processes(
        daemon_present=False
    )
    assert error_without_daemon["status"] == "query_error_without_mps_daemon"
    assert clients == []


def test_mps_client_attribution(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = {
        "compute_processes": [
            {
                "pid": 10,
                "process_name": "/usr/bin/nvidia-cuda-mps-server",
                "used_memory_mib": 52,
                "kind": "compute",
            }
        ],
        "graphics_processes": [],
        "mps_compute_processes": [
            {
                "pid": 11,
                "process_name": "/fixture/owned-client",
                "used_memory_mib": 10,
                "kind": "mps_compute",
            },
            {
                "pid": 12,
                "process_name": "/fixture/foreign-client",
                "used_memory_mib": 20,
                "kind": "mps_compute",
            },
        ],
    }
    assert [item["pid"] for item in guard.attributable_gpu_processes(gpu)] == [11, 12]
    monkeypatch.setattr(
        guard,
        "gpu_process_matches_workload",
        lambda process, unused_identity, unused_allowed=None: int(process["pid"]) == 11,
    )
    assert [
        item["pid"]
        for item in guard.external_gpu_processes(gpu, workload_identity=(100, 200))
    ] == [12]


def wait_for_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), f"timed out waiting for {path}"


def wait_for_nonempty_path(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while (
        not path.exists() or path.stat().st_size == 0
    ) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists() and path.stat().st_size > 0, f"timed out waiting for {path}"


def assert_terminal_evidence(output: Path, expected_status: str) -> dict[str, object]:
    report = json.loads(output.read_text())
    sample_log = output.with_name(output.stem + ".samples.jsonl")
    assert report["status"] == expected_status
    assert report["sample_log"]["bytes"] == sample_log.stat().st_size
    assert report["sample_log"]["sha256"] == guard.sha256_file(sample_log)
    lines = [json.loads(line) for line in sample_log.read_text().splitlines()]
    assert [line["sample_index"] for line in lines] == list(range(len(lines)))
    assert report["sample_count"] == len(lines)
    assert all(
        (line.get("host_load") or line.get("cpu", {}).get("host_load"))["cpu_count"] > 0
        for line in lines
        if line.get("sample_error") is None
    )
    return report


def test_clean_monitor_wrapper_has_two_validated_terminal_samples(
    tmp_path: Path,
) -> None:
    output = tmp_path / "clean-true-gpu-monitor.json"
    child_command = ["/bin/true"]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from test_refined_gpu_guards import controller_fixture_entry; "
                "controller_fixture_entry()"
            ),
            "monitor",
            str(output),
            *child_command,
        ],
        cwd=BUNDLE_ROOT,
        timeout=15.0,
        check=False,
    )
    assert completed.returncode == 0
    report = assert_terminal_evidence(output, "passed")
    assert report["post_exit_sample_indices"] == [
        report["sample_count"] - 2,
        report["sample_count"] - 1,
    ]
    terminal = [
        report["samples"][index] for index in report["post_exit_sample_indices"]
    ]
    assert [sample["post_exit_ordinal"] for sample in terminal] == [0, 1]
    assert all(sample["post_exit_telemetry"] is True for sample in terminal)
    gap = (terminal[1]["monotonic_ns"] - terminal[0]["monotonic_ns"]) / 1e9
    assert 0.19 <= gap <= 1.0

    runner = import_script(
        "fixture_clean_monitor_validator",
        "run_pynv_endpoint_high_concurrency_matrix_refined.py",
    )
    validated, sample_audit = runner.validate_monitor_evidence(
        output,
        expected_command=child_command,
        watchdog_pair=(3600.0, 120.0),
        conflicting_controller_roots=[FIXTURE_CONTROLLER_ROOT],
    )
    assert validated["status"] == "passed"
    assert sample_audit["sample_count"] == report["sample_count"]


def write_detached_workload_scripts(
    directory: Path, *, leader_lifetime_seconds: float
) -> tuple[Path, Path]:
    grandchild = directory / "e2e_rtx.py"
    grandchild.write_text(
        "import time\n"
        "try:\n"
        "    time.sleep(60)\n"
        "except KeyboardInterrupt:\n"
        "    pass\n"
    )
    pid_file = directory / "grandchild.pid"
    leader = directory / "leader.py"
    leader.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, {str(grandchild)!r}], "
        "start_new_session=True)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid) + '\\n')\n"
        f"time.sleep({leader_lifetime_seconds!r})\n"
    )
    return leader, pid_file


@pytest.mark.parametrize(
    "signum",
    [
        signal.SIGINT,
        signal.SIGTERM,
        *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else []),
    ],
)
def test_monitor_signal_terminal_evidence_and_detached_cleanup(
    tmp_path: Path, signum: int
) -> None:
    leader, pid_file = write_detached_workload_scripts(
        tmp_path, leader_lifetime_seconds=60.0
    )
    output = tmp_path / f"monitor-{signal.Signals(signum).name}.json"
    command = [
        sys.executable,
        "-c",
        (
            "from test_refined_gpu_guards import controller_fixture_entry; "
            "controller_fixture_entry()"
        ),
        "monitor",
        str(output),
        sys.executable,
        str(leader),
    ]
    controller = subprocess.Popen(command, cwd=BUNDLE_ROOT)
    try:
        wait_for_nonempty_path(pid_file)
        grandchild_pid = int(pid_file.read_text().strip())
        time.sleep(0.6)
        os.kill(controller.pid, signum)
        assert controller.wait(timeout=15.0) == 128 + signum
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5.0)
    report = assert_terminal_evidence(output, "interrupted")
    assert report["termination_signal"] == signal.Signals(signum).name
    assert report["timeout_cleanup"]["completed"] is True
    assert report["post_exit_sample_indices"]
    assert not guard.identity_still_matches(
        grandchild_pid,
        next(
            item["start_time_ticks"]
            for item in report["owned_process_registry"]
            if item["pid"] == grandchild_pid
        ),
    )


@pytest.mark.parametrize(
    "signum",
    [
        signal.SIGINT,
        signal.SIGTERM,
        *([signal.SIGHUP] if hasattr(signal, "SIGHUP") else []),
    ],
)
def test_gate_signal_terminal_evidence(tmp_path: Path, signum: int) -> None:
    output = tmp_path / f"gate-{signal.Signals(signum).name}.json"
    command = [
        sys.executable,
        "-c",
        (
            "from test_refined_gpu_guards import controller_fixture_entry; "
            "controller_fixture_entry()"
        ),
        "gate",
        str(output),
    ]
    controller = subprocess.Popen(command, cwd=BUNDLE_ROOT)
    try:
        wait_for_path(output)
        time.sleep(0.3)
        os.kill(controller.pid, signum)
        assert controller.wait(timeout=10.0) == 128 + signum
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait(timeout=5.0)
    report = assert_terminal_evidence(output, "interrupted")
    assert report["termination_signal"] == signal.Signals(signum).name
    assert report["configuration"]["idle_max_load_1m_per_cpu"] == 0.25


def test_natural_leader_exit_cleans_detached_session(tmp_path: Path) -> None:
    leader, pid_file = write_detached_workload_scripts(
        tmp_path, leader_lifetime_seconds=0.8
    )
    output = tmp_path / "monitor-natural-exit.json"
    command = [
        sys.executable,
        "-c",
        (
            "from test_refined_gpu_guards import controller_fixture_entry; "
            "controller_fixture_entry()"
        ),
        "monitor",
        str(output),
        sys.executable,
        str(leader),
    ]
    controller = subprocess.run(command, cwd=BUNDLE_ROOT, timeout=15.0, check=False)
    assert controller.returncode == 98
    grandchild_pid = int(pid_file.read_text().strip())
    report = assert_terminal_evidence(output, "monitor_error")
    assert any(
        item.get("reason") == "owned_process_survived_natural_harness_exit"
        for item in report["monitor_errors"]
    )
    assert report["timeout_cleanup"]["completed"] is True
    owned_grandchild = next(
        item
        for item in report["owned_process_registry"]
        if item["pid"] == grandchild_pid
    )
    assert not guard.identity_still_matches(
        grandchild_pid, owned_grandchild["start_time_ticks"]
    )


def test_identity_initialization_race_cleans_generic_adopted_child(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "generic-sleep.pid"
    output = tmp_path / "monitor-identity-race.json"
    leader_code = (
        "import subprocess; "
        "child=subprocess.Popen(['/usr/bin/sleep','60'],start_new_session=True); "
        f"open({str(pid_file)!r},'w').write(str(child.pid)+'\\n')"
    )
    command = [
        sys.executable,
        "-c",
        (
            "from test_refined_gpu_guards import controller_fixture_entry; "
            "controller_fixture_entry()"
        ),
        "monitor_identity_race",
        str(output),
        sys.executable,
        "-c",
        leader_code,
    ]
    environment = dict(os.environ)
    environment["PYNV_IDENTITY_RACE_PID_FILE"] = str(pid_file)
    completed = subprocess.run(
        command,
        cwd=BUNDLE_ROOT,
        env=environment,
        timeout=15.0,
        check=False,
    )
    assert completed.returncode == 98
    sleep_pid = int(pid_file.read_text().strip())
    report = assert_terminal_evidence(output, "monitor_error")
    assert report["post_popen_adopted_child_audit"]["ran"] is True
    assert (
        report["post_popen_adopted_child_audit"]["root_identity_initialized"] is False
    )
    probe_identity = report["probe_worker_excluded_identity"]
    assert probe_identity["pid"] == report["controller_pid"]
    assert all(
        (item["pid"], item["start_time_ticks"])
        != (probe_identity["pid"], probe_identity["start_time_ticks"])
        for item in report["owned_process_registry"]
    )
    owned_sleep = next(
        item for item in report["owned_process_registry"] if item["pid"] == sleep_pid
    )
    assert owned_sleep["comm"] == "sleep"
    assert owned_sleep["ownership_reason"] in {
        "owned_subreaper_adopted_process",
        "owned_process_descendant",
    }
    assert report["timeout_cleanup"]["completed"] is True
    assert not guard.identity_still_matches(sleep_pid, owned_sleep["start_time_ticks"])


def test_identity_race_preserves_distinct_live_probe_while_cleaning_child(
    tmp_path: Path,
) -> None:
    detached_pid_path = tmp_path / "generic-sleep.pid"
    probe_state_path = tmp_path / "distinct-probe-state.json"
    output = tmp_path / "monitor-identity-race-distinct-probe.json"
    leader_code = (
        "import subprocess; "
        "child=subprocess.Popen(['/usr/bin/sleep','60'],start_new_session=True); "
        f"open({str(detached_pid_path)!r},'w').write(str(child.pid)+'\\n')"
    )
    command = [
        sys.executable,
        "-c",
        (
            "from test_refined_gpu_guards import controller_fixture_entry; "
            "controller_fixture_entry()"
        ),
        "monitor_identity_race_distinct_probe",
        str(output),
        sys.executable,
        "-c",
        leader_code,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "PYNV_IDENTITY_RACE_PID_FILE": str(detached_pid_path),
            "PYNV_DISTINCT_PROBE_STATE_FILE": str(probe_state_path),
        }
    )
    completed = subprocess.run(
        command,
        cwd=BUNDLE_ROOT,
        env=environment,
        timeout=15.0,
        check=False,
    )
    assert completed.returncode == 98
    report = assert_terminal_evidence(output, "monitor_error")
    probe_state = json.loads(probe_state_path.read_text())
    probe_identity = report["probe_worker_excluded_identity"]
    assert probe_state["alive_before_close"] is True
    assert (probe_state["pid"], probe_state["start_time_ticks"]) == (
        probe_identity["pid"],
        probe_identity["start_time_ticks"],
    )
    assert probe_identity["pid"] != report["controller_pid"]
    assert all(
        (item["pid"], item["start_time_ticks"])
        != (probe_identity["pid"], probe_identity["start_time_ticks"])
        for item in report["owned_process_registry"]
    )

    detached_pid = int(detached_pid_path.read_text().strip())
    owned_detached = next(
        item for item in report["owned_process_registry"] if item["pid"] == detached_pid
    )
    assert owned_detached["comm"] == "sleep"
    assert report["timeout_cleanup"]["completed"] is True
    assert not guard.identity_still_matches(
        detached_pid, owned_detached["start_time_ticks"]
    )
    assert not guard.identity_still_matches(
        probe_state["pid"], probe_state["start_time_ticks"]
    )


if __name__ == "__main__":
    main()
