# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import pynvml

    del pynvml
except ModuleNotFoundError:
    sys.modules["pynvml"] = types.ModuleType("pynvml")

BUNDLE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return load(
        "rtx_endpoint_runner", "run_pynv_endpoint_high_concurrency_matrix_refined.py"
    )


@pytest.fixture(scope="module")
def harness():
    return load("rtx_persistent_harness", "benchmark_pynvvideocodec_e2e_persistent.py")


@pytest.fixture(scope="module")
def preflight():
    return load(
        "rtx_endpoint_preflight_contract",
        "run_pynv_endpoint_persistent_preflight.py",
    )


@pytest.fixture(scope="module")
def monitor():
    return load("rtx_refined_monitor_contract", "run_with_gpu_monitor_refined.py")


def test_schedule_and_endpoint_command_contract(
    runner, harness, tmp_path: Path
) -> None:
    assert runner.CAMPAIGN_HARNESS_SHA256 == runner.sha256_file(
        BUNDLE / "benchmark_pynvvideocodec_e2e_persistent.py"
    )
    assert len(runner.SCHEDULE) == 6
    assert sum(len(variants) for _, _, variants in runner.SCHEDULE) == 12
    assert [variants for _, _, variants in runner.SCHEDULE] == [
        ["upstream", "pr-head"],
        ["pr-head", "upstream"],
        ["upstream", "pr-head"],
        ["pr-head", "upstream"],
        ["upstream", "pr-head"],
        ["pr-head", "upstream"],
    ]
    for concurrency in (8, 16, 32):
        positions = [order.index(concurrency) for _, order, _ in runner.SCHEDULE]
        assert sorted(positions) == [0, 0, 1, 1, 2, 2]

    args = SimpleNamespace(
        python=Path("/venv/bin/python"),
        harness=BUNDLE / "benchmark_pynvvideocodec_e2e_persistent.py",
        root=Path("/source"),
        transformers_root=Path("/overlay"),
        corpus=Path("/corpus"),
        port=18600,
    )
    videos = [Path(f"/corpus/traffic1080-{index:02d}.mp4") for index in range(8)]
    for variant, expected_backend, expected_server_argv in (
        ("upstream", {"hw_decoders": 2}, ["--no-mm-device-do-normalize"]),
        (
            "pr-head",
            {"hw_decoders": 2, "output_layout": "tchw"},
            ["--mm-device-do-normalize"],
        ),
    ):
        command = runner.build_harness_command(
            args,
            variant=variant,
            result_path=tmp_path / f"{variant}.json",
            videos=videos,
            concurrencies=[8, 16, 32],
        )
        parsed = harness.parse_args(command[2:])
        assert parsed.variant == variant
        assert parsed.backend == "pynvvideocodec"
        assert parsed.backend_kwargs == expected_backend
        assert parsed.server_arg == expected_server_argv
        assert parsed.concurrency == [8, 16, 32]
        assert parsed.warmup_requests == 1
        assert parsed.warmup_requests_by_concurrency == {"8": 24, "16": 48, "32": 96}
        assert parsed.requests_by_concurrency == {"8": 64, "16": 128, "32": 256}
        assert parsed.video_pixel_budget == (1024, 576)
        assert parsed.frames == 32
        assert parsed.output_len == 32
        assert parsed.max_num_seqs == 32
        assert parsed.max_num_batched_tokens == 9216
        assert parsed.kv_cache_memory_bytes == 40 * 1024**3
        assert parsed.settle_seconds == 1.0
        assert parsed.video == videos


def test_direct_actual_head_preflight_has_no_legacy_endpoint_dependency(
    preflight,
) -> None:
    assert preflight.PILOT_VARIANTS == ("upstream", "pr-head")
    source = (BUNDLE / "run_pynv_endpoint_persistent_preflight.py").read_text()
    for forbidden in (
        "pr-base-",
        "final-",
        "baseline-",
        "validate_runtime_equivalence_legacy",
        "superseded-results",
        "benchmark-methodology-runner",
    ):
        assert forbidden not in source


def test_all_monitor_invocations_use_approved_watchdog_pairs(
    runner, preflight, monitor, tmp_path: Path
) -> None:
    expected = {(1200.0, 120.0), (3600.0, 120.0)}
    assert set(monitor.APPROVED_WATCHDOG_PAIRS) == expected
    assert set(runner.APPROVED_MONITOR_WATCHDOG_PAIRS) == expected
    used = {
        runner.TIMING_MONITOR_WATCHDOG_PAIR,
        preflight.PIXEL_MONITOR_WATCHDOG_PAIR,
        preflight.PILOT_MONITOR_WATCHDOG_PAIR,
    }
    assert used == expected
    for pair in used:
        assert monitor.validate_watchdog_pair(*pair) == pair
        command = runner.build_monitored_command(
            python=Path("/venv/bin/python"),
            monitor=Path("/assets/run_with_gpu_monitor_refined.py"),
            output=tmp_path / f"monitor-{int(pair[0])}.json",
            child_command=["/venv/bin/python", "workload.py"],
            watchdog_pair=pair,
            conflicting_controller_roots=[Path("/fixture/controllers")],
        )
        assert command[command.index("--timeout-seconds") + 1] == f"{pair[0]:g}"
        assert command[command.index("--timeout-grace-seconds") + 1] == (f"{pair[1]:g}")
    with pytest.raises(ValueError, match="not approved"):
        monitor.validate_watchdog_pair(1800, 120)
    with pytest.raises(ValueError, match="unapproved"):
        runner.build_monitored_command(
            python=Path("/venv/bin/python"),
            monitor=Path("/assets/run_with_gpu_monitor_refined.py"),
            output=tmp_path / "bad.json",
            child_command=["workload"],
            watchdog_pair=(1800, 120),
            conflicting_controller_roots=[Path("/fixture/controllers")],
        )
    assert (
        "--timeout-seconds"
        not in (BUNDLE / "run_pynv_endpoint_persistent_preflight.py").read_text()
    )
    assert (
        BUNDLE / "run_pynv_endpoint_high_concurrency_matrix_refined.py"
    ).read_text().count('"--timeout-seconds"') == 1


def test_fixed_cell_idle_gate_contract(runner, tmp_path: Path) -> None:
    assert runner.validate_cell_idle_pair(30, 1800) == (30.0, 1800.0)
    for pair in ((0.001, 1), (30, 7200), (1200, 21600)):
        with pytest.raises(ValueError, match="exactly"):
            runner.validate_cell_idle_pair(*pair)
    command = runner.build_idle_gate_command(
        python=Path("/venv/bin/python"),
        idle_gate=Path("/assets/wait_for_exclusive_gpu_refined.py"),
        output=tmp_path / "cell-idle-gate.json",
        seconds=30,
        timeout=1800,
        conflicting_controller_roots=[Path("/fixture/controllers")],
    )
    assert command[command.index("--seconds") + 1] == "30"
    assert command[command.index("--timeout") + 1] == "1800"
    assert command[command.index("--conflicting-controller-root") + 1] == (
        "/fixture/controllers"
    )


def make_contamination_retry_fixture(runner, tmp_path: Path, name: str):
    report_path = tmp_path / f"{name}-gpu-monitor.json"
    sample_path = tmp_path / f"{name}-gpu-monitor.samples.jsonl"
    monitor_script = BUNDLE / "run_with_gpu_monitor_refined.py"
    child_command = [sys.executable, "fixture-child.py"]
    roots = [tmp_path / "controller-root"]
    wrapper_command = runner.build_monitored_command(
        python=Path(sys.executable),
        monitor=monitor_script,
        output=report_path,
        child_command=child_command,
        watchdog_pair=(3600.0, 120.0),
        conflicting_controller_roots=roots,
    )
    app = {
        "pid": 424242,
        "process_name": "/fixture/foreign",
        "used_memory_mib": 64,
        "kind": "compute",
    }
    sample = {
        "sample_index": 0,
        "utc": "fixture",
        "time_ns": 1,
        "monotonic_ns": 1,
        "sample_gap_seconds": None,
        "external_gpu_processes": [app],
        "cpu_conflicts": [],
        "monitor_errors": ["gpu_compute_process"],
    }
    sample_path.write_text(json.dumps(sample, separators=(",", ":")) + "\n")
    event = runner._expected_foreign_event_for_sample(
        sample, initial_contamination=True
    )
    report = {
        "status": "contaminated",
        "contaminated": True,
        "timed_out": False,
        "returncode": None,
        "command": child_command,
        "timeout_seconds": 3600.0,
        "timeout_grace_seconds": 120.0,
        "monitor_errors": [],
        "foreign_events": [event],
        "configuration": {
            "sample_interval_seconds": 0.2,
            "maximum_sample_gap_seconds": 1.0,
            "initial_idle_memory_ceiling_mib": 1024,
            "device_index": 0,
            "conflicting_controller_roots": [str(path.resolve()) for path in roots],
            "telemetry": "direct NVML; no external telemetry commands or pgrep",
            "workload_ownership": (
                "new process session/group plus PID/start_ticks ancestry"
            ),
        },
        "process": {
            "executable": sys.executable,
            "argv": wrapper_command[1:],
            "script_path": str(monitor_script),
            "script_sha256": runner.sha256_file(monitor_script),
        },
        "guard_helper": {
            "path": str(BUNDLE / "pynv_gpu_guard.py"),
            "sha256": runner.sha256_file(BUNDLE / "pynv_gpu_guard.py"),
        },
        "sample_count": 1,
        "samples": [sample],
        "sample_log": {
            "path": str(sample_path),
            "bytes": sample_path.stat().st_size,
            "sha256": runner.sha256_file(sample_path),
        },
    }
    runner.write_json(report_path, report)
    return report_path, sample_path, report, wrapper_command, child_command, roots


def test_strict_contamination_retry_requires_rehashed_telemetry(
    runner, tmp_path: Path
) -> None:
    fixture = make_contamination_retry_fixture(runner, tmp_path, "valid")
    report_path, _sample_path, _report, wrapper, child, roots = fixture
    audit = runner.validate_contamination_retry_evidence(
        wrapper_returncode=99,
        report_path=report_path,
        expected_wrapper_command=wrapper,
        expected_child_command=child,
        watchdog_pair=(3600.0, 120.0),
        conflicting_controller_roots=roots,
    )
    assert audit["status"] == "validated_contamination_retry"
    assert audit["foreign_event_sample_indices"] == [0]

    for index, mutation in enumerate(
        (
            lambda report: report.update(status="passed", contaminated=False),
            lambda report: report.update(foreign_events=[]),
            lambda report: report["foreign_events"][0]["apps"][0].update(pid=7),
            lambda report: report["guard_helper"].update(sha256="0" * 64),
            lambda report: report["process"].update(script_sha256="0" * 64),
            lambda report: report["configuration"].update(device_index=1),
            lambda report: report.update(command=["/bin/true"]),
        )
    ):
        report_path, _sample_path, report, wrapper, child, roots = (
            make_contamination_retry_fixture(runner, tmp_path, f"bad-{index}")
        )
        mutation(report)
        runner.write_json(report_path, report)
        with pytest.raises(RuntimeError):
            runner.validate_contamination_retry_evidence(
                wrapper_returncode=99,
                report_path=report_path,
                expected_wrapper_command=wrapper,
                expected_child_command=child,
                watchdog_pair=(3600.0, 120.0),
                conflicting_controller_roots=roots,
            )

    report_path, sample_path, _report, wrapper, child, roots = (
        make_contamination_retry_fixture(runner, tmp_path, "changed-sample")
    )
    sample_path.write_text(sample_path.read_text() + "{}\n")
    with pytest.raises(RuntimeError, match="hash/size"):
        runner.validate_contamination_retry_evidence(
            wrapper_returncode=99,
            report_path=report_path,
            expected_wrapper_command=wrapper,
            expected_child_command=child,
            watchdog_pair=(3600.0, 120.0),
            conflicting_controller_roots=roots,
        )

    report_path, sample_path, _report, wrapper, child, roots = (
        make_contamination_retry_fixture(runner, tmp_path, "missing-sample")
    )
    sample_path.unlink()
    with pytest.raises(RuntimeError, match="sample-log path"):
        runner.validate_contamination_retry_evidence(
            wrapper_returncode=99,
            report_path=report_path,
            expected_wrapper_command=wrapper,
            expected_child_command=child,
            watchdog_pair=(3600.0, 120.0),
            conflicting_controller_roots=roots,
        )


def test_clean_child_exit_99_is_not_retryable(runner, tmp_path: Path) -> None:
    report_path, _sample_path, report, wrapper, child, roots = (
        make_contamination_retry_fixture(runner, tmp_path, "clean-99")
    )
    report.update(
        status="passed",
        contaminated=False,
        returncode=99,
        foreign_events=[],
    )
    runner.write_json(report_path, report)
    with pytest.raises(RuntimeError, match="not strict contamination"):
        runner.validate_contamination_retry_evidence(
            wrapper_returncode=99,
            report_path=report_path,
            expected_wrapper_command=wrapper,
            expected_child_command=child,
            watchdog_pair=(3600.0, 120.0),
            conflicting_controller_roots=roots,
        )


def make_launched_contamination_retry_fixture(
    runner, tmp_path: Path, name: str, *, phase: str
):
    report_path, sample_path, report, wrapper, child, roots = (
        make_contamination_retry_fixture(runner, tmp_path, name)
    )
    foreign_app = report["samples"][0]["external_gpu_processes"][0]
    samples = [
        {
            "sample_index": 0,
            "utc": "fixture-0",
            "time_ns": 1_000_000_000,
            "monotonic_ns": 1_000_000_000,
            "sample_gap_seconds": None,
            "external_gpu_processes": ([foreign_app] if phase == "in_flight" else []),
            "cpu_conflicts": [],
            "monitor_errors": [],
        },
        {
            "sample_index": 1,
            "utc": "fixture-1",
            "time_ns": 1_100_000_000,
            "monotonic_ns": 1_100_000_000,
            "sample_gap_seconds": 0.1,
            "external_gpu_processes": ([foreign_app] if phase == "post_exit" else []),
            "cpu_conflicts": [],
            "monitor_errors": [],
            "post_exit_telemetry": True,
            "post_exit_ordinal": 0,
        },
        {
            "sample_index": 2,
            "utc": "fixture-2",
            "time_ns": 1_300_000_000,
            "monotonic_ns": 1_300_000_000,
            "sample_gap_seconds": 0.2,
            "external_gpu_processes": [],
            "cpu_conflicts": [],
            "monitor_errors": [],
            "post_exit_telemetry": True,
            "post_exit_ordinal": 1,
        },
    ]
    events = [
        runner._expected_foreign_event_for_sample(
            samples[0 if phase == "in_flight" else 1],
            initial_contamination=False,
        )
    ]
    report.update(
        returncode=130 if phase == "in_flight" else 0,
        command_pid=31337,
        workload_identity={"pid": 31337, "start_ticks": 99},
        samples=samples,
        sample_count=len(samples),
        post_exit_sample_indices=[1, 2],
        foreign_events=events,
        post_popen_adopted_child_audit={
            "ran": True,
            "capture_errors": [],
            "identity_errors": [],
            "survivors_before_cleanup": [],
            "process_groups_alive_before_cleanup": [],
        },
        timeout_cleanup=(
            {
                "reason": "foreign_workload_detected",
                "completed": True,
                "signal_actions": [{"signal": "SIGINT", "signal_sent": True}],
            }
            if phase == "in_flight"
            else None
        ),
    )
    sample_path.write_text(
        "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample in samples)
    )
    report["sample_log"] = {
        "path": str(sample_path),
        "bytes": sample_path.stat().st_size,
        "sha256": runner.sha256_file(sample_path),
    }
    runner.write_json(report_path, report)
    return report_path, report, wrapper, child, roots


@pytest.mark.parametrize(
    ("phase", "semantics"),
    (
        ("in_flight", "terminated_by_monitor_for_foreign_workload"),
        ("post_exit", "successful_child_then_post_exit_contamination"),
    ),
)
def test_launched_contamination_retry_requires_consistent_child_semantics(
    runner, tmp_path: Path, phase: str, semantics: str
) -> None:
    report_path, report, wrapper, child, roots = (
        make_launched_contamination_retry_fixture(
            runner, tmp_path, f"launched-{phase}", phase=phase
        )
    )
    audit = runner.validate_contamination_retry_evidence(
        wrapper_returncode=99,
        report_path=report_path,
        expected_wrapper_command=wrapper,
        expected_child_command=child,
        watchdog_pair=(3600.0, 120.0),
        conflicting_controller_roots=roots,
    )
    assert audit["child_semantics"] == semantics
    assert audit["terminal_post_exit_samples"]["ordinals"] == [0, 1]

    report["returncode"] = 0 if phase == "in_flight" else 130
    runner.write_json(report_path, report)
    with pytest.raises(RuntimeError, match="child"):
        runner.validate_contamination_retry_evidence(
            wrapper_returncode=99,
            report_path=report_path,
            expected_wrapper_command=wrapper,
            expected_child_command=child,
            watchdog_pair=(3600.0, 120.0),
            conflicting_controller_roots=roots,
        )


def test_monitor_coverage_and_vram_use_only_monotonic_time(runner) -> None:
    start = 10_000_000_000
    finish = 12_000_000_000
    samples = [
        {
            "monotonic_ns": start,
            "time_ns": 9_000_000_000_000,
            "memory_used_mib": 100,
            "utilization_percent": 0,
            "utc": "2099-01-01T00:00:00+00:00",
            "compute_apps": [],
        },
        {
            "monotonic_ns": 11_000_000_000,
            "time_ns": 1,
            "memory_used_mib": 300,
            "utilization_percent": 20,
            "utc": "1970-01-01T00:00:00+00:00",
            "compute_apps": [],
        },
        {
            "monotonic_ns": finish,
            "time_ns": 8_000_000_000_000,
            "memory_used_mib": 200,
            "utilization_percent": 10,
            "utc": "2080-01-01T00:00:00+00:00",
            "compute_apps": [],
        },
    ]
    block = {
        "concurrency": 8,
        "measured": {
            "started_monotonic_ns": start,
            "finished_monotonic_ns": finish,
            "started_at": "2099-01-01T00:00:00+00:00",
            "finished_at": "1970-01-01T00:00:00+00:00",
        },
    }
    monitor = {"samples": samples}
    coverage = runner.monitor_coverage_audit({"concurrency_blocks": [block]}, monitor)
    assert coverage["passed"] is True
    assert coverage["blocks"][0]["sample_count"] == 3
    assert coverage["blocks"][0]["maximum_boundary_inclusive_gap_seconds"] == 1.0
    vram = runner.measured_window_vram(monitor, block)
    assert vram["sample_count"] == 3
    assert vram["peak_total_gpu_memory_used_mib"] == 300


def make_token_batch(runner):
    payload = {"model": "fixture", "messages": [{"role": "user"}]}
    prompt_ids = [1, 2, 3]
    completion_ids = [4, 5]
    raw_response = {
        "id": "response-1",
        "model": "fixture",
        "prompt_token_ids": prompt_ids,
        "choices": [
            {
                "token_ids": completion_ids,
                "message": {"content": "caption", "reasoning": None},
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
    }
    response = {
        "id": "response-1",
        "model": "fixture",
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "prompt_token_count": len(prompt_ids),
        "completion_token_count": len(completion_ids),
        "prompt_token_ids_sha256": runner.sha256_json(prompt_ids),
        "completion_token_ids_sha256": runner.sha256_json(completion_ids),
        "prompt_and_completion_token_ids_sha256": runner.sha256_json(
            {"prompt": prompt_ids, "completion": completion_ids}
        ),
        "text": "caption",
        "text_sha256": runner.sha256_json("caption"),
        "reasoning_content": None,
        "reasoning_content_sha256": runner.sha256_json(None),
        "finish_reason": "length",
        "stop_reason": None,
        "raw_response": raw_response,
        "raw_response_sha256": runner.sha256_json(raw_response),
    }
    record = {
        "status": "passed",
        "request_index": 0,
        "video_index": 0,
        "video_path": "/fixture/video.mp4",
        "video_sha256": "f" * 64,
        "payload": payload,
        "request_payload_sha256": runner.sha256_json(payload),
        "response": response,
    }
    fingerprint = {
        "request_index": 0,
        "video_index": 0,
        "video_path": "/fixture/video.mp4",
        "prompt_token_ids_sha256": response["prompt_token_ids_sha256"],
        "completion_token_ids_sha256": response["completion_token_ids_sha256"],
        "prompt_and_completion_token_ids_sha256": response[
            "prompt_and_completion_token_ids_sha256"
        ],
    }
    aggregate = {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(completion_ids),
        "all_tokens": len(prompt_ids) + len(completion_ids),
        "response_token_fingerprints_by_request": [fingerprint],
        "ordered_response_token_fingerprints_sha256": runner.sha256_json([fingerprint]),
        "ordered_response_token_ids_sha256": runner.sha256_json(
            [
                {
                    "prompt": response["prompt_token_ids_sha256"],
                    "completion": response["completion_token_ids_sha256"],
                }
            ]
        ),
        "completion_token_ids_sha256_counts": {
            response["completion_token_ids_sha256"]: 1
        },
    }
    return {"records": [record], "aggregate": aggregate}


def make_validate_result_fixture(
    runner, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "source"
    transformers_root = tmp_path / "overlay"
    corpus = tmp_path / "corpus"
    source_root.joinpath("vllm").mkdir(parents=True)
    transformers_root.joinpath("transformers").mkdir(parents=True)
    corpus.mkdir()
    source_root.joinpath("vllm/__init__.py").write_text("fixture\n")
    transformers_init = transformers_root / "transformers/__init__.py"
    transformers_init.write_text("fixture transformers\n")
    monkeypatch.setattr(
        runner, "TRANSFORMERS_INIT_SHA256", runner.sha256_file(transformers_init)
    )
    video_bytes = b"fixture-video"
    monkeypatch.setattr(runner, "VIDEO_BYTES", len(video_bytes))
    monkeypatch.setattr(runner, "VIDEO_SHA256", hashlib.sha256(video_bytes).hexdigest())
    videos = []
    for index in range(8):
        path = corpus / f"traffic1080-{index:02d}.mp4"
        path.write_bytes(video_bytes)
        path_stat = path.stat()
        videos.append(
            {
                "video_index": index,
                "path": str(path),
                "file_uri": path.resolve().as_uri(),
                "bytes": len(video_bytes),
                "sha256": runner.VIDEO_SHA256,
                "device": path_stat.st_dev,
                "inode": path_stat.st_ino,
                "mtime_ns": path_stat.st_mtime_ns,
                "probe": {"width": 1920, "height": 1080, "frame_count": 914},
            }
        )

    pynv_root = tmp_path / "PyNvVideoCodec"
    pynv_root.mkdir()
    pynv_artifacts = {}
    runtime_artifacts = []
    for index, name in enumerate(("PyNvVideoCodec.py", "_PyNvVideoCodec.so")):
        path = pynv_root / name
        path.write_bytes(f"pynv-{index}".encode())
        digest = runner.sha256_file(path)
        pynv_artifacts[name] = digest
        runtime_artifacts.append(
            {
                "path": str(path),
                "resolved_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    monkeypatch.setattr(runner, "PYNV_RUNTIME_ARTIFACT_SHA256", pynv_artifacts)
    runtime_artifacts.append(
        {
            "path": str(transformers_init),
            "resolved_path": str(transformers_init.resolve()),
            "bytes": transformers_init.stat().st_size,
            "sha256": runner.sha256_file(transformers_init),
        }
    )

    prompt_ids = [1, 2, 3]
    monkeypatch.setattr(runner, "EXPECTED_PROMPT_TOKENS", len(prompt_ids))
    monkeypatch.setattr(
        runner, "EXPECTED_PROMPT_TOKEN_IDS_SHA256", runner.sha256_json(prompt_ids)
    )
    fixture_backend_kwargs = runner.variant_backend_kwargs("upstream")
    fixture_media_kwargs = {
        "video_backend": "qwen3_vl",
        "min_frames": 32,
        "max_frames": 32,
        "backend": "pynvvideocodec",
        **fixture_backend_kwargs,
    }

    def phase_batch(phase: str, start: int, finish: int, global_index: int):
        batch = make_token_batch(runner)
        batch.update(
            {
                "status": "passed",
                "started_at": f"fixture-{phase}-start",
                "finished_at": f"fixture-{phase}-finish",
                "started_monotonic_ns": start,
                "finished_monotonic_ns": finish,
                "measured_window_seconds": (finish - start) / 1e9,
                "requested_concurrency": 1,
                "effective_client_workers": 1,
            }
        )
        record = batch["records"][0]
        payload = runner.expected_chat_payload(videos[0]["file_uri"])
        completion_ids = list(range(4, 4 + runner.OUTPUT_LENGTH))
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_ids) + len(completion_ids),
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
        raw_response = {
            "id": f"chatcmpl-fixture-{global_index}",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": runner.SERVED_MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "token_ids": completion_ids,
                    "message": {
                        "role": "assistant",
                        "content": "caption",
                        "refusal": None,
                        "annotations": None,
                        "audio": None,
                        "function_call": None,
                        "reasoning": None,
                    },
                    "logprobs": None,
                    "finish_reason": "length",
                    "stop_reason": None,
                    "routed_experts": None,
                }
            ],
            "service_tier": None,
            "system_fingerprint": None,
            "usage": usage,
            "prompt_logprobs": None,
            "prompt_token_ids": prompt_ids,
            "prompt_text": None,
            "kv_transfer_params": None,
            "ec_transfer_params": None,
            "metrics": None,
        }
        response = {
            "id": f"chatcmpl-fixture-{global_index}",
            "model": runner.SERVED_MODEL_NAME,
            "prompt_token_ids": prompt_ids,
            "completion_token_ids": completion_ids,
            "prompt_token_count": len(prompt_ids),
            "completion_token_count": len(completion_ids),
            "prompt_token_ids_sha256": runner.sha256_json(prompt_ids),
            "completion_token_ids_sha256": runner.sha256_json(completion_ids),
            "prompt_and_completion_token_ids_sha256": runner.sha256_json(
                {"prompt": prompt_ids, "completion": completion_ids}
            ),
            "text": "caption",
            "text_sha256": runner.sha256_json("caption"),
            "reasoning_content": None,
            "reasoning_content_sha256": runner.sha256_json(None),
            "finish_reason": "length",
            "stop_reason": None,
            "usage": usage,
            "server_metrics": None,
            "raw_response": raw_response,
            "raw_response_sha256": runner.sha256_json(raw_response),
        }

        record.update(
            {
                "phase": phase,
                "block_index": 0,
                "concurrency": 1,
                "request_index": 0,
                "global_request_index": global_index,
                "video_index": 0,
                "video_path": videos[0]["path"],
                "video_file_uri": videos[0]["file_uri"],
                "video_sha256": runner.VIDEO_SHA256,
                "video_work": runner.independently_expected_video_work(
                    videos[0], fixture_media_kwargs
                ),
                "payload": payload,
                "request_payload_sha256": runner.sha256_json(payload),
                "status": "passed",
                "http_status": 200,
                "response": response,
                "started_at": "2023-11-14T22:13:19.900000+00:00",
                "finished_at": "2023-11-14T22:13:20.100000+00:00",
                "started_monotonic_ns": start,
                "finished_monotonic_ns": finish,
                "start_offset_seconds": 0.0,
                "finish_offset_seconds": (finish - start) / 1e9,
                "latency_seconds": (finish - start) / 1e9,
                "latency_ms": (finish - start) / 1e6,
                "transport": {
                    "pool_slot_id": 0,
                    "phase": phase,
                    "seeded_first_wave": True,
                    "connection_generation": 1,
                    "request_ordinal_on_generation": 1 if phase == "warmup" else 2,
                    "connection_reused": phase == "measured",
                    "prewarmed_for_measurement": phase == "measured",
                    "request_connection_header": "keep-alive",
                    "response_http_version": 11,
                    "response_connection_header": None,
                    "response_will_close": False,
                    "response_persistent": True,
                },
            }
        )
        batch["aggregate"] = harness.batch_aggregate(
            batch["records"], (finish - start) / 1e9
        )
        cumulative_requests = 1 if phase == "warmup" else 2
        batch["aggregate"]["persistent_transport_audit"] = {
            "status": "passed",
            "phase": phase,
            "pool_size": 1,
            "request_count": 1,
            "used_slot_ids": [0],
            "seeded_first_wave_request_to_slot": {"0": 0},
            "reasons": [],
            "counts_at_phase_end": {
                "open_count": 1,
                "reuse_count": cumulative_requests - 1,
                "close_count": 0,
            },
            "slot_snapshots_at_phase_end": [
                {
                    "slot_id": 0,
                    "current_generation": 1,
                    "warmed_generation": 1,
                    "request_ordinal_on_current_generation": cumulative_requests,
                    "open_count": 1,
                    "reuse_count": cumulative_requests - 1,
                    "close_count": 0,
                    "close_reasons": {},
                    "currently_open": True,
                }
            ],
        }
        return batch

    warmup = phase_batch("warmup", 100, 200, 0)
    measured = phase_batch("measured", 300, 400, 1)
    block = {
        "status": "passed",
        "block_index": 0,
        "concurrency": 1,
        "requested_warmup_requests": 1,
        "effective_warmup_requests": 1,
        "requested_measured_requests": 1,
        "warmup": warmup,
        "measured": measured,
        "persistent_http_pool": {
            "implementation": "stdlib http.client.HTTPConnection HTTP/1.1",
            "pool_size": 1,
            "connection_scope": "one pool per concurrency block",
            "phase_scope": "same slots span warmup, settle, and measured phases",
            "request_streaming": False,
            "request_retry_count": 0,
            "closed": True,
            "counts": {"open_count": 1, "reuse_count": 1, "close_count": 1},
            "slots": [
                {
                    "slot_id": 0,
                    "current_generation": 1,
                    "warmed_generation": 1,
                    "request_ordinal_on_current_generation": 2,
                    "open_count": 1,
                    "reuse_count": 1,
                    "close_count": 1,
                    "currently_open": False,
                    "close_reasons": {"pool_close": 1},
                }
            ],
            "phase_audits": {
                "warmup": warmup["aggregate"]["persistent_transport_audit"],
                "measured": measured["aggregate"]["persistent_transport_audit"],
            },
        },
        "aggregate": copy.deepcopy(measured["aggregate"]),
    }
    server_log_path = tmp_path / "cell.server.log"
    server_log_path.write_text(
        "GPU KV cache size: 336,560 tokens, Maximum concurrency for "
        "32,768 tokens per request: 10.27x\n"
    )
    harness_path = BUNDLE / "benchmark_pynvvideocodec_e2e_persistent.py"
    backend_kwargs = runner.variant_backend_kwargs("upstream")
    server_argv = runner.variant_server_argv("upstream")
    media_kwargs = {
        "video_backend": "qwen3_vl",
        "min_frames": 32,
        "max_frames": 32,
        "backend": "pynvvideocodec",
        **backend_kwargs,
    }
    result = {
        "schema": "vllm-qwen3-vl-video-e2e-throughput-v3-persistent-http",
        "status": "passed",
        "configuration": {
            "variant": "upstream",
            "model": runner.MODEL,
            "revision": runner.REVISION,
            "prompt": runner.PROMPT,
            "prompt_sha256": runner.sha256_json(runner.PROMPT),
            "output_len": 32,
            "frame_target": 32,
            "warmup_requests_by_concurrency": [
                {"concurrency": 1, "requested": 1, "effective": 1}
            ],
            "measured_requests_per_concurrency": [{"concurrency": 1, "requests": 1}],
            "concurrency_order": [1],
            "max_num_seqs": 32,
            "max_num_batched_tokens": 9216,
            "kv_cache_memory_bytes": runner.KV_CACHE_MEMORY_BYTES,
            "mm_ipc_gpu_memory_gb": 2.0,
            "backend_argument": "pynvvideocodec",
            "backend_kwargs": backend_kwargs,
            "extra_server_argv": server_argv,
            "dtype": "bfloat16",
            "seed": 0,
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
            "mm_processor_cache_gb": 0,
            "prefix_caching": False,
            "gpu_memory_utilization": None,
            "request_media_io_kwargs": {},
            "server_mm_processor_kwargs": {"max_pixels": runner.TOTAL_MAX_PIXELS},
            "server_limit_mm_per_prompt": {"image": 0, "video": 1},
            "request_timeout_seconds": 1200.0,
            "startup_timeout_seconds": 600.0,
            "shutdown_timeout_seconds": 60.0,
            "video_count": 8,
            "video_cycle_policy": (
                "video_index = phase-local request_index modulo video count; "
                "reset for each warmup and measured batch"
            ),
            "parity_reference": None,
            "pythonpath_extra": [str(transformers_root)],
            "allowed_local_media_path": str(corpus),
            "client_http_protocol": {
                "implementation": "stdlib http.client.HTTPConnection",
                "http_version": "HTTP/1.1",
                "streaming": False,
                "request_connection_header": "keep-alive",
                "pool_size": "exact requested concurrency for each block",
                "pool_lifetime": "warmup through settle and measurement; then close",
                "slot_seeding": (
                    "the first wave leases every slot exactly once before any slot "
                    "can be reused"
                ),
                "request_retries": 0,
                "measured_connection_requirement": (
                    "same successful warmup generation, reused and persistent"
                ),
            },
            "video_pixel_budget": {
                "reference_width": 1024,
                "reference_height": 576,
                "max_pixels_per_sampled_frame": 1024 * 576,
                "sampled_frames": 32,
                "max_pixels_total": runner.TOTAL_MAX_PIXELS,
            },
            "server_media_io_kwargs": {"video": media_kwargs},
            "video_kwargs_for_metric_derivation": media_kwargs,
            "video_kwargs_for_metric_derivation_unavailable_reason": None,
        },
        "provenance": {
            "source": {
                "root": str(source_root),
                "commit": runner.COMMITS["upstream"],
                "tree": runner.TREES["upstream"],
                "tracked_diff_bytes": 0,
                "untracked_files": [],
            },
            "harness": {
                "path": str(harness_path),
                "sha256": runner.sha256_file(harness_path),
            },
            "hardware": {
                "cuda_visible_devices": "0",
                "nvidia_smi_output": (
                    "0, NVIDIA RTX PRO 6000, GPU-fixture, 999.0, 98304 MiB, "
                    "12.0, 0000:01:00.0, P0, 2100 MHz, 1593 MHz"
                ),
            },
            "python": {
                "packages": {"PyNvVideoCodec": "2.0.4", "transformers": "5.14.1"},
                "module_origins": {
                    "transformers": str(transformers_init),
                    "vllm": str(source_root / "vllm/__init__.py"),
                },
                "runtime_artifacts": runtime_artifacts,
            },
        },
        "server": {
            "command": ["vllm", "serve", *server_argv],
            "log": harness.server_log_record(server_log_path),
        },
        "videos": videos,
        "request_payloads_by_video": [
            {
                "video_index": index,
                "video_path": video["path"],
                "payload": (payload := runner.expected_chat_payload(video["file_uri"])),
                "payload_sha256": runner.sha256_json(payload),
            }
            for index, video in enumerate(videos)
        ],
        "concurrency_blocks": [block],
    }
    monitor = {
        "contaminated": False,
        "foreign_events": [],
        "returncode": 0,
        "timed_out": False,
        "timeout_seconds": 3600.0,
        "command": ["fixture-harness"],
        "device": {"index": 0, "name": "NVIDIA RTX PRO 6000", "uuid": "GPU-fixture"},
        "samples": [
            {
                "monotonic_ns": 300,
                "utc": "fixture",
                "memory_used_mib": 100,
                "utilization_percent": 1,
                "compute_apps": [],
            },
            {
                "monotonic_ns": 400,
                "utc": "fixture",
                "memory_used_mib": 200,
                "utilization_percent": 2,
                "compute_apps": [],
            },
        ],
        "peak_memory_used_mib": 200,
        "sample_count": 2,
    }
    monkeypatch.setattr(
        runner,
        "canonical_runtime_fingerprint",
        lambda _result: {
            "schema": "pynv-runtime-hardware-fingerprint-v1",
            "canonical": {"fixture": True},
            "sha256": runner.sha256_json({"fixture": True}),
        },
    )
    kwargs = {
        "commit": runner.COMMITS["upstream"],
        "variant": "upstream",
        "concurrency_order": [1],
        "harness": harness_path,
        "harness_sha256": runner.sha256_file(harness_path),
        "expected_monitor_command": ["fixture-harness"],
        "corpus": corpus,
        "transformers_root": transformers_root,
        "source_root": source_root,
        "server_log_path": server_log_path,
        "warmup_requests": {1: 1},
        "measured_requests": {1: 1},
    }
    return result, monitor, kwargs


def reaggregate_result(harness, result) -> None:
    for block in result["concurrency_blocks"]:
        for phase in ("warmup", "measured"):
            batch = block[phase]
            transport_audit = copy.deepcopy(
                batch["aggregate"]["persistent_transport_audit"]
            )
            batch["aggregate"] = harness.batch_aggregate(
                batch["records"], batch["measured_window_seconds"]
            )
            batch["aggregate"]["persistent_transport_audit"] = transport_audit
        block["aggregate"] = copy.deepcopy(block["measured"]["aggregate"])


PASSED_RECORD_SEMANTIC_FIELDS = (
    "phase",
    "block_index",
    "concurrency",
    "request_index",
    "global_request_index",
    "video_index",
    "video_path",
    "video_file_uri",
    "video_sha256",
    "video_work",
    "request_payload_sha256",
    "payload",
    "status",
    "http_status",
    "transport",
    "response",
    "started_at",
    "finished_at",
    "started_monotonic_ns",
    "finished_monotonic_ns",
    "start_offset_seconds",
    "finish_offset_seconds",
    "latency_seconds",
    "latency_ms",
)


@pytest.mark.parametrize("field", PASSED_RECORD_SEMANTIC_FIELDS)
def test_validate_result_rejects_each_record_field_mutation_after_reaggregation(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    assert set(PASSED_RECORD_SEMANTIC_FIELDS) == runner.PASSED_REQUEST_RECORD_FIELDS
    record = result["concurrency_blocks"][0]["measured"]["records"][0]
    if field == "phase":
        record[field] = "warmup"
    elif field in {
        "block_index",
        "concurrency",
        "request_index",
        "global_request_index",
        "video_index",
    }:
        record[field] += 1
    elif field == "video_path":
        record[field] = result["videos"][1]["path"]
    elif field == "video_file_uri":
        record[field] = result["videos"][1]["file_uri"]
    elif field == "video_sha256":
        record[field] = "0" * 64
    elif field == "video_work":
        record[field]["sampled_source_megapixels_estimate"] += 1.0
    elif field == "request_payload_sha256":
        record[field] = "0" * 64
    elif field == "payload":
        record[field]["model"] = "tampered-model"
        record["request_payload_sha256"] = runner.sha256_json(record[field])
    elif field == "status":
        record[field] = "failed"
    elif field == "http_status":
        record[field] = 201
    elif field == "transport":
        record[field]["response_connection_header"] = "keep-alive"
    elif field == "response":
        record[field]["raw_response"]["model"] = "tampered-model"
        record[field] = harness.response_record(
            record[field]["raw_response"], runner.OUTPUT_LENGTH
        )
    elif field == "started_at":
        record[field] = "not-an-ISO-timestamp"
    elif field == "finished_at":
        record[field] = "2023-11-14T22:13:18+00:00"
    else:
        record[field] += 1
    reaggregate_result(harness, result)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


EXACT_SCHEMA_FIELDS = {
    "record": PASSED_RECORD_SEMANTIC_FIELDS,
    "normalized": (
        "id",
        "model",
        "prompt_token_count",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "completion_token_count",
        "completion_token_ids",
        "completion_token_ids_sha256",
        "prompt_and_completion_token_ids_sha256",
        "text",
        "text_sha256",
        "reasoning_content",
        "reasoning_content_sha256",
        "finish_reason",
        "stop_reason",
        "usage",
        "server_metrics",
        "raw_response_sha256",
        "raw_response",
    ),
    "raw": (
        "id",
        "object",
        "created",
        "model",
        "choices",
        "service_tier",
        "system_fingerprint",
        "usage",
        "prompt_logprobs",
        "prompt_token_ids",
        "prompt_text",
        "kv_transfer_params",
        "ec_transfer_params",
        "metrics",
    ),
    "choice": (
        "index",
        "message",
        "logprobs",
        "finish_reason",
        "stop_reason",
        "token_ids",
        "routed_experts",
    ),
    "message": (
        "role",
        "content",
        "refusal",
        "annotations",
        "audio",
        "function_call",
        "reasoning",
    ),
    "usage": (
        "prompt_tokens",
        "total_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
    ),
    "transport": (
        "pool_slot_id",
        "phase",
        "seeded_first_wave",
        "connection_generation",
        "request_ordinal_on_generation",
        "connection_reused",
        "prewarmed_for_measurement",
        "request_connection_header",
        "response_http_version",
        "response_connection_header",
        "response_will_close",
        "response_persistent",
    ),
}


@pytest.mark.parametrize(
    ("layer", "field"),
    [
        (layer, field)
        for layer, fields in EXACT_SCHEMA_FIELDS.items()
        for field in fields
    ],
)
def test_validate_result_rejects_each_missing_nested_schema_field(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    expected_schema_constants = {
        "record": runner.PASSED_REQUEST_RECORD_FIELDS,
        "normalized": runner.NORMALIZED_RESPONSE_FIELDS,
        "raw": runner.RAW_CHAT_RESPONSE_FIELDS,
        "choice": runner.RAW_CHAT_CHOICE_FIELDS,
        "message": runner.RAW_CHAT_MESSAGE_FIELDS,
        "usage": runner.RAW_USAGE_FIELDS,
        "transport": runner.PERSISTENT_TRANSPORT_FIELDS,
    }
    assert set(EXACT_SCHEMA_FIELDS[layer]) == expected_schema_constants[layer]
    record = result["concurrency_blocks"][0]["measured"]["records"][0]
    response = record["response"]
    raw = response["raw_response"]
    containers = {
        "record": record,
        "normalized": response,
        "raw": raw,
        "choice": raw["choices"][0],
        "message": raw["choices"][0]["message"],
        "usage": raw["usage"],
        "transport": record["transport"],
    }
    del containers[layer][field]
    if layer in {"raw", "choice", "message", "usage"}:
        response["raw_response_sha256"] = runner.sha256_json(raw)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


@pytest.mark.parametrize(
    ("layer", "field"),
    (
        ("record", "response_bytes"),
        ("record", "response_body_evidence"),
        ("transport", "response_content_length_header"),
        ("normalized", "unexpected_normalized_field"),
        ("raw", "unexpected_raw_field"),
        ("choice", "unexpected_choice_field"),
        ("message", "tool_calls"),
        ("usage", "unexpected_usage_field"),
    ),
)
def test_validate_result_rejects_reintroduced_or_extra_evidence_fields(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer: str,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    record = result["concurrency_blocks"][0]["measured"]["records"][0]
    response = record["response"]
    raw = response["raw_response"]
    containers = {
        "record": record,
        "normalized": response,
        "raw": raw,
        "choice": raw["choices"][0],
        "message": raw["choices"][0]["message"],
        "usage": raw["usage"],
        "transport": record["transport"],
    }
    containers[layer][field] = None
    if layer in {"raw", "choice", "message", "usage"}:
        response["raw_response_sha256"] = runner.sha256_json(raw)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


RAW_FIXED_VALUE_MUTATIONS = (
    "duplicate_id",
    "object",
    "created",
    "model",
    "service_tier",
    "system_fingerprint",
    "prompt_logprobs",
    "prompt_token_ids",
    "prompt_text",
    "kv_transfer_params",
    "ec_transfer_params",
    "metrics",
    "choice_index",
    "choice_logprobs",
    "choice_finish_reason",
    "choice_stop_reason",
    "choice_token_ids",
    "choice_routed_experts",
    "message_role",
    "message_content",
    "message_refusal",
    "message_annotations",
    "message_audio",
    "message_function_call",
    "message_reasoning",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "usage_prompt_tokens_details",
    "usage_completion_tokens_details",
)


@pytest.mark.parametrize("case", RAW_FIXED_VALUE_MUTATIONS)
def test_validate_result_rejects_each_fixed_raw_protocol_value_coherently(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    block = result["concurrency_blocks"][0]
    record = block["measured"]["records"][0]
    response = record["response"]
    raw = response["raw_response"]
    choice = raw["choices"][0]
    message = choice["message"]
    usage = raw["usage"]
    if case == "duplicate_id":
        raw["id"] = block["warmup"]["records"][0]["response"]["id"]
    elif case == "object":
        raw["object"] = "chat.completion.chunk"
    elif case == "created":
        raw["created"] -= 10
    elif case == "model":
        raw["model"] = "tampered-model"
    elif case == "service_tier":
        raw["service_tier"] = "default"
    elif case == "system_fingerprint":
        raw["system_fingerprint"] = "tampered"
    elif case == "prompt_logprobs":
        raw["prompt_logprobs"] = []
    elif case == "prompt_token_ids":
        raw["prompt_token_ids"][0] = True
    elif case == "prompt_text":
        raw["prompt_text"] = "tampered"
    elif case == "kv_transfer_params":
        raw["kv_transfer_params"] = {}
    elif case == "ec_transfer_params":
        raw["ec_transfer_params"] = {}
    elif case == "metrics":
        raw["metrics"] = {field: None for field in runner.RAW_SERVER_METRICS_FIELDS}
    elif case == "choice_index":
        choice["index"] = 1
    elif case == "choice_logprobs":
        choice["logprobs"] = {}
    elif case == "choice_finish_reason":
        choice["finish_reason"] = "stop"
    elif case == "choice_stop_reason":
        choice["stop_reason"] = "tampered"
    elif case == "choice_token_ids":
        choice["token_ids"][0] = True
    elif case == "choice_routed_experts":
        choice["routed_experts"] = "tampered"
    elif case == "message_role":
        message["role"] = "user"
    elif case == "message_content":
        message["content"] = None
    elif case == "message_refusal":
        message["refusal"] = "tampered"
    elif case == "message_annotations":
        message["annotations"] = []
    elif case == "message_audio":
        message["audio"] = {}
    elif case == "message_function_call":
        message["function_call"] = {}
    elif case == "message_reasoning":
        message["reasoning"] = "tampered"
    elif case == "usage_prompt_tokens":
        usage["prompt_tokens"] += 1
    elif case == "usage_completion_tokens":
        usage["completion_tokens"] = float(usage["completion_tokens"])
    elif case == "usage_total_tokens":
        usage["total_tokens"] += 1
    elif case == "usage_prompt_tokens_details":
        usage["prompt_tokens_details"] = {
            "cached_tokens": None,
            "created_cache_tokens": None,
            "multimodal_tokens": None,
        }
    elif case == "usage_completion_tokens_details":
        usage["completion_tokens_details"] = {"reasoning_tokens": 0}
    else:  # pragma: no cover - parameter list and dispatch must stay paired
        raise AssertionError(case)
    record["response"] = harness.response_record(raw, runner.OUTPUT_LENGTH)
    reaggregate_result(harness, result)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


@pytest.mark.parametrize("field", EXACT_SCHEMA_FIELDS["transport"])
def test_validate_result_rejects_each_transport_value_mutation(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    transport = result["concurrency_blocks"][0]["measured"]["records"][0]["transport"]
    mutations = {
        "pool_slot_id": 1,
        "phase": "warmup",
        "seeded_first_wave": False,
        "connection_generation": 2,
        "request_ordinal_on_generation": 3,
        "connection_reused": False,
        "prewarmed_for_measurement": False,
        "request_connection_header": "close",
        "response_http_version": 10,
        "response_connection_header": "keep-alive",
        "response_will_close": True,
        "response_persistent": False,
    }
    transport[field] = mutations[field]
    reaggregate_result(harness, result)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


TRANSPORT_AUDIT_FIELDS = (
    "status",
    "phase",
    "pool_size",
    "request_count",
    "used_slot_ids",
    "seeded_first_wave_request_to_slot",
    "reasons",
    "counts_at_phase_end",
    "slot_snapshots_at_phase_end",
)
TRANSPORT_SLOT_FIELDS = (
    "slot_id",
    "current_generation",
    "warmed_generation",
    "request_ordinal_on_current_generation",
    "open_count",
    "reuse_count",
    "close_count",
    "close_reasons",
    "currently_open",
)
TRANSPORT_POOL_FIELDS = (
    "implementation",
    "pool_size",
    "connection_scope",
    "phase_scope",
    "request_streaming",
    "request_retry_count",
    "counts",
    "closed",
    "slots",
    "phase_audits",
)


@pytest.mark.parametrize("field", TRANSPORT_AUDIT_FIELDS)
def test_validate_result_recomputes_each_phase_transport_audit_field(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    assert set(TRANSPORT_AUDIT_FIELDS) == runner.PERSISTENT_PHASE_AUDIT_FIELDS
    audit = result["concurrency_blocks"][0]["measured"]["aggregate"][
        "persistent_transport_audit"
    ]
    mutations = {
        "status": "failed",
        "phase": "warmup",
        "pool_size": 2,
        "request_count": 2,
        "used_slot_ids": [],
        "seeded_first_wave_request_to_slot": {},
        "reasons": ["tampered"],
        "counts_at_phase_end": {
            "open_count": 2,
            "reuse_count": 1,
            "close_count": 0,
        },
        "slot_snapshots_at_phase_end": [],
    }
    audit[field] = mutations[field]
    reaggregate_result(harness, result)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


@pytest.mark.parametrize("field", TRANSPORT_SLOT_FIELDS)
def test_validate_result_recomputes_each_phase_transport_slot_field(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    assert set(TRANSPORT_SLOT_FIELDS) == runner.PERSISTENT_SLOT_SNAPSHOT_FIELDS
    slot = result["concurrency_blocks"][0]["measured"]["aggregate"][
        "persistent_transport_audit"
    ]["slot_snapshots_at_phase_end"][0]
    mutations = {
        "slot_id": 1,
        "current_generation": 2,
        "warmed_generation": 2,
        "request_ordinal_on_current_generation": 3,
        "open_count": 2,
        "reuse_count": 2,
        "close_count": 1,
        "close_reasons": {"tampered": 1},
        "currently_open": False,
    }
    slot[field] = mutations[field]
    reaggregate_result(harness, result)
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


@pytest.mark.parametrize("field", TRANSPORT_POOL_FIELDS)
def test_validate_result_recomputes_each_final_transport_pool_field(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    assert set(TRANSPORT_POOL_FIELDS) == runner.PERSISTENT_POOL_FIELDS
    pool = result["concurrency_blocks"][0]["persistent_http_pool"]
    mutations = {
        "implementation": "tampered",
        "pool_size": 2,
        "connection_scope": "tampered",
        "phase_scope": "tampered",
        "request_streaming": True,
        "request_retry_count": 1,
        "counts": {"open_count": 2, "reuse_count": 1, "close_count": 1},
        "closed": False,
        "slots": [],
        "phase_audits": {},
    }
    pool[field] = mutations[field]
    with pytest.raises(RuntimeError):
        runner.validate_result(result, monitor, **kwargs)


def test_transport_ordinals_must_be_unique_and_contiguous_per_slot(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _monitor, _kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    block = result["concurrency_blocks"][0]
    records = [
        block["warmup"]["records"][0],
        block["measured"]["records"][0],
    ]
    records[0]["finished_monotonic_ns"] = 500
    records[1]["finished_monotonic_ns"] = 400
    snapshots = runner._expected_slot_snapshots(records, concurrency=1, closed=True)
    assert snapshots[0]["request_ordinal_on_current_generation"] == 2
    records[1]["transport"]["request_ordinal_on_generation"] = 1
    with pytest.raises(RuntimeError, match="unique/contiguous"):
        runner._expected_slot_snapshots(records, concurrency=1, closed=True)


def test_validate_result_reconstructs_top_level_payload_graph(
    runner,
    harness,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    entry = result["request_payloads_by_video"][0]
    entry["payload"]["model"] = "tampered-model"
    entry["payload_sha256"] = runner.sha256_json(entry["payload"])
    with pytest.raises(RuntimeError, match="payload graph"):
        runner.validate_result(result, monitor, **kwargs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda batch: batch["records"][0]["payload"].update(extra=True),
        lambda batch: batch["records"][0]["response"]["prompt_token_ids"].append(9),
        lambda batch: batch["records"][0]["response"]["completion_token_ids"].append(9),
        lambda batch: batch["records"][0]["response"].update(text="changed"),
        lambda batch: batch["records"][0]["response"].update(
            reasoning_content="changed"
        ),
        lambda batch: batch["records"][0]["response"]["raw_response"].update(
            id="changed"
        ),
        lambda batch: batch["aggregate"].update(
            ordered_response_token_ids_sha256="0" * 64
        ),
    ],
)
def test_token_evidence_rejects_every_stale_hash_class(runner, mutation) -> None:
    batch = make_token_batch(runner)
    assert runner.validate_batch_token_evidence(batch, context="fixture")["status"] == (
        "passed"
    )
    tampered = copy.deepcopy(batch)
    mutation(tampered)
    with pytest.raises(RuntimeError, match="mismatch"):
        runner.validate_batch_token_evidence(tampered, context="tampered fixture")


def test_validate_result_rechecks_token_evidence_and_variant_label(
    runner, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    validated = runner.validate_result(result, monitor, **kwargs)
    assert (
        validated["blocks"][0]["warmup"]["token_evidence_audit"]["status"] == "passed"
    )

    tampered = copy.deepcopy(result)
    tampered["concurrency_blocks"][0]["measured"]["records"][0]["response"][
        "text"
    ] = "tampered-with-stale-hash"
    with pytest.raises(RuntimeError, match="response text_sha256 mismatch"):
        runner.validate_result(tampered, monitor, **kwargs)

    pilot_label = "pilot-upstream"
    pilot = copy.deepcopy(result)
    pilot["configuration"]["variant"] = pilot_label
    with pytest.raises(RuntimeError, match="configuration variant mismatch"):
        runner.validate_result(pilot, monitor, **kwargs)
    assert (
        runner.validate_result(
            pilot, monitor, **kwargs, result_variant_label=pilot_label
        )["blocks"][0]["measured"]["token_evidence_audit"]["status"]
        == "passed"
    )


@pytest.mark.parametrize(
    "field",
    [
        "status",
        "attempted_requests",
        "successful_requests",
        "failed_requests",
        "measured_window_seconds",
        "attempted_request_throughput_per_second",
        "request_throughput_per_second",
        "prompt_tokens",
        "generated_tokens",
        "all_tokens",
        "prompt_token_throughput_per_second",
        "generated_token_throughput_per_second",
        "all_token_throughput_per_second",
        "sampled_source_megapixels_estimate",
        "sampled_source_megapixels_estimate_per_second",
        "video_megapixel_estimate_method",
        "video_megapixel_estimate_unavailable",
        "achieved_mean_in_flight_requests",
        "achieved_peak_in_flight_requests",
        "response_token_fingerprints_by_request",
        "ordered_response_token_fingerprints_sha256",
        "ordered_response_token_ids_sha256",
        "completion_token_ids_sha256_counts",
        "failures",
        "latency_ms.count",
        "latency_ms.min",
        "latency_ms.mean",
        "latency_ms.median",
        "latency_ms.p50",
        "latency_ms.p90",
        "latency_ms.p95",
        "latency_ms.p99",
        "latency_ms.max",
        "latency_ms.population_stdev",
        "latency_ms.percentile_method",
    ],
)
def test_independent_aggregate_recomputation_rejects_every_field_mutation(
    runner, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    result, _monitor, _kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    batch = result["concurrency_blocks"][0]["measured"]
    assert (
        runner.independently_recompute_batch_aggregate(batch, context="fixture")[
            "status"
        ]
        == "passed"
    )
    tampered = copy.deepcopy(batch)
    if field.startswith("latency_ms."):
        tampered["aggregate"]["latency_ms"][field.split(".", 1)[1]] = None
    else:
        tampered["aggregate"][field] = None
    with pytest.raises(RuntimeError, match="independently recomputed"):
        runner.independently_recompute_batch_aggregate(
            tampered, context="tampered fixture"
        )


@pytest.mark.parametrize(
    "field",
    [
        "started_monotonic_ns",
        "finished_monotonic_ns",
        "start_offset_seconds",
        "finish_offset_seconds",
        "latency_seconds",
        "latency_ms",
    ],
)
def test_independent_aggregate_recomputation_rejects_record_timing_mutation(
    runner, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    result, _monitor, _kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    batch = copy.deepcopy(result["concurrency_blocks"][0]["measured"])
    batch["records"][0][field] = 999
    with pytest.raises(RuntimeError, match="monotonic boundaries|record .* mismatch"):
        runner.independently_recompute_batch_aggregate(batch, context="tampered timing")


def test_validate_result_rejects_block_aggregate_splice(
    runner, harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, monitor, kwargs = make_validate_result_fixture(
        runner, harness, tmp_path, monkeypatch
    )
    result["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(
        result["concurrency_blocks"][0]["measured"]["aggregate"]
    )
    result["concurrency_blocks"][0]["aggregate"]["request_throughput_per_second"] += 1.0
    with pytest.raises(RuntimeError, match="block aggregate differs"):
        runner.validate_result(result, monitor, **kwargs)


def test_full_preflight_graph_replay_rejects_checkpoint_and_artifact_tampering(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    artifact_names = (
        "harness",
        "monitor",
        "idle_gate",
        "guard_helper",
        "runtime_manifest_tool",
        "runtime_manifest_test",
        "pilot_runner",
        "pixel_preflight",
    )
    artifact_paths = {}
    for name in artifact_names:
        path = tmp_path / f"{name}.py"
        path.write_text(f"# fixture {name}\n")
        artifact_paths[name] = path
    current_artifacts = {
        "driver": Path(runner.__file__).resolve(),
        **artifact_paths,
    }
    trusted_python = tmp_path / "source/.venv/bin/python"
    artifact_hashes = {
        name: runner.sha256_file(path) for name, path in current_artifacts.items()
    }
    monkeypatch.setattr(runner, "CAMPAIGN_HARNESS_SHA256", artifact_hashes["harness"])
    monkeypatch.setattr(runner, "GPU_MONITOR_SHA256", artifact_hashes["monitor"])
    monkeypatch.setattr(runner, "IDLE_GATE_SHA256", artifact_hashes["idle_gate"])
    monkeypatch.setattr(runner, "GUARD_HELPER_SHA256", artifact_hashes["guard_helper"])
    monkeypatch.setattr(
        runner,
        "RUNTIME_TREE_MANIFEST_TOOL_SHA256",
        artifact_hashes["runtime_manifest_tool"],
    )
    monkeypatch.setattr(
        runner,
        "RUNTIME_TREE_MANIFEST_TEST_SHA256",
        artifact_hashes["runtime_manifest_test"],
    )
    monkeypatch.setattr(
        runner, "PREFLIGHT_RUNNER_SHA256", artifact_hashes["pilot_runner"]
    )
    monkeypatch.setattr(
        runner, "PIXEL_PREFLIGHT_SHA256", artifact_hashes["pixel_preflight"]
    )

    manifests = {
        "transformers_overlay": {"manifest_sha256": "1" * 64},
        "transformers_package": {"manifest_sha256": "2" * 64},
        "hf_snapshot": {"manifest_sha256": "3" * 64},
    }

    def checkpoint(label: str):
        return {
            "status": "passed",
            "label": label,
            "validated_utc": "fixture",
            "evidence_sha256": runner.sha256_json(manifests),
            "manifests": manifests,
        }

    def source(variant: str):
        return {
            "commit": runner.COMMITS[variant],
            "tree": runner.TREES[variant],
            "status": "",
            "source_harness_exists": False,
            "source_harness_sha256": None,
            "ignored_python_bytecode_or_cache_paths": [],
        }

    def write_monitor(path: Path, sample_audit: dict):
        report = {
            "status": "passed",
            "contaminated": False,
            "sample_audit": sample_audit,
        }
        path.write_text(json.dumps(report, sort_keys=True) + "\n")
        return report

    monkeypatch.setattr(
        runner,
        "validate_jsonl_binding",
        lambda report, **_kwargs: report["sample_audit"],
    )
    ingress_report_path = tmp_path / "preflight-ingress-idle-gate.json"
    ingress_report_path.write_text("{}\n")

    def fake_idle_evidence(evidence, **_kwargs):
        return {
            "fixture": evidence["fixture"],
            "report_path": evidence.get("report_path", str(ingress_report_path)),
            "report_sha256": runner.sha256_file(ingress_report_path),
            "sample_log_audit": {"fixture": "ingress-samples"},
        }

    monkeypatch.setattr(runner, "validate_idle_gate_evidence", fake_idle_evidence)
    monkeypatch.setattr(
        runner,
        "validate_monitor_evidence",
        lambda path, **_kwargs: (
            json.loads(Path(path).read_text()),
            json.loads(Path(path).read_text())["sample_audit"],
        ),
    )
    coverage = {"passed": True, "blocks": [{"fixture": True}]}
    monkeypatch.setattr(runner, "monitor_coverage_audit", lambda *_args: coverage)
    live_artifacts = {"fixture": "live-runtime-artifacts"}
    monkeypatch.setattr(
        runner,
        "revalidate_live_runtime_artifact_manifest_binding",
        lambda manifest, **_kwargs: dict(manifest),
    )
    fingerprint = {
        "schema": "pynv-runtime-hardware-fingerprint-v1",
        "canonical": {"fixture": True},
        "sha256": runner.sha256_json({"fixture": True}),
        "live_runtime_artifact_manifest": live_artifacts,
    }

    def fake_validate_result(
        result, _monitor, *, variant, result_variant_label, **_kwargs
    ):
        assert result["configuration"]["variant"] == result_variant_label == variant
        return {"runtime_hardware_fingerprint": fingerprint, "blocks": []}

    monkeypatch.setattr(runner, "validate_result", fake_validate_result)

    tensor_paths = {}
    worker_paths = {}
    common_tensors = {
        "raw_processor_pixels": torch.tensor([1, 2], dtype=torch.uint8),
        "model_visible_pixels": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "video_grid_thw": torch.tensor([[16, 36, 64]], dtype=torch.int64),
        "output_prompt_token_ids": torch.tensor([1, 2], dtype=torch.int64),
        "placeholder_is_embed": torch.tensor([True]),
    }
    for variant in runner.COMMITS:
        worker_path = tmp_path / f"pixel-parity-a01-{variant}-worker.json"
        worker_path.write_text(json.dumps({"fixture": variant}) + "\n")
        path = worker_path.with_suffix(".tensors.pt")
        torch.save(common_tensors, path)
        tensor_paths[variant] = path
        worker_paths[variant] = worker_path

    pixel_result_path = tmp_path / "pixel-parity-a01.json"
    pixel_result = {
        "schema": "pynv-endpoint-pixel-parity-v2",
        "status": "passed",
        "commits": runner.COMMITS,
        "variants": {},
    }
    for variant, commit in runner.COMMITS.items():
        pixel_result["variants"][variant] = {
            "commit": commit,
            "source": {"tree": runner.TREES[variant]},
            "backend_kwargs": runner.variant_backend_kwargs(variant),
            "worker_result_artifact": {
                "path": str(worker_paths[variant]),
                "bytes": worker_paths[variant].stat().st_size,
                "sha256": runner.sha256_file(worker_paths[variant]),
            },
            "processor": {
                "configured_max_pixels_per_frame": 1024 * 576,
                "configured_max_pixels_total": runner.TOTAL_MAX_PIXELS,
                "processed_width": 1024,
                "processed_height": 576,
                "tensor_artifact": {
                    "path": str(tensor_paths[variant]),
                    "sha256": runner.sha256_file(tensor_paths[variant]),
                },
            },
            "canonical_thwc": {"sha256": "4" * 64},
            "metadata": {"frames_indices": list(range(32)), "total_num_frames": 914},
        }
    pixel_result_path.write_text(json.dumps(pixel_result, sort_keys=True) + "\n")
    pixel_monitor_path = tmp_path / "pixel-parity-a01-gpu-monitor.json"
    pixel_sample_audit = {"fixture": "pixel-monitor"}
    write_monitor(pixel_monitor_path, pixel_sample_audit)
    pixel_log_path = tmp_path / "pixel-parity-a01.log"
    pixel_log_path.write_text("fixture pixel log\n")
    pixel_command = runner.build_pixel_preflight_command(
        python=trusted_python,
        pixel_preflight=artifact_paths["pixel_preflight"],
        source_root=tmp_path / "source",
        transformers_root=tmp_path / "overlay",
        video=tmp_path / "corpus/traffic1080-00.mp4",
        result_path=pixel_result_path,
    )
    pixel_attempt = {
        "attempt": 1,
        "result": str(pixel_result_path),
        "monitor": str(pixel_monitor_path),
        "log": str(pixel_log_path),
        "returncode": 0,
        "command": pixel_command,
        "contaminated": False,
        "runtime_manifest_before": checkpoint("pixel-parity-a01:before_attempt"),
        "runtime_manifest_after": checkpoint("pixel-parity-a01:after_attempt"),
        "idle_evidence": {
            "fixture": "pixel-idle",
            "report_path": str(tmp_path / "pixel-parity-a01-idle-gate.json"),
        },
        "monitor_sample_audit": pixel_sample_audit,
        "live_runtime_artifacts_before": live_artifacts,
        "live_runtime_artifacts_after": live_artifacts,
        "source_after_attempt": source("pr-head"),
        "result_sha256": runner.sha256_file(pixel_result_path),
        "monitor_sha256": runner.sha256_file(pixel_monitor_path),
        "log_sha256": runner.sha256_file(pixel_log_path),
    }

    def pilot_result(variant: str):
        blocks = []
        for concurrency in (1, 8, 32):
            phases = {}
            for phase in ("warmup", "measured"):
                batch = make_token_batch(runner)
                batch["records"][0]["video_sha256"] = "f" * 64
                phases[phase] = batch
            blocks.append({"concurrency": concurrency, **phases})
        return {
            "schema": "vllm-qwen3-vl-video-e2e-throughput-v3-persistent-http",
            "status": "passed",
            "provenance": {
                "source": {
                    "commit": runner.COMMITS[variant],
                    "tree": runner.TREES[variant],
                }
            },
            "configuration": {
                "variant": variant,
                "backend_kwargs": runner.variant_backend_kwargs(variant),
                "extra_server_argv": runner.variant_server_argv(variant),
            },
            "concurrency_blocks": blocks,
        }

    pilots = []
    pilot_attempts = []
    for variant, commit in runner.COMMITS.items():
        stem = f"pilot-{variant}-c1-8-32-a01"
        result_path = tmp_path / f"{stem}.json"
        result_path.write_text(json.dumps(pilot_result(variant), sort_keys=True) + "\n")
        monitor_path = tmp_path / f"{stem}-gpu-monitor.json"
        sample_audit = {"fixture": f"{variant}-monitor"}
        write_monitor(monitor_path, sample_audit)
        log_path = tmp_path / f"{stem}.log"
        log_path.write_text("fixture pilot log\n")
        server_log_path = tmp_path / f"{stem}.server.log"
        server_log_path.write_text("fixture server log\n")
        command = runner.build_harness_command(
            SimpleNamespace(
                python=trusted_python,
                harness=artifact_paths["harness"],
                root=tmp_path / "source",
                transformers_root=tmp_path / "overlay",
                corpus=tmp_path / "corpus",
                port=18600,
            ),
            variant=variant,
            result_path=result_path,
            videos=[
                tmp_path / f"corpus/traffic1080-{index:02d}.mp4" for index in range(8)
            ],
            concurrencies=[1, 8, 32],
            warmup_requests={1: 8, 8: 8, 32: 32},
            measured_requests={1: 8, 8: 8, 32: 32},
        )
        validated_result = {
            "runtime_hardware_fingerprint": fingerprint,
            "blocks": [],
        }
        pilot_attempts.append(
            {
                "variant": variant,
                "commit": commit,
                "attempt": 1,
                "result": str(result_path),
                "monitor": str(monitor_path),
                "log": str(log_path),
                "server_log": str(server_log_path),
                "returncode": 0,
                "command": command,
                "contaminated": False,
                "runtime_manifest_before": checkpoint(f"{stem}:before_attempt"),
                "runtime_manifest_after": checkpoint(f"{stem}:after_attempt"),
                "idle_evidence": {
                    "fixture": f"{variant}-idle",
                    "report_path": str(tmp_path / f"{stem}-idle-gate.json"),
                },
                "monitor_sample_audit": sample_audit,
                "live_runtime_artifacts_before": live_artifacts,
                "live_runtime_artifacts_after": live_artifacts,
                "source_after_attempt": source(variant),
                "result_sha256": runner.sha256_file(result_path),
                "server_log_sha256": runner.sha256_file(server_log_path),
                "monitor_sha256": runner.sha256_file(monitor_path),
                "log_sha256": runner.sha256_file(log_path),
            }
        )
        pilots.append(
            {
                "variant": variant,
                "commit": commit,
                "attempt": 1,
                "result": str(result_path),
                "result_sha256": runner.sha256_file(result_path),
                "monitor": str(monitor_path),
                "server_log": str(server_log_path),
                "server_log_sha256": runner.sha256_file(server_log_path),
                "validated_result": validated_result,
                "monitor_coverage_audit": coverage,
            }
        )

    artifact_bindings = {
        name: {"path": str(path), "sha256": artifact_hashes[name]}
        for name, path in current_artifacts.items()
        if name not in {"pilot_runner", "pixel_preflight"}
    }
    summary = {
        "schema": "pynv-endpoint-persistent-preflight-v1",
        "status": "passed",
        "harness_sha256": artifact_hashes["harness"],
        "evidence_namespace": {
            "root": str(tmp_path),
            "summary": str(tmp_path / "pilot-summary.json"),
            "fresh_at_collection_start": True,
            "cross_namespace_sidecars_forbidden": True,
        },
        "runner_sha256": artifact_hashes["driver"],
        "pilot_runner": {
            "path": str(artifact_paths["pilot_runner"]),
            "sha256": artifact_hashes["pilot_runner"],
        },
        "pixel_preflight_artifact": {
            "path": str(artifact_paths["pixel_preflight"]),
            "sha256": artifact_hashes["pixel_preflight"],
        },
        "artifacts": artifact_bindings,
        "runtime_manifests": manifests,
        "runtime_manifest_checkpoints": [
            checkpoint("preflight_start"),
            checkpoint("preflight_end"),
        ],
        "configuration": {
            "model": runner.MODEL,
            "revision": runner.REVISION,
            "frames": runner.FRAMES,
            "pixel_budget_per_frame": list(runner.PIXEL_BUDGET),
            "max_pixels_total": runner.TOTAL_MAX_PIXELS,
            "warmups": {"1": 8, "8": 8, "32": 32},
            "measured": {"1": 8, "8": 8, "32": 32},
            "concurrencies": [1, 8, 32],
            "max_num_seqs": runner.MAX_NUM_SEQS,
        },
        "ingress_idle_gate": {
            "fixture": "ingress-idle",
            "report_path": str(ingress_report_path),
        },
        "pixel_preflight": {
            "attempt": 1,
            "result": str(pixel_result_path),
            "result_sha256": runner.sha256_file(pixel_result_path),
            "monitor": str(pixel_monitor_path),
            "log": str(pixel_log_path),
            "worker_artifacts": {
                variant: {
                    "worker_result": pixel_result["variants"][variant][
                        "worker_result_artifact"
                    ],
                    "tensor": pixel_result["variants"][variant]["processor"][
                        "tensor_artifact"
                    ],
                }
                for variant in runner.COMMITS
            },
        },
        "pixel_attempts": [pixel_attempt],
        "pilots": pilots,
        "pilot_attempts": pilot_attempts,
        "terminal_source_revalidation": source("pr-head"),
        "terminal_live_runtime_artifact_revalidation": live_artifacts,
        "runtime_hardware_fingerprint_contract": {
            "status": "passed",
            "schema": fingerprint["schema"],
            "sha256": fingerprint["sha256"],
            "canonical": fingerprint["canonical"],
            "variants": {variant: fingerprint["sha256"] for variant in runner.COMMITS},
        },
    }
    summary_path = tmp_path / "pilot-summary.json"
    runner.write_json(summary_path, summary)
    kwargs = {
        "python": trusted_python,
        "harness_sha256": artifact_hashes["harness"],
        "source_root": tmp_path / "source",
        "corpus": tmp_path / "corpus",
        "transformers_root": tmp_path / "overlay",
        "current_artifacts": current_artifacts,
        "current_runtime_manifests": manifests,
        "conflicting_controller_roots": [tmp_path / "controllers"],
    }
    audit = runner.validate_preflight_summary(summary_path, **kwargs)
    assert audit["critical_parity_recomputed_from_raw_artifacts"] is True
    assert audit["pilot_prompt_pair_count"] == 6

    checkpoint_tamper = copy.deepcopy(summary)
    checkpoint_tamper["runtime_manifest_checkpoints"][1]["evidence_sha256"] = "0" * 64
    runner.write_json(summary_path, checkpoint_tamper)
    with pytest.raises(RuntimeError, match="runtime-manifest checkpoint mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    artifact_tamper = copy.deepcopy(summary)
    artifact_tamper["pilot_attempts"][0]["monitor_sha256"] = "0" * 64
    runner.write_json(summary_path, artifact_tamper)
    with pytest.raises(RuntimeError, match="endpoint pilot artifact mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    command_substitution = copy.deepcopy(summary)
    command_substitution["pixel_attempts"][0]["command"] = ["/bin/true"]
    runner.write_json(summary_path, command_substitution)
    with pytest.raises(RuntimeError, match="pixel attempt artifact binding mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    result_splice = copy.deepcopy(summary)
    result_splice["pixel_attempts"][0]["result"] = result_splice["pilot_attempts"][0][
        "result"
    ]
    runner.write_json(summary_path, result_splice)
    with pytest.raises(RuntimeError, match="pixel attempt artifact binding mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    monitor_splice = copy.deepcopy(summary)
    monitor_splice["pilot_attempts"][0]["monitor"] = monitor_splice["pilot_attempts"][
        1
    ]["monitor"]
    runner.write_json(summary_path, monitor_splice)
    with pytest.raises(RuntimeError, match="endpoint pilot artifact mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    alternate_sidecar = copy.deepcopy(summary)
    alternate_sidecar["pilot_attempts"][0]["log"] = str(
        tmp_path / "pilot-upstream-c1-8-32-a99.log"
    )
    runner.write_json(summary_path, alternate_sidecar)
    with pytest.raises(RuntimeError, match="endpoint pilot artifact mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)

    stale_namespace = copy.deepcopy(summary)
    stale_namespace["evidence_namespace"]["root"] = str(tmp_path / "stale")
    runner.write_json(summary_path, stale_namespace)
    with pytest.raises(RuntimeError, match="identity/status mismatch"):
        runner.validate_preflight_summary(summary_path, **kwargs)


def test_runtime_fingerprint_reads_server_environment_and_detects_drift(
    runner, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    files = {
        "python": tmp_path / "python",
        "torch": tmp_path / "torch.so",
        "torch_native": tmp_path / "torch._C.so",
        "numpy": tmp_path / "numpy.so",
        "numpy_native": tmp_path / "numpy._multiarray_umath.so",
        "transformers": tmp_path / "transformers.py",
        "pynv": tmp_path / "PyNvVideoCodec.so",
        "vllm": source_root / "vllm" / "_C.so",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())

    def artifact(path: Path):
        return {
            "path": str(path),
            "resolved_path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    performance_environment = {
        name: "1"
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "PYTHONHASHSEED",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "TOKENIZERS_PARALLELISM",
            "VLLM_WORKER_MULTIPROC_METHOD",
        )
    }
    result = {
        "provenance": {
            "source": {"root": str(source_root)},
            "python": {
                "implementation": "CPython",
                "python_version": "3.12.0",
                "executable": str(files["python"]),
                "packages": {
                    "vllm": "1",
                    "torch": "2",
                    "numpy": "3",
                    "transformers": "5.14.1",
                    "PyNvVideoCodec": "2.0.4",
                },
                "module_origins": {
                    "torch": str(files["torch"]),
                    "numpy": str(files["numpy"]),
                    "transformers": str(files["transformers"]),
                    "PyNvVideoCodec": str(files["pynv"]),
                },
                "native_module_origins": {
                    "torch._C": str(files["torch_native"]),
                    "numpy._core._multiarray_umath": str(files["numpy_native"]),
                },
                "runtime_artifacts": [artifact(path) for path in files.values()],
                "torch_runtime": {
                    "torch_version": "2",
                    "compiled_cuda_version": "13.0",
                    "cudnn_version": 9000,
                    "nvcc": None,
                },
            },
            "hardware": {
                "nvidia_smi_output": (
                    "0, NVIDIA RTX PRO 6000, GPU-fixture, 999.0, 98304 MiB, "
                    "12.0, 0000:01:00.0, P0, 2100 MHz, 1593 MHz"
                ),
                "logical_cpus": 32,
                "cuda_visible_devices": "0",
            },
        },
        "server": {"performance_environment": performance_environment},
    }
    fingerprint = runner.canonical_runtime_fingerprint(result)
    assert fingerprint["schema"] == "pynv-runtime-hardware-fingerprint-v1"
    tampered = copy.deepcopy(result)
    tampered["server"]["performance_environment"]["HF_HOME"] = "/other-cache"
    assert (
        runner.canonical_runtime_fingerprint(tampered)["sha256"]
        != fingerprint["sha256"]
    )
    extra_environment = copy.deepcopy(result)
    extra_environment["server"]["performance_environment"][
        "VLLM_ATTENTION_BACKEND"
    ] = "FLASH_ATTN"
    assert (
        runner.canonical_runtime_fingerprint(extra_environment)["sha256"]
        != fingerprint["sha256"]
    )
    assert set(fingerprint["canonical"]["python"]["native_module_artifacts"]) == {
        "torch._C",
        "numpy._core._multiarray_umath",
    }
    for artifact_name in ("python", "torch_native", "numpy_native", "pynv", "vllm"):
        path = files[artifact_name]
        original = path.read_bytes()
        path.write_bytes(original + b"-mutated")
        try:
            with pytest.raises(RuntimeError, match="size/SHA-256 claim mismatch"):
                runner.canonical_runtime_fingerprint(result)
        finally:
            path.write_bytes(original)
    for extra_native in (
        source_root / "vllm" / "omitted.so",
        files["pynv"].parent / "omitted.so.1",
    ):
        extra_native.write_bytes(b"unreported-native-artifact")
        try:
            with pytest.raises(RuntimeError, match="omits"):
                runner.canonical_runtime_fingerprint(result)
        finally:
            extra_native.unlink()
    live_manifest = copy.deepcopy(fingerprint["live_runtime_artifact_manifest"])
    del live_manifest["required_bindings"]["python"]
    canonical = {
        field: live_manifest[field]
        for field in (
            "artifacts",
            "required_bindings",
            "pynv_native_paths",
            "vllm_native_paths",
            "nvcc",
        )
    }
    live_manifest["sha256"] = runner.sha256_json(canonical)
    with pytest.raises(RuntimeError, match="required artifact set"):
        runner.revalidate_live_runtime_artifact_manifest_binding(
            live_manifest, label="omitted-python"
        )
    missing = copy.deepcopy(result)
    del missing["server"]["performance_environment"]
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.canonical_runtime_fingerprint(missing)


def test_runtime_manifest_accepts_source_native_symlink_to_precompiled_target(
    runner, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    vllm_root = source_root / "vllm"
    vllm_root.mkdir(parents=True)
    precompiled_root = tmp_path / "zz-precompiled"
    precompiled_root.mkdir()

    files = {
        "python": tmp_path / "python",
        "torch": tmp_path / "torch.py",
        "torch_native": tmp_path / "torch._C.so",
        "numpy": tmp_path / "numpy.py",
        "numpy_native": tmp_path / "numpy._multiarray_umath.so",
        "transformers": tmp_path / "transformers.py",
        "pynv": tmp_path / "PyNvVideoCodec" / "__init__.py",
        "pynv_native": tmp_path / "PyNvVideoCodec" / "_PyNvVideoCodec.so",
        "vllm_target": precompiled_root / "_C.abi3.so",
    }
    for name, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    vllm_link = vllm_root / "_C.abi3.so"
    vllm_link.symlink_to(files["vllm_target"])
    nested_target = (
        tmp_path / "aa-precompiled" / "vllm_flash_attn" / "_vllm_fa2_C.abi3.so"
    )
    nested_target.parent.mkdir(parents=True)
    nested_target.write_bytes(b"nested-vllm")
    nested_link = vllm_root / "vllm_flash_attn" / "_vllm_fa2_C.abi3.so"
    nested_link.parent.mkdir()
    nested_link.symlink_to(nested_target)

    def artifact(path: Path) -> dict[str, object]:
        resolved = path.resolve(strict=True)
        return {
            "path": str(path),
            "resolved_path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": runner.sha256_file(resolved),
        }

    provenance = {
        "executable": str(files["python"]),
        "implementation": "CPython",
        "python_version": "3.12.0",
        "packages": {
            "vllm": "1",
            "torch": "2",
            "numpy": "3",
            "transformers": "5.14.1",
            "PyNvVideoCodec": "2.0.4",
        },
        "module_origins": {
            "torch": str(files["torch"]),
            "numpy": str(files["numpy"]),
            "transformers": str(files["transformers"]),
            "PyNvVideoCodec": str(files["pynv"]),
        },
        "native_module_origins": {
            "torch._C": str(files["torch_native"]),
            "numpy._core._multiarray_umath": str(files["numpy_native"]),
        },
        "runtime_artifacts": [
            artifact(path)
            for path in (
                files["python"],
                files["torch"],
                files["torch_native"],
                files["numpy"],
                files["numpy_native"],
                files["transformers"],
                files["pynv"],
                files["pynv_native"],
                vllm_link,
                nested_link,
            )
        ],
        "torch_runtime": {
            "torch_version": "2",
            "compiled_cuda_version": "13.0",
            "cudnn_version": 9000,
            "nvcc": None,
        },
    }

    manifest = runner.revalidate_runtime_artifact_manifest(
        provenance, source_root=source_root
    )
    assert manifest["vllm_native_paths"] == sorted(
        [str(files["vllm_target"].resolve()), str(nested_target.resolve())]
    )
    vllm_artifact = next(
        item
        for item in manifest["artifacts"]
        if item["resolved_path"] == str(files["vllm_target"].resolve())
    )
    assert vllm_artifact["path"] == str(vllm_link)
    assert vllm_artifact["path_identity_before_after"] is True
    nested_artifact = next(
        item
        for item in manifest["artifacts"]
        if item["resolved_path"] == str(nested_target.resolve())
    )
    assert nested_artifact["path"] == str(nested_link)
    assert (
        runner.revalidate_live_runtime_artifact_manifest_binding(
            manifest, label="symlinked-precompiled-vllm"
        )
        == manifest
    )
    performance_environment = {
        name: "1"
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "PYTHONHASHSEED",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "TOKENIZERS_PARALLELISM",
            "VLLM_WORKER_MULTIPROC_METHOD",
        )
    }
    result = {
        "provenance": {
            "source": {"root": str(source_root)},
            "python": provenance,
            "hardware": {
                "nvidia_smi_output": (
                    "0, NVIDIA RTX PRO 6000, GPU-fixture, 999.0, 98304 MiB, "
                    "12.0, 0000:01:00.0, P0, 2100 MHz, 1593 MHz"
                ),
                "logical_cpus": 32,
                "cuda_visible_devices": "0",
            },
        },
        "server": {"performance_environment": performance_environment},
    }
    fingerprint = runner.canonical_runtime_fingerprint(result)
    assert len(fingerprint["canonical"]["python"]["vllm_compiled_artifacts"]) == 2

    traversal = copy.deepcopy(manifest)
    traversal_artifact = next(
        item
        for item in traversal["artifacts"]
        if item["resolved_path"] == str(files["vllm_target"].resolve())
    )
    traversal_artifact["path"] = str(
        vllm_root / ".." / ".." / "precompiled" / files["vllm_target"].name
    )
    canonical = {
        field: traversal[field]
        for field in (
            "artifacts",
            "required_bindings",
            "pynv_native_paths",
            "vllm_native_paths",
            "nvcc",
        )
    }
    traversal["sha256"] = runner.sha256_json(canonical)
    with pytest.raises(RuntimeError, match="not normalized"):
        runner.revalidate_live_runtime_artifact_manifest_binding(
            traversal, label="traversal-alias"
        )


def make_strict_audit_cells(runner, tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    common_configuration = {
        field: f"fixture-{field}" for field in runner.COMMON_PARITY_CONFIGURATION_FIELDS
    }
    cells = []
    for rep in range(1, 7):
        for variant in runner.COMMITS:
            configuration = {
                **common_configuration,
                "backend_kwargs": runner.variant_backend_kwargs(variant),
                "extra_server_argv": runner.variant_server_argv(variant),
            }
            result = {
                "configuration": configuration,
                "concurrency_blocks": [
                    {
                        "concurrency": 1,
                        "warmup": make_token_batch(runner),
                        "measured": make_token_batch(runner),
                    }
                ],
            }
            path = tmp_path / f"r{rep}-{variant}.json"
            path.write_text(json.dumps(result, sort_keys=True) + "\n")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            cells.append(
                {
                    "rep": rep,
                    "variant": variant,
                    "winning_attempt": 1,
                    "output": str(path),
                    "attempts": [
                        {"attempt": 1, "accepted": True, "result_sha256": digest}
                    ],
                }
            )
    return cells


def test_strict_audit_revalidates_result_sha_and_internal_token_hashes(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "WARMUP_REQUESTS", {1: 1})
    monkeypatch.setattr(runner, "MEASURED_REQUESTS", {1: 1})
    cells = make_strict_audit_cells(runner, tmp_path)
    audit = runner.strict_token_audit(cells)
    assert audit["status"] == "passed_exact"
    assert audit["compared_response_pair_count"] == 12

    changed_path = Path(cells[0]["output"])
    changed = json.loads(changed_path.read_text())
    changed["concurrency_blocks"][0]["warmup"]["records"][0]["response"][
        "text"
    ] = "stale-hash tamper"
    changed_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    cells[0]["attempts"][0]["result_sha256"] = hashlib.sha256(
        changed_path.read_bytes()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="response text_sha256 mismatch"):
        runner.strict_token_audit(cells)

    cells = make_strict_audit_cells(runner, tmp_path / "fresh")
    changed_path = Path(cells[0]["output"])
    changed_path.write_text(changed_path.read_text() + " ")
    with pytest.raises(RuntimeError, match="changed after acceptance"):
        runner.strict_token_audit(cells)


def test_terminal_failure_record_is_atomic_sanitized_and_non_reusable(
    runner, preflight, tmp_path: Path
) -> None:
    matrix_manifest = tmp_path / "matrix-manifest.json"
    runner.write_json(
        matrix_manifest,
        {
            "status": "collection_running",
            "cells": [
                {
                    "rep": 2,
                    "position": 4,
                    "variant": "upstream",
                    "status": "running",
                    "attempts": [{"attempt": 1}],
                }
            ],
        },
    )
    runner._ACTIVE_MANIFEST_PATH = matrix_manifest
    runner.record_terminal_collection_failure(RuntimeError("private absolute path"))
    failed = json.loads(matrix_manifest.read_text())
    assert failed["status"] == "collection_failed"
    assert failed["failure"]["category"] == "runtime_validation_or_workload_failure"
    assert "private absolute path" not in matrix_manifest.read_text()
    assert not matrix_manifest.with_suffix(".json.tmp").exists()

    summary_path = tmp_path / "pilot-summary.json"
    summary_path.write_text(
        json.dumps({"status": "running", "pixel_attempts": [], "pilot_attempts": []})
        + "\n"
    )
    preflight._ACTIVE_SUMMARY_PATH = summary_path
    preflight.record_terminal_collection_failure(ValueError("private input"))
    failed_summary = json.loads(summary_path.read_text())
    assert failed_summary["status"] == "collection_failed"
    assert failed_summary["failure"]["category"] == "contract_validation_failure"
    assert "private input" not in summary_path.read_text()


def test_strict_audit_count_and_status_contract(runner) -> None:
    assert runner.strict_expected_response_pair_count() == 3696
    empty = {
        "common_configuration": 0,
        "treatment_configuration": 0,
        "request_identity": 0,
        "prompt_token_ids": 0,
        "completion_token_ids": 0,
        "text_sha256": 0,
        "reasoning_content_sha256": 0,
        "finish_reason": 0,
        "stop_reason": 0,
    }
    assert runner.strict_token_status(empty) == "passed_exact"
    for generation_field in (
        "completion_token_ids",
        "text_sha256",
        "reasoning_content_sha256",
        "finish_reason",
        "stop_reason",
    ):
        counts = dict(empty)
        counts[generation_field] = 1
        assert runner.strict_token_status(counts) == "completion_or_text_mismatch"
    for input_field in (
        "common_configuration",
        "treatment_configuration",
        "request_identity",
        "prompt_token_ids",
    ):
        counts = dict(empty)
        counts[input_field] = 1
        assert runner.strict_token_status(counts) == "failed_input_parity"


def test_full_server_log_sidecar_binding(runner, harness, tmp_path: Path) -> None:
    path = tmp_path / "cell.server.log"
    path.write_text(
        "startup\nGPU KV cache size: 336,560 tokens, Maximum concurrency for "
        "32,768 tokens per request: 10.27x\nshutdown\n"
    )
    record = harness.server_log_record(path)
    assert record["storage"] == "append-only full server-log sidecar"
    result = {"server": {"log": record}}
    audit = runner.validate_full_server_log_binding(result, path)
    assert audit["full_server_log"]["full_file_scanned"] is True
    assert audit["full_server_log"]["sha256"] == runner.sha256_file(path)
    assert audit["server_log_audit"]["oom_line_count"] == 0
    assert audit["server_log_audit"]["preemption_line_count"] == 0
    path.write_text(path.read_text() + "CUDA out of memory\n")
    with pytest.raises(RuntimeError, match="binding mismatch"):
        runner.validate_full_server_log_binding(result, path)


def test_all_three_runtime_tree_manifest_contracts(
    runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_validate(**kwargs):
        calls.append(kwargs)
        return {"kind": kwargs["kind"], "root": str(kwargs["root"])}

    monkeypatch.setattr(runner, "validate_runtime_tree_manifest", fake_validate)
    args = SimpleNamespace(
        python=Path("/venv/bin/python"),
        runtime_manifest_tool=Path("/assets/capture_runtime_tree_manifest.py"),
        transformers_root=tmp_path / runner.TRANSFORMERS_OVERLAY_BASENAME,
        transformers_overlay_manifest_jsonl=Path("/evidence/overlay.jsonl"),
        transformers_overlay_manifest_summary=Path("/evidence/overlay.json"),
        transformers_manifest_jsonl=Path("/evidence/package.jsonl"),
        transformers_manifest_summary=Path("/evidence/package.json"),
        hf_snapshot_root=Path("/hf") / runner.REVISION,
        hf_manifest_jsonl=Path("/evidence/hf.jsonl"),
        hf_manifest_summary=Path("/evidence/hf.json"),
    )
    manifests = runner.validate_all_runtime_tree_manifests(args)
    assert set(manifests) == {
        "transformers_overlay",
        "transformers_package",
        "hf_snapshot",
    }
    assert [call["kind"] for call in calls] == [
        "transformers-overlay",
        "transformers",
        "hf-snapshot",
    ]
    assert calls[0]["expected_root_basename"] == runner.TRANSFORMERS_OVERLAY_BASENAME
    assert calls[0]["anchor_relative_path"] == "transformers/__init__.py"
    assert calls[1]["expected_root_basename"] == "transformers"
    assert calls[1]["anchor_relative_path"] == "__init__.py"
    assert calls[2]["root"].name == runner.REVISION
