# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_driver(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("highc_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PILOT_WARMUPS = {1: 8, 8: 8, 32: 32}
PILOT_MEASURED = {1: 8, 8: 8, 32: 32}
PILOT_CONCURRENCIES = [1, 8, 32]
PIXEL_PREFLIGHT_SHA256 = (
    "e4cb333cd47f3015ccf3aa510e3f6c26364cc4947b63d89053decc0f8156addb"
)
PIXEL_MONITOR_WATCHDOG_PAIR = (1200.0, 120.0)
PILOT_MONITOR_WATCHDOG_PAIR = (3600.0, 120.0)
PILOT_VARIANTS = ("upstream", "pr-head")
PILOT_PAIRWISE_COMPARISONS = (("upstream_to_head", "upstream", "pr-head"),)
_ACTIVE_SUMMARY_PATH: Path | None = None


def run_idle_gate(
    python: Path,
    idle_gate: Path,
    output: Path,
    environment: dict[str, str],
    driver: Any,
    *,
    seconds: float = 30.0,
    timeout: float = 7200.0,
    conflicting_controller_roots: tuple[Path, ...],
) -> dict[str, Any]:
    completed = subprocess.run(
        driver.build_idle_gate_command(
            python=python,
            idle_gate=idle_gate,
            output=output,
            seconds=seconds,
            timeout=timeout,
            conflicting_controller_roots=conflicting_controller_roots,
        ),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    report = json.loads(output.read_text())
    if (
        report.get("passed") is not True
        or report.get("guard_helper", {}).get("sha256") != driver.GUARD_HELPER_SHA256
        or report.get("configuration", {}).get("required_idle_seconds") != seconds
        or report.get("configuration", {}).get("timeout_seconds") != timeout
        or report.get("configuration", {}).get("idle_max_load_1m_per_cpu") != 0.25
        or report.get("configuration", {}).get("device_index") != 0
        or report.get("configuration", {}).get("conflicting_controller_roots")
        != [str(path) for path in conflicting_controller_roots]
    ):
        raise RuntimeError(f"idle gate did not produce accepted evidence: {output}")
    evidence = {
        "report_path": str(output),
        "report": report,
        "report_sha256": driver.sha256_file(output),
        "sample_log_audit": driver.validate_jsonl_binding(
            report,
            expected_path=output.with_name(output.stem + ".samples.jsonl"),
            expected_suffix="-idle-gate.samples.jsonl",
        ),
    }
    driver.validate_idle_gate_evidence(
        evidence,
        expected_seconds=seconds,
        expected_timeout=timeout,
        conflicting_controller_roots=conflicting_controller_roots,
    )
    return evidence


def compare_tokens(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(results) != set(PILOT_VARIANTS):
        raise RuntimeError(f"pilot variants differ: {sorted(results)}")
    pairwise: dict[str, Any] = {}
    all_prompt_exact = True
    all_completion_exact = True
    all_text_exact = True
    all_reasoning_exact = True
    all_finish_reason_exact = True
    all_stop_reason_exact = True
    all_request_identity_exact = True
    all_c1_completion_exact = True
    for (
        comparison_name,
        baseline_variant,
        candidate_variant,
    ) in PILOT_PAIRWISE_COMPARISONS:
        by_concurrency: dict[str, Any] = {}
        for concurrency in PILOT_CONCURRENCIES:
            blocks = {}
            for variant in (baseline_variant, candidate_variant):
                blocks[variant] = next(
                    block
                    for block in results[variant]["concurrency_blocks"]
                    if int(block["concurrency"]) == concurrency
                )
            by_phase: dict[str, Any] = {}
            for phase in ("warmup", "measured"):
                baseline_records = {
                    int(record["request_index"]): record
                    for record in blocks[baseline_variant][phase]["records"]
                }
                candidate_records = {
                    int(record["request_index"]): record
                    for record in blocks[candidate_variant][phase]["records"]
                }
                if baseline_records.keys() != candidate_records.keys():
                    raise RuntimeError(
                        f"{comparison_name} c{concurrency} {phase} request index "
                        "sets differ"
                    )
                mismatches = []
                prompt_mismatches = 0
                completion_mismatches = 0
                text_mismatches = 0
                reasoning_mismatches = 0
                finish_reason_mismatches = 0
                stop_reason_mismatches = 0
                casefold_text_mismatches = 0
                identity_mismatches = 0
                for request_index in baseline_records:
                    baseline_record = baseline_records[request_index]
                    candidate_record = candidate_records[request_index]
                    identity_fields = (
                        "phase",
                        "block_index",
                        "concurrency",
                        "request_index",
                        "global_request_index",
                        "video_index",
                        "video_path",
                        "video_file_uri",
                        "video_sha256",
                        "request_payload_sha256",
                        "status",
                    )
                    identity_exact = all(
                        baseline_record.get(field) == candidate_record.get(field)
                        for field in identity_fields
                    )
                    baseline_response = baseline_record["response"]
                    candidate_response = candidate_record["response"]
                    prompt_exact = (
                        baseline_response["prompt_token_ids"]
                        == candidate_response["prompt_token_ids"]
                    )
                    completion_exact = (
                        baseline_response["completion_token_ids"]
                        == candidate_response["completion_token_ids"]
                    )
                    text_exact = (
                        baseline_response["text"] == candidate_response["text"]
                        and baseline_response["text_sha256"]
                        == candidate_response["text_sha256"]
                    )
                    casefold_exact = (
                        baseline_response["text"].casefold()
                        == candidate_response["text"].casefold()
                    )
                    reasoning_exact = (
                        baseline_response["reasoning_content_sha256"]
                        == candidate_response["reasoning_content_sha256"]
                    )
                    finish_reason_exact = (
                        baseline_response["finish_reason"]
                        == candidate_response["finish_reason"]
                    )
                    stop_reason_exact = (
                        baseline_response["stop_reason"]
                        == candidate_response["stop_reason"]
                    )
                    prompt_mismatches += int(not prompt_exact)
                    completion_mismatches += int(not completion_exact)
                    text_mismatches += int(not text_exact)
                    reasoning_mismatches += int(not reasoning_exact)
                    finish_reason_mismatches += int(not finish_reason_exact)
                    stop_reason_mismatches += int(not stop_reason_exact)
                    casefold_text_mismatches += int(not casefold_exact)
                    identity_mismatches += int(not identity_exact)
                    if not (
                        identity_exact
                        and prompt_exact
                        and completion_exact
                        and text_exact
                        and reasoning_exact
                        and finish_reason_exact
                        and stop_reason_exact
                    ):
                        baseline_completion_ids = baseline_response[
                            "completion_token_ids"
                        ]
                        candidate_completion_ids = candidate_response[
                            "completion_token_ids"
                        ]
                        differing_positions = [
                            index
                            for index, (baseline_token, candidate_token) in enumerate(
                                zip(
                                    baseline_completion_ids,
                                    candidate_completion_ids,
                                )
                            )
                            if baseline_token != candidate_token
                        ]
                        mismatches.append(
                            {
                                "request_index": request_index,
                                "video_index": baseline_record["video_index"],
                                "identity_exact": identity_exact,
                                "prompt_exact": prompt_exact,
                                "completion_exact": completion_exact,
                                "text_exact": text_exact,
                                "casefold_text_exact": casefold_exact,
                                "reasoning_content_sha256_exact": reasoning_exact,
                                "finish_reason_exact": finish_reason_exact,
                                "stop_reason_exact": stop_reason_exact,
                                "differing_completion_token_positions": (
                                    differing_positions
                                ),
                                "baseline_completion_token_count": len(
                                    baseline_completion_ids
                                ),
                                "candidate_completion_token_count": len(
                                    candidate_completion_ids
                                ),
                                "baseline_completion_tokens_at_differences": [
                                    baseline_response["completion_token_ids"][index]
                                    for index in differing_positions
                                ],
                                "candidate_completion_tokens_at_differences": [
                                    candidate_response["completion_token_ids"][index]
                                    for index in differing_positions
                                ],
                                "baseline_text_sha256": baseline_response[
                                    "text_sha256"
                                ],
                                "candidate_text_sha256": candidate_response[
                                    "text_sha256"
                                ],
                                "baseline_reasoning_content_sha256": (
                                    baseline_response["reasoning_content_sha256"]
                                ),
                                "candidate_reasoning_content_sha256": (
                                    candidate_response["reasoning_content_sha256"]
                                ),
                            }
                        )
                all_prompt_exact &= prompt_mismatches == 0
                all_completion_exact &= completion_mismatches == 0
                all_text_exact &= text_mismatches == 0
                all_reasoning_exact &= reasoning_mismatches == 0
                all_finish_reason_exact &= finish_reason_mismatches == 0
                all_stop_reason_exact &= stop_reason_mismatches == 0
                all_request_identity_exact &= identity_mismatches == 0
                if concurrency == 1:
                    all_c1_completion_exact &= completion_mismatches == 0
                by_phase[phase] = {
                    "requests": len(baseline_records),
                    "request_identity_mismatches": identity_mismatches,
                    "prompt_token_id_mismatches": prompt_mismatches,
                    "completion_token_id_mismatches": completion_mismatches,
                    "text_mismatches": text_mismatches,
                    "reasoning_content_mismatches": reasoning_mismatches,
                    "finish_reason_mismatches": finish_reason_mismatches,
                    "stop_reason_mismatches": stop_reason_mismatches,
                    "casefold_text_mismatches": casefold_text_mismatches,
                    "mismatch_details": mismatches,
                }
            by_concurrency[str(concurrency)] = {"by_phase": by_phase}
        pairwise[comparison_name] = {
            "baseline": baseline_variant,
            "candidate": candidate_variant,
            "by_concurrency": by_concurrency,
        }
    return {
        "all_request_identity_exact": all_request_identity_exact,
        "all_prompt_token_ids_exact": all_prompt_exact,
        "all_completion_token_ids_exact": all_completion_exact,
        "all_text_exact": all_text_exact,
        "all_reasoning_content_exact": all_reasoning_exact,
        "all_finish_reason_exact": all_finish_reason_exact,
        "all_stop_reason_exact": all_stop_reason_exact,
        "all_c1_completion_token_ids_exact": all_c1_completion_exact,
        "policy": (
            "exact request identity, full prompt/completion ID arrays, and text "
            "hash/content across every warmup and measured pilot request"
        ),
        "pairwise": pairwise,
    }


def run_main() -> None:
    global _ACTIVE_SUMMARY_PATH
    _ACTIVE_SUMMARY_PATH = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-tool", type=Path, required=True)
    parser.add_argument("--runtime-manifest-test", type=Path, required=True)
    parser.add_argument(
        "--transformers-overlay-manifest-jsonl", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-overlay-manifest-summary", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-package-manifest-jsonl", type=Path, required=True
    )
    parser.add_argument(
        "--transformers-package-manifest-summary", type=Path, required=True
    )
    parser.add_argument("--hf-snapshot-root", type=Path, required=True)
    parser.add_argument("--hf-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--hf-manifest-summary", type=Path, required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--expected-driver-sha256", required=True)
    parser.add_argument(
        "--conflicting-controller-root", type=Path, action="append", required=True
    )
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.assets = args.assets.resolve()
    args.results = args.results.resolve()
    args.corpus = args.corpus.resolve()
    args.transformers_root = args.transformers_root.resolve()
    args.runtime_manifest_tool = args.runtime_manifest_tool.resolve()
    args.runtime_manifest_test = args.runtime_manifest_test.resolve()
    args.transformers_overlay_manifest_jsonl = (
        args.transformers_overlay_manifest_jsonl.resolve()
    )
    args.transformers_overlay_manifest_summary = (
        args.transformers_overlay_manifest_summary.resolve()
    )
    args.transformers_package_manifest_jsonl = (
        args.transformers_package_manifest_jsonl.resolve()
    )
    args.transformers_package_manifest_summary = (
        args.transformers_package_manifest_summary.resolve()
    )
    args.hf_snapshot_root = args.hf_snapshot_root.resolve()
    args.hf_manifest_jsonl = args.hf_manifest_jsonl.resolve()
    args.hf_manifest_summary = args.hf_manifest_summary.resolve()
    args.conflicting_controller_root = tuple(
        sorted(
            {path.resolve(strict=False) for path in args.conflicting_controller_root},
            key=str,
        )
    )
    python = args.root / ".venv/bin/python"
    if args.results.exists():
        raise FileExistsError(args.results)
    driver_path = args.assets / "run_pynv_endpoint_high_concurrency_matrix_refined.py"
    driver = load_driver(driver_path)
    harness = args.assets / "benchmark_pynvvideocodec_e2e_persistent.py"
    monitor = args.assets / "run_with_gpu_monitor_refined.py"
    idle_gate = args.assets / "wait_for_exclusive_gpu_refined.py"
    guard_helper = args.assets / "pynv_gpu_guard.py"
    pixel_preflight = args.assets / "preflight_pynv_endpoint_pixel_parity.py"
    if driver.sha256_file(driver_path) != args.expected_driver_sha256:
        raise RuntimeError("endpoint matrix driver SHA mismatch")
    if driver.sha256_file(harness) != args.harness_sha256:
        raise RuntimeError("campaign harness SHA mismatch")
    if args.harness_sha256 != driver.CAMPAIGN_HARNESS_SHA256:
        raise RuntimeError("campaign harness is not the frozen artifact")
    if driver.sha256_file(monitor) != driver.GPU_MONITOR_SHA256:
        raise RuntimeError("GPU monitor SHA mismatch")
    if driver.sha256_file(idle_gate) != driver.IDLE_GATE_SHA256:
        raise RuntimeError("idle gate SHA mismatch")
    if driver.sha256_file(guard_helper) != driver.GUARD_HELPER_SHA256:
        raise RuntimeError("guard helper SHA mismatch")
    if (
        driver.sha256_file(args.runtime_manifest_tool)
        != driver.RUNTIME_TREE_MANIFEST_TOOL_SHA256
        or driver.sha256_file(args.runtime_manifest_test)
        != driver.RUNTIME_TREE_MANIFEST_TEST_SHA256
    ):
        raise RuntimeError("runtime manifest tool/test SHA mismatch")
    if driver.sha256_file(pixel_preflight) != PIXEL_PREFLIGHT_SHA256:
        raise RuntimeError("pixel preflight SHA mismatch")
    runtime_manifest_validation_kwargs = {
        "python": python,
        "tool": args.runtime_manifest_tool,
        "transformers_root": args.transformers_root,
        "transformers_overlay_jsonl": args.transformers_overlay_manifest_jsonl,
        "transformers_overlay_summary": args.transformers_overlay_manifest_summary,
        "transformers_package_jsonl": args.transformers_package_manifest_jsonl,
        "transformers_package_summary": args.transformers_package_manifest_summary,
        "hf_snapshot_root": args.hf_snapshot_root,
        "hf_jsonl": args.hf_manifest_jsonl,
        "hf_summary": args.hf_manifest_summary,
    }
    runtime_manifests = driver.validate_runtime_manifests(
        **runtime_manifest_validation_kwargs
    )
    videos = [args.corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    for video in videos:
        if (
            not video.is_file()
            or video.stat().st_size != driver.VIDEO_BYTES
            or driver.sha256_file(video) != driver.VIDEO_SHA256
        ):
            raise RuntimeError(f"video corpus mismatch: {video}")
    args.results.mkdir(parents=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    environment.update(driver.hf_cache_environment(args.hf_snapshot_root))
    args.python = python
    args.harness = harness
    args.monitor = monitor
    args.idle_gate = idle_gate
    args.guard_helper = guard_helper
    args.port = 18600
    summary: dict[str, Any] = {
        "schema": "pynv-endpoint-persistent-preflight-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_namespace": {
            "root": str(args.results),
            "summary": str(args.results / "pilot-summary.json"),
            "fresh_at_collection_start": True,
            "cross_namespace_sidecars_forbidden": True,
        },
        "pilot_runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": driver.sha256_file(Path(__file__).resolve()),
        },
        "harness_sha256": args.harness_sha256,
        "runner_sha256": driver.sha256_file(driver_path),
        "artifacts": {
            "driver": {
                "path": str(driver_path),
                "sha256": driver.sha256_file(driver_path),
            },
            "harness": {"path": str(harness), "sha256": driver.sha256_file(harness)},
            "monitor": {"path": str(monitor), "sha256": driver.sha256_file(monitor)},
            "idle_gate": {
                "path": str(idle_gate),
                "sha256": driver.sha256_file(idle_gate),
            },
            "guard_helper": {
                "path": str(guard_helper),
                "sha256": driver.sha256_file(guard_helper),
            },
            "runtime_manifest_tool": {
                "path": str(args.runtime_manifest_tool),
                "sha256": driver.sha256_file(args.runtime_manifest_tool),
            },
            "runtime_manifest_test": {
                "path": str(args.runtime_manifest_test),
                "sha256": driver.sha256_file(args.runtime_manifest_test),
            },
        },
        "runtime_manifests": runtime_manifests,
        "runtime_manifest_checkpoints": [
            {
                "status": "passed",
                "label": "preflight_start",
                "validated_utc": datetime.now(timezone.utc).isoformat(),
                "evidence_sha256": driver.sha256_json(runtime_manifests),
                "manifests": runtime_manifests,
            }
        ],
        "pixel_preflight_artifact": {
            "path": str(pixel_preflight),
            "sha256": driver.sha256_file(pixel_preflight),
        },
        "configuration": {
            "model": driver.MODEL,
            "revision": driver.REVISION,
            "frames": driver.FRAMES,
            "pixel_budget_per_frame": list(driver.PIXEL_BUDGET),
            "max_pixels_total": driver.TOTAL_MAX_PIXELS,
            "warmups": PILOT_WARMUPS,
            "measured": PILOT_MEASURED,
            "concurrencies": PILOT_CONCURRENCIES,
            "max_num_seqs": driver.MAX_NUM_SEQS,
        },
        "pixel_preflight": None,
        "pixel_attempts": [],
        "pilots": [],
        "pilot_attempts": [],
    }
    summary_path = args.results / "pilot-summary.json"
    driver.write_json(summary_path, summary)
    _ACTIVE_SUMMARY_PATH = summary_path

    ingress_idle_path = args.results / "preflight-ingress-idle-gate.json"
    summary["ingress_idle_gate"] = run_idle_gate(
        python,
        idle_gate,
        ingress_idle_path,
        environment,
        driver,
        seconds=1200.0,
        timeout=21600.0,
        conflicting_controller_roots=args.conflicting_controller_root,
    )
    driver.write_json(summary_path, summary)

    for attempt in range(1, 21):
        stem = f"pixel-parity-a{attempt:02d}"
        idle_path = args.results / f"{stem}-idle-gate.json"
        idle_samples_path = args.results / f"{stem}-idle-gate.samples.jsonl"
        result_path = args.results / f"{stem}.json"
        monitor_path = args.results / f"{stem}-gpu-monitor.json"
        monitor_samples_path = args.results / f"{stem}-gpu-monitor.samples.jsonl"
        log_path = args.results / f"{stem}.log"
        preexisting = [
            path
            for path in (
                idle_path,
                idle_samples_path,
                result_path,
                monitor_path,
                monitor_samples_path,
                log_path,
            )
            if path.exists()
        ]
        if preexisting:
            raise FileExistsError(
                f"refusing to overwrite preflight evidence: {preexisting}"
            )
        before_runtime_manifest = driver.runtime_manifest_checkpoint(
            expected=runtime_manifests,
            label=f"{stem}:before_attempt",
            validation_kwargs=runtime_manifest_validation_kwargs,
        )
        live_runtime_artifacts_before = driver.capture_live_runtime_artifact_manifest(
            harness=harness,
            python=python,
            source_root=args.root,
            pythonpath_extras=[args.transformers_root],
            environment=environment,
        )
        idle_evidence = run_idle_gate(
            python,
            idle_gate,
            idle_path,
            environment,
            driver,
            seconds=30.0,
            timeout=1800.0,
            conflicting_controller_roots=args.conflicting_controller_root,
        )
        pixel_command = driver.build_pixel_preflight_command(
            python=python,
            pixel_preflight=pixel_preflight,
            source_root=args.root,
            transformers_root=args.transformers_root,
            video=videos[0],
            result_path=result_path,
        )
        command = driver.build_monitored_command(
            python=python,
            monitor=monitor,
            output=monitor_path,
            child_command=pixel_command,
            watchdog_pair=PIXEL_MONITOR_WATCHDOG_PAIR,
            conflicting_controller_roots=args.conflicting_controller_root,
        )
        print(f"RUN {stem}", flush=True)
        with log_path.open("x") as log:
            completed = subprocess.run(
                command,
                check=False,
                cwd=args.root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        post_attempt_integrity = driver.post_attempt_integrity_checks(
            root=args.root,
            commit=driver.COMMITS["pr-head"],
            variant="pr-head",
            runtime_manifests=runtime_manifests,
            runtime_label=f"{stem}:after_attempt",
            runtime_validation_kwargs=runtime_manifest_validation_kwargs,
        )
        source_after_attempt = post_attempt_integrity["source"]
        after_runtime_manifest = post_attempt_integrity["runtime_manifests"]
        monitor_result = json.loads(monitor_path.read_text())
        monitor_sample_audit = driver.validate_jsonl_binding(
            monitor_result,
            expected_path=monitor_samples_path,
            expected_suffix="-gpu-monitor.samples.jsonl",
        )
        pixel_attempt = {
            "attempt": attempt,
            "result": str(result_path),
            "monitor": str(monitor_path),
            "log": str(log_path),
            "returncode": completed.returncode,
            "command": pixel_command,
            "contaminated": monitor_result.get("contaminated"),
            "runtime_manifest_before": before_runtime_manifest,
            "runtime_manifest_after": after_runtime_manifest,
            "live_runtime_artifacts_before": live_runtime_artifacts_before,
            "idle_evidence": idle_evidence,
            "monitor_sample_audit": monitor_sample_audit,
            "source_after_attempt": source_after_attempt,
            "result_sha256": (
                driver.sha256_file(result_path) if result_path.is_file() else None
            ),
            "monitor_sha256": driver.sha256_file(monitor_path),
            "log_sha256": driver.sha256_file(log_path),
        }
        live_runtime_artifacts_after = driver.capture_live_runtime_artifact_manifest(
            harness=harness,
            python=python,
            source_root=args.root,
            pythonpath_extras=[args.transformers_root],
            environment=environment,
        )
        pixel_attempt["live_runtime_artifacts_after"] = live_runtime_artifacts_after
        if live_runtime_artifacts_after != live_runtime_artifacts_before:
            raise RuntimeError(f"{stem} live runtime artifacts changed")
        summary["pixel_attempts"].append(pixel_attempt)
        driver.write_json(summary_path, summary)
        if monitor_result.get("timed_out"):
            raise RuntimeError(f"{stem} timed out")
        if completed.returncode == 99 or monitor_result.get("contaminated") is True:
            pixel_attempt["contamination_retry_evidence"] = (
                driver.validate_contamination_retry_evidence(
                    wrapper_returncode=completed.returncode,
                    report_path=monitor_path,
                    expected_wrapper_command=command,
                    expected_child_command=pixel_command,
                    watchdog_pair=PIXEL_MONITOR_WATCHDOG_PAIR,
                    conflicting_controller_roots=args.conflicting_controller_root,
                )
            )
            driver.write_json(summary_path, summary)
            print(f"CONTAMINATED {stem}", flush=True)
            continue
        monitor_result, accepted_monitor_sample_audit = (
            driver.validate_monitor_evidence(
                monitor_path,
                expected_command=pixel_command,
                watchdog_pair=PIXEL_MONITOR_WATCHDOG_PAIR,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
        )
        if accepted_monitor_sample_audit != monitor_sample_audit:
            raise RuntimeError(f"{stem} accepted monitor sample audit changed")
        if monitor_result.get("contaminated") is not False:
            raise RuntimeError(f"{stem} monitor contamination state is not false")
        if monitor_result.get("foreign_events") != []:
            raise RuntimeError(f"{stem} monitor recorded foreign events")
        if monitor_result.get("returncode") != 0:
            raise RuntimeError(f"{stem} monitor child return code mismatch")
        expected_pixel_child = command[command.index("--") + 1 :]
        if monitor_result.get("command") != expected_pixel_child:
            raise RuntimeError(f"{stem} monitor child command mismatch")
        if monitor_result.get("timeout_seconds") != 1200.0:
            raise RuntimeError(f"{stem} monitor watchdog mismatch")
        if completed.returncode:
            raise RuntimeError(log_path.read_text(errors="replace")[-16000:])
        pixel_result = json.loads(result_path.read_text())
        if (
            pixel_result.get("status") != "passed"
            or pixel_result.get("schema") != "pynv-endpoint-pixel-parity-v2"
        ):
            raise RuntimeError(f"{stem} status is not passed")
        summary["pixel_preflight"] = {
            "attempt": attempt,
            "result": str(result_path),
            "result_sha256": driver.sha256_file(result_path),
            "monitor": str(monitor_path),
            "log": str(log_path),
            "parity": pixel_result["parity"],
            "model_visible_comparison": pixel_result["model_visible_comparison"],
            "peak_total_gpu_memory_used_mib": monitor_result["peak_memory_used_mib"],
            "runtime_manifest_before": before_runtime_manifest,
            "runtime_manifest_after": after_runtime_manifest,
            "idle_evidence": idle_evidence,
            "monitor_sample_audit": monitor_sample_audit,
            "worker_artifacts": {
                variant: {
                    "worker_result": pixel_result["variants"][variant][
                        "worker_result_artifact"
                    ],
                    "tensor": {
                        "path": pixel_result["variants"][variant]["processor"][
                            "tensor_artifact"
                        ]["path"],
                        "sha256": pixel_result["variants"][variant]["processor"][
                            "tensor_artifact"
                        ]["sha256"],
                    },
                }
                for variant in PILOT_VARIANTS
            },
        }
        driver.write_json(summary_path, summary)
        print(f"PASS {stem}", flush=True)
        break
    else:
        raise RuntimeError("too many contaminated pixel parity attempts")

    pilot_results: dict[str, dict[str, Any]] = {}
    pilot_runtime_fingerprints: dict[str, dict[str, Any]] = {}
    for variant, commit in driver.COMMITS.items():
        checked_out = subprocess.run(
            ["git", "-C", str(args.root), "checkout", "--quiet", "--detach", commit],
            check=False,
            capture_output=True,
            text=True,
        )
        if checked_out.returncode:
            raise RuntimeError(checked_out.stderr or checked_out.stdout)
        driver.validate_source(args.root, commit, variant=variant)
        for attempt in range(1, 21):
            stem = f"pilot-{variant}-c1-8-32-a{attempt:02d}"
            idle_path = args.results / f"{stem}-idle-gate.json"
            idle_samples_path = args.results / f"{stem}-idle-gate.samples.jsonl"
            result_path = args.results / f"{stem}.json"
            server_log_path = args.results / f"{stem}.server.log"
            monitor_path = args.results / f"{stem}-gpu-monitor.json"
            monitor_samples_path = args.results / f"{stem}-gpu-monitor.samples.jsonl"
            log_path = args.results / f"{stem}.log"
            preexisting = [
                path
                for path in (
                    idle_path,
                    idle_samples_path,
                    result_path,
                    server_log_path,
                    monitor_path,
                    monitor_samples_path,
                    log_path,
                )
                if path.exists()
            ]
            if preexisting:
                raise FileExistsError(
                    f"refusing to overwrite pilot evidence: {preexisting}"
                )
            before_runtime_manifest = driver.runtime_manifest_checkpoint(
                expected=runtime_manifests,
                label=f"{stem}:before_attempt",
                validation_kwargs=runtime_manifest_validation_kwargs,
            )
            live_runtime_artifacts_before = (
                driver.capture_live_runtime_artifact_manifest(
                    harness=harness,
                    python=python,
                    source_root=args.root,
                    pythonpath_extras=[args.transformers_root],
                    environment=environment,
                )
            )
            idle_evidence = run_idle_gate(
                python,
                idle_gate,
                idle_path,
                environment,
                driver,
                seconds=30.0,
                timeout=1800.0,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
            harness_command = [
                str(python),
                str(harness),
                "--source-root",
                str(args.root),
                "--python",
                str(python),
                "--pythonpath-extra",
                str(args.transformers_root),
                "--variant",
                variant,
                "--allowed-local-media-path",
                str(args.corpus),
                "--backend",
                "pynvvideocodec",
                "--backend-kwargs",
                json.dumps(
                    driver.variant_backend_kwargs(variant),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "--model",
                driver.MODEL,
                "--revision",
                driver.REVISION,
                "--frames",
                "32",
                "--video-pixel-budget",
                "1024x576",
                "--warmup-requests",
                "1",
                "--warmup-requests-by-concurrency",
                json.dumps(PILOT_WARMUPS, separators=(",", ":"), sort_keys=True),
                "--requests",
                "1",
                "--requests-by-concurrency",
                json.dumps(PILOT_MEASURED, separators=(",", ":"), sort_keys=True),
                "--output-len",
                "32",
                "--max-model-len",
                "32768",
                "--max-num-batched-tokens",
                "9216",
                "--max-num-seqs",
                "32",
                "--mm-ipc-gpu-memory-gb",
                "2",
                "--kv-cache-memory-bytes",
                str(driver.KV_CACHE_MEMORY_BYTES),
                "--settle-seconds",
                "1.0",
                "--request-timeout",
                "1200",
                "--startup-timeout",
                "600",
                "--shutdown-timeout",
                "60",
                "--port",
                "18600",
                "--output",
                str(result_path),
            ]
            for server_argument in driver.variant_server_argv(variant):
                harness_command.append(f"--server-arg={server_argument}")
            for video in videos:
                harness_command.extend(["--video", str(video)])
            for concurrency in PILOT_CONCURRENCIES:
                harness_command.extend(["--concurrency", str(concurrency)])
            reconstructed_harness_command = driver.build_harness_command(
                args,
                variant=variant,
                result_path=result_path,
                videos=videos,
                concurrencies=PILOT_CONCURRENCIES,
                warmup_requests=PILOT_WARMUPS,
                measured_requests=PILOT_MEASURED,
            )
            if harness_command != reconstructed_harness_command:
                raise RuntimeError(f"{stem} internal harness command drift")
            harness_command = reconstructed_harness_command
            command = driver.build_monitored_command(
                python=python,
                monitor=monitor,
                output=monitor_path,
                child_command=harness_command,
                watchdog_pair=PILOT_MONITOR_WATCHDOG_PAIR,
                conflicting_controller_roots=args.conflicting_controller_root,
            )
            print(f"RUN {stem}", flush=True)
            with log_path.open("x") as log:
                completed = subprocess.run(
                    command,
                    check=False,
                    cwd=args.root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            post_attempt_integrity = driver.post_attempt_integrity_checks(
                root=args.root,
                commit=commit,
                variant=variant,
                runtime_manifests=runtime_manifests,
                runtime_label=f"{stem}:after_attempt",
                runtime_validation_kwargs=runtime_manifest_validation_kwargs,
            )
            source_after_attempt = post_attempt_integrity["source"]
            after_runtime_manifest = post_attempt_integrity["runtime_manifests"]
            monitor_result = json.loads(monitor_path.read_text())
            monitor_sample_audit = driver.validate_jsonl_binding(
                monitor_result,
                expected_path=monitor_samples_path,
                expected_suffix="-gpu-monitor.samples.jsonl",
            )
            summary["pilot_attempts"].append(
                {
                    "variant": variant,
                    "commit": commit,
                    "attempt": attempt,
                    "result": str(result_path),
                    "monitor": str(monitor_path),
                    "log": str(log_path),
                    "server_log": str(server_log_path),
                    "returncode": completed.returncode,
                    "command": harness_command,
                    "contaminated": monitor_result.get("contaminated"),
                    "runtime_manifest_before": before_runtime_manifest,
                    "runtime_manifest_after": after_runtime_manifest,
                    "live_runtime_artifacts_before": live_runtime_artifacts_before,
                    "idle_evidence": idle_evidence,
                    "monitor_sample_audit": monitor_sample_audit,
                    "source_after_attempt": source_after_attempt,
                    "result_sha256": (
                        driver.sha256_file(result_path)
                        if result_path.is_file()
                        else None
                    ),
                    "server_log_sha256": (
                        driver.sha256_file(server_log_path)
                        if server_log_path.is_file()
                        else None
                    ),
                    "monitor_sha256": driver.sha256_file(monitor_path),
                    "log_sha256": driver.sha256_file(log_path),
                }
            )
            live_runtime_artifacts_after = (
                driver.capture_live_runtime_artifact_manifest(
                    harness=harness,
                    python=python,
                    source_root=args.root,
                    pythonpath_extras=[args.transformers_root],
                    environment=environment,
                )
            )
            summary["pilot_attempts"][-1][
                "live_runtime_artifacts_after"
            ] = live_runtime_artifacts_after
            if live_runtime_artifacts_after != live_runtime_artifacts_before:
                raise RuntimeError(f"{stem} live runtime artifacts changed")
            driver.write_json(summary_path, summary)
            if monitor_result.get("timed_out"):
                raise RuntimeError(f"{stem} timed out")
            if completed.returncode == 99 or monitor_result.get("contaminated") is True:
                summary["pilot_attempts"][-1]["contamination_retry_evidence"] = (
                    driver.validate_contamination_retry_evidence(
                        wrapper_returncode=completed.returncode,
                        report_path=monitor_path,
                        expected_wrapper_command=command,
                        expected_child_command=harness_command,
                        watchdog_pair=PILOT_MONITOR_WATCHDOG_PAIR,
                        conflicting_controller_roots=(args.conflicting_controller_root),
                    )
                )
                driver.write_json(summary_path, summary)
                print(f"CONTAMINATED {stem}", flush=True)
                continue
            monitor_result, accepted_monitor_sample_audit = (
                driver.validate_monitor_evidence(
                    monitor_path,
                    expected_command=harness_command,
                    watchdog_pair=PILOT_MONITOR_WATCHDOG_PAIR,
                    conflicting_controller_roots=args.conflicting_controller_root,
                )
            )
            if accepted_monitor_sample_audit != monitor_sample_audit:
                raise RuntimeError(f"{stem} accepted monitor sample audit changed")
            if monitor_result.get("contaminated") is not False:
                raise RuntimeError(f"{stem} monitor contamination state is not false")
            if monitor_result.get("foreign_events") != []:
                raise RuntimeError(f"{stem} monitor recorded foreign events")
            if monitor_result.get("returncode") != 0:
                raise RuntimeError(f"{stem} monitor child return code mismatch")
            if monitor_result.get("command") != harness_command:
                raise RuntimeError(f"{stem} monitor child command mismatch")
            if monitor_result.get("timeout_seconds") != 3600.0:
                raise RuntimeError(f"{stem} monitor watchdog mismatch")
            if completed.returncode:
                raise RuntimeError(log_path.read_text(errors="replace")[-16000:])
            result = json.loads(result_path.read_text())
            if result["status"] != "passed":
                raise RuntimeError(f"{stem} result status is not passed")
            validated_result = driver.validate_result(
                result,
                monitor_result,
                commit=commit,
                variant=variant,
                concurrency_order=PILOT_CONCURRENCIES,
                harness=harness,
                harness_sha256=args.harness_sha256,
                expected_monitor_command=harness_command,
                corpus=args.corpus,
                transformers_root=args.transformers_root,
                source_root=args.root,
                server_log_path=server_log_path,
                warmup_requests=PILOT_WARMUPS,
                measured_requests=PILOT_MEASURED,
                result_variant_label=variant,
            )
            if (
                validated_result["runtime_hardware_fingerprint"].get(
                    "live_runtime_artifact_manifest"
                )
                != live_runtime_artifacts_before
            ):
                raise RuntimeError(f"{stem} result live runtime artifacts differ")
            monitor_coverage = driver.monitor_coverage_audit(result, monitor_result)
            if not monitor_coverage["passed"]:
                raise RuntimeError(f"{stem} monitor coverage failed")
            configuration = result["configuration"]
            expected_configuration = {
                "model": driver.MODEL,
                "revision": driver.REVISION,
                "dtype": "bfloat16",
                "seed": 0,
                "tensor_parallel_size": 1,
                "output_len": driver.OUTPUT_LENGTH,
                "frame_target": driver.FRAMES,
                "max_model_len": 32768,
                "max_num_batched_tokens": 9216,
                "max_num_seqs": driver.MAX_NUM_SEQS,
                "mm_ipc_gpu_memory_gb": 2.0,
                "mm_processor_cache_gb": 0,
                "kv_cache_memory_bytes": driver.KV_CACHE_MEMORY_BYTES,
                "prefix_caching": False,
                "gpu_memory_utilization": None,
                "request_media_io_kwargs": {},
                "server_mm_processor_kwargs": {"max_pixels": driver.TOTAL_MAX_PIXELS},
                "server_limit_mm_per_prompt": {"image": 0, "video": 1},
                "request_timeout_seconds": 1200.0,
                "startup_timeout_seconds": 600.0,
                "shutdown_timeout_seconds": 60.0,
                "video_count": 8,
            }
            for key, expected in expected_configuration.items():
                if configuration.get(key) != expected:
                    raise RuntimeError(
                        f"{stem} configuration {key} mismatch: "
                        f"{configuration.get(key)!r} != {expected!r}"
                    )
            if configuration["backend_kwargs"] != driver.variant_backend_kwargs(
                variant
            ):
                raise RuntimeError(f"{stem} backend kwargs mismatch")
            if configuration["extra_server_argv"] != driver.variant_server_argv(
                variant
            ):
                raise RuntimeError(f"{stem} server argv mismatch")
            if configuration["max_num_seqs"] != driver.MAX_NUM_SEQS:
                raise RuntimeError(f"{stem} max_num_seqs mismatch")
            expected_video_kwargs = {
                "video_backend": "qwen3_vl",
                "min_frames": driver.FRAMES,
                "max_frames": driver.FRAMES,
                "backend": "pynvvideocodec",
                **driver.variant_backend_kwargs(variant),
            }
            if configuration["server_media_io_kwargs"] != {
                "video": expected_video_kwargs
            }:
                raise RuntimeError(f"{stem} server media-I/O mismatch")
            provenance = result["provenance"]
            source = provenance["source"]
            if (
                source["commit"] != commit
                or source["tree"] != driver.TREES[variant]
                or source["tracked_diff_bytes"] != 0
                or source["untracked_files"]
                or Path(source["root"]).resolve() != args.root
            ):
                raise RuntimeError(f"{stem} source provenance mismatch")
            python_provenance = provenance["python"]
            if python_provenance["packages"].get("PyNvVideoCodec") != "2.0.4":
                raise RuntimeError(f"{stem} PyNv distribution mismatch")
            if python_provenance["packages"].get("transformers") != "5.14.1":
                raise RuntimeError(f"{stem} Transformers distribution mismatch")
            expected_transformers_origin = (
                args.transformers_root / "transformers/__init__.py"
            ).resolve()
            if (
                Path(python_provenance["module_origins"]["transformers"]).resolve()
                != expected_transformers_origin
            ):
                raise RuntimeError(f"{stem} Transformers origin mismatch")
            runtime_artifacts = python_provenance["runtime_artifacts"]
            pynv_artifacts = {
                Path(artifact["path"]).name: artifact["sha256"]
                for artifact in runtime_artifacts
                if "PyNvVideoCodec" in Path(artifact["path"]).parts
                and Path(artifact["path"]).name in driver.PYNV_RUNTIME_ARTIFACT_SHA256
            }
            if pynv_artifacts != driver.PYNV_RUNTIME_ARTIFACT_SHA256:
                raise RuntimeError(f"{stem} PyNv artifact mismatch")
            transformers_artifacts = [
                artifact
                for artifact in runtime_artifacts
                if Path(artifact["path"]).resolve() == expected_transformers_origin
            ]
            if (
                len(transformers_artifacts) != 1
                or transformers_artifacts[0]["sha256"]
                != driver.TRANSFORMERS_INIT_SHA256
            ):
                raise RuntimeError(f"{stem} Transformers artifact mismatch")
            if len(result["videos"]) != 8:
                raise RuntimeError(f"{stem} video provenance count mismatch")
            for video_index, video in enumerate(result["videos"]):
                if (
                    Path(video["path"]).resolve() != videos[video_index]
                    or video["sha256"] != driver.VIDEO_SHA256
                    or video["bytes"] != driver.VIDEO_BYTES
                    or (
                        video["probe"]["width"],
                        video["probe"]["height"],
                        video["probe"]["frame_count"],
                    )
                    != (1920, 1080, 914)
                ):
                    raise RuntimeError(f"{stem} video {video_index} mismatch")
            pixel_budget = configuration["video_pixel_budget"]
            if pixel_budget["reference_width"] != driver.PIXEL_BUDGET[0]:
                raise RuntimeError(f"{stem} pixel width mismatch")
            if pixel_budget["reference_height"] != driver.PIXEL_BUDGET[1]:
                raise RuntimeError(f"{stem} pixel height mismatch")
            if pixel_budget["max_pixels_total"] != driver.TOTAL_MAX_PIXELS:
                raise RuntimeError(f"{stem} total pixel budget mismatch")
            if [
                int(block["concurrency"]) for block in result["concurrency_blocks"]
            ] != PILOT_CONCURRENCIES:
                raise RuntimeError(f"{stem} concurrency order mismatch")
            block_records = []
            for block in result["concurrency_blocks"]:
                concurrency = int(block["concurrency"])
                for phase, count in (
                    ("warmup", PILOT_WARMUPS[concurrency]),
                    ("measured", PILOT_MEASURED[concurrency]),
                ):
                    batch = block[phase]
                    aggregate = batch["aggregate"]
                    if aggregate["attempted_requests"] != count:
                        raise RuntimeError(f"{stem} {phase} attempted count mismatch")
                    if aggregate["successful_requests"] != count:
                        raise RuntimeError(f"{stem} {phase} success count mismatch")
                    if aggregate["failed_requests"] != 0:
                        raise RuntimeError(f"{stem} {phase} failures")
                    if aggregate["achieved_peak_in_flight_requests"] != concurrency:
                        raise RuntimeError(f"{stem} {phase} peak concurrency mismatch")
                    prompt_counts = {
                        record["response"]["prompt_token_count"]
                        for record in batch["records"]
                    }
                    prompt_hashes = {
                        record["response"]["prompt_token_ids_sha256"]
                        for record in batch["records"]
                    }
                    if prompt_counts != {driver.EXPECTED_PROMPT_TOKENS}:
                        raise RuntimeError(
                            f"{stem} {phase} prompt counts {prompt_counts}"
                        )
                    if prompt_hashes != {driver.EXPECTED_PROMPT_TOKEN_IDS_SHA256}:
                        raise RuntimeError(
                            f"{stem} {phase} prompt hashes {prompt_hashes}"
                        )
                block_records.append(
                    {
                        "concurrency": concurrency,
                        "warmup_requests": PILOT_WARMUPS[concurrency],
                        "measured_requests": PILOT_MEASURED[concurrency],
                        "achieved_mean_in_flight_requests": block["aggregate"][
                            "achieved_mean_in_flight_requests"
                        ],
                        "achieved_peak_in_flight_requests": block["aggregate"][
                            "achieved_peak_in_flight_requests"
                        ],
                        "failed_requests": block["aggregate"]["failed_requests"],
                        "request_throughput_per_second": block["aggregate"][
                            "request_throughput_per_second"
                        ],
                        "e2e_latency_ms": {
                            key: block["aggregate"]["latency_ms"][key]
                            for key in ("p50", "p95")
                        },
                        "measured_window_vram": driver.measured_window_vram(
                            monitor_result, block
                        ),
                    }
                )
            log_audit = driver.parse_server_log(
                server_log_path.read_text(errors="replace")
            )
            if not log_audit["gpu_kv_capacity"]:
                raise RuntimeError(f"{stem} lacks KV capacity log")
            expected_kv = {
                "tokens": 336560,
                "per_request_tokens": 32768,
                "maximum_concurrency": 10.27,
            }
            if not any(
                all(capacity[key] == value for key, value in expected_kv.items())
                for capacity in log_audit["gpu_kv_capacity"]
            ):
                raise RuntimeError(f"{stem} exact KV capacity mismatch")
            if log_audit["preemption_line_count"] or log_audit["oom_line_count"]:
                raise RuntimeError(f"{stem} preemption/OOM in server log")
            record = {
                "variant": variant,
                "commit": commit,
                "attempt": attempt,
                "result": str(result_path),
                "result_sha256": driver.sha256_file(result_path),
                "monitor": str(monitor_path),
                "monitor_sha256": driver.sha256_file(monitor_path),
                "log": str(log_path),
                "log_sha256": driver.sha256_file(log_path),
                "server_log": str(server_log_path),
                "server_log_sha256": driver.sha256_file(server_log_path),
                "blocks": block_records,
                "server_log_audit": log_audit,
                "validated_result": validated_result,
                "monitor_coverage_audit": monitor_coverage,
                "whole_run_peak_total_gpu_memory_used_mib": monitor_result[
                    "peak_memory_used_mib"
                ],
                "runtime_manifest_before": before_runtime_manifest,
                "runtime_manifest_after": after_runtime_manifest,
                "source_after_winning_attempt": source_after_attempt,
            }
            summary["pilots"].append(record)
            pilot_results[variant] = result
            pilot_runtime_fingerprints[variant] = validated_result[
                "runtime_hardware_fingerprint"
            ]
            driver.write_json(summary_path, summary)
            print(f"PASS {stem}", flush=True)
            break
        else:
            raise RuntimeError(f"too many contaminated {variant} pilots")
    token_parity = compare_tokens(pilot_results)
    if not token_parity["all_request_identity_exact"]:
        raise RuntimeError("endpoint pilot request identity differs")
    if not token_parity["all_prompt_token_ids_exact"]:
        raise RuntimeError("endpoint pilot prompt token IDs differ")
    if not token_parity["all_c1_completion_token_ids_exact"]:
        raise RuntimeError("endpoint pilot C1 completion token IDs differ")
    summary["token_parity"] = token_parity
    fingerprint_hashes = {
        fingerprint["sha256"] for fingerprint in pilot_runtime_fingerprints.values()
    }
    if len(fingerprint_hashes) != 1:
        raise RuntimeError(
            "endpoint pilot runtime/hardware fingerprints differ: "
            f"{pilot_runtime_fingerprints}"
        )
    reference_fingerprint = pilot_runtime_fingerprints[PILOT_VARIANTS[0]]
    summary["runtime_hardware_fingerprint_contract"] = {
        "status": "passed",
        "schema": reference_fingerprint["schema"],
        "sha256": reference_fingerprint["sha256"],
        "canonical": reference_fingerprint["canonical"],
        "variants": {
            variant: fingerprint["sha256"]
            for variant, fingerprint in pilot_runtime_fingerprints.items()
        },
    }
    terminal_live_runtime_artifacts = driver.capture_live_runtime_artifact_manifest(
        harness=harness,
        python=python,
        source_root=args.root,
        pythonpath_extras=[args.transformers_root],
        environment=environment,
    )
    if terminal_live_runtime_artifacts != live_runtime_artifacts_after:
        raise RuntimeError("live runtime artifacts changed at preflight end")
    summary["terminal_live_runtime_artifact_revalidation"] = (
        terminal_live_runtime_artifacts
    )
    terminal_integrity = driver.post_attempt_integrity_checks(
        root=args.root,
        commit=driver.COMMITS[PILOT_VARIANTS[-1]],
        variant=PILOT_VARIANTS[-1],
        runtime_manifests=runtime_manifests,
        runtime_label="preflight_end",
        runtime_validation_kwargs=runtime_manifest_validation_kwargs,
    )
    summary["runtime_manifest_checkpoints"].append(
        terminal_integrity["runtime_manifests"]
    )
    summary["terminal_source_revalidation"] = terminal_integrity["source"]
    summary["status"] = "passed"
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    driver.write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def sanitized_failure_category(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "controller_interrupted"
    if isinstance(error, FileExistsError):
        return "evidence_collision"
    if isinstance(error, TimeoutError):
        return "bounded_timeout"
    if isinstance(error, (ValueError, KeyError, TypeError, AssertionError)):
        return "contract_validation_failure"
    if isinstance(error, RuntimeError):
        return "runtime_validation_or_workload_failure"
    return "unexpected_controller_failure"


def record_terminal_collection_failure(error: BaseException) -> None:
    path = _ACTIVE_SUMMARY_PATH
    if path is None or not path.is_file():
        return
    summary = json.loads(path.read_text())
    if summary.get("status") != "running":
        return
    summary.update(
        {
            "status": "collection_failed",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {
                "category": sanitized_failure_category(error),
                "exception_type": type(error).__name__,
                "pixel_attempt_count": len(summary.get("pixel_attempts", [])),
                "pilot_attempt_count": len(summary.get("pilot_attempts", [])),
                "message_omitted_from_summary": True,
                "results_must_not_be_reused": True,
            },
        }
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    try:
        run_main()
    except BaseException as error:
        try:
            record_terminal_collection_failure(error)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    main()
