# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType


def load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_response(
    runner: ModuleType,
    prompt_ids: list[int],
    completion_ids: list[int],
    text: str,
) -> dict[str, object]:
    reasoning = None
    usage = {
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "total_tokens": len(prompt_ids) + len(completion_ids),
    }
    raw_response = {
        "id": "fixture-id",
        "model": runner.SERVED_MODEL_NAME,
        "prompt_token_ids": prompt_ids,
        "choices": [
            {
                "token_ids": completion_ids,
                "message": {"content": text, "reasoning_content": reasoning},
                "finish_reason": "length",
                "stop_reason": None,
            }
        ],
        "usage": usage,
        "metrics": {"fixture": True},
    }
    return {
        "id": raw_response["id"],
        "model": raw_response["model"],
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": canonical_sha256(prompt_ids),
        "completion_token_count": len(completion_ids),
        "completion_token_ids": completion_ids,
        "completion_token_ids_sha256": canonical_sha256(completion_ids),
        "prompt_and_completion_token_ids_sha256": canonical_sha256(
            {"prompt": prompt_ids, "completion": completion_ids}
        ),
        "text": text,
        "text_sha256": canonical_sha256(text),
        "reasoning_content": reasoning,
        "reasoning_content_sha256": canonical_sha256(reasoning),
        "finish_reason": "length",
        "stop_reason": None,
        "usage": usage,
        "server_metrics": raw_response["metrics"],
        "raw_response_sha256": canonical_sha256(raw_response),
        "raw_response": raw_response,
    }


def add_persistent_transport_fixture(
    batch: dict[str, object],
    *,
    phase: str,
    concurrency: int,
    prior_requests_per_slot: int,
) -> None:
    records = batch["records"]
    assert isinstance(records, list)
    for request_index, record in enumerate(records):
        assert isinstance(record, dict)
        slot = request_index % concurrency
        ordinal = prior_requests_per_slot + request_index // concurrency + 1
        record["transport"] = {
            "pool_slot_id": slot,
            "phase": phase,
            "seeded_first_wave": request_index < concurrency,
            "connection_generation": 1,
            "request_ordinal_on_generation": ordinal,
            "connection_reused": ordinal > 1,
            "prewarmed_for_measurement": phase == "measured",
            "request_connection_header": "keep-alive",
            "response_http_version": 11,
            "response_connection_header": None,
            "response_will_close": False,
            "response_persistent": True,
        }
    requests_per_slot = prior_requests_per_slot + len(records) // concurrency
    transport_audit = {
        "status": "passed",
        "phase": phase,
        "pool_size": concurrency,
        "request_count": len(records),
        "used_slot_ids": list(range(concurrency)),
        "seeded_first_wave_request_to_slot": {
            str(slot): slot for slot in range(concurrency)
        },
        "reasons": [],
        "counts_at_phase_end": {
            "open_count": concurrency,
            "reuse_count": requests_per_slot * concurrency - concurrency,
            "close_count": 0,
        },
        "slot_snapshots_at_phase_end": [
            {
                "slot_id": slot,
                "current_generation": 1,
                "warmed_generation": 1,
                "request_ordinal_on_current_generation": requests_per_slot,
                "open_count": 1,
                "reuse_count": requests_per_slot - 1,
                "close_count": 0,
                "close_reasons": {},
                "currently_open": True,
            }
            for slot in range(concurrency)
        ],
    }
    aggregate = batch["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["persistent_transport_audit"] = transport_audit


def timeout_values_in_literal_commands(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    values: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        elements = node.elts
        for index, element in enumerate(elements[:-1]):
            if (
                isinstance(element, ast.Constant)
                and element.value == "--timeout-seconds"
                and isinstance(elements[index + 1], ast.Constant)
            ):
                values.append(str(elements[index + 1].value))
    return values


def exercise_idle_evidence_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        sample_path = Path(temporary) / "outer-quiet-gate.samples.jsonl"
        base_time_ns = 1_800_000_000_000_000_000
        samples = []
        for index in range(1201):
            samples.append(
                {
                    "sample_index": index,
                    "utc": f"fixture-{index}",
                    "time_ns": base_time_ns + index * 1_000_000_000,
                    "monotonic_ns": index * 1_000_000_000,
                    "sample_gap_seconds": None if index == 0 else 1.0,
                    "reset_reasons": [],
                    "sample_error": None,
                    "gpu": {"fixture": True},
                    "cpu": {"conflicts": [], "errors": []},
                }
            )
        sample_path.write_text(
            "".join(
                json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                for sample in samples
            )
        )
        report = {
            "status": "passed",
            "passed": True,
            "process": {"script_sha256": runner.IDLE_GATE_SHA256},
            "guard_helper": {"sha256": runner.GUARD_HELPER_SHA256},
            "device": {"index": 0, "name": "fixture", "uuid": "fixture"},
            "configuration": {
                "device_index": 0,
                "required_idle_seconds": 1200.0,
                "timeout_seconds": 21600.0,
                "sample_interval_seconds": 0.2,
                "maximum_sample_gap_seconds": 1.0,
                "idle_memory_ceiling_mib": 1024,
                "idle_max_load_1m_per_cpu": 0.25,
            },
            "sample_log": {
                "path": str(sample_path),
                "bytes": sample_path.stat().st_size,
                "sha256": sha256_file(sample_path),
            },
            "sample_count": len(samples),
            "finished_time_ns": samples[-1]["time_ns"],
            "finished_utc": samples[-1]["utc"],
            "quiet_interval": {
                "sample_start_index": 0,
                "sample_end_index_inclusive": len(samples) - 1,
                "sample_count": len(samples),
                "started_time_ns": samples[0]["time_ns"],
                "finished_time_ns": samples[-1]["time_ns"],
                "started_utc": samples[0]["utc"],
                "finished_utc": samples[-1]["utc"],
                "maximum_sample_gap_seconds": 1.0,
                "duration_seconds": 1200.0,
                "all_samples_clean": True,
            },
        }
        runner.validate_idle_gate_report(
            report,
            required_idle_seconds=1200.0,
            required_timeout_seconds=21600.0,
        )
        audit = runner.validate_jsonl_binding(
            report,
            expected_suffix="outer-quiet-gate.samples.jsonl",
            expected_path=sample_path,
        )
        assert audit["sample_count"] == 1201
        assert audit["quiet_interval_recomputed"] is True
        report["configuration"]["idle_max_load_1m_per_cpu"] = 0.251
        try:
            runner.validate_idle_gate_report(
                report,
                required_idle_seconds=1200.0,
                required_timeout_seconds=21600.0,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("idle report accepted a changed host-load threshold")


def exercise_monotonic_telemetry_contract(runner: ModuleType) -> None:
    measured = {
        "started_at": "wall-clock-intentionally-not-used",
        "finished_at": "wall-clock-intentionally-not-used",
        "started_monotonic_ns": 10_000_000_000,
        "finished_monotonic_ns": 12_000_000_000,
        "measured_window_seconds": 2.0,
        "aggregate": {"measured_window_seconds": 2.0},
    }
    result = {
        "concurrency_blocks": [{"concurrency": 8, "measured": measured}],
    }
    monitor = {
        "samples": [
            {
                "time_ns": 1,
                "monotonic_ns": 10_100_000_000,
                "utc": "fixture-a",
                "memory_used_mib": 100,
                "utilization_percent": 1,
                "compute_apps": [],
                "mps_compute_apps": [],
            },
            {
                "time_ns": 9_999_999_999_999,
                "monotonic_ns": 11_000_000_000,
                "utc": "fixture-b",
                "memory_used_mib": 200,
                "utilization_percent": 2,
                "compute_apps": [],
                "mps_compute_apps": [],
            },
            {
                "time_ns": 2,
                "monotonic_ns": 11_900_000_000,
                "utc": "fixture-c",
                "memory_used_mib": 150,
                "utilization_percent": 1,
                "compute_apps": [],
                "mps_compute_apps": [],
            },
        ]
    }
    coverage = runner.monitor_coverage_audit(result, monitor)
    assert coverage["passed"] is True
    assert coverage["blocks"][0]["sample_count"] == 3
    vram = runner.measured_window_vram(monitor, result["concurrency_blocks"][0])
    assert vram["peak_total_gpu_memory_used_mib"] == 200
    assert vram["clock_policy"] == "host monotonic clock only"
    measured["finished_monotonic_ns"] = 13_000_000_000
    try:
        runner.batch_monotonic_window(measured)
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched monotonic duration was accepted")


def exercise_monitor_jsonl_terminal_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-monitor-jsonl-") as raw:
        path = Path(raw) / "fixture-gpu-monitor.samples.jsonl"
        samples = []
        for index in range(3):
            samples.append(
                {
                    "sample_index": index,
                    "utc": f"fixture-{index}",
                    "time_ns": 1_000_000_000 + index * 200_000_000,
                    "monotonic_ns": 2_000_000_000 + index * 200_000_000,
                    "sample_gap_seconds": None if index == 0 else 0.2,
                    "post_exit_telemetry": index >= 1,
                    "post_exit_ordinal": index - 1 if index >= 1 else None,
                    "monitor_errors": [],
                    "sample_error": None,
                    "owned_processes": [],
                    "process_inspection_errors": [],
                    "cpu_conflicts": [],
                    "external_gpu_processes": [],
                    "mps_compute_apps": [],
                }
            )
        path.write_text(
            "".join(
                json.dumps(sample, separators=(",", ":"), sort_keys=True) + "\n"
                for sample in samples
            )
        )
        report = {
            "configuration": {
                "maximum_sample_gap_seconds": (
                    runner.MAXIMUM_MONITOR_COVERAGE_GAP_SECONDS
                )
            },
            "sample_log": {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            },
            "sample_count": len(samples),
            "samples": samples,
            "post_exit_sample_indices": [1, 2],
        }
        audit = runner.validate_jsonl_binding(
            report,
            expected_suffix="-gpu-monitor.samples.jsonl",
            expected_path=path,
        )
        assert audit["sample_count"] == 3
        terminal_audit = runner.validate_terminal_post_exit_samples(report)
        assert terminal_audit["ordinals"] == [0, 1]
        assert terminal_audit["monotonic_gap_seconds"] == 0.2
        one_post_exit = copy.deepcopy(report)
        one_post_exit["post_exit_sample_indices"] = [2]
        try:
            runner.validate_terminal_post_exit_samples(one_post_exit)
        except RuntimeError:
            pass
        else:
            raise AssertionError("single post-exit monitor sample was accepted")
        nonterminal = copy.deepcopy(report)
        nonterminal["post_exit_sample_indices"] = [0, 2]
        try:
            runner.validate_terminal_post_exit_samples(nonterminal)
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-contiguous post-exit samples were accepted")


def exercise_huggingface_cache_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-hf-layout-fixture-") as raw:
        hf_home = Path(raw) / "huggingface"
        snapshot = (
            hf_home
            / "hub"
            / "models--Qwen--Qwen3-VL-2B-Instruct"
            / "snapshots"
            / runner.REVISION
        )
        snapshot.mkdir(parents=True)
        assert runner.huggingface_cache_environment(snapshot) == {
            "HF_HOME": str(hf_home.resolve()),
            "HF_HUB_CACHE": str((hf_home / "hub").resolve()),
            "HUGGINGFACE_HUB_CACHE": str((hf_home / "hub").resolve()),
        }
        wrong_snapshot = (
            hf_home / "hub" / "models--wrong" / "snapshots" / runner.REVISION
        )
        wrong_snapshot.mkdir(parents=True)
        try:
            runner.huggingface_cache_environment(wrong_snapshot)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong model cache root was accepted")


def exercise_runtime_hardware_fingerprint_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-runtime-fingerprint-") as raw:
        root = Path(raw)
        source_root = root / "source"
        paths = {
            "python": root / "python",
            "torch": root / "torch.so",
            "numpy": root / "numpy.so",
            "transformers": root / "transformers.py",
            "torch._C": root / "torch_C.so",
            "numpy._core._multiarray_umath": root / "numpy_multiarray_umath.so",
            "PyNvVideoCodec": root / "PyNvVideoCodec.so",
            "vllm": source_root / "vllm/_C.abi3.so",
            "nvcc": root / "nvcc",
        }
        artifacts = []
        for label, path in paths.items():
            content = f"fixture-{label}".encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            artifacts.append(
                {
                    "path": str(path),
                    "resolved_path": str(path.resolve()),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        vllm_link = paths["vllm"]
        vllm_artifact = next(
            item for item in artifacts if item["path"] == str(vllm_link)
        )
        vllm_content = vllm_link.read_bytes()
        vllm_link.unlink()
        vllm_target = root / "zz-precompiled" / "_C.abi3.so"
        vllm_target.parent.mkdir(parents=True)
        vllm_target.write_bytes(vllm_content)
        vllm_link.symlink_to(vllm_target)
        vllm_artifact["resolved_path"] = str(vllm_target.resolve())

        nested_target = (
            root / "aa-precompiled" / "vllm_flash_attn" / "_vllm_fa2_C.abi3.so"
        )
        nested_target.parent.mkdir(parents=True)
        nested_target.write_bytes(b"fixture-vllm-nested")
        nested_link = source_root / "vllm" / "vllm_flash_attn" / nested_target.name
        nested_link.parent.mkdir(parents=True)
        nested_link.symlink_to(nested_target)
        paths["vllm_nested"] = nested_link
        artifacts.append(
            {
                "path": str(nested_link),
                "resolved_path": str(nested_target.resolve()),
                "bytes": nested_target.stat().st_size,
                "sha256": sha256_file(nested_target),
            }
        )
        environment = {
            name: "fixture"
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
        environment["OMP_NUM_THREADS"] = "8"
        result = {
            "provenance": {
                "source": {"root": str(source_root)},
                "python": {
                    "implementation": "CPython",
                    "python_version": "3.12.0",
                    "executable": str(paths["python"]),
                    "packages": {
                        "vllm": "fixture",
                        "torch": "2.0",
                        "numpy": "2.1",
                        "transformers": "5.14.1",
                        "PyNvVideoCodec": "2.0.4",
                    },
                    "module_origins": {
                        name: str(paths[name])
                        for name in (
                            "torch",
                            "numpy",
                            "transformers",
                            "PyNvVideoCodec",
                        )
                    },
                    "native_module_origins": {
                        name: str(paths[name])
                        for name in ("torch._C", "numpy._core._multiarray_umath")
                    },
                    "runtime_artifacts": artifacts,
                    "torch_runtime": {
                        "torch_version": "2.0",
                        "compiled_cuda_version": "13.0",
                        "cudnn_version": 9999,
                        "nvcc": {
                            "path": str(paths["nvcc"]),
                            "resolved_path": str(paths["nvcc"]),
                            "bytes": paths["nvcc"].stat().st_size,
                            "sha256": sha256_file(paths["nvcc"]),
                            "version_output": "Cuda compilation tools, release 13.0",
                        },
                    },
                },
                "hardware": {
                    "nvidia_smi_output": (
                        "0, NVIDIA A100-SXM4-80GB, GPU-fixture, 999.0, "
                        "81920, 8.0, 00000000:00:00.0, P0, 0, 0"
                    ),
                    "logical_cpus": 64,
                    "cuda_visible_devices": "0",
                    "platform": "Linux-fixture",
                    "uname": ["Linux", "fixture", "1", "1", "x86_64", ""],
                },
            },
            "server": {"performance_environment": environment},
        }
        first = runner.canonical_runtime_fingerprint(result)
        second = runner.canonical_runtime_fingerprint(copy.deepcopy(result))
        assert first == second
        assert first["schema"] == "pynv-runtime-hardware-fingerprint-v1"
        live_manifest = first["live_runtime_artifact_manifest"]
        assert live_manifest["vllm_native_paths"] == sorted(
            [str(vllm_target.resolve()), str(nested_target.resolve())]
        )
        assert (
            runner.revalidate_live_runtime_artifact_manifest_binding(
                live_manifest, label="external-native-links"
            )
            == live_manifest
        )
        assert len(first["canonical"]["python"]["vllm_compiled_artifacts"]) == 2

        omitted_nested = copy.deepcopy(live_manifest)
        omitted_nested["vllm_native_paths"].remove(str(nested_target.resolve()))
        omitted_nested["artifacts"] = [
            item
            for item in omitted_nested["artifacts"]
            if item["resolved_path"] != str(nested_target.resolve())
        ]
        omitted_nested["sha256"] = runner.sha256_json(
            {
                field: omitted_nested[field]
                for field in (
                    "artifacts",
                    "required_bindings",
                    "pynv_native_paths",
                    "vllm_native_paths",
                    "nvcc",
                )
            }
        )
        try:
            runner.revalidate_live_runtime_artifact_manifest_binding(
                omitted_nested, label="omitted-nested-native"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("omitted nested vLLM native was accepted")

        direct_target = copy.deepcopy(live_manifest)
        direct_artifact = next(
            item
            for item in direct_target["artifacts"]
            if item["resolved_path"] == str(vllm_target.resolve())
        )
        direct_artifact["path"] = direct_artifact["resolved_path"]
        direct_target["sha256"] = runner.sha256_json(
            {
                field: direct_target[field]
                for field in (
                    "artifacts",
                    "required_bindings",
                    "pynv_native_paths",
                    "vllm_native_paths",
                    "nvcc",
                )
            }
        )
        try:
            runner.revalidate_live_runtime_artifact_manifest_binding(
                direct_target, label="direct-resolved-native"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("direct resolved vLLM native path was accepted")

        traversal_alias = copy.deepcopy(live_manifest)
        aliased_artifact = next(
            item
            for item in traversal_alias["artifacts"]
            if item["resolved_path"] == str(vllm_target.resolve())
        )
        aliased_artifact["path"] = str(
            source_root / "vllm" / ".." / "vllm" / vllm_link.name
        )
        traversal_alias["sha256"] = runner.sha256_json(
            {
                field: traversal_alias[field]
                for field in (
                    "artifacts",
                    "required_bindings",
                    "pynv_native_paths",
                    "vllm_native_paths",
                    "nvcc",
                )
            }
        )
        try:
            runner.revalidate_live_runtime_artifact_manifest_binding(
                traversal_alias, label="traversal-alias"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("non-normalized vLLM native alias was accepted")
        changed = copy.deepcopy(result)
        changed["provenance"]["hardware"]["nvidia_smi_output"] = changed["provenance"][
            "hardware"
        ]["nvidia_smi_output"].replace("999.0", "999.1")
        assert (
            runner.canonical_runtime_fingerprint(changed)["sha256"] != first["sha256"]
        )
        native_changed = copy.deepcopy(result)
        native_path = paths["torch._C"]
        native_artifact = next(
            item
            for item in native_changed["provenance"]["python"]["runtime_artifacts"]
            if Path(item["resolved_path"]) == native_path.resolve()
        )
        native_artifact["sha256"] = "f" * 64
        try:
            runner.canonical_runtime_fingerprint(native_changed)
        except RuntimeError:
            pass
        else:
            raise AssertionError("forged native artifact hash was accepted")
        for artifact_label in (
            "python",
            "torch._C",
            "numpy._core._multiarray_umath",
            "transformers",
            "PyNvVideoCodec",
            "vllm",
            "vllm_nested",
            "nvcc",
        ):
            artifact_path = paths[artifact_label]
            original = artifact_path.read_bytes()
            artifact_path.write_bytes(original + b"-mutated")
            try:
                runner.canonical_runtime_fingerprint(result)
            except RuntimeError:
                pass
            else:
                raise AssertionError(
                    f"on-disk runtime mutation was accepted: {artifact_label}"
                )
            finally:
                artifact_path.write_bytes(original)
        invalid_mutations = (
            lambda value: value["provenance"]["python"][
                "runtime_artifacts"
            ].__setitem__(
                slice(None),
                [
                    item
                    for item in value["provenance"]["python"]["runtime_artifacts"]
                    if not str(item["resolved_path"]).endswith("_C.abi3.so")
                ],
            ),
            lambda value: value["provenance"]["python"]["native_module_origins"].pop(
                "torch._C"
            ),
        )
        for mutation in invalid_mutations:
            invalid = copy.deepcopy(result)
            mutation(invalid)
            try:
                runner.canonical_runtime_fingerprint(invalid)
            except RuntimeError:
                pass
            else:
                raise AssertionError("incomplete runtime/HW fingerprint was accepted")
        changed = copy.deepcopy(result)
        changed["server"]["performance_environment"]["PYTHONHASHSEED"] = "1"
        assert (
            runner.canonical_runtime_fingerprint(changed)["sha256"] != first["sha256"]
        )
        changed = copy.deepcopy(result)
        changed["server"]["performance_environment"]["OMP_NUM_THREADS"] = "16"
        assert (
            runner.canonical_runtime_fingerprint(changed)["sha256"] != first["sha256"]
        )
        symlink_target_a = root / "symlink-target-a.so"
        symlink_target_b = root / "symlink-target-b.so"
        symlink_path = root / "runtime-link.so"
        symlink_target_a.write_bytes(b"same-content")
        symlink_target_b.write_bytes(b"same-content")
        symlink_path.symlink_to(symlink_target_a)
        symlink_binding = {
            "path": str(symlink_path),
            "resolved_path": str(symlink_target_a.resolve()),
            "bytes": symlink_target_a.stat().st_size,
            "sha256": sha256_file(symlink_target_a),
        }
        runner.stable_rehash_artifact(symlink_binding, label="symlink-fixture")
        symlink_path.unlink()
        symlink_path.symlink_to(symlink_target_b)
        try:
            runner.stable_rehash_artifact(symlink_binding, label="symlink-fixture")
        except RuntimeError:
            pass
        else:
            raise AssertionError("runtime artifact symlink substitution was accepted")


def exercise_recorded_source_contract(runner: ModuleType) -> None:
    variant = "pr-head"
    record = {
        "variant": variant,
        "commit": runner.COMMITS[variant],
        "tree": runner.TREES[variant],
        "status": "",
        "source_harness_exists": False,
        "chat_completion_protocol_artifacts": {
            "vllm/entrypoints/openai/chat_completion/protocol.py": (
                runner.CHAT_COMPLETION_PROTOCOL_SHA256
            ),
            "vllm/entrypoints/openai/chat_completion/serving.py": (
                runner.CHAT_COMPLETION_SERVING_SHA256
            ),
        },
        "ignored_python_bytecode_scan": {
            "scope": "vllm/**",
            "excluded_shared_venv": ".venv",
            "matched_paths": [],
            "passed": True,
        },
    }
    assert runner.validate_recorded_source(record)["variant"] == variant
    assert (
        runner.validate_recorded_source(record, expected_variant=variant)["commit"]
        == runner.COMMITS[variant]
    )
    tampered = copy.deepcopy(record)
    tampered["ignored_python_bytecode_scan"]["matched_paths"] = [
        "vllm/__pycache__/stale.pyc"
    ]
    try:
        runner.validate_recorded_source(tampered)
    except RuntimeError:
        pass
    else:
        raise AssertionError("recorded stale source bytecode was accepted")


def exercise_monitor_completion_contract(runner: ModuleType) -> None:
    assert callable(runner.validate_contamination_retry_evidence)
    assert not hasattr(runner, "classify_monitor_completion")


def exercise_actual_clean_monitor_contract(runner: ModuleType, monitor: Path) -> None:
    bundle_root = monitor.resolve().parent
    fixture_controller_root = (bundle_root / "fixture-controller-root").resolve()
    with tempfile.TemporaryDirectory(prefix="a100-clean-monitor-") as raw:
        output = Path(raw) / "clean-true-gpu-monitor.json"
        child_command = ["/bin/true"]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import test_refined_gpu_guards as fixture; "
                    "fixture.guard.ProbeWorker=fixture.FixtureProbeWorker; "
                    "fixture.guard.scan_cpu_processes=lambda **unused: "
                    "{'conflicts': [], 'observers': [], 'errors': [], "
                    "'owned_processes': [], 'host_load': "
                    "{'cpu_count': 1, 'load_1m': 0.0, "
                    "'load_1m_per_cpu': 0.0}}; "
                    "monitor=fixture.import_script("
                    "'a100_fixture_monitor', "
                    "'run_with_gpu_monitor_refined.py'); "
                    "monitor.capture_owned_tree=lambda **unused: []; "
                    "monitor.sys.argv=[str(fixture.BUNDLE_ROOT / "
                    "'run_with_gpu_monitor_refined.py'), '--output', "
                    "sys.argv[2], '--conflicting-controller-root', "
                    "str(fixture.FIXTURE_CONTROLLER_ROOT), '--', *sys.argv[3:]]; "
                    "monitor.main()"
                ),
                "monitor",
                str(output),
                *child_command,
            ],
            cwd=bundle_root,
            check=False,
            timeout=15.0,
        )
        assert completed.returncode == 0
        report, sample_audit = runner.validate_monitor_evidence(
            output,
            expected_command=child_command,
            watchdog_pair=(3600.0, 120.0),
            conflicting_controller_roots=[fixture_controller_root],
        )
        assert report["status"] == "passed"
        assert sample_audit["sample_count"] == report["sample_count"]
        terminal_indices = report["post_exit_sample_indices"]
        assert terminal_indices == [
            report["sample_count"] - 2,
            report["sample_count"] - 1,
        ]
        terminal = [report["samples"][index] for index in terminal_indices]
        assert [sample["post_exit_ordinal"] for sample in terminal] == [0, 1]
        assert all(sample["post_exit_telemetry"] is True for sample in terminal)
        assert (
            0.0
            < (terminal[1]["monotonic_ns"] - terminal[0]["monotonic_ns"]) / 1e9
            <= 1.0
        )


def exercise_preflight_attempt_reconstruction_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-preflight-layout-") as raw:
        root = Path(raw).resolve()
        python = root / "python"
        monitor = root / "monitor.py"
        source_root = root / "source"
        transformers_root = root / "transformers-overlay"
        pixel_preflight = root / "pixel.py"
        harness = root / "harness.py"
        corpus = root / "corpus"
        videos = [corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]

        def attempt(kind: str, variant: str | None = None) -> dict[str, object]:
            index = 3
            if kind == "pixel":
                stem = f"pixel-parity-a{index:02d}"
                result = root / f"{stem}.json"
                child = [
                    str(python),
                    str(pixel_preflight),
                    "--root",
                    str(source_root),
                    "--python",
                    str(python),
                    "--transformers-root",
                    str(transformers_root),
                    "--video",
                    str(videos[0]),
                    "--output",
                    str(result),
                ]
                record: dict[str, object] = {"attempt": index, "stem": stem}
            else:
                assert variant is not None
                stem = f"pilot-{variant}-c1-8-32-a{index:02d}"
                result = root / f"{stem}.json"
                child = runner.build_harness_command(
                    python=python,
                    harness=harness,
                    source_root=source_root,
                    transformers_root=transformers_root,
                    variant=variant,
                    variant_label=f"pilot-{variant}",
                    corpus=corpus,
                    videos=videos,
                    concurrencies=[1, 8, 32],
                    port=18600,
                    result_path=result,
                    warmup_requests={1: 8, 8: 8, 32: 32},
                    measured_requests={1: 8, 8: 8, 32: 32},
                )
                record = {
                    "attempt": index,
                    "stem": stem,
                    "variant": variant,
                    "commit": runner.COMMITS[variant],
                    "server_log": str(root / f"{stem}.server.log"),
                }
            monitor_path = root / f"{stem}-gpu-monitor.json"
            record.update(
                {
                    "idle_gate": str(root / f"{stem}-idle-gate.json"),
                    "result": str(result),
                    "monitor": str(monitor_path),
                    "log": str(root / f"{stem}.log"),
                    "idle_gate_sample_log_audit": {
                        "path": str(root / f"{stem}-idle-gate.samples.jsonl")
                    },
                    "monitor_sample_log_audit": {
                        "path": str(root / f"{stem}-gpu-monitor.samples.jsonl")
                    },
                    "command": child,
                }
            )
            return record

        common = {
            "preflight_root": root,
            "python": python,
            "monitor": monitor,
            "source_root": source_root,
            "transformers_root": transformers_root,
            "pixel_preflight": pixel_preflight,
            "harness": harness,
            "corpus": corpus,
            "videos": videos,
            "port": 18600,
            "conflicting_controller_roots": [root / "foreign"],
        }
        pixel = attempt("pixel")
        runner.validate_preflight_attempt_command_and_paths(
            pixel, kind="pixel", **common
        )
        pilot = attempt("pilot", "pr-head")
        runner.validate_preflight_attempt_command_and_paths(
            pilot, kind="pilot", **common
        )
        mutations = []
        changed_command = copy.deepcopy(pilot)
        changed_command["command"][-1] = "spliced"
        mutations.append(changed_command)
        changed_result = copy.deepcopy(pilot)
        changed_result["result"] = str(root / "spliced.json")
        mutations.append(changed_result)
        changed_monitor_sample = copy.deepcopy(pilot)
        changed_monitor_sample["monitor_sample_log_audit"]["path"] = str(
            root / "spliced.samples.jsonl"
        )
        mutations.append(changed_monitor_sample)
        changed_identity = copy.deepcopy(pilot)
        changed_identity["commit"] = runner.COMMITS["upstream"]
        mutations.append(changed_identity)
        for tampered in mutations:
            try:
                runner.validate_preflight_attempt_command_and_paths(
                    tampered, kind="pilot", **common
                )
            except RuntimeError:
                pass
            else:
                raise AssertionError("spliced preflight evidence was accepted")


def exercise_result_integrity_tamper_contract(
    runner: ModuleType, harness: ModuleType
) -> None:
    videos = [
        {
            "video_index": video_index,
            "path": f"/fixture/video-{video_index}.mp4",
            "file_uri": f"file:///fixture/video-{video_index}.mp4",
            "sha256": f"fixture-video-{video_index}",
            "probe": {
                "width": 250,
                "height": 125,
                "frame_count": 32,
                "frames_per_second": 30.0,
            },
        }
        for video_index in range(8)
    ]
    payload_records = []
    for video_index in range(8):
        payload = runner.expected_chat_payload(videos[video_index]["file_uri"])
        payload_records.append(
            {
                "video_index": video_index,
                "video_path": f"/fixture/video-{video_index}.mp4",
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
            }
        )

    def batch(phase: str, start_ns: int) -> dict[str, object]:
        prompt_ids = [1, 2, 3]
        completion_ids = [4, 5]
        response = synthetic_response(runner, prompt_ids, completion_ids, "fixture")
        request_started_ns = start_ns + 100_000_000
        request_finished_ns = start_ns + 900_000_000
        record = {
            "phase": phase,
            "block_index": 0,
            "concurrency": 1,
            "request_index": 0,
            "global_request_index": 0 if phase == "warmup" else 1,
            "video_index": 0,
            "video_path": "/fixture/video-0.mp4",
            "video_file_uri": "file:///fixture/video-0.mp4",
            "video_sha256": "fixture-video-0",
            "video_work": {
                "source_width": 250,
                "source_height": 125,
                "source_frame_count": 32,
                "sampled_frames": 32,
                "sampled_source_megapixels_estimate": 1.0,
                "derivation": "Qwen3-VL equal min/max frame clamp",
            },
            "request_payload_sha256": payload_records[0]["payload_sha256"],
            "payload": payload_records[0]["payload"],
            "status": "passed",
            "response": response,
            "started_monotonic_ns": request_started_ns,
            "finished_monotonic_ns": request_finished_ns,
            "start_offset_seconds": 0.1,
            "finish_offset_seconds": 0.9,
            "latency_seconds": 0.8,
            "latency_ms": 800.0,
        }
        aggregate = harness.batch_aggregate([record], 1.0)
        result_batch = {
            "started_monotonic_ns": start_ns,
            "finished_monotonic_ns": start_ns + 1_000_000_000,
            "measured_window_seconds": 1.0,
            "records": [record],
            "aggregate": aggregate,
        }
        add_persistent_transport_fixture(
            result_batch,
            phase=phase,
            concurrency=1,
            prior_requests_per_slot=0 if phase == "warmup" else 1,
        )
        return result_batch

    measured_batch = batch("measured", 12_000_000_000)
    result = {
        "videos": videos,
        "request_payloads_by_video": payload_records,
        "concurrency_blocks": [
            {
                "concurrency": 1,
                "warmup": batch("warmup", 10_000_000_000),
                "measured": measured_batch,
                "aggregate": copy.deepcopy(measured_batch["aggregate"]),
            }
        ],
    }
    audit = runner.validate_result_integrity(result)
    assert audit == {
        "status": "passed",
        "payload_count": 8,
        "batch_count": 2,
        "response_count": 2,
        "policy": "all payload, token/text, aggregate hash/count fields recomputed",
    }
    mutations = (
        lambda value: value["request_payloads_by_video"][0].update(
            payload={"changed": True}
        ),
        lambda value: value["concurrency_blocks"][0]["measured"]["records"][0][
            "response"
        ].update(prompt_token_ids_sha256="0" * 64),
        lambda value: value["concurrency_blocks"][0]["measured"]["aggregate"].update(
            attempted_requests=999
        ),
        lambda value: value["concurrency_blocks"][0]["measured"]["records"][0].update(
            global_request_index=999
        ),
        lambda value: value["concurrency_blocks"][0]["measured"]["records"][0].update(
            started_monotonic_ns=11_000_000_000
        ),
        lambda value: value["concurrency_blocks"][0]["measured"]["records"][0][
            "response"
        ]["raw_response"].update(model="changed"),
        lambda value: value["videos"][0].update(file_uri="file:///fixture/other.mp4"),
    )
    for mutation in mutations:
        tampered = copy.deepcopy(result)
        mutation(tampered)
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError("tampered result integrity evidence was accepted")

    measured_aggregate = result["concurrency_blocks"][0]["measured"]["aggregate"]

    def changed(value: object) -> object:
        if value is None:
            return 1
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return value + 1
        if isinstance(value, str):
            return value + "-changed"
        if isinstance(value, list):
            return [*value, {"changed": True}]
        if isinstance(value, dict):
            return {**value, "changed": True}
        raise AssertionError(f"unhandled aggregate fixture value: {value!r}")

    for field in measured_aggregate:
        if field == "latency_ms":
            continue
        tampered = copy.deepcopy(result)
        aggregate = tampered["concurrency_blocks"][0]["measured"]["aggregate"]
        aggregate[field] = changed(aggregate[field])
        tampered["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(aggregate)
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"tampered aggregate field was accepted: {field}")

    for field in measured_aggregate["latency_ms"]:
        tampered = copy.deepcopy(result)
        aggregate = tampered["concurrency_blocks"][0]["measured"]["aggregate"]
        aggregate["latency_ms"][field] = changed(aggregate["latency_ms"][field])
        tampered["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(aggregate)
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"tampered latency field was accepted: {field}")

    transport_audit = measured_aggregate["persistent_transport_audit"]
    for field in transport_audit:
        if field in {"counts_at_phase_end", "slot_snapshots_at_phase_end"}:
            continue
        tampered = copy.deepcopy(result)
        audit = tampered["concurrency_blocks"][0]["measured"]["aggregate"][
            "persistent_transport_audit"
        ]
        audit[field] = changed(audit[field])
        tampered["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(
            tampered["concurrency_blocks"][0]["measured"]["aggregate"]
        )
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"tampered transport aggregate field was accepted: {field}"
            )
    for field in transport_audit["counts_at_phase_end"]:
        tampered = copy.deepcopy(result)
        audit = tampered["concurrency_blocks"][0]["measured"]["aggregate"][
            "persistent_transport_audit"
        ]
        audit["counts_at_phase_end"][field] = changed(
            audit["counts_at_phase_end"][field]
        )
        tampered["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(
            tampered["concurrency_blocks"][0]["measured"]["aggregate"]
        )
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"tampered transport count was accepted: {field}")
    for field in transport_audit["slot_snapshots_at_phase_end"][0]:
        tampered = copy.deepcopy(result)
        audit = tampered["concurrency_blocks"][0]["measured"]["aggregate"][
            "persistent_transport_audit"
        ]
        audit["slot_snapshots_at_phase_end"][0][field] = changed(
            audit["slot_snapshots_at_phase_end"][0][field]
        )
        tampered["concurrency_blocks"][0]["aggregate"] = copy.deepcopy(
            tampered["concurrency_blocks"][0]["measured"]["aggregate"]
        )
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"tampered transport slot snapshot was accepted: {field}"
            )
    raw_transport = result["concurrency_blocks"][0]["measured"]["records"][0][
        "transport"
    ]
    for field in raw_transport:
        tampered = copy.deepcopy(result)
        transport = tampered["concurrency_blocks"][0]["measured"]["records"][0][
            "transport"
        ]
        transport[field] = changed(transport[field])
        try:
            runner.validate_result_integrity(tampered)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"tampered raw transport field was accepted: {field}")

    tampered = copy.deepcopy(result)
    tampered["concurrency_blocks"][0]["aggregate"]["attempted_requests"] = 2
    try:
        runner.validate_result_integrity(tampered)
    except RuntimeError:
        pass
    else:
        raise AssertionError("block/measured aggregate mismatch was accepted")


def exercise_terminal_failure_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-terminal-fixture-") as raw:
        results = Path(raw) / "results"
        results.mkdir()
        assert (
            runner.configured_results_root([f"--results={results}"])
            == results.resolve()
        )
        assert (
            runner.configured_results_root(
                ["--results", str(results), f"--results={results}"]
            )
            is None
        )
        previous_argv = sys.argv
        try:
            sys.argv = ["runner", "--results", str(results)]
            runner.record_terminal_failure(RuntimeError("private path must not leak"))
        finally:
            sys.argv = previous_argv
        manifest = json.loads((results / "matrix-manifest.json").read_text())
        assert manifest["status"] == "collection_failed"
        assert manifest["terminal_failure"] == {
            "category": "validation_or_workload_failure",
            "exception_type": "RuntimeError",
            "message_recorded": False,
            "traceback_recorded": False,
        }
        assert "private path must not leak" not in json.dumps(manifest)


def exercise_preflight_terminal_failure_contract(pilot: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-preflight-terminal-fixture-") as raw:
        results = Path(raw) / "results"
        results.mkdir()
        assert (
            pilot.configured_results_root([f"--results={results}"]) == results.resolve()
        )
        previous_argv = sys.argv
        try:
            sys.argv = ["pilot", "--results", str(results)]
            pilot.record_terminal_failure(RuntimeError("private value"))
        finally:
            sys.argv = previous_argv
        summary_path = results / "pilot-summary.json"
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "preflight_failed"
        assert "private value" not in summary_path.read_text()
        summary_path.write_text('{"status":"invalid_input_parity"}\n')
        try:
            sys.argv = ["pilot", "--results", str(results)]
            pilot.record_terminal_failure(RuntimeError("private token value"))
        finally:
            sys.argv = previous_argv
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "invalid_input_parity"
        assert summary["terminal_failure"]["category"] == "input_parity_failure"
        assert summary["terminal_failure"]["message_recorded"] is False
        summary_path.write_text('{"status":"timing_passed_completion_mismatch"}\n')
        try:
            sys.argv = ["pilot", "--results", str(results)]
            pilot.record_terminal_failure(RuntimeError("private completion value"))
        finally:
            sys.argv = previous_argv
        summary = json.loads(summary_path.read_text())
        assert summary["status"] == "timing_passed_completion_mismatch"
        assert summary["terminal_failure"]["category"] == (
            "completion_or_text_mismatch"
        )
        assert "private completion value" not in summary_path.read_text()


def synthetic_cells(
    runner: ModuleType, harness: ModuleType, root: Path
) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    videos = [
        {
            "video_index": video_index,
            "path": f"/fixture/video-{video_index}.mp4",
            "file_uri": f"file:///fixture/video-{video_index}.mp4",
            "sha256": f"video-{video_index}",
            "probe": {
                "width": 250,
                "height": 125,
                "frame_count": 32,
                "frames_per_second": 30.0,
            },
        }
        for video_index in range(8)
    ]
    payload_records = []
    for video_index in range(8):
        payload = runner.expected_chat_payload(videos[video_index]["file_uri"])
        payload_records.append(
            {
                "video_index": video_index,
                "video_path": f"/fixture/video-{video_index}.mp4",
                "payload": payload,
                "payload_sha256": canonical_sha256(payload),
            }
        )
    for rep in range(1, runner.REPETITIONS + 1):
        for variant in runner.COMMITS:
            configuration = {
                field: None for field in runner.WORKLOAD_PARITY_CONFIGURATION_FIELDS
            }
            configuration.update(runner.expected_treatment_configuration(variant))
            blocks = []
            global_request_index = 0
            for block_index, concurrency in enumerate(runner.MEASURED_REQUESTS):
                block = {"concurrency": concurrency}
                for phase, count in (
                    ("warmup", runner.WARMUP_REQUESTS[concurrency]),
                    ("measured", runner.MEASURED_REQUESTS[concurrency]),
                ):
                    elapsed_seconds = count / concurrency
                    monotonic_start_ns = (
                        1_000_000_000_000
                        + rep * 100_000_000_000
                        + block_index * 10_000_000_000
                        + (0 if phase == "warmup" else 5_000_000_000)
                    )
                    records = []
                    for request_index in range(count):
                        video_index = request_index % 8
                        prompt_ids = [11, video_index, request_index]
                        completion_ids = [21, 22, 23]
                        text = "fixture completion"
                        start_offset = request_index // concurrency
                        request_started_ns = monotonic_start_ns + int(
                            start_offset * 1e9
                        )
                        request_finished_ns = request_started_ns + 900_000_000
                        records.append(
                            {
                                "phase": phase,
                                "block_index": block_index,
                                "concurrency": concurrency,
                                "request_index": request_index,
                                "global_request_index": global_request_index,
                                "video_index": video_index,
                                "video_path": f"/fixture/video-{video_index}.mp4",
                                "video_file_uri": f"file:///fixture/video-{video_index}.mp4",
                                "video_sha256": f"video-{video_index}",
                                "video_work": {
                                    "source_width": 250,
                                    "source_height": 125,
                                    "source_frame_count": 32,
                                    "sampled_frames": 32,
                                    "sampled_source_megapixels_estimate": 1.0,
                                    "derivation": (
                                        "Qwen3-VL equal min/max frame clamp"
                                    ),
                                },
                                "request_payload_sha256": payload_records[video_index][
                                    "payload_sha256"
                                ],
                                "payload": payload_records[video_index]["payload"],
                                "status": "passed",
                                "response": synthetic_response(
                                    runner, prompt_ids, completion_ids, text
                                ),
                                "started_monotonic_ns": request_started_ns,
                                "finished_monotonic_ns": request_finished_ns,
                                "start_offset_seconds": float(start_offset),
                                "finish_offset_seconds": float(start_offset + 0.9),
                                "latency_seconds": 0.9,
                                "latency_ms": 900.0,
                            }
                        )
                        global_request_index += 1
                    aggregate = harness.batch_aggregate(records, elapsed_seconds)
                    phase_batch = {
                        "started_monotonic_ns": monotonic_start_ns,
                        "finished_monotonic_ns": monotonic_start_ns
                        + int(elapsed_seconds * 1e9),
                        "measured_window_seconds": elapsed_seconds,
                        "records": records,
                        "aggregate": aggregate,
                    }
                    add_persistent_transport_fixture(
                        phase_batch,
                        phase=phase,
                        concurrency=concurrency,
                        prior_requests_per_slot=(
                            0
                            if phase == "warmup"
                            else runner.WARMUP_REQUESTS[concurrency] // concurrency
                        ),
                    )
                    block[phase] = phase_batch
                block["aggregate"] = copy.deepcopy(block["measured"]["aggregate"])
                blocks.append(block)
            result = {
                "configuration": configuration,
                "videos": copy.deepcopy(videos),
                "request_payloads_by_video": copy.deepcopy(payload_records),
                "concurrency_blocks": blocks,
            }
            result_path = root / f"r{rep:02d}-{variant}.json"
            result_path.write_text(json.dumps(result, sort_keys=True) + "\n")
            result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
            cells.append(
                {
                    "rep": rep,
                    "variant": variant,
                    "output": str(result_path),
                    "winning_attempt": 1,
                    "attempts": [
                        {
                            "attempt": 1,
                            "accepted": True,
                            "result_sha256": result_sha256,
                        }
                    ],
                }
            )
    return cells


def update_result_and_acceptance(
    cell: dict[str, object], mutate: object, harness: ModuleType
) -> None:
    path = Path(str(cell["output"]))
    result = json.loads(path.read_text())
    mutate(result)
    for block in result["concurrency_blocks"]:
        concurrency = int(block["concurrency"])
        for phase in ("warmup", "measured"):
            batch = block[phase]
            batch["aggregate"] = harness.batch_aggregate(
                batch["records"], batch["measured_window_seconds"]
            )
            add_persistent_transport_fixture(
                batch,
                phase=phase,
                concurrency=concurrency,
                prior_requests_per_slot=(
                    0
                    if phase == "warmup"
                    else len(block["warmup"]["records"]) // concurrency
                ),
            )
        block["aggregate"] = copy.deepcopy(block["measured"]["aggregate"])
    path.write_text(json.dumps(result, sort_keys=True) + "\n")
    attempts = cell["attempts"]
    assert isinstance(attempts, list)
    attempts[0]["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def exercise_strict_token_statuses(runner: ModuleType, harness: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-three-arm-token-fixture-") as raw:
        cells = synthetic_cells(runner, harness, Path(raw))
        exact = runner.strict_token_text_audit(cells)
        assert exact["schema"] == "pynv-endpoint-strict-token-parity-v1"
        assert exact["status"] == "passed_exact"
        assert exact["compared_response_pair_count"] == 11_088
        assert exact["accepted_result_count"] == 18
        rep_one_results = {
            str(cell["variant"]): json.loads(Path(str(cell["output"])).read_text())
            for cell in cells
            if cell["rep"] == 1
        }
        pilot_parity = runner.validate_three_way_pilot_parity(rep_one_results)
        assert pilot_parity["status"] == "passed_exact"
        assert pilot_parity["compared_response_pair_count"] == 1848
        assert set(pilot_parity["comparisons"]) == {
            comparison[0] for comparison in runner.PAIRWISE_COMPARISONS
        }
        generation_divergence = copy.deepcopy(rep_one_results)
        generation_response = generation_divergence["pr-head"]["concurrency_blocks"][0][
            "measured"
        ]["records"][0]["response"]
        generation_response["completion_token_ids"] = [999]
        generation_status = runner.validate_three_way_pilot_parity(
            generation_divergence
        )
        assert generation_status["status"] == "completion_or_text_mismatch"
        assert "fixture completion" not in json.dumps(
            generation_status["mismatch_details"]
        )
        input_divergence = copy.deepcopy(rep_one_results)
        input_response = input_divergence["pr-head"]["concurrency_blocks"][0][
            "measured"
        ]["records"][0]["response"]
        input_response["prompt_token_ids"] = [999]
        assert (
            runner.validate_three_way_pilot_parity(input_divergence)["status"]
            == "failed_input_parity"
        )

        head_cell = next(
            cell for cell in cells if cell["rep"] == 1 and cell["variant"] == "pr-head"
        )

        def mutate_completion(result: dict[str, object]) -> None:
            response = result["concurrency_blocks"][0]["measured"]["records"][0][
                "response"
            ]
            response["completion_token_ids"] = [21, 999, 23]
            response["completion_token_ids_sha256"] = canonical_sha256(
                response["completion_token_ids"]
            )
            response["prompt_and_completion_token_ids_sha256"] = canonical_sha256(
                {
                    "prompt": response["prompt_token_ids"],
                    "completion": response["completion_token_ids"],
                }
            )
            response["raw_response"]["choices"][0]["token_ids"] = response[
                "completion_token_ids"
            ]
            response["raw_response_sha256"] = canonical_sha256(response["raw_response"])

        update_result_and_acceptance(head_cell, mutate_completion, harness)
        completion_mismatch = runner.strict_token_text_audit(cells)
        assert completion_mismatch["status"] == "completion_or_text_mismatch"
        assert completion_mismatch["mismatch_counts"]["completion_token_ids"] == 2

        def mutate_prompt(result: dict[str, object]) -> None:
            response = result["concurrency_blocks"][0]["measured"]["records"][1][
                "response"
            ]
            response["prompt_token_ids"] = [11, 888, 1]
            response["prompt_token_ids_sha256"] = canonical_sha256(
                response["prompt_token_ids"]
            )
            response["prompt_and_completion_token_ids_sha256"] = canonical_sha256(
                {
                    "prompt": response["prompt_token_ids"],
                    "completion": response["completion_token_ids"],
                }
            )
            response["raw_response"]["prompt_token_ids"] = response["prompt_token_ids"]
            response["raw_response_sha256"] = canonical_sha256(response["raw_response"])

        update_result_and_acceptance(head_cell, mutate_prompt, harness)
        input_mismatch = runner.strict_token_text_audit(cells)
        assert input_mismatch["status"] == "failed_input_parity"
        assert input_mismatch["mismatch_counts"]["prompt_token_ids"] == 2

        Path(str(head_cell["output"])).write_bytes(b"{}\n")
        try:
            runner.strict_token_text_audit(cells)
        except RuntimeError as error:
            assert "result changed after acceptance" in str(error)
        else:
            raise AssertionError("accepted result SHA drift was not rejected")


def exercise_attempt_integrity_context_contract(runner: ModuleType) -> None:
    with tempfile.TemporaryDirectory(prefix="pynv-attempt-context-") as raw:
        root = Path(raw)
        state_path = root / "state.json"
        evidence = root / "result.json"
        state: dict[str, object] = {"attempts": []}
        record: dict[str, object] = {"attempt": 1}
        original_capture = runner.capture_live_runtime_artifact_manifest
        original_checkpoint = runner.runtime_manifest_checkpoint
        original_source_after = runner._source_after_attempt
        calls = 0

        def capture(**unused: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            persisted = json.loads(state_path.read_text())
            assert len(persisted["attempts"]) == 1
            return {
                "schema": "pynv-live-runtime-artifact-manifest-v1",
                "sha256": "a" * 64,
            }

        runner.capture_live_runtime_artifact_manifest = capture
        runner.runtime_manifest_checkpoint = lambda **unused: {"status": "passed"}
        runner._source_after_attempt = lambda *unused, **kwargs: {"status": "passed"}
        try:
            with runner.attempt_integrity_context(
                record_container=state["attempts"],
                record=record,
                state=state,
                state_path=state_path,
                runtime_manifests={"fixture": True},
                runtime_validation_kwargs={},
                live_runtime_capture_kwargs={},
                source_root=root,
                stem="fixture-a01",
                commit=None,
                variant=None,
                evidence_paths={"result": evidence},
            ):
                evidence.write_text("fixture\n")
        finally:
            runner.capture_live_runtime_artifact_manifest = original_capture
            runner.runtime_manifest_checkpoint = original_checkpoint
            runner._source_after_attempt = original_source_after
        assert calls == 2
        assert record["attempt_state"] == "finished"
        assert record["body_status"] == "completed"
        assert record["post_attempt_integrity_status"] == "passed"
        assert record["result_sha256"] == sha256_file(evidence)

        state = {"attempts": []}
        record = {"attempt": 2}
        calls = 0

        def changing_capture(**unused: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {
                "schema": "pynv-live-runtime-artifact-manifest-v1",
                "sha256": ("a" if calls == 1 else "b") * 64,
            }

        runner.capture_live_runtime_artifact_manifest = changing_capture
        runner.runtime_manifest_checkpoint = lambda **unused: {"status": "passed"}
        runner._source_after_attempt = lambda *unused, **kwargs: {"status": "passed"}
        try:
            try:
                with runner.attempt_integrity_context(
                    record_container=state["attempts"],
                    record=record,
                    state=state,
                    state_path=state_path,
                    runtime_manifests={"fixture": True},
                    runtime_validation_kwargs={},
                    live_runtime_capture_kwargs={},
                    source_root=root,
                    stem="fixture-a02",
                    commit=None,
                    variant=None,
                    evidence_paths={"result": evidence},
                ):
                    pass
            except RuntimeError as error:
                assert "post-attempt integrity validation failed" in str(error)
            else:
                raise AssertionError("runtime drift after attempt was accepted")
        finally:
            runner.capture_live_runtime_artifact_manifest = original_capture
            runner.runtime_manifest_checkpoint = original_checkpoint
            runner._source_after_attempt = original_source_after
        assert record["post_attempt_integrity_status"] == "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--pilot-runner", type=Path, required=True)
    parser.add_argument("--pixel-preflight", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--campaign-contract", type=Path, required=True)
    args = parser.parse_args()
    runner = load_module("a100_three_arm_runner_fixture", args.runner.resolve())
    harness = load_module("persistent_harness_fixture", args.harness.resolve())
    pilot = load_module("a100_three_arm_pilot_fixture", args.pilot_runner.resolve())
    contract = json.loads(args.campaign_contract.read_text())
    assert contract["schema"] == "pynv-a100-three-arm-publication-contract-v1"
    assert contract["status"] == "frozen_launch_authorized_not_launched"
    assert contract["launch_authorization"] is True
    assert {
        variant: endpoint["commit"]
        for variant, endpoint in contract["endpoints"].items()
    } == runner.COMMITS
    assert {
        variant: endpoint["tree"] for variant, endpoint in contract["endpoints"].items()
    } == runner.TREES
    assert [
        (
            entry["rep"],
            entry["concurrency_order"],
            entry["endpoint_order"],
        )
        for entry in contract["schedule"]
    ] == runner.SCHEDULE
    assert contract["workload"]["pixel_budget_per_frame"] == list(runner.PIXEL_BUDGET)
    assert contract["workload"]["max_pixels_total"] == runner.TOTAL_MAX_PIXELS
    assert contract["workload"]["warmup_requests_by_concurrency"] == {
        str(key): value for key, value in runner.WARMUP_REQUESTS.items()
    }
    assert contract["workload"]["measured_requests_by_concurrency"] == {
        str(key): value for key, value in runner.MEASURED_REQUESTS.items()
    }
    assert contract["evidence_policy"]["ingress_idle_gate_seconds"] == 1200
    assert contract["evidence_policy"]["terminal_post_exit_samples"] == 2
    assert "outer_quiet_gate_seconds" not in contract["evidence_policy"]
    for variant, endpoint in contract["endpoints"].items():
        assert endpoint["video_backend"] == "pynvvideocodec"
        assert endpoint["backend_kwargs"] == runner.variant_backend_kwargs(variant)
        assert endpoint["server_argv"] == runner.variant_server_argv(variant)
    contract_bytes = args.campaign_contract.read_text()
    assert "/home/" not in contract_bytes and "/tmp/" not in contract_bytes
    shared_manifest = args.campaign_contract.parent / "SHARED_V4_ARTIFACT_MANIFEST.json"
    assert contract["shared_v4"]["manifest_sha256"] == sha256_file(shared_manifest)
    assert contract["shared_v4"]["harness_sha256"] == sha256_file(
        args.harness.resolve()
    )
    assert runner.CAMPAIGN_HARNESS_SHA256 == sha256_file(args.harness.resolve())
    assert runner.PREFLIGHT_RUNNER_SHA256 == sha256_file(args.pilot_runner.resolve())
    assert runner.PREFLIGHT_RUNNER_FILENAME == args.pilot_runner.name
    assert runner.PIXEL_PREFLIGHT_SHA256 == sha256_file(args.pixel_preflight.resolve())
    assert runner.PIXEL_PREFLIGHT_FILENAME == args.pixel_preflight.name
    assert pilot.PIXEL_PREFLIGHT_SHA256 == sha256_file(args.pixel_preflight.resolve())
    hardlink_identity = [{"device": 11, "inode": 22} for _ in range(8)]
    runner.validate_video_hardlink_identity(hardlink_identity)
    hardlink_identity[-1] = {"device": 11, "inode": 23}
    try:
        runner.validate_video_hardlink_identity(hardlink_identity)
    except RuntimeError:
        pass
    else:
        raise AssertionError("distinct video corpus inode was accepted")
    pilot_source = args.pilot_runner.read_text()
    runner_source = args.runner.read_text()
    assert "seconds=1200.0" in pilot_source
    assert '"ingress_idle_gate"' in pilot_source
    assert '"outer_quiet_gate"' not in pilot_source
    assert "after_outer_quiet_gate" not in pilot_source
    assert "completion_tokens_at_differences" not in pilot_source
    assert pilot_source.count('result_variant_label=f"pilot-{variant}"') == 1
    assert pilot_source.index("ingress_idle_path =") < pilot_source.index(
        "for attempt in range(1, 21):"
    )
    assert pilot_source.index(
        "initial_source = driver.validate_source_at_any_endpoint"
    ) < pilot_source.index("args.results.mkdir(parents=True)")
    assert timeout_values_in_literal_commands(args.pilot_runner.resolve()) == []
    assert pilot.run_idle_gate.__kwdefaults__ == {
        "seconds": 30.0,
        "timeout": 1800.0,
    }
    assert runner_source.index(
        "runtime_manifests = validate_runtime_manifests"
    ) < runner_source.index("args.results.mkdir(parents=True)")
    assert runner_source.index(
        "initial_source = validate_source_at_any_endpoint"
    ) < runner_source.index("args.results.mkdir(parents=True)")
    assert runner_source.index("preflight_pilot_parity_audit =") < runner_source.index(
        "args.results.mkdir(parents=True)"
    )
    assert (
        "validate_cell_idle_pair(args.idle_seconds, args.idle_timeout)" in runner_source
    )
    for required_source_audit in (
        'record["source_after_attempt"]',
        '"source_after_cell": source_after_cell',
        'manifest["terminal_source"]',
        'manifest["terminal_source_after_audit"]',
    ):
        assert required_source_audit in runner_source
    monitor_source = args.monitor.read_text()
    assert "validate_watchdog_pair(" in monitor_source
    assert "(1200.0, 120.0)" in monitor_source
    assert "(3600.0, 120.0)" in monitor_source

    endpoint_positions: Counter[tuple[str, int]] = Counter()
    concurrency_positions: Counter[tuple[int, int]] = Counter()
    endpoint_adjacencies: Counter[tuple[str, str]] = Counter()
    concurrency_adjacencies: Counter[tuple[int, int]] = Counter()
    endpoint_orders = set()
    concurrency_orders = set()
    parsed_cells = 0
    videos = [Path(f"/fixture/traffic1080-{index:02d}.mp4") for index in range(8)]
    for rep, concurrencies, variants in runner.SCHEDULE:
        del rep
        endpoint_orders.add(tuple(variants))
        concurrency_orders.add(tuple(concurrencies))
        endpoint_adjacencies.update(zip(variants, variants[1:]))
        concurrency_adjacencies.update(zip(concurrencies, concurrencies[1:]))
        for position, variant in enumerate(variants, start=1):
            endpoint_positions[(variant, position)] += 1
            command = runner.build_harness_command(
                python=Path("/fixture/.venv/bin/python"),
                harness=args.harness.resolve(),
                source_root=Path("/fixture/source"),
                transformers_root=Path("/fixture/transformers"),
                variant=variant,
                corpus=Path("/fixture"),
                videos=videos,
                concurrencies=concurrencies,
                port=18600,
                result_path=Path("/fixture/result.json"),
            )
            parsed = harness.parse_args(command[2:])
            assert parsed.variant == variant
            assert parsed.model == runner.MODEL
            assert parsed.revision == runner.REVISION
            assert parsed.prompt == runner.PROMPT
            assert parsed.backend == "pynvvideocodec"
            assert parsed.backend_kwargs == runner.variant_backend_kwargs(variant)
            assert parsed.server_arg == runner.variant_server_argv(variant)
            assert parsed.warmup_requests == 1
            assert parsed.warmup_requests_by_concurrency == {
                str(key): value for key, value in runner.WARMUP_REQUESTS.items()
            }
            assert parsed.requests == 1
            assert parsed.requests_by_concurrency == {
                str(key): value for key, value in runner.MEASURED_REQUESTS.items()
            }
            assert parsed.concurrency == concurrencies
            assert parsed.video == videos
            assert parsed.frames == 32
            assert parsed.video_pixel_budget == (1024, 576)
            assert parsed.output_len == 32
            assert parsed.max_num_seqs == 32
            assert parsed.max_num_batched_tokens == 9216
            assert parsed.kv_cache_memory_bytes == 40 * 1024**3
            assert parsed.mm_ipc_gpu_memory_gb == 2.0
            assert parsed.settle_seconds == 1.0
            assert parsed.request_timeout == 1200.0
            assert parsed.startup_timeout == 600.0
            assert parsed.shutdown_timeout == 60.0
            assert parsed.parity_reference is None
            parsed_cells += 1
        for position, concurrency in enumerate(concurrencies, start=1):
            concurrency_positions[(concurrency, position)] += 1

    assert parsed_cells == 18
    assert len(endpoint_orders) == 6
    assert len(concurrency_orders) == 6
    assert set(endpoint_positions.values()) == {2}
    assert set(concurrency_positions.values()) == {2}
    assert len(endpoint_adjacencies) == 6
    assert set(endpoint_adjacencies.values()) == {2}
    assert len(concurrency_adjacencies) == 6
    assert set(concurrency_adjacencies.values()) == {2}

    for variant in runner.COMMITS:
        pilot_command = pilot.build_pilot_harness_command(
            driver=runner,
            python=Path("/fixture/.venv/bin/python"),
            harness=args.harness.resolve(),
            root=Path("/fixture/source"),
            transformers_root=Path("/fixture/transformers"),
            variant=variant,
            corpus=Path("/fixture"),
            videos=videos,
            result_path=Path("/fixture/pilot-result.json"),
        )
        parsed = harness.parse_args(pilot_command[2:])
        assert parsed.variant == f"pilot-{variant}"
        assert parsed.backend_kwargs == runner.variant_backend_kwargs(variant)
        assert parsed.server_arg == runner.variant_server_argv(variant)
        assert parsed.warmup_requests_by_concurrency == {
            str(key): value for key, value in pilot.PILOT_WARMUPS.items()
        }
        assert parsed.requests_by_concurrency == {
            str(key): value for key, value in pilot.PILOT_MEASURED.items()
        }
        assert parsed.concurrency == pilot.PILOT_CONCURRENCIES
        monitor_command = pilot.build_monitor_command(
            driver=runner,
            python=Path("/fixture/.venv/bin/python"),
            monitor=args.monitor.resolve(),
            output=Path("/fixture/monitor.json"),
            child_command=pilot_command,
            watchdog_pair=pilot.PILOT_MONITOR_WATCHDOG_PAIR,
            conflicting_controller_roots=(Path("/fixture/foreign"),),
        )
        assert monitor_command[monitor_command.index("--timeout-seconds") + 1] == "3600"
        assert monitor_command[
            monitor_command.index("--timeout-grace-seconds") + 1
        ] == ("120")
        assert monitor_command[monitor_command.index("--") + 1 :] == pilot_command

    sample_command = runner.build_harness_command(
        python=Path("/fixture/.venv/bin/python"),
        harness=args.harness.resolve(),
        source_root=Path("/fixture/source"),
        transformers_root=Path("/fixture/transformers"),
        variant="pr-head",
        corpus=Path("/fixture"),
        videos=videos,
        concurrencies=[8, 16, 32],
        port=18600,
        result_path=Path("/fixture/result.json"),
    )
    warmup_index = sample_command.index("--warmup-requests") + 1
    sample_command[warmup_index] = "0"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            harness.parse_args(sample_command[2:])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("persistent harness parser accepted zero warmup")

    exercise_strict_token_statuses(runner, harness)
    exercise_idle_evidence_contract(runner)
    exercise_monotonic_telemetry_contract(runner)
    exercise_monitor_jsonl_terminal_contract(runner)
    exercise_huggingface_cache_contract(runner)
    exercise_runtime_hardware_fingerprint_contract(runner)
    exercise_recorded_source_contract(runner)
    exercise_monitor_completion_contract(runner)
    exercise_actual_clean_monitor_contract(runner, args.monitor)
    exercise_preflight_attempt_reconstruction_contract(runner)
    exercise_result_integrity_tamper_contract(runner, harness)
    exercise_attempt_integrity_context_contract(runner)
    exercise_terminal_failure_contract(runner)
    exercise_preflight_terminal_failure_contract(pilot)

    print("A100 persistent three-arm parser/token audit fixtures passed")


if __name__ == "__main__":
    main()
