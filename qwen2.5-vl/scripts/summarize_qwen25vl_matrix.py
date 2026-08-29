# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize paired A100/RTX Qwen2.5-VL matrix results.

Completion-token/text differences are reported separately from timing validity.

Example::

    python summarize_qwen25vl_matrix.py --a100 results-a100 --rtx results-rtx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "vllm-qwen25-vl-video-e2e-throughput-v1-persistent-http"
EXPECTED_MANIFEST_SCHEMA = "qwen25-vl-pynvvideocodec-paired-matrix-v1"
EXPECTED_REPETITIONS = range(1, 7)
EXPECTED_VARIANTS = ("base", "head")
EXPECTED_CONCURRENCIES = (8, 16, 32)
EXPECTED_COMMITS = {
    "base": "d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302",
    "head": "fc52204ce7e0203456ceca030b90283dde28232a",
}
EXPECTED_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
EXPECTED_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
EXPECTED_BACKEND_KWARGS = {
    "hw_decoders": 2,
    "max_frames": 32,
    "min_frames": 32,
    "video_backend": "qwen2_vl",
}
EXPECTED_SCHEDULE = [
    [1, [8, 16, 32], ["base", "head"]],
    [2, [32, 16, 8], ["head", "base"]],
    [3, [16, 32, 8], ["base", "head"]],
    [4, [8, 32, 16], ["head", "base"]],
    [5, [32, 8, 16], ["base", "head"]],
    [6, [16, 8, 32], ["head", "base"]],
]
EXPECTED_WARMUP_REQUESTS = {8: 24, 16: 48, 32: 96}
EXPECTED_MEASURED_REQUESTS = {8: 64, 16: 128, 32: 256}
THROUGHPUT_FIELD = "request_throughput_per_second"

# Kept in sync with the harness fields that define performance parity. Treatment
# fields are also equal in this source-revision A/B campaign.
PAIRED_CONFIGURATION_FIELDS = (
    "model",
    "revision",
    "prompt_sha256",
    "output_len",
    "seed",
    "frame_target",
    "video_count",
    "video_pixel_budget",
    "backend_argument",
    "request_media_io_kwargs",
    "server_mm_processor_kwargs",
    "server_limit_mm_per_prompt",
    "warmup_requests_by_concurrency",
    "measured_requests_per_concurrency",
    "concurrency_order",
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "mm_ipc_gpu_memory_gb",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "mm_processor_cache_gb",
    "prefix_caching",
    "request_timeout_seconds",
    "startup_timeout_seconds",
    "shutdown_timeout_seconds",
    "settle_seconds",
    "client_http_protocol",
    "backend_kwargs",
    "server_media_io_kwargs",
    "video_kwargs_for_metric_derivation",
    "extra_server_argv",
)


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"could not read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def normalized_configuration(
    configuration: Mapping[str, Any], context: str
) -> dict[str, Any]:
    missing = [
        field for field in PAIRED_CONFIGURATION_FIELDS if field not in configuration
    ]
    require(not missing, f"{context}: missing configuration fields: {missing}")
    normalized = {field: configuration[field] for field in PAIRED_CONFIGURATION_FIELDS}
    order = normalized["concurrency_order"]
    require(isinstance(order, list), f"{context}: invalid concurrency order")
    normalized["concurrency_order"] = sorted(order)
    for field in (
        "warmup_requests_by_concurrency",
        "measured_requests_per_concurrency",
    ):
        values = normalized[field]
        require(
            isinstance(values, list)
            and all(isinstance(value, Mapping) for value in values),
            f"{context}: invalid {field}",
        )
        normalized[field] = sorted(values, key=lambda value: value.get("concurrency"))
    return normalized


def validate_locked_configuration(
    configuration: Mapping[str, Any], order: Sequence[int], context: str
) -> None:
    expected_video_kwargs = {
        "backend": "pynvvideocodec",
        "hw_decoders": 2,
        "max_frames": 32,
        "min_frames": 32,
        "video_backend": "qwen2_vl",
    }
    expected = {
        "model": EXPECTED_MODEL,
        "revision": EXPECTED_REVISION,
        "prompt_sha256": sha256_json("Describe this video concisely and factually."),
        "output_len": 32,
        "seed": 0,
        "frame_target": 32,
        "video_count": 8,
        "backend_argument": "pynvvideocodec",
        "backend_kwargs": EXPECTED_BACKEND_KWARGS,
        "server_media_io_kwargs": {"video": expected_video_kwargs},
        "request_media_io_kwargs": {},
        "video_kwargs_for_metric_derivation": expected_video_kwargs,
        "server_mm_processor_kwargs": {"max_pixels": 1024 * 576},
        "server_limit_mm_per_prompt": {"image": 0, "video": 1},
        "concurrency_order": list(order),
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 32768,
        "max_num_batched_tokens": 12288,
        "max_num_seqs": 32,
        "mm_ipc_gpu_memory_gb": 2.0,
        "gpu_memory_utilization": None,
        "kv_cache_memory_bytes": 42949672960,
        "mm_processor_cache_gb": 0,
        "prefix_caching": False,
        "request_timeout_seconds": 1200.0,
        "startup_timeout_seconds": 600.0,
        "shutdown_timeout_seconds": 60.0,
        "settle_seconds": 1.0,
        "extra_server_argv": ["--mm-device-do-normalize"],
    }
    mismatched = [
        field for field, value in expected.items() if configuration.get(field) != value
    ]
    require(
        not mismatched,
        f"{context}: fields differ from the locked campaign: {mismatched}",
    )
    expected_warmup = [
        {
            "concurrency": concurrency,
            "requested": EXPECTED_WARMUP_REQUESTS[concurrency],
            "effective": EXPECTED_WARMUP_REQUESTS[concurrency],
        }
        for concurrency in order
    ]
    expected_measured = [
        {
            "concurrency": concurrency,
            "requests": EXPECTED_MEASURED_REQUESTS[concurrency],
        }
        for concurrency in order
    ]
    require(
        configuration.get("warmup_requests_by_concurrency") == expected_warmup
        and configuration.get("measured_requests_per_concurrency") == expected_measured,
        f"{context}: configured request counts differ from the locked campaign",
    )
    pixel_budget = configuration.get("video_pixel_budget")
    require(
        isinstance(pixel_budget, Mapping)
        and pixel_budget.get("reference_width") == 1024
        and pixel_budget.get("reference_height") == 576
        and pixel_budget.get("sampled_frames") == 32,
        f"{context}: video pixel budget differs from the locked campaign",
    )


def validate_phase(
    phase: Mapping[str, Any],
    concurrency: int,
    block_index: int,
    expected_requests: int,
    phase_name: str,
    video_hashes: Sequence[str],
    context: str,
) -> list[Mapping[str, Any]]:
    require(phase.get("status") == "passed", f"{context}: phase did not pass")
    aggregate = phase.get("aggregate")
    require(isinstance(aggregate, Mapping), f"{context}: missing aggregate")
    require(aggregate.get("status") == "passed", f"{context}: aggregate did not pass")

    attempted = aggregate.get("attempted_requests")
    successful = aggregate.get("successful_requests")
    require(
        attempted == expected_requests,
        f"{context}: expected {expected_requests} requests, got {attempted!r}",
    )
    require(successful == attempted, f"{context}: not every request succeeded")
    require(aggregate.get("failed_requests") == 0, f"{context}: requests failed")
    require(aggregate.get("failures") == [], f"{context}: failure records are present")

    records = phase.get("records")
    require(
        isinstance(records, list) and len(records) == attempted,
        f"{context}: request records do not match attempted count",
    )
    require(
        all(
            isinstance(record, Mapping) and record.get("status") == "passed"
            for record in records
        ),
        f"{context}: a request record did not pass",
    )
    require(
        phase.get("requested_concurrency") == concurrency
        and phase.get("effective_client_workers") == concurrency,
        f"{context}: client concurrency metadata differs",
    )
    for request_index, record in enumerate(records):
        video_index = request_index % len(video_hashes)
        require(
            record.get("phase") == phase_name
            and record.get("block_index") == block_index
            and record.get("concurrency") == concurrency
            and record.get("request_index") == request_index,
            f"{context}: request identity/order differs at index {request_index}",
        )
        require(
            record.get("video_index") == video_index
            and record.get("video_sha256") == video_hashes[video_index],
            f"{context}: request video cycle differs at index {request_index}",
        )
        transport = record.get("transport")
        require(
            isinstance(transport, Mapping),
            f"{context}: request is missing transport metadata",
        )
        require(
            transport.get("response_http_version") == 11
            and transport.get("response_persistent") is True,
            f"{context}: request did not use persistent HTTP/1.1",
        )
        if phase_name == "measured":
            require(
                transport.get("connection_reused") is True
                and transport.get("prewarmed_for_measurement") is True,
                f"{context}: measured request did not reuse a prewarmed connection",
            )

    audit = aggregate.get("persistent_transport_audit")
    require(
        isinstance(audit, Mapping), f"{context}: missing persistent transport audit"
    )
    require(audit.get("status") == "passed", f"{context}: transport audit did not pass")
    require(
        audit.get("reasons") == [], f"{context}: transport audit has failure reasons"
    )
    require(
        audit.get("pool_size") == concurrency, f"{context}: wrong transport pool size"
    )
    require(
        audit.get("request_count") == attempted,
        f"{context}: wrong audited request count",
    )
    require(
        audit.get("used_slot_ids") == list(range(concurrency)),
        f"{context}: transport audit did not use every pool slot",
    )
    return records


def response_outputs(
    records: Sequence[Mapping[str, Any]], context: str
) -> list[dict[str, Any]]:
    outputs = []
    for expected_index, record in enumerate(records):
        require(isinstance(record, Mapping), f"{context}: invalid request record")
        require(
            record.get("request_index") == expected_index,
            f"{context}: request indexes are not contiguous and ordered",
        )
        response = record.get("response")
        require(isinstance(response, Mapping), f"{context}: missing response")
        prompt_ids = response.get("prompt_token_ids")
        completion_ids = response.get("completion_token_ids")
        require(
            isinstance(prompt_ids, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in prompt_ids
            ),
            f"{context}: invalid prompt token IDs",
        )
        require(
            isinstance(completion_ids, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in completion_ids
            ),
            f"{context}: invalid completion token IDs",
        )
        require(
            response.get("prompt_token_count") == len(prompt_ids)
            and response.get("completion_token_count") == len(completion_ids),
            f"{context}: token counts disagree with token IDs",
        )
        require(
            isinstance(response.get("text"), str),
            f"{context}: response text is missing",
        )
        required_values = (
            record.get("video_index"),
            record.get("video_sha256"),
            response.get("prompt_token_ids_sha256"),
            response.get("completion_token_ids_sha256"),
            response.get("text_sha256"),
            response.get("finish_reason"),
        )
        require(
            all(value is not None for value in required_values),
            f"{context}: incomplete response fingerprint",
        )
        require(
            response["prompt_token_ids_sha256"] == sha256_json(prompt_ids)
            and response["completion_token_ids_sha256"] == sha256_json(completion_ids)
            and response["text_sha256"] == sha256_json(response["text"]),
            f"{context}: response fingerprint does not match its value",
        )
        outputs.append(
            {
                "request_index": expected_index,
                "video_index": record["video_index"],
                "video_sha256": record["video_sha256"],
                "prompt_token_count": len(prompt_ids),
                "prompt_token_ids_sha256": response["prompt_token_ids_sha256"],
                "completion_token_count": len(completion_ids),
                "completion_token_ids_sha256": response["completion_token_ids_sha256"],
                "text_sha256": response["text_sha256"],
                "finish_reason": response["finish_reason"],
            }
        )
    return outputs


def validate_result(
    result: Mapping[str, Any],
    variant: str,
    concurrency_order: Sequence[int],
    context: str,
) -> dict[str, Any]:
    require(
        result.get("schema") == EXPECTED_SCHEMA, f"{context}: unexpected result schema"
    )
    require(result.get("status") == "passed", f"{context}: result did not pass")
    require("error" not in result, f"{context}: result contains an error")

    configuration = result.get("configuration")
    require(isinstance(configuration, Mapping), f"{context}: missing configuration")
    require(configuration.get("variant") == variant, f"{context}: wrong variant")
    validate_locked_configuration(configuration, concurrency_order, context)

    videos = result.get("videos")
    require(
        isinstance(videos, list)
        and len(videos) == 8
        and all(isinstance(video, Mapping) for video in videos),
        f"{context}: expected an eight-video manifest",
    )
    video_hashes = [video.get("sha256") for video in videos]
    require(
        all(isinstance(value, str) and len(value) == 64 for value in video_hashes)
        and len(set(video_hashes)) == 1,
        f"{context}: video hashes are missing or do not identify one hard-linked clip",
    )

    blocks = result.get("concurrency_blocks")
    require(
        isinstance(blocks, list) and len(blocks) == 3,
        f"{context}: expected three blocks",
    )
    by_concurrency: dict[int, dict[str, Any]] = {}
    for block_index, block in enumerate(blocks):
        require(isinstance(block, Mapping), f"{context}: invalid concurrency block")
        concurrency = block.get("concurrency")
        require(
            concurrency == concurrency_order[block_index]
            and block.get("block_index") == block_index
            and concurrency not in by_concurrency,
            f"{context}: concurrency block order/index differs at block {block_index}",
        )
        block_context = f"{context}/C{concurrency}"
        require(block.get("status") == "passed", f"{block_context}: block did not pass")
        warmup = block.get("warmup")
        measured = block.get("measured")
        require(isinstance(warmup, Mapping), f"{block_context}: missing warmup")
        require(isinstance(measured, Mapping), f"{block_context}: missing measurement")
        require(
            block.get("requested_warmup_requests")
            == EXPECTED_WARMUP_REQUESTS[concurrency]
            and block.get("effective_warmup_requests")
            == EXPECTED_WARMUP_REQUESTS[concurrency]
            and block.get("requested_measured_requests")
            == EXPECTED_MEASURED_REQUESTS[concurrency],
            f"{block_context}: request-count metadata differs from the full matrix",
        )
        warmup_records = validate_phase(
            warmup,
            concurrency,
            block_index,
            EXPECTED_WARMUP_REQUESTS[concurrency],
            "warmup",
            video_hashes,
            f"{block_context}/warmup",
        )
        measured_records = validate_phase(
            measured,
            concurrency,
            block_index,
            EXPECTED_MEASURED_REQUESTS[concurrency],
            "measured",
            video_hashes,
            f"{block_context}/measured",
        )

        aggregate = block.get("aggregate")
        require(
            isinstance(aggregate, Mapping) and aggregate == measured.get("aggregate"),
            f"{block_context}: block and measured aggregates differ",
        )
        throughput = aggregate.get(THROUGHPUT_FIELD)
        measured_window = aggregate.get("measured_window_seconds")
        require(
            isinstance(throughput, (int, float))
            and not isinstance(throughput, bool)
            and math.isfinite(throughput)
            and throughput > 0,
            f"{block_context}: invalid measured request throughput",
        )
        require(
            isinstance(measured_window, (int, float))
            and not isinstance(measured_window, bool)
            and math.isfinite(measured_window)
            and measured_window > 0
            and measured.get("measured_window_seconds") == measured_window
            and math.isclose(
                throughput,
                aggregate["successful_requests"] / measured_window,
                rel_tol=1e-12,
                abs_tol=0.0,
            ),
            f"{block_context}: throughput disagrees with request/window evidence",
        )

        pool = block.get("persistent_http_pool")
        require(
            isinstance(pool, Mapping), f"{block_context}: missing persistent HTTP pool"
        )
        require(
            pool.get("pool_size") == concurrency, f"{block_context}: wrong pool size"
        )
        require(
            pool.get("closed") is True, f"{block_context}: HTTP pool was not closed"
        )
        phase_audits = pool.get("phase_audits")
        require(
            isinstance(phase_audits, Mapping), f"{block_context}: missing pool audits"
        )
        for phase_name in ("warmup", "measured"):
            audit = phase_audits.get(phase_name)
            require(
                isinstance(audit, Mapping)
                and audit.get("status") == "passed"
                and audit.get("reasons") == [],
                f"{block_context}: pool {phase_name} audit did not pass",
            )

        by_concurrency[concurrency] = {
            "throughput": float(throughput),
            "outputs": {
                "warmup": response_outputs(warmup_records, f"{block_context}/warmup"),
                "measured": response_outputs(
                    measured_records, f"{block_context}/measured"
                ),
            },
        }

    require(
        set(by_concurrency) == set(EXPECTED_CONCURRENCIES),
        f"{context}: missing expected concurrency",
    )
    return {
        "configuration": configuration,
        "blocks": by_concurrency,
        "video_hashes": video_hashes,
    }


def compare_pair(
    base: Mapping[str, Any], head: Mapping[str, Any], context: str
) -> dict[int, dict[str, int]]:
    base_configuration = base["configuration"]
    head_configuration = head["configuration"]
    missing = [
        field
        for field in PAIRED_CONFIGURATION_FIELDS
        if field not in base_configuration or field not in head_configuration
    ]
    require(not missing, f"{context}: missing paired configuration fields: {missing}")
    mismatched = [
        field
        for field in PAIRED_CONFIGURATION_FIELDS
        if base_configuration[field] != head_configuration[field]
    ]
    require(not mismatched, f"{context}: configuration fields differ: {mismatched}")

    require(
        base["video_hashes"] == head["video_hashes"],
        f"{context}: paired video manifests differ",
    )

    comparison: dict[int, dict[str, int]] = {}
    output_len = base_configuration["output_len"]
    require(
        isinstance(output_len, int) and output_len > 0,
        f"{context}: invalid configured output length",
    )
    for concurrency in EXPECTED_CONCURRENCIES:
        pair_context = f"{context}/C{concurrency}"
        completion_mismatches = 0
        text_mismatches = 0
        paired_requests = 0
        for phase_name in ("warmup", "measured"):
            base_outputs = base["blocks"][concurrency]["outputs"][phase_name]
            head_outputs = head["blocks"][concurrency]["outputs"][phase_name]
            require(
                len(base_outputs) == len(head_outputs),
                f"{pair_context}/{phase_name}: paired request counts differ",
            )
            paired_requests += len(base_outputs)
            for base_output, head_output in zip(
                base_outputs, head_outputs, strict=True
            ):
                for field in ("request_index", "video_index", "video_sha256"):
                    require(
                        base_output[field] == head_output[field],
                        f"{pair_context}/{phase_name}: paired field differs: {field}",
                    )
                require(
                    base_output["prompt_token_count"]
                    == head_output["prompt_token_count"]
                    and base_output["prompt_token_ids_sha256"]
                    == head_output["prompt_token_ids_sha256"],
                    f"{pair_context}/{phase_name}: prompt token IDs differ",
                )
                require(
                    base_output["completion_token_count"]
                    == head_output["completion_token_count"]
                    == output_len,
                    f"{pair_context}/{phase_name}: completion token counts differ",
                )
                require(
                    base_output["finish_reason"] == head_output["finish_reason"],
                    f"{pair_context}/{phase_name}: finish reasons differ",
                )
                completion_mismatches += int(
                    base_output["completion_token_ids_sha256"]
                    != head_output["completion_token_ids_sha256"]
                )
                text_mismatches += int(
                    base_output["text_sha256"] != head_output["text_sha256"]
                )
        comparison[concurrency] = {
            "paired_requests": paired_requests,
            "completion_token_id_mismatches": completion_mismatches,
            "text_mismatches": text_mismatches,
        }
    return comparison


def load_gpu_results(
    label: str, directory: Path
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[int, dict[str, int]]]:
    require(
        directory.is_dir(), f"{label}: result directory does not exist: {directory}"
    )
    manifest = load_object(directory / "matrix-manifest.json")
    require(
        manifest.get("schema") == EXPECTED_MANIFEST_SCHEMA,
        f"{label}: unexpected matrix manifest schema",
    )
    require(manifest.get("status") == "passed", f"{label}: matrix did not pass")
    require(
        manifest.get("pilot") is False, f"{label}: pilot output is not a full matrix"
    )
    require(
        manifest.get("commits") == EXPECTED_COMMITS,
        f"{label}: base/head commits differ from the locked matrix",
    )
    require(
        manifest.get("model") == EXPECTED_MODEL
        and manifest.get("revision") == EXPECTED_REVISION,
        f"{label}: model or revision differs from the locked matrix",
    )
    require(
        manifest.get("backend_kwargs") == EXPECTED_BACKEND_KWARGS
        and manifest.get("sampled_frames") == 32
        and manifest.get("device_normalization") is True,
        f"{label}: video treatment differs from the locked matrix",
    )
    require(
        manifest.get("schedule") == EXPECTED_SCHEDULE,
        f"{label}: schedule differs from the locked six-repetition matrix",
    )
    runs = manifest.get("runs")
    require(isinstance(runs, list) and len(runs) == 12, f"{label}: expected 12 runs")

    expected_keys = {
        (repetition, variant)
        for repetition in EXPECTED_REPETITIONS
        for variant in EXPECTED_VARIANTS
    }
    loaded: dict[tuple[int, str], dict[str, Any]] = {}
    expected_run_order = [
        (repetition, variant)
        for repetition, _, variants in EXPECTED_SCHEDULE
        for variant in variants
    ]
    seen_paths: set[Path] = set()
    seen_artifact_hashes: set[str] = set()
    for expected_position, (run, expected_key) in enumerate(
        zip(runs, expected_run_order, strict=True), start=1
    ):
        require(isinstance(run, Mapping), f"{label}: invalid manifest run")
        key = (run.get("repetition"), run.get("variant"))
        require(
            key == expected_key and key in expected_keys and key not in loaded,
            f"{label}: unexpected or duplicate run {key}",
        )
        require(
            run.get("position") == expected_position,
            f"{label}/r{key[0]}/{key[1]}: wrong schedule position",
        )
        require(
            run.get("status") == "passed",
            f"{label}/r{key[0]}/{key[1]}: run did not pass",
        )
        require(
            run.get("exit_code") == 0, f"{label}/r{key[0]}/{key[1]}: nonzero exit code"
        )
        concurrency_order = run.get("concurrency_order")
        expected_order = next(
            order for repetition, order, _ in EXPECTED_SCHEDULE if repetition == key[0]
        )
        require(
            isinstance(concurrency_order, list) and concurrency_order == expected_order,
            f"{label}/r{key[0]}/{key[1]}: wrong concurrency coverage",
        )
        require(
            run.get("commit") == EXPECTED_COMMITS[key[1]],
            f"{label}/r{key[0]}/{key[1]}: wrong source commit",
        )
        output = run.get("output")
        require(
            isinstance(output, str), f"{label}/r{key[0]}/{key[1]}: missing output path"
        )
        path = directory / Path(output).name
        require(
            path not in seen_paths, f"{label}: result artifact is reused: {path.name}"
        )
        seen_paths.add(path)
        context = f"{label}/r{key[0]}/{key[1]}"
        result = load_object(path)
        artifact_hash = sha256_file(path)
        require(
            artifact_hash not in seen_artifact_hashes,
            f"{context}: result duplicates a different repetition artifact",
        )
        seen_artifact_hashes.add(artifact_hash)
        provenance = result.get("provenance")
        require(isinstance(provenance, Mapping), f"{context}: missing provenance")
        source = provenance.get("source")
        require(
            isinstance(source, Mapping) and source.get("commit") == run.get("commit"),
            f"{context}: source commit differs from manifest",
        )
        hardware = provenance.get("hardware")
        require(
            isinstance(hardware, Mapping), f"{context}: missing hardware provenance"
        )
        gpu_description = hardware.get("nvidia_smi_output")
        require(
            isinstance(gpu_description, str)
            and label.casefold() in gpu_description.casefold(),
            f"{context}: hardware provenance does not identify an {label} GPU",
        )
        loaded[key] = validate_result(result, key[1], concurrency_order, context)

    require(set(loaded) == expected_keys, f"{label}: matrix coverage is incomplete")
    output_comparison = {
        concurrency: {
            "paired_requests": 0,
            "completion_token_id_mismatches": 0,
            "text_mismatches": 0,
        }
        for concurrency in EXPECTED_CONCURRENCIES
    }
    for repetition in EXPECTED_REPETITIONS:
        pair_comparison = compare_pair(
            loaded[(repetition, "base")],
            loaded[(repetition, "head")],
            f"{label}/r{repetition}",
        )
        for concurrency, counts in pair_comparison.items():
            for field, value in counts.items():
                output_comparison[concurrency][field] += value
    reference_configuration = normalized_configuration(
        loaded[(1, "base")]["configuration"], f"{label}/r1/base"
    )
    reference_videos = loaded[(1, "base")]["video_hashes"]
    for (repetition, variant), result in loaded.items():
        require(
            normalized_configuration(
                result["configuration"], f"{label}/r{repetition}/{variant}"
            )
            == reference_configuration,
            f"{label}/r{repetition}/{variant}: configuration differs across repetitions",
        )
        require(
            result["video_hashes"] == reference_videos,
            f"{label}/r{repetition}/{variant}: video corpus differs across repetitions",
        )
    return loaded, output_comparison


def summarize_gpu(
    label: str,
    results: Mapping[tuple[int, str], Mapping[str, Any]],
    output_comparison: Mapping[int, Mapping[str, int]],
) -> list[dict[str, Any]]:
    groups = []
    for concurrency in EXPECTED_CONCURRENCIES:
        base = [
            results[(rep, "base")]["blocks"][concurrency]["throughput"]
            for rep in EXPECTED_REPETITIONS
        ]
        head = [
            results[(rep, "head")]["blocks"][concurrency]["throughput"]
            for rep in EXPECTED_REPETITIONS
        ]
        deltas = [
            (head_value / base_value - 1.0) * 100.0
            for base_value, head_value in zip(base, head, strict=True)
        ]
        groups.append(
            {
                "gpu": label,
                "concurrency": concurrency,
                "sample_order": "repetitions 1 through 6",
                "base_request_throughput_per_second": base,
                "head_request_throughput_per_second": head,
                "paired_percent_deltas": deltas,
                "base_statistics": {
                    "median": statistics.median(base),
                    "min": min(base),
                    "max": max(base),
                    "population_stdev": statistics.pstdev(base),
                },
                "head_statistics": {
                    "median": statistics.median(head),
                    "min": min(head),
                    "max": max(head),
                    "population_stdev": statistics.pstdev(head),
                },
                "exact_output_comparison": {
                    "status": (
                        "passed_exact"
                        if output_comparison[concurrency][
                            "completion_token_id_mismatches"
                        ]
                        == 0
                        and output_comparison[concurrency]["text_mismatches"] == 0
                        else "completion_or_text_mismatch"
                    ),
                    **output_comparison[concurrency],
                },
                "paired_percent_delta_statistics": {
                    "median": statistics.median(deltas),
                    "min": min(deltas),
                    "max": max(deltas),
                    "population_stdev": statistics.pstdev(deltas),
                },
            }
        )
    return groups


def print_text(summary: Mapping[str, Any]) -> None:
    comparison = summary["exact_output_comparison"]
    print(
        f"Status: {summary['status']} (timing validation passed; "
        "12 runs and 36 measured blocks per GPU)"
    )
    print(
        f"Exact output parity (warmup + measured): {comparison['status']} "
        f"(completion IDs {comparison['completion_token_id_mismatches']}/"
        f"{comparison['paired_requests']}, text {comparison['text_mismatches']}/"
        f"{comparison['paired_requests']})"
    )
    print(
        "Metric: measured request throughput (requests/second); sample order is r1..r6"
    )
    for group in summary["groups"]:
        stats = group["paired_percent_delta_statistics"]
        base_stats = group["base_statistics"]
        head_stats = group["head_statistics"]
        comparison = group["exact_output_comparison"]
        print(f"\n{group['gpu']} C{group['concurrency']}")
        print(
            "  base:  "
            + ", ".join(
                f"{value:.4f}" for value in group["base_request_throughput_per_second"]
            )
        )
        print(
            "  head:  "
            + ", ".join(
                f"{value:.4f}" for value in group["head_request_throughput_per_second"]
            )
        )
        print(
            "  delta: "
            + ", ".join(f"{value:+.2f}%" for value in group["paired_percent_deltas"])
        )
        print(
            "  delta stats: "
            f"median={stats['median']:+.2f}%, min={stats['min']:+.2f}%, "
            f"max={stats['max']:+.2f}%, population_stdev={stats['population_stdev']:.2f} pp"
        )
        print(
            "  throughput stats: "
            f"base median={base_stats['median']:.4f} "
            f"[{base_stats['min']:.4f}, {base_stats['max']:.4f}], "
            f"head median={head_stats['median']:.4f} "
            f"[{head_stats['min']:.4f}, {head_stats['max']:.4f}]"
        )
        print(
            f"  output parity: {comparison['status']}; completion IDs "
            f"{comparison['completion_token_id_mismatches']}/"
            f"{comparison['paired_requests']}, text {comparison['text_mismatches']}/"
            f"{comparison['paired_requests']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a100", type=Path, required=True, help="copied A100 result directory"
    )
    parser.add_argument(
        "--rtx", type=Path, required=True, help="copied RTX result directory"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require(
            args.a100.resolve() != args.rtx.resolve(),
            "A100 and RTX inputs are the same directory",
        )
        a100_results, a100_comparison = load_gpu_results("A100", args.a100)
        rtx_results, rtx_comparison = load_gpu_results("RTX", args.rtx)
        require(
            normalized_configuration(
                a100_results[(1, "base")]["configuration"], "A100/r1/base"
            )
            == normalized_configuration(
                rtx_results[(1, "base")]["configuration"], "RTX/r1/base"
            ),
            "A100 and RTX performance configurations differ",
        )
        require(
            a100_results[(1, "base")]["video_hashes"]
            == rtx_results[(1, "base")]["video_hashes"],
            "A100 and RTX video corpora differ",
        )
        groups = [
            *summarize_gpu("A100", a100_results, a100_comparison),
            *summarize_gpu("RTX", rtx_results, rtx_comparison),
        ]
        paired_requests = sum(
            group["exact_output_comparison"]["paired_requests"] for group in groups
        )
        completion_mismatches = sum(
            group["exact_output_comparison"]["completion_token_id_mismatches"]
            for group in groups
        )
        text_mismatches = sum(
            group["exact_output_comparison"]["text_mismatches"] for group in groups
        )
        summary = {
            "schema": "qwen25-vl-pynvvideocodec-paired-summary-v1",
            "status": (
                "passed_exact"
                if completion_mismatches == 0 and text_mismatches == 0
                else "timing_passed_completion_mismatch"
            ),
            "timing_validation": "passed",
            "exact_output_comparison": {
                "status": (
                    "passed_exact"
                    if completion_mismatches == 0 and text_mismatches == 0
                    else "completion_or_text_mismatch"
                ),
                "paired_requests": paired_requests,
                "completion_token_id_mismatches": completion_mismatches,
                "text_mismatches": text_mismatches,
            },
            "throughput_metric": THROUGHPUT_FIELD,
            "groups": groups,
        }
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
