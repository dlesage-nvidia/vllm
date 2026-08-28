# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

HARNESS_PATH = Path(__file__).with_name("benchmark_pynvvideocodec_e2e_persistent.py")
SPEC = importlib.util.spec_from_file_location("persistent_harness", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
harness = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = harness
SPEC.loader.exec_module(harness)

RESPONSE = {
    "id": "fixture",
    "model": "fixture-model",
    "prompt_token_ids": [11, 12],
    "choices": [
        {
            "message": {"content": "fixture response", "reasoning": None},
            "token_ids": [21, 22],
            "finish_reason": "length",
            "stop_reason": None,
        }
    ],
    "usage": {"completion_tokens": 2},
}


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.connection_count = 0
        self.request_count = 0
        self.mode = "keep-alive"
        self.block_request_started = threading.Event()
        self.block_request_release = threading.Event()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "persistent-harness-fixture"

    def setup(self) -> None:
        super().setup()
        with self.server.state.lock:  # type: ignore[attr-defined]
            self.server.state.connection_count += 1  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        assert request["fixture"] is True
        with self.server.state.lock:  # type: ignore[attr-defined]
            self.server.state.request_count += 1  # type: ignore[attr-defined]
            mode = self.server.state.mode  # type: ignore[attr-defined]
        if mode == "drop":
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if mode == "block":
            self.server.state.block_request_started.set()  # type: ignore[attr-defined]
            self.server.state.block_request_release.wait(30.0)  # type: ignore[attr-defined]
            return
        body = harness.canonical_json_bytes(RESPONSE)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if mode == "close":
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, unused_format: str, *unused_args: object) -> None:
        return


def specification(index: int, phase: str, concurrency: int) -> dict[str, Any]:
    return {
        "phase": phase,
        "block_index": 0,
        "concurrency": concurrency,
        "request_index": index,
        "global_request_index": index,
        "video_index": 0,
        "video_path": "/fixture/video.mp4",
        "video_file_uri": "file:///fixture/video.mp4",
        "video_sha256": "fixture-video",
        "video_work": {"sampled_source_megapixels_estimate": 1.0},
        "request_payload_sha256": "fixture-payload",
        "payload": {"fixture": True},
    }


def test_measured_window_closes_immediately_after_request_futures() -> None:
    source = HARNESS_PATH.read_text()
    for forbidden in (
        "response_body_evidence",
        "response_bytes",
        "response_content_length_header",
        "zlib",
        "base64",
    ):
        assert forbidden not in source
    tree = ast.parse(inspect.getsource(harness.execute_batch))
    try_node = next(node for node in ast.walk(tree) if isinstance(node, ast.Try))
    targets = []
    for statement in try_node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            targets.append(target.id if isinstance(target, ast.Name) else None)
        else:
            targets.append(None)
    records_index = targets.index("records")
    assert targets[records_index + 1] == "finished_ns"


def batch(
    pool: Any, args: argparse.Namespace, *, phase: str, concurrency: int, count: int
) -> dict[str, Any]:
    return harness.execute_batch(
        args,
        [specification(index, phase, concurrency) for index in range(count)],
        concurrency=concurrency,
        client_pool=pool,
    )


def run_lifecycle_fixture() -> None:
    state = State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    args = argparse.Namespace(port=port, request_timeout=2.0, output_len=2)
    try:
        pool = harness.PersistentHttpClientPool(size=4, port=port, timeout=2.0)
        warmup = batch(pool, args, phase="warmup", concurrency=4, count=8)
        assert warmup["status"] == "passed"
        assert state.connection_count == 4
        assert warmup["aggregate"]["persistent_transport_audit"]["used_slot_ids"] == [
            0,
            1,
            2,
            3,
        ]
        assert warmup["aggregate"]["persistent_transport_audit"][
            "seeded_first_wave_request_to_slot"
        ] == {0: 0, 1: 1, 2: 2, 3: 3}
        measured = batch(pool, args, phase="measured", concurrency=4, count=8)
        assert measured["status"] == "passed"
        assert (
            isinstance(measured["started_monotonic_ns"], int)
            and isinstance(measured["finished_monotonic_ns"], int)
            and measured["finished_monotonic_ns"] > measured["started_monotonic_ns"]
        )
        assert all(
            measured["started_monotonic_ns"]
            <= record["started_monotonic_ns"]
            <= record["finished_monotonic_ns"]
            <= measured["finished_monotonic_ns"]
            for record in measured["records"]
        )
        assert state.connection_count == 4
        assert all(
            record["transport"]["connection_reused"]
            and record["transport"]["prewarmed_for_measurement"]
            and record["transport"]["response_persistent"]
            for record in measured["records"]
        )
        assert all(
            record["response"]["prompt_token_ids"] == [11, 12]
            for record in measured["records"]
        )
        pool.close()
        snapshot = pool.snapshot()
        assert snapshot["implementation"] == (
            "stdlib http.client.HTTPConnection HTTP/1.1"
        )
        assert snapshot["connection_scope"] == "one pool per concurrency block"
        assert snapshot["request_retry_count"] == 0
        assert snapshot["counts"] == {
            "open_count": 4,
            "reuse_count": 12,
            "close_count": 4,
        }

        second_pool = harness.PersistentHttpClientPool(size=2, port=port, timeout=2.0)
        second_warmup = batch(second_pool, args, phase="warmup", concurrency=2, count=2)
        assert second_warmup["status"] == "passed"
        assert state.connection_count == 6
        assert all(
            slot["current_generation"] == 1 for slot in second_pool.snapshot()["slots"]
        )
        second_pool.close()

        mismatched_pool = harness.PersistentHttpClientPool(
            size=2, port=port, timeout=2.0
        )
        try:
            batch(
                mismatched_pool,
                args,
                phase="warmup",
                concurrency=4,
                count=4,
            )
        except RuntimeError as error:
            assert "does not match requested concurrency" in str(error)
        else:
            raise AssertionError("mismatched pool size was accepted")
        mismatched_pool.close()

        state.mode = "close"
        closing_pool = harness.PersistentHttpClientPool(size=1, port=port, timeout=2.0)
        closing = batch(closing_pool, args, phase="warmup", concurrency=1, count=1)
        assert closing["status"] == "failed"
        assert (
            "response_not_persistent"
            in closing["aggregate"]["persistent_transport_audit"]["reasons"]
        )
        assert closing_pool.counts()["open_count"] == 1
        closing_pool.close()

        state.mode = "drop"
        requests_before_drop = state.request_count
        drop_pool = harness.PersistentHttpClientPool(size=1, port=port, timeout=2.0)
        dropped = batch(drop_pool, args, phase="warmup", concurrency=1, count=1)
        assert dropped["status"] == "failed"
        assert dropped["aggregate"]["failed_requests"] == 1
        assert state.request_count == requests_before_drop + 1
        assert drop_pool.counts()["open_count"] == 1
        drop_pool.close()

        state.mode = "block"
        state.block_request_started.clear()
        state.block_request_release.clear()
        blocked_pool = harness.PersistentHttpClientPool(size=1, port=port, timeout=30.0)

        def interrupt_main() -> None:
            assert state.block_request_started.wait(2.0)
            os.kill(os.getpid(), signal.SIGINT)

        interrupter = threading.Thread(target=interrupt_main, daemon=True)
        interrupter.start()
        started = time.monotonic()
        try:
            with harness.termination_handlers():
                batch(
                    blocked_pool,
                    args,
                    phase="warmup",
                    concurrency=1,
                    count=1,
                )
        except harness.TerminationRequested as error:
            assert error.signum == signal.SIGINT
        else:
            raise AssertionError("TerminationRequested was not preserved")
        assert time.monotonic() - started < 2.0
        state.block_request_release.set()
        interrupter.join(timeout=2.0)
        deadline = time.monotonic() + 2.0
        while (
            any(
                thread.name.startswith("vllm-e2e-client")
                for thread in threading.enumerate()
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert not any(
            thread.name.startswith("vllm-e2e-client")
            for thread in threading.enumerate()
        )
        assert blocked_pool.closed
        assert blocked_pool.snapshot()["slots"][0]["close_reasons"] == {"pool_abort": 1}
    finally:
        state.block_request_release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
    print("persistent HTTP harness lifecycle fixtures passed")


def test_persistent_http_pool_lifecycle() -> None:
    run_lifecycle_fixture()


def test_persistent_config_fingerprint_and_warmup_contract() -> None:
    try:
        harness.effective_warmup_request_count(0, 8, 8)
    except ValueError as error:
        assert "positive warmup" in str(error)
    else:
        raise AssertionError("zero warmup was accepted for persistent HTTP")

    configuration = {
        field: f"fixture-{field}"
        for field in harness.PERFORMANCE_PARITY_CONFIGURATION_FIELDS
    }
    current = {
        "status": "passed",
        "configuration": dict(configuration),
        "videos": [],
        "concurrency_blocks": [],
    }
    reference = json.loads(json.dumps(current))
    current["configuration"]["backend_kwargs"] = {"hw_decoders": 2}
    reference["configuration"]["backend_kwargs"] = {
        "hw_decoders": 2,
        "output_layout": "tchw",
    }
    passed = harness.result_token_parity(
        current, reference, Path("/fixture/reference.json"), "fixture-sha"
    )
    assert passed["status"] == "passed"
    assert (
        passed["current_performance_configuration_fingerprint"]["sha256"]
        == passed["reference_performance_configuration_fingerprint"]["sha256"]
    )
    assert (
        passed["endpoint_treatment_configuration"]["current"]
        != passed["endpoint_treatment_configuration"]["reference"]
    )
    reference["configuration"]["dtype"] = "different"
    failed = harness.result_token_parity(
        current, reference, Path("/fixture/reference.json"), "fixture-sha"
    )
    assert failed["status"] == "failed"
    assert any(
        mismatch.get("kind") == "configuration" and mismatch.get("field") == "dtype"
        for mismatch in failed["mismatches"]
    )
    assert any(
        mismatch.get("kind") == "performance_configuration_fingerprint"
        for mismatch in failed["mismatches"]
    )

    cleanup_pool = harness.PersistentHttpClientPool(size=1, port=1, timeout=1.0)
    original_abort = cleanup_pool.slots[0].abort

    def broken_abort() -> None:
        raise RuntimeError("fixture cleanup failure")

    cleanup_pool.slots[0].abort = broken_abort
    original_error = harness.TerminationRequested(signal.SIGINT)
    harness.abort_pool_preserving_exception(cleanup_pool, original_error)
    assert isinstance(original_error, harness.TerminationRequested)
    assert any(
        "fixture cleanup failure" in note for note in original_error.cleanup_notes
    )
    cleanup_pool.slots[0].abort = original_abort
    cleanup_pool.close()


def test_batch_boundaries_ignore_wall_clock_jumps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = argparse.Namespace(
        port=server.server_address[1], request_timeout=2.0, output_len=2
    )
    wall_values = iter(
        [
            "2099-01-01T00:00:00+00:00",
            "1970-01-01T00:00:00+00:00",
            "2099-01-01T00:00:01+00:00",
            "1970-01-01T00:00:01+00:00",
        ]
    )
    monkeypatch.setattr(harness, "utc_now", lambda: next(wall_values))
    pool = harness.PersistentHttpClientPool(
        size=1, port=server.server_address[1], timeout=2.0
    )
    try:
        result = batch(pool, args, phase="warmup", concurrency=1, count=1)
        assert result["started_at"] > result["finished_at"]
        assert result["finished_monotonic_ns"] > result["started_monotonic_ns"]
        assert result["measured_window_seconds"] > 0
    finally:
        pool.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_partial_submission_failure_releases_workers(monkeypatch: Any) -> None:
    real_executor_type = harness.concurrent.futures.ThreadPoolExecutor

    class PartialSubmitExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.executor = real_executor_type(*args, **kwargs)
            self.submission_count = 0

        def submit(self, *args: Any, **kwargs: Any) -> Any:
            self.submission_count += 1
            if self.submission_count == 2:
                raise RuntimeError("fixture partial submission failure")
            return self.executor.submit(*args, **kwargs)

        def shutdown(self, *args: Any, **kwargs: Any) -> None:
            self.executor.shutdown(*args, **kwargs)

    monkeypatch.setattr(
        harness.concurrent.futures, "ThreadPoolExecutor", PartialSubmitExecutor
    )
    args = argparse.Namespace(port=1, request_timeout=30.0, output_len=2)
    pool = harness.PersistentHttpClientPool(size=2, port=1, timeout=30.0)
    started = time.monotonic()
    try:
        batch(pool, args, phase="warmup", concurrency=2, count=2)
    except RuntimeError as error:
        assert str(error) == "fixture partial submission failure"
    else:
        raise AssertionError("partial submission failure was not preserved")
    assert time.monotonic() - started < 2.0
    assert pool.closed
    deadline = time.monotonic() + 2.0
    while (
        any(
            thread.name.startswith("vllm-e2e-client")
            for thread in threading.enumerate()
        )
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert not any(
        thread.name.startswith("vllm-e2e-client") for thread in threading.enumerate()
    )


def test_server_log_sidecar_is_exclusive_and_fully_bound(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    (source_root / "vllm").mkdir(parents=True)
    media_root = tmp_path / "media"
    media_root.mkdir()
    video = media_root / "video.mp4"
    video.write_bytes(b"fixture video")
    output = tmp_path / "result.json"
    args = argparse.Namespace(
        source_root=source_root,
        python=Path(sys.executable),
        video=[video],
        output=output,
        pythonpath_extra=[],
        overwrite=False,
        allowed_local_media_path=media_root,
    )
    harness.validate_paths(args)
    expected_sidecar = tmp_path / "result.server.log"
    assert args.server_log_path == expected_sidecar
    expected_sidecar.write_bytes(b"startup\nCUDA out of memory\nshutdown\n")
    record = harness.server_log_record(expected_sidecar)
    assert record["path"] == str(expected_sidecar)
    assert record["bytes"] == expected_sidecar.stat().st_size
    assert record["sha256"] == harness.sha256_file(expected_sidecar)
    assert record["storage"] == "append-only full server-log sidecar"
    try:
        harness.validate_paths(args)
    except FileExistsError as error:
        assert "append-only server log sidecar" in str(error)
    else:
        raise AssertionError("preexisting full server log sidecar was accepted")


def main() -> None:
    run_lifecycle_fixture()
    test_persistent_config_fingerprint_and_warmup_contract()


if __name__ == "__main__":
    main()
