# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc

COMMITS = {
    "base": "d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302",
    "head": "fc52204ce7e0203456ceca030b90283dde28232a",
}
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
SCHEDULE = [
    (1, [8, 16, 32], ["base", "head"]),
    (2, [32, 16, 8], ["head", "base"]),
    (3, [16, 32, 8], ["base", "head"]),
    (4, [8, 32, 16], ["head", "base"]),
    (5, [32, 8, 16], ["base", "head"]),
    (6, [16, 8, 32], ["head", "base"]),
]
WARMUP_REQUESTS = {8: 24, 16: 48, 32: 96}
MEASURED_REQUESTS = {8: 64, 16: 128, 32: 256}
BACKEND_KWARGS = {
    "hw_decoders": 2,
    "max_frames": 32,
    "min_frames": 32,
    "video_backend": "qwen2_vl",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(source: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *args], text=True
    ).strip()


def validate_inputs(args: argparse.Namespace) -> list[Path]:
    for path in (
        args.source_root,
        args.python,
        args.transformers_root,
        args.hf_hub_cache,
        args.corpus,
        args.harness,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.results.exists():
        raise FileExistsError(f"result directory already exists: {args.results}")
    if git(args.source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source checkout is not clean")
    for commit in COMMITS.values():
        subprocess.run(
            ["git", "-C", str(args.source_root), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
        )
    videos = [args.corpus / f"traffic1080-{index:02d}.mp4" for index in range(8)]
    if not all(path.is_file() for path in videos):
        raise FileNotFoundError("the eight traffic1080 hard-link paths are required")
    identities = {(path.stat().st_dev, path.stat().st_ino) for path in videos}
    if len(identities) != 1:
        raise RuntimeError("the eight workload paths are not hard links to one clip")
    return videos


def harness_command(
    args: argparse.Namespace,
    *,
    variant: str,
    concurrencies: list[int],
    output: Path,
    videos: list[Path],
    pilot: bool,
) -> list[str]:
    if pilot:
        warmups = {1: 8, 8: 8, 32: 32}
        requests = {1: 8, 8: 8, 32: 32}
    else:
        warmups = WARMUP_REQUESTS
        requests = MEASURED_REQUESTS
    command = [
        str(args.python),
        str(args.harness),
        "--source-root",
        str(args.source_root),
        "--python",
        str(args.python),
        "--pythonpath-extra",
        str(args.transformers_root),
        "--variant",
        variant,
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--allowed-local-media-path",
        str(args.corpus),
        "--backend",
        "pynvvideocodec",
        "--backend-kwargs",
        json.dumps(BACKEND_KWARGS, separators=(",", ":"), sort_keys=True),
        "--frames",
        "32",
        "--video-pixel-budget",
        "1024x576",
        "--warmup-requests",
        "1",
        "--warmup-requests-by-concurrency",
        json.dumps(warmups, separators=(",", ":"), sort_keys=True),
        "--requests",
        "1",
        "--requests-by-concurrency",
        json.dumps(requests, separators=(",", ":"), sort_keys=True),
        "--output-len",
        "32",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "12288",
        "--max-num-seqs",
        "32",
        "--mm-ipc-gpu-memory-gb",
        "2",
        "--kv-cache-memory-bytes",
        "42949672960",
        "--settle-seconds",
        "1",
        "--request-timeout",
        "1200",
        "--startup-timeout",
        "600",
        "--shutdown-timeout",
        "60",
        "--port",
        str(args.port),
        "--output",
        str(output),
        "--server-arg=--mm-device-do-normalize",
    ]
    for video in videos:
        command.extend(["--video", str(video)])
    for concurrency in concurrencies:
        command.extend(["--concurrency", str(concurrency)])
    return command


def run(args: argparse.Namespace) -> None:
    videos = validate_inputs(args)
    args.results.mkdir(parents=True)
    manifest_path = args.results / "matrix-manifest.json"
    schedule = [(1, [1, 8, 32], ["base", "head"])] if args.pilot else SCHEDULE
    manifest: dict[str, Any] = {
        "schema": "qwen25-vl-pynvvideocodec-paired-matrix-v1",
        "status": "running",
        "started_at": now(),
        "model": MODEL,
        "revision": REVISION,
        "hf_hub_cache": str(args.hf_hub_cache),
        "commits": COMMITS,
        "pixel_budget_per_frame": {"width": 1024, "height": 576, "pixels": 589824},
        "sampled_frames": 32,
        "backend_kwargs": BACKEND_KWARGS,
        "device_normalization": True,
        "schedule": schedule,
        "pilot": args.pilot,
        "runs": [],
    }
    write_json(manifest_path, manifest)
    position = 0
    try:
        for repetition, concurrencies, variants in schedule:
            for variant in variants:
                position += 1
                commit = COMMITS[variant]
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(args.source_root),
                        "switch",
                        "--quiet",
                        "--detach",
                        commit,
                    ],
                    check=True,
                )
                if git(args.source_root, "rev-parse", "HEAD") != commit:
                    raise RuntimeError("checkout did not select the requested commit")
                if git(
                    args.source_root,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ):
                    raise RuntimeError("source checkout became dirty")
                order = "-".join(str(value) for value in concurrencies)
                stem = f"r{repetition:02d}-p{position:02d}-{variant}-c{order}"
                output = args.results / f"{stem}.json"
                log_path = args.results / f"{stem}.log"
                command = harness_command(
                    args,
                    variant=variant,
                    concurrencies=concurrencies,
                    output=output,
                    videos=videos,
                    pilot=args.pilot,
                )
                record = {
                    "repetition": repetition,
                    "position": position,
                    "variant": variant,
                    "commit": commit,
                    "concurrency_order": concurrencies,
                    "output": str(output),
                    "log": str(log_path),
                    "command": command,
                    "started_at": now(),
                    "status": "running",
                }
                manifest["runs"].append(record)
                write_json(manifest_path, manifest)
                print(f"RUN {stem}", flush=True)
                environment = dict(os.environ)
                environment.update(
                    {
                        "HF_HUB_OFFLINE": "1",
                        "HF_HUB_CACHE": str(args.hf_hub_cache),
                        "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONNOUSERSITE": "1",
                        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": "3600",
                    }
                )
                with log_path.open("x") as log_file:
                    completed = subprocess.run(
                        command,
                        cwd=args.source_root,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                record["finished_at"] = now()
                record["exit_code"] = completed.returncode
                record["status"] = "passed" if completed.returncode == 0 else "failed"
                write_json(manifest_path, manifest)
                if completed.returncode:
                    raise RuntimeError(f"benchmark failed: {stem}; see {log_path}")
        manifest["status"] = "passed"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_at"] = now()
        write_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--transformers-root", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
