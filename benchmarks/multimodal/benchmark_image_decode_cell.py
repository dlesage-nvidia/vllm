# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run one fresh-process image decode-only benchmark cell.

The Pillow path intentionally imports only APIs present in the vanilla source
tree. nvImageCodec-only imports are delayed until that backend is selected.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import psutil
import pynvml
from PIL import Image, JpegImagePlugin

import vllm
from vllm.multimodal.media.connector import global_thread_pool
from vllm.multimodal.media.image import ImageMediaIO


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"samples": 0}
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        resolution = (int(width_text), int(height_text))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "resolution must have the form WIDTHxHEIGHT"
        ) from error
    if min(resolution) <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return resolution


def git_output(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.strip()


def source_identity(
    source_root: Path, expected_commit: str, backend: str
) -> dict[str, object]:
    source_root = source_root.resolve()
    imported = Path(vllm.__file__).resolve()
    git_root = Path(git_output(source_root, "rev-parse", "--show-toplevel")).resolve()
    if source_root != git_root or not imported.is_relative_to(source_root):
        raise RuntimeError(
            f"vLLM imported from {imported}, not requested root {source_root}"
        )

    pythonpath_roots = {
        Path(entry).expanduser().resolve()
        for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    }
    if source_root not in pythonpath_roots:
        raise RuntimeError(f"{source_root} is not an explicit PYTHONPATH entry")

    expected = git_output(source_root, "rev-parse", f"{expected_commit}^{{commit}}")
    actual = git_output(source_root, "rev-parse", "HEAD^{commit}")
    if actual != expected:
        raise RuntimeError(f"expected source commit {expected}, found {actual}")
    tracked_status = git_output(
        source_root, "status", "--porcelain=v1", "--untracked-files=no"
    )
    if tracked_status:
        raise RuntimeError(
            f"source has tracked modifications: {source_root}\n{tracked_status}"
        )

    feature_paths = [
        source_root / "vllm/multimodal/image_decoders/nvimagecodec.py",
        source_root / "vllm/multimodal/media/image_decode_service.py",
    ]
    feature_present = all(path.is_file() for path in feature_paths)
    parameters = inspect.signature(ImageMediaIO.__init__).parameters
    feature_parameters = {
        "backend",
        "decoders",
        "batch_size",
        "pipeline_depth",
        "coalesce_timeout_ms",
    }
    signature_has_feature = feature_parameters.issubset(parameters)
    if backend == "pillow" and (feature_present or signature_has_feature):
        raise RuntimeError("Pillow baseline source contains nvImageCodec feature code")
    if backend == "nvimagecodec" and not (feature_present and signature_has_feature):
        raise RuntimeError("nvImageCodec source does not contain the expected feature")

    return {
        "root": str(source_root),
        "vllm_import": str(imported),
        "commit": actual,
        "tree": git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "commit_subject": git_output(source_root, "show", "-s", "--format=%s"),
        "commit_time": git_output(source_root, "show", "-s", "--format=%cI"),
        "branch": git_output(source_root, "branch", "--show-current"),
        "tracked_worktree_clean": True,
        "nvimagecodec_feature_present": feature_present,
    }


def dependency_versions(backend: str) -> dict[str, str]:
    names = ["vllm", "torch", "pillow", "nvidia-ml-py", "psutil"]
    if backend == "nvimagecodec":
        names.extend(
            [
                "nvidia-nvimgcodec-cu12",
                "nvidia-nvjpeg2k-cu12",
                "nvidia-nvtiff-cu12",
            ]
        )
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def selected_environment() -> dict[str, str | None]:
    names = [
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_LOG_DIRECTORY",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "VLLM_IMAGE_LOADER_BACKEND",
        "VLLM_MAX_IMAGE_PIXELS",
        "VLLM_MEDIA_LOADING_THREAD_COUNT",
    ]
    return {name: os.environ.get(name) for name in names}


def nvml_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def device_identity(device_index: int) -> dict[str, object]:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result: dict[str, object] = {
            "index": device_index,
            "name": nvml_text(pynvml.nvmlDeviceGetName(handle)),
            "uuid": nvml_text(pynvml.nvmlDeviceGetUUID(handle)),
            "pci_bus_id": nvml_text(pci.busId),
            "driver_version": nvml_text(pynvml.nvmlSystemGetDriverVersion()),
            "memory_total_bytes": int(memory.total),
        }
        try:
            result["cuda_driver_version"] = int(
                pynvml.nvmlSystemGetCudaDriverVersion_v2()
            )
        except (AttributeError, pynvml.NVMLError):
            result["cuda_driver_version"] = None
        return result
    finally:
        pynvml.nvmlShutdown()


class NvmlSampler:
    """Collect device-global utilization and memory during the timed region."""

    def __init__(self, device_index: int, interval_seconds: float) -> None:
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.interval_seconds = interval_seconds
        self.initial_memory_used_bytes = int(
            pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        )
        self.samples: list[dict[str, float | int]] = []
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                jpeg, period = pynvml.nvmlDeviceGetJpgUtilization(self.handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append(
                    {
                        "monotonic_ns": time.monotonic_ns(),
                        "jpeg": float(jpeg),
                        "period_us": float(period),
                        "gpu": float(utilization.gpu),
                        "memory": float(utilization.memory),
                        "memory_used_bytes": int(memory.used),
                    }
                )
                self.stop_event.wait(self.interval_seconds)
        except BaseException as error:
            self.error = error

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()
        if self.error is not None:
            raise RuntimeError("NVML sampler failed") from self.error

    def between(
        self, start_ns: int, end_ns: int, *, full_utilization_period: bool
    ) -> list[dict[str, float | int]]:
        samples = [
            sample
            for sample in self.samples
            if start_ns <= int(sample["monotonic_ns"]) <= end_ns
        ]
        if full_utilization_period:
            samples = [
                sample
                for sample in samples
                if int(sample["monotonic_ns"]) - int(float(sample["period_us"]) * 1000)
                >= start_ns
            ]
        return samples


def process_cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return float(times.user + times.system)


def process_tree_cpu_snapshot(
    root: psutil.Process,
) -> dict[tuple[int, float], dict[str, object]]:
    snapshot: dict[tuple[int, float], dict[str, object]] = {}
    processes = [root]
    with contextlib.suppress(psutil.Error):
        processes.extend(root.children(recursive=True))
    for process in processes:
        with contextlib.suppress(psutil.Error, OSError):
            create_time = float(process.create_time())
            key = (process.pid, create_time)
            snapshot[key] = {
                "pid": process.pid,
                "create_time": create_time,
                "name": process.name(),
                "cmdline": " ".join(process.cmdline())[:500],
                "cpu_seconds": process_cpu_seconds(process),
            }
    return snapshot


def mps_cpu_snapshot() -> dict[tuple[int, float], dict[str, object]]:
    snapshot: dict[tuple[int, float], dict[str, object]] = {}
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            identity = (
                f"{process.info['name']} {' '.join(process.info['cmdline'] or [])}"
            ).lower()
            if "nvidia-cuda-mps-server" not in identity:
                continue
            create_time = float(process.info["create_time"])
            key = (process.pid, create_time)
            snapshot[key] = {
                "pid": process.pid,
                "create_time": create_time,
                "name": process.info["name"],
                "cmdline": " ".join(process.info["cmdline"] or [])[:500],
                "cpu_seconds": process_cpu_seconds(process),
            }
        except (psutil.Error, OSError, TypeError):
            continue
    return snapshot


def cpu_delta(
    start: dict[tuple[int, float], dict[str, object]],
    finish: dict[tuple[int, float], dict[str, object]],
) -> float:
    return sum(
        max(
            0.0,
            float(record["cpu_seconds"])
            - float(start.get(identity, {"cpu_seconds": 0.0})["cpu_seconds"]),
        )
        for identity, record in finish.items()
    )


class PillowRunner:
    backend = "pillow"

    def __init__(self) -> None:
        self.loader = ImageMediaIO()

    async def load(self, payload: bytes) -> tuple[object, bool]:
        loop = asyncio.get_running_loop()
        item = await loop.run_in_executor(
            global_thread_pool, self.loader.load_bytes, payload
        )
        return item, False

    def begin_accounting(self) -> None:
        return None

    def end_accounting(self, _token: None, images: int) -> dict[str, object]:
        return {
            "submitted_images": images,
            "completed_images": images,
            "accounting_gap": 0,
            "service_width_histogram": {},
            "native_width_histogram": {},
        }

    def close(self) -> None:
        return None


class NativeTrace:
    def __init__(self, backend_class: type) -> None:
        self.backend_class = backend_class
        self.original = backend_class._decode_native
        self.widths: Counter[int] = Counter()
        self.lock = threading.Lock()

    def install(self) -> None:
        trace = self

        def wrapped(decoder, items, params, *, cuda_stream: int):
            with trace.lock:
                trace.widths[len(items)] += 1
            return trace.original(decoder, items, params, cuda_stream=cuda_stream)

        self.backend_class._decode_native = staticmethod(wrapped)

    def restore(self) -> dict[str, int]:
        self.backend_class._decode_native = staticmethod(self.original)
        with self.lock:
            return {str(width): count for width, count in sorted(self.widths.items())}


class NvImageCodecRunner:
    backend = "nvimagecodec"

    def __init__(
        self,
        *,
        decoders: int,
        batch_size: int,
        pipeline_depth: int,
        coalesce_timeout_ms: float,
        gpu_pool_bytes: int,
    ) -> None:
        gpu_memory = importlib.import_module("vllm.multimodal.gpu_ipc_memory")
        decoder_module = importlib.import_module(
            "vllm.multimodal.image_decoders.nvimagecodec"
        )
        service = importlib.import_module("vllm.multimodal.media.image_decode_service")
        self.backend_class = decoder_module.NvImageCodecBackend
        self.get_stats = service.get_nvimagecodec_decode_service_stats
        self.load_async = service.load_images_with_service_async
        self.shutdown_service = service.shutdown_nvimagecodec_decode_service
        self.set_pool = gpu_memory.set_mm_gpu_ipc_pool
        self.pool = gpu_memory.MultiModalGPUMemoryPool(gpu_pool_bytes)
        self.shutdown_service()
        self.set_pool(self.pool)
        self.loader = ImageMediaIO(
            backend="nvimagecodec",
            decoders=decoders,
            batch_size=batch_size,
            pipeline_depth=pipeline_depth,
            coalesce_timeout_ms=coalesce_timeout_ms,
        )
        self.batch_size = batch_size
        self.pipeline_depth = pipeline_depth

    async def load(self, payload: bytes) -> tuple[object, bool]:
        loaded = await self.load_async(self.loader, [payload])
        if len(loaded) != 1:
            raise RuntimeError(f"expected one decoded image, got {len(loaded)}")
        item = loaded[0]
        fallback = (item.io_config or {}).get("backend") != "nvimagecodec"
        return item, fallback

    def begin_accounting(self) -> tuple[object, NativeTrace, int]:
        if self.pool.available_bytes != self.pool.total_bytes:
            raise RuntimeError("GPU memory leases remain after warmup")
        trace = NativeTrace(self.backend_class)
        trace.install()
        return self.get_stats(), trace, self.pool.available_bytes

    def end_accounting(
        self, token: tuple[object, NativeTrace, int], images: int
    ) -> dict[str, object]:
        before_stats, trace, pool_available_before = token
        native_widths = trace.restore()
        after_stats = self.get_stats()
        service_widths = Counter(after_stats.batch_widths)
        service_widths.subtract(before_stats.batch_widths)
        service_widths = Counter(
            {width: jobs for width, jobs in service_widths.items() if jobs}
        )
        submitted = after_stats.submitted_images - before_stats.submitted_images
        direct_jobs = after_stats.direct_jobs - before_stats.direct_jobs
        service_images = direct_jobs + sum(
            width * jobs for width, jobs in service_widths.items()
        )
        native_images = sum(int(width) * jobs for width, jobs in native_widths.items())
        pool_available_after = self.pool.available_bytes
        accounting = {
            "submitted_images": submitted,
            "completed_images": images,
            "direct_jobs": direct_jobs,
            "service_accounted_images": service_images,
            "native_accounted_images": native_images,
            "accounting_gap": submitted - images,
            "service_accounting_gap": submitted - service_images,
            "native_accounting_gap": submitted - native_images,
            "service_width_histogram": {
                str(width): jobs for width, jobs in sorted(service_widths.items())
            },
            "native_width_histogram": native_widths,
            "queue_wait_seconds": (
                after_stats.queue_wait_seconds - before_stats.queue_wait_seconds
            ),
            "gpu_pool": {
                "total_bytes": self.pool.total_bytes,
                "available_before_bytes": pool_available_before,
                "available_after_bytes": pool_available_after,
                "outstanding_after_bytes": (
                    self.pool.total_bytes - pool_available_after
                ),
            },
        }
        failures = [
            name
            for name, value in (
                ("submitted/completed", submitted - images),
                ("service", submitted - service_images),
                ("native", submitted - native_images),
                (
                    "gpu_pool",
                    self.pool.total_bytes - pool_available_after,
                ),
            )
            if value
        ]
        if failures:
            raise RuntimeError("nvImageCodec accounting failed: " + ", ".join(failures))
        maximum_claim_width = self.batch_size * self.pipeline_depth
        if service_widths and max(service_widths) > maximum_claim_width:
            raise RuntimeError(
                f"service claim width {max(service_widths)} exceeds "
                f"{maximum_claim_width}"
            )
        return accounting

    def close(self) -> None:
        self.shutdown_service()
        self.set_pool(None)


def close_item(item: object) -> None:
    media = item.media
    media.close()


def reference_pixels(
    payloads: list[bytes], expected_size: tuple[int, int]
) -> tuple[list[list[tuple[int, int, int]]], list[tuple[int, int]]]:
    width, height = expected_size
    coordinates = [
        (0, 0),
        (width // 3, height // 3),
        (width // 2, height // 2),
        (width - 1, height - 1),
    ]
    references = []
    for payload in payloads:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            converted = image.convert("RGB")
            try:
                references.append([converted.getpixel(point) for point in coordinates])
            finally:
                if converted is not image:
                    converted.close()
    return references, coordinates


async def validate_backend_outputs(
    runner: PillowRunner | NvImageCodecRunner,
    payloads: list[bytes],
    expected_size: tuple[int, int],
    references: list[list[tuple[int, int, int]]],
    coordinates: list[tuple[int, int]],
    *,
    max_abs_error: int,
) -> dict[str, object]:
    async def check(index: int) -> tuple[int, int, bool]:
        item, fallback = await runner.load(payloads[index])
        try:
            media = item.media
            if media.mode != "RGB" or media.size != expected_size:
                raise RuntimeError(
                    f"image {index} decoded as {media.mode}/{media.size}"
                )
            observed = [media.getpixel(point) for point in coordinates]
            error = max(
                abs(actual - expected)
                for actual_pixel, expected_pixel in zip(observed, references[index])
                for actual, expected in zip(actual_pixel, expected_pixel)
            )
            checksum = sum(channel for pixel in observed for channel in pixel)
            return error, checksum, fallback
        finally:
            close_item(item)

    checked = await asyncio.gather(*(check(index) for index in range(len(payloads))))
    observed_max = max(result[0] for result in checked)
    fallbacks = sum(result[2] for result in checked)
    if observed_max > max_abs_error:
        raise RuntimeError(
            f"sampled-pixel error {observed_max} exceeds {max_abs_error}"
        )
    if fallbacks:
        raise RuntimeError(f"correctness preflight used {fallbacks} fallbacks")
    return {
        "images": len(checked),
        "pixels_per_image": len(coordinates),
        "max_abs_channel_error": observed_max,
        "allowed_max_abs_channel_error": max_abs_error,
        "sampled_pixel_checksum": sum(result[1] for result in checked),
        "fallbacks": fallbacks,
    }


async def run_load(
    runner: PillowRunner | NvImageCodecRunner,
    payloads: list[bytes],
    expected_size: tuple[int, int],
    *,
    concurrency: int,
    duration: float,
) -> dict[str, object]:
    start = asyncio.Event()
    deadline = [0.0]

    async def worker(worker_index: int) -> dict[str, object]:
        cursor = worker_index * 11
        count = 0
        fallbacks = 0
        checksum = 0
        latencies: list[float] = []
        payload_counts: Counter[int] = Counter()
        await start.wait()
        while time.monotonic() < deadline[0]:
            payload_index = cursor % len(payloads)
            before = time.perf_counter()
            item, fallback = await runner.load(payloads[payload_index])
            latencies.append((time.perf_counter() - before) * 1000)
            try:
                media = item.media
                if media.mode != "RGB" or media.size != expected_size:
                    raise RuntimeError(
                        f"unexpected decoded image {media.mode}/{media.size}"
                    )
                pixel = media.getpixel(
                    (payload_index % expected_size[0], payload_index % expected_size[1])
                )
                checksum += sum(pixel)
            finally:
                close_item(item)
            fallbacks += int(fallback)
            payload_counts[payload_index] += 1
            cursor += 1
            count += 1
        return {
            "images": count,
            "fallbacks": fallbacks,
            "checksum": checksum,
            "latencies": latencies,
            "payload_counts": payload_counts,
        }

    tasks = [asyncio.create_task(worker(index)) for index in range(concurrency)]
    deadline[0] = time.monotonic() + duration
    start.set()
    results = await asyncio.gather(*tasks)
    payload_counts: Counter[int] = Counter()
    for result in results:
        payload_counts.update(result["payload_counts"])
    return {
        "images": sum(int(result["images"]) for result in results),
        "fallbacks": sum(int(result["fallbacks"]) for result in results),
        "checksum": sum(int(result["checksum"]) for result in results),
        "latencies": [latency for result in results for latency in result["latencies"]],
        "payload_counts": payload_counts,
    }


def validate_corpus(
    jpeg_dir: Path, expected_size: tuple[int, int]
) -> tuple[list[bytes], list[str], dict[str, object]]:
    paths = sorted(jpeg_dir.glob("*.jpg"))
    if len(paths) < 32:
        raise RuntimeError(f"expected at least 32 JPEGs, found {len(paths)}")
    payloads = [path.read_bytes() for path in paths]
    properties: set[tuple[object, ...]] = set()
    file_digests: list[dict[str, object]] = []
    corpus_digest = hashlib.sha256()
    for path, payload in zip(paths, payloads):
        with Image.open(BytesIO(payload)) as image:
            properties.add(
                (
                    image.format,
                    image.size,
                    image.mode,
                    getattr(image, "bits", None),
                    JpegImagePlugin.get_sampling(image),
                )
            )
        payload_digest = hashlib.sha256(payload).hexdigest()
        file_digests.append(
            {"name": path.name, "bytes": len(payload), "sha256": payload_digest}
        )
        corpus_digest.update(path.name.encode())
        corpus_digest.update(b"\0")
        corpus_digest.update(bytes.fromhex(payload_digest))
    if len(properties) != 1:
        raise RuntimeError(f"heterogeneous corpus: {properties}")
    image_format, actual_size, mode, bits, sampling = next(iter(properties))
    if actual_size != expected_size:
        raise RuntimeError(f"corpus resolution {actual_size} != {expected_size}")
    if (image_format, mode, bits, sampling) != ("JPEG", "RGB", 8, 2):
        raise RuntimeError(
            "corpus is not baseline 8-bit RGB 4:2:0 JPEG: "
            f"{image_format}/{mode}/{bits}/{sampling}"
        )
    return (
        payloads,
        [path.name for path in paths],
        {
            "path": str(jpeg_dir.resolve()),
            "count": len(payloads),
            "sha256": corpus_digest.hexdigest(),
            "width": expected_size[0],
            "height": expected_size[1],
            "format": image_format,
            "mode": mode,
            "bits_per_channel": bits,
            "chroma_subsampling": "4:2:0",
            "files": file_digests,
        },
    )


def payload_mix_digest(names: list[str], counts: Counter[int]) -> str:
    digest = hashlib.sha256()
    for index, name in enumerate(names):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(counts[index]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jpeg_dir", type=Path)
    parser.add_argument("--backend", choices=("pillow", "nvimagecodec"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--resolution", type=parse_resolution, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup", type=float, default=10)
    parser.add_argument("--window", type=float, default=20)
    parser.add_argument("--telemetry-interval", type=float, default=0.1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--decoders", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--coalesce-timeout-ms", type=float, default=0.25)
    parser.add_argument("--gpu-pool-gb", type=float, default=8)
    parser.add_argument("--correctness-atol", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        min(
            args.concurrency,
            args.decoders,
            args.batch_size,
            args.pipeline_depth,
        )
        < 1
    ):
        parser.error("counts must be positive")
    if (
        min(
            args.warmup,
            args.window,
            args.telemetry_interval,
            args.gpu_pool_gb,
        )
        <= 0
    ):
        parser.error("timings and GPU pool size must be positive")
    if args.correctness_atol < 0:
        parser.error("--correctness-atol must be nonnegative")

    source = source_identity(args.source_root, args.expected_commit, args.backend)
    payloads, payload_names, corpus = validate_corpus(args.jpeg_dir, args.resolution)
    references, coordinates = reference_pixels(payloads, args.resolution)
    gpu_pool_bytes = int(args.gpu_pool_gb * (1 << 30))
    runner: PillowRunner | NvImageCodecRunner
    if args.backend == "pillow":
        runner = PillowRunner()
    else:
        runner = NvImageCodecRunner(
            decoders=args.decoders,
            batch_size=args.batch_size,
            pipeline_depth=args.pipeline_depth,
            coalesce_timeout_ms=args.coalesce_timeout_ms,
            gpu_pool_bytes=gpu_pool_bytes,
        )

    try:
        correctness = await validate_backend_outputs(
            runner,
            payloads,
            args.resolution,
            references,
            coordinates,
            max_abs_error=args.correctness_atol,
        )
        warmup_start_ns = time.monotonic_ns()
        warmup = await run_load(
            runner,
            payloads,
            args.resolution,
            concurrency=args.concurrency,
            duration=args.warmup,
        )
        warmup_end_ns = time.monotonic_ns()
        if warmup["fallbacks"]:
            raise RuntimeError(f"warmup used {warmup['fallbacks']} Pillow fallbacks")

        accounting_token = runner.begin_accounting()
        sampler = NvmlSampler(args.device_index, args.telemetry_interval)
        sampler.start()
        root_process = psutil.Process()
        worker_start = process_tree_cpu_snapshot(root_process)
        mps_start = mps_cpu_snapshot()
        wall_start_ns = time.monotonic_ns()
        measured: dict[str, object] | None = None
        try:
            measured = await run_load(
                runner,
                payloads,
                args.resolution,
                concurrency=args.concurrency,
                duration=args.window,
            )
            wall_end_ns = time.monotonic_ns()
            worker_finish = process_tree_cpu_snapshot(root_process)
            mps_finish = mps_cpu_snapshot()
        finally:
            sampler.stop()
        assert measured is not None
        accounting = runner.end_accounting(accounting_token, int(measured["images"]))
    finally:
        runner.close()

    utilization_samples = sampler.between(
        wall_start_ns, wall_end_ns, full_utilization_period=True
    )
    memory_samples = sampler.between(
        wall_start_ns, wall_end_ns, full_utilization_period=False
    )
    if not utilization_samples or not memory_samples:
        raise RuntimeError("no boundary-valid NVML samples")
    if measured["fallbacks"]:
        raise RuntimeError(f"measurement used {measured['fallbacks']} fallbacks")
    latencies = measured["latencies"]
    assert isinstance(latencies, list)
    payload_counts = measured["payload_counts"]
    assert isinstance(payload_counts, Counter)
    if len(payload_counts) != len(payloads) or min(payload_counts.values()) <= 0:
        raise RuntimeError("timed window did not exercise every corpus image")

    wall_seconds = (wall_end_ns - wall_start_ns) / 1e9
    worker_cpu = cpu_delta(worker_start, worker_finish)
    mps_cpu = cpu_delta(mps_start, mps_finish)
    memory_values = [int(sample["memory_used_bytes"]) for sample in memory_samples]
    nvjpg_values = [float(sample["jpeg"]) for sample in utilization_samples]
    nvjpg_nonzero_samples = sum(value > 0 for value in nvjpg_values)
    width, height = args.resolution
    point = {
        "backend": args.backend,
        "concurrency": args.concurrency,
        "images": measured["images"],
        "wall_seconds": wall_seconds,
        "target_window_seconds": args.window,
        "drainage_seconds": max(0.0, wall_seconds - args.window),
        "images_per_second": int(measured["images"]) / wall_seconds,
        "gpixels_per_second": (
            int(measured["images"]) * width * height / wall_seconds / 1e9
        ),
        "fallbacks": measured["fallbacks"],
        "sampled_pixel_checksum": measured["checksum"],
        "request_latency_ms": summary(latencies),
        "cpu": {
            "seconds": {
                "server_worker_tree": worker_cpu,
                "mps_server": mps_cpu,
                "server_worker_plus_mps": worker_cpu + mps_cpu,
            },
            "average_cores": {
                "server_worker_tree": worker_cpu / wall_seconds,
                "mps_server": mps_cpu / wall_seconds,
                "server_worker_plus_mps": (worker_cpu + mps_cpu) / wall_seconds,
            },
            "server_worker_processes_start": sorted(
                worker_start.values(), key=lambda record: int(record["pid"])
            ),
            "server_worker_processes_finish": sorted(
                worker_finish.values(), key=lambda record: int(record["pid"])
            ),
            "mps_processes_start": sorted(
                mps_start.values(), key=lambda record: int(record["pid"])
            ),
            "mps_processes_finish": sorted(
                mps_finish.values(), key=lambda record: int(record["pid"])
            ),
        },
        "nvjpg_utilization": {
            **summary(nvjpg_values),
            "nonzero_samples": nvjpg_nonzero_samples,
            "nonzero_percent": 100 * nvjpg_nonzero_samples / len(nvjpg_values),
        },
        "gpu_utilization": summary(
            [float(sample["gpu"]) for sample in utilization_samples]
        ),
        "memory_utilization": summary(
            [float(sample["memory"]) for sample in utilization_samples]
        ),
        "device_memory": {
            "scope": "device_global",
            "used_bytes_at_sampler_start": sampler.initial_memory_used_bytes,
            "used_bytes_first_timed_sample": memory_values[0],
            "used_bytes_last_timed_sample": memory_values[-1],
            "used_bytes_peak": max(memory_values),
            "peak_minus_first_timed_sample_bytes": (
                max(memory_values) - memory_values[0]
            ),
            "used_bytes": summary([float(value) for value in memory_values]),
        },
        "nvml_sampling_period_us": summary(
            [float(sample["period_us"]) for sample in utilization_samples]
        ),
        "nvml_sample_counts": {
            "utilization": len(utilization_samples),
            "memory": len(memory_samples),
        },
        "payload_decode_counts": {
            payload_names[index]: payload_counts[index]
            for index in range(len(payload_names))
        },
        "payload_mix_sha256": payload_mix_digest(payload_names, payload_counts),
        "accounting": accounting,
    }

    script = Path(__file__).resolve()
    process = psutil.Process()
    result = {
        "schema": "vllm-matched-image-decode-cell-v1",
        "source": source,
        "device": device_identity(args.device_index),
        "dependencies": dependency_versions(args.backend),
        "environment": selected_environment(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "process": {
            "pid": process.pid,
            "create_time": process.create_time(),
        },
        "harness": {
            "path": str(script),
            "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "argv": sys.argv,
        },
        "host_logical_cpus": psutil.cpu_count(logical=True),
        "corpus": corpus,
        "configuration": {
            "backend": args.backend,
            "concurrency": args.concurrency,
            "warmup_seconds": args.warmup,
            "window_seconds": args.window,
            "telemetry_interval_seconds": args.telemetry_interval,
            "device_index": args.device_index,
            "decoders": args.decoders if args.backend == "nvimagecodec" else None,
            "batch_size": args.batch_size if args.backend == "nvimagecodec" else 1,
            "pipeline_depth": (
                args.pipeline_depth if args.backend == "nvimagecodec" else 1
            ),
            "coalesce_timeout_ms": (
                args.coalesce_timeout_ms if args.backend == "nvimagecodec" else 0.0
            ),
            "gpu_pool_bytes": (gpu_pool_bytes if args.backend == "nvimagecodec" else 0),
            "process_isolation": "one fresh process for this cell",
            "load_model": "closed-loop, one image per request",
        },
        "correctness": correctness,
        "warmup": {
            "images": warmup["images"],
            "wall_seconds": (warmup_end_ns - warmup_start_ns) / 1e9,
            "fallbacks": warmup["fallbacks"],
            "sampled_pixel_checksum": warmup["checksum"],
        },
        "validation": {
            "source_commit_exact": source["commit"]
            == git_output(
                args.source_root, "rev-parse", f"{args.expected_commit}^{{commit}}"
            ),
            "tracked_worktree_clean": source["tracked_worktree_clean"],
            "correctness_within_tolerance": (
                correctness["max_abs_channel_error"]
                <= correctness["allowed_max_abs_channel_error"]
            ),
            "preflight_fallbacks_zero": correctness["fallbacks"] == 0,
            "warmup_fallbacks_zero": warmup["fallbacks"] == 0,
            "measurement_fallbacks_zero": measured["fallbacks"] == 0,
            "accounting_gap_zero": accounting["accounting_gap"] == 0,
            "all_corpus_images_exercised": len(payload_counts) == len(payloads),
            "nvml_utilization_samples_nonzero": bool(utilization_samples),
            "nvml_memory_samples_nonzero": bool(memory_samples),
        },
        "point": point,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    cpu_cores = point["cpu"]["average_cores"]["server_worker_plus_mps"]
    print(
        f"{args.backend} c{args.concurrency}: "
        f"{point['images_per_second']:.3f} img/s, "
        f"CPU={cpu_cores:.3f}, "
        f"NVJPG={point['nvjpg_utilization']['mean']:.2f}% mean/"
        f"{point['nvjpg_utilization']['p95']:.2f}% p95, "
        f"VRAM_peak={point['device_memory']['used_bytes_peak']} bytes",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
