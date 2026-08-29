# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU JPEG decoding through nvImageCodec, with Pillow as the overflow valve.

The whole design in one sentence: a process-global pool of retained decoder
slots is borrowed **non-blockingly** by whatever thread is already running
``ImageMediaIO.load_bytes``, an image is admitted only if a header-only
predicate proves Pillow parity, and *every* unhappy path returns ``None`` for
that position, which means "Pillow decodes this one, exactly as it does today".

Two consequences are worth stating explicitly, because they are what keep this
module small:

* **The GPU layer is never allowed to fail.** It has no exception type, no
  retry, no queue and no error taxonomy, because a failure is indistinguishable
  from "not eligible" -- both are ``None`` and both are handled by the caller's
  existing Pillow call, at the right position, in the caller's own task.
* **No thread ever blocks on a resource this module owns.** Slots are taken
  with a non-blocking ``popleft`` and GPU bytes with ``try_acquire``. The
  wait-for graph therefore has no edges, so deadlock is impossible by
  construction rather than by protocol.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Event
from typing import Any

import numpy as np

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

NVIMGCODEC_BACKEND = "nvimgcodec"
PILLOW_BACKEND = "pillow"

# Eight, from measurement, not intuition. At concurrency 256 with 8 slots the
# outcome counters read {"gpu": 4864, "native_width_1": 4864} -- every image on
# the GPU, zero shed. At 4 slots ~18% of images time out parking and fall back
# to Pillow, costing ~7% end-to-end. Decode-only microbenchmarks prefer 4 (more
# raw img/s), but this feature exists for serving, so serving wins the default.
# Costs 8 * DECODER_WORKSPACE_BYTES + one CUDA context, reserved at startup.
DEFAULT_NUM_DECODERS = 8

# Measured on A100 (nvimgcodec 0.9.0), not estimated. Creating the first CUDA
# stream in a process that has no context builds one: measured 434 MiB of
# device memory attributed to the process. It is per-process and shared with any
# other GPU media backend, so callers must take max(), never sum, across them.
CUDA_CONTEXT_BYTES = 448 * 1024 * 1024
# Marginal retained decoder state per slot, measured across 1..8 warmed slots:
# ~11 MiB/slot at 1080p and ~20 MiB/slot at 4K. Sized for the 4K case with
# headroom; nvJPEG grows its workspace to the largest image a slot has seen.
# Largest raster the accelerated path will admit. nvJPEG grows its retained
# workspace with the biggest image a slot has seen, so the reservation is only
# an upper bound if the eligible domain is bounded too. Images above this go to
# Pillow, which makes DECODER_WORKSPACE_BYTES (measured at 4K, with headroom) a
# real ceiling rather than a sample.
MAX_ELIGIBLE_PIXELS = 3840 * 2160
DECODER_WORKSPACE_BYTES = 32 * 1024 * 1024
# Each slot also retains one decoded-image buffer between calls. nvImageCodec
# allocates it from its own device allocator, so it is invisible to torch's
# accounting and has to be reserved explicitly or it comes silently out of the
# KV cache. Measured directly: the buffer grows by reallocation to the largest
# raster a slot has decoded and never shrinks, so the bound is one raster at the
# eligibility ceiling, not a sum over sizes.
DECODER_RETAINED_RASTER_BYTES = MAX_ELIGIBLE_PIXELS * 3

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"

# Which host layout the accelerated path hands back. "pil" is the safe default
# and the only one that needs no processor capability: the other two bypass the
# PIL-only normalization in multimodal/parse.py, so they are selected ONLY by
# probe_output_layout() proving a specific processor produces identical tensors.
#   "chw" -> (3, H, W)   requested from the decoder as P_RGB, so the GPU emits
#                        planar directly and no host transpose is needed.
#                        Measured faster than HWC through the real processor.
#   "hwc" -> (H, W, 3)   I_RGB, the interleaved layout.
#
# Both array layouts are ZERO-COPY VIEWS of a decoder-allocated host buffer:
# owndata=False, and `base` is the nvImageCodec host Image that owns the
# storage. That is lifetime-safe -- the buffer is fresh per decode, is not
# reused by later decodes, and outlives decoder shutdown because the array
# holds the reference -- but it is not ordinary NumPy storage, so a caller
# requiring owned memory must use "pil" or copy. Pinned by
# tests/multimodal/test_image_decoders_cuda.py::test_output_ownership_contract.
#   "pil" -> PIL.Image   no bypass, no capability required
_OUTPUT_LAYOUT = "pil"

# Widest native call a leader will assemble. This tracks the number of hardware
# JPEG engines on the device, because that is what a wide call parallelises
# across. Measured per-image throughput vs width:
#   A100 (5 NVJPG engines):  1.00x / 1.89x / 3.38x / 4.01x / 1.13x  at 1/2/4/5/8
#   RTX PRO 6000 Blackwell:  1.00x / 1.89x / 3.52x / 1.52x / 0.84x  at 1/2/4/5/8
# Both peak and then collapse, because one decoder has a single internal CUDA
# stream: past the engine count the batch serialises instead of spreading. Five
# is the A100 optimum; four is Blackwell's. Overridable per deployment.
# On a device with no hardware engines the curve is flat (1.03x at every width)
# and this value is irrelevant -- see the backends list in _Slot.
DEFAULT_COALESCE_WIDTH = 5

# Smallest image the accelerator will accept. Zero by default: nothing in the
# defaults declines hardware decoding.
#
# It exists because the trade is host-vs-GPU, and which side is scarce is a
# property of the deployment. Measured end to end at a 1024x576 pixel budget,
# three arms in one session: on an RTX PRO 6000 at 1080p this backend is 1.17x
# upstream, while on an A100 at 1080p it is 0.989x -- reproducibly below
# parity, with the tightest error bars in that matrix (+-0.07 and +-0.03 over
# three runs). The A100 sits at 98-99% GPU in every arm there, so it has no
# headroom in which to spend the 23% of host CPU this frees, and the remaining
# cost is putting decode work on an already-saturated device. Such a deployment
# can raise this to keep hardware decoding for the large images that pay for
# themselves while leaving small ones on Pillow.
DEFAULT_MIN_GPU_PIXELS = 0
MIN_GPU_PIXELS = DEFAULT_MIN_GPU_PIXELS
COALESCE_WIDTH = DEFAULT_COALESCE_WIDTH
# How long a caller parks hoping to be adopted. A fixed 2 ms was measured to be
# self-defeating: a leader adopts only at the *start* of its native call, so
# under the contention that makes callers park at all, the next adoption point
# is a whole decode away -- always longer than the window. The result was that
# parked callers reliably timed out instead of being batched (measured: achieved
# width pinned at 1.00, and at 16 callers 43% of eligible images shed to Pillow
# while slots kept going idle). The window must therefore span at least one
# decode, which is workload- and GPU-dependent, so it is derived from a measured
# EWMA of native decode duration rather than guessed. Bounds keep it honest:
# never shorter than the old constant, never long enough to starve the shared
# media executor thread this parks.
COALESCE_WAIT_SECONDS = 0.002
MAX_COALESCE_WAIT_SECONDS = 0.030
# Seeded so that the derived window (2x this) starts at exactly the floor, and
# the very first callers behave as they did before the window became adaptive.
_DECODE_EWMA_SECONDS = COALESCE_WAIT_SECONDS / 2.0
_DECODE_EWMA_ALPHA = 0.2
# A leader that already claimed an entry is mid-native-call; withdrawing then
# would race it, so the waiter grants this much extra time. Kept short because
# this parks a thread from the *shared* media executor, which also serves audio,
# video and cache I/O -- a long grace here starves unrelated work.
_CLAIMED_GRACE_SECONDS = 0.05
# Parked callers retain their *encoded* bytes (plus a CodeStream), not a
# decoded raster, so the cap must count encoded size. Counting raster bytes
# instead would let a burst of tiny-dimension JPEGs carrying large metadata
# retain far more than this nominal limit.
MAX_PARKED_BYTES = 64 * 1024 * 1024


@dataclass(eq=False)
class _Waiter:
    """One parked single-image request awaiting adoption by a slot owner."""

    data: bytes
    stream: Any
    raster_bytes: int
    done: Event
    result: np.ndarray | None = field(default=None)

# --- process-global state -------------------------------------------------
# _PID is a plain int read WITHOUT the lock: R8 requires that fork detection
# never acquires a lock inherited from the parent.
_PID = os.getpid()
_LOCK = threading.Lock()
_FREE: deque[Any] = deque()
_PARKED: deque[_Waiter] = deque()
_PARKED_BYTES = 0
_CREATED = 0
_ACTIVE = 0
_GENERATION = 0
_NUM_DECODERS = 0
_MAX_PARKED = 0
_CLOSED = True
_DISABLED = False
_COUNTERS: Counter[str] = Counter()


def _count_width(width: int) -> None:
    """R10: the achieved native batch width, which is what the design turns on.

    Recorded per native call, not per claim, so it cannot flatter itself.
    """
    _COUNTERS[f"native_width_{width}"] += 1


def _count(reason: str) -> None:
    # Plain Counter mutation under the GIL; contention here is irrelevant and a
    # lost increment in a stats counter is not worth a lock on the hot path.
    _COUNTERS[reason] += 1
    _maybe_log_outcomes()


# R10 is only satisfied if the outcomes actually reach a log. Emitting them
# solely from shutdown() ties that to a clean renderer teardown, which does not
# happen for every API server process: with --api-server-count 2 and 4, serving
# 2560 images each, no outcome line was emitted at all. A run where the backend
# silently declined everything was therefore indistinguishable from one where it
# worked -- the exact confusion the counters exist to prevent. Emit periodically
# as well, so the record survives any teardown path.
_OUTCOME_LOG_EVERY = 4096
_decisions_since_log = 0


def _maybe_log_outcomes() -> None:
    # A running total rather than a scan of _COUNTERS: this sits on the
    # per-image path, and summing the dict here measured 769 ns/call against
    # 63 ns for the bare increment it guards.
    global _decisions_since_log
    _decisions_since_log += 1
    if _decisions_since_log < _OUTCOME_LOG_EVERY:
        return
    _decisions_since_log = 0
    _log_outcomes("nvImageCodec image decode outcomes (running)")


def _log_outcomes(prefix: str) -> None:
    outcomes = stats()
    if outcomes:
        logger.info("%s: %s", prefix, json.dumps(dict(sorted(outcomes.items()))))


def output_layout() -> str:
    """The host layout the accelerated path currently returns."""
    return _OUTPUT_LAYOUT


def stats() -> dict[str, int]:
    """R10: decode outcomes by reason, so a silent shed is never invisible."""
    return dict(_COUNTERS)


def configure(
    num_decoders: int = DEFAULT_NUM_DECODERS,
    output_layout: str = "pil",
    coalesce_width: int = DEFAULT_COALESCE_WIDTH,
    min_gpu_pixels: int = DEFAULT_MIN_GPU_PIXELS,
) -> None:
    """Enable the backend for this process. Idempotent within a generation."""
    global _NUM_DECODERS, _CLOSED, _PID, _CREATED, _FREE, _DISABLED
    global _MAX_PARKED, _PARKED, _GENERATION, _ACTIVE, _OUTPUT_LAYOUT, _PARKED_BYTES
    global COALESCE_WIDTH, _DECODE_EWMA_SECONDS, MIN_GPU_PIXELS
    if not isinstance(num_decoders, int) or isinstance(num_decoders, bool):
        raise ValueError("num_decoders must be an integer")
    if not 1 <= num_decoders <= 16:
        raise ValueError("num_decoders must be between 1 and 16")
    if output_layout not in ("pil", "hwc", "chw"):
        raise ValueError("output_layout must be 'pil', 'hwc' or 'chw'")
    if not isinstance(coalesce_width, int) or not 1 <= coalesce_width <= 16:
        raise ValueError("coalesce_width must be an integer between 1 and 16")
    with _LOCK:
        _PID = os.getpid()
        _NUM_DECODERS = num_decoders
        # Enough room for every caller a leader could adopt, and no more: a
        # deeper queue only adds latency, since a leader takes COALESCE_WIDTH.
        _MAX_PARKED = num_decoders * coalesce_width
        _PARKED = deque()
        _PARKED_BYTES = 0
        # Reconfiguration may change resolution mix or decoder count; a stale
        # decode estimate would size the next park window from the old regime.
        _DECODE_EWMA_SECONDS = COALESCE_WAIT_SECONDS / 2.0
        _FREE = deque()
        _CREATED = 0
        _ACTIVE = 0
        # A new generation. Slots borrowed before this point belong to the old
        # one and must never re-enter the pool -- see _release_slot.
        _GENERATION += 1
        _OUTPUT_LAYOUT = output_layout
        COALESCE_WIDTH = coalesce_width
        MIN_GPU_PIXELS = int(min_gpu_pixels)
        _DISABLED = False
        _CLOSED = False


def shutdown() -> None:
    """Release retained decoders. Idempotent; a fresh generation may follow."""
    global _CREATED, _CLOSED, _FREE, _PARKED, _GENERATION, _ACTIVE, _PARKED_BYTES
    if os.getpid() != _PID:
        # Never touch decoders or a lock inherited from the parent process.
        return
    with _LOCK:
        if _CLOSED:
            return
        _CLOSED = True
        # Retiring the generation is what makes an in-flight borrow safe: its
        # slot is dropped on release instead of rejoining a later pool.
        _GENERATION += 1
        _ACTIVE = 0
        slots = list(_FREE)
        _FREE = deque()
        # Wake every parked caller with no result: they fall back to Pillow
        # rather than waiting out their timeout against a closed generation.
        stranded, _PARKED = list(_PARKED), deque()
        _PARKED_BYTES = 0
        _CREATED = 0
    # In-flight decodes keep their own slot alive by reference and append it to
    # a deque nobody reads, so a slot can never be freed under a live call.
    for waiter in stranded:
        waiter.result = None
        waiter.done.set()
    del slots
    # R10. Without this a run where every image quietly fell back to Pillow is
    # indistinguishable from a run where the backend simply did not help.
    _log_outcomes("nvImageCodec image decode outcomes")


def _load_nvimgcodec():
    from nvidia import nvimgcodec

    return nvimgcodec


class _Slot:
    """One retained decoder plus its decode params.

    A slot is owned *exclusively* by whichever thread borrowed it. That is a
    memory-safety requirement, not a resource policy: sharing one
    ``nvimgcodec.Decoder`` across threads reproducibly corrupts the heap
    ("double free or corruption", "corrupted size vs. prev_size") rather than
    raising, so exclusivity is made structural here.
    """

    def __init__(self, generation: int) -> None:
        import torch

        self.generation = generation
        nvimgcodec = _load_nvimgcodec()
        # A private CUDA stream per slot. Without one, every slot submits to the
        # legacy default stream, which serialises them on the device: N decoder
        # slots would give N-way host concurrency and 1-way device concurrency,
        # silently capping the pool at one decoder's throughput.
        self.stream = torch.cuda.Stream(device=0)
        # One decoded-image buffer retained between calls. Without it every
        # decode allocates fresh device memory, and cudaMalloc synchronizes the
        # whole device: on an instance whose GPU already serves the model that
        # stall is paid by the forward pass. Measured at 1080p on an A100,
        # accelerator decoding was 0.988x of baseline with the NVJPG engines at
        # 5.7% -- idle engines and a real loss point at the allocation, not the
        # decode. See DECODER_RETAINED_RASTER_BYTES for the memory it costs.
        self.reusable: list[Any] = []
        self.decoder = nvimgcodec.Decoder(
            device_id=0,
            # MUST stay 1. In nvimgcodec 0.9.0 the HYBRID_CPU_GPU backend
            # deadlocks forever on a native batch of exactly ONE image when the
            # decoder is built with more than one CPU helper thread. Reproduced
            # for T in (2, 4, 8) x B == 1; T == 1 is safe at every batch size,
            # and B >= 2 is safe at every T. Width-one batches are the common
            # case here, so this value is load-bearing. See
            # tests/multimodal/test_image_decoders_cuda.py for the guard test.
            max_num_cpu_threads=1,
            options=":num_cuda_streams=1",
            # Prefer the dedicated NVJPG hardware engines, then the GPU kernel
            # path, and only then the CPU-assisted hybrid. Naming these is NOT
            # cosmetic: with no list, nvImageCodec selects the hybrid backend,
            # which decodes entropy on the CPU. Measured on an NVJPG-equipped
            # GPU, the hybrid path is ~2x slower at width 1 and -- decisively --
            # does not scale with batch width at all (1.03x from width 1 to 16),
            # whereas this list scales 3.5x from width 1 to 4. Losing the
            # hardware engines silently removes the entire reason for batching.
            backends=[
                nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY),
                nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
                nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
            ],
        )
        self.params = nvimgcodec.DecodeParams(
            # vLLM applies EXIF orientation itself so the transform is provably
            # identical to Pillow's; see _apply_orientation.
            apply_exif_orientation=False,
            color_spec=nvimgcodec.ColorSpec.SRGB,
            # vLLM's image contract is uint8. Higher precision is declined by
            # the eligibility predicate rather than silently rescaled here.
            allow_any_depth=False,
            # Planar directly from the decoder when CHW was negotiated: the
            # GPU emits (3, H, W) at no cost, where converting on the host costs
            # a full-raster transpose (measured 1.14 ms at 1080p, 5.08 ms at 4K
            # per image). Verified bit-identical to transposing I_RGB.
            sample_format=(
                nvimgcodec.SampleFormat.P_RGB
                if _OUTPUT_LAYOUT == "chw"
                else nvimgcodec.SampleFormat.I_RGB
            ),
        )


# EXIF orientation -> the numpy transform on an HWC array. These mirror
# PIL.ImageOps.exif_transpose exactly (PIL rotations are counter-clockwise);
# equality is proved per tag by the differential parity corpus, not asserted.
# Same transforms as _ORIENTATION_OPS but for (3, H, W): the spatial axes move
# from (0, 1) to (1, 2). Equality with the HWC table is proved per tag by the
# differential parity corpus, which runs both layouts.
_ORIENTATION_OPS_CHW = {
    2: lambda a: a[:, :, ::-1],
    3: lambda a: a[:, ::-1, ::-1],
    4: lambda a: a[:, ::-1, :],
    5: lambda a: a.transpose(0, 2, 1),
    6: lambda a: np.rot90(a, k=-1, axes=(1, 2)),
    7: lambda a: a.transpose(0, 2, 1)[:, ::-1, ::-1],
    8: lambda a: np.rot90(a, k=1, axes=(1, 2)),
}

_ORIENTATION_OPS = {
    2: lambda a: a[:, ::-1],
    3: lambda a: a[::-1, ::-1],
    4: lambda a: a[::-1, :],
    5: lambda a: a.transpose(1, 0, 2),
    6: lambda a: np.rot90(a, k=-1),
    7: lambda a: a.transpose(1, 0, 2)[::-1, ::-1],
    8: lambda a: np.rot90(a, k=1),
}


def _has_image_id(data: bytes) -> bool:
    """True when the JPEG carries an EXIF ImageID.

    MultiModalHasher gives such images a UUID-derived cache identity that only
    exists on the Pillow branch. Accelerating them would silently change a
    documented caching behaviour, so they are declined.
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            return Image.ExifTags.Base.ImageID in image.getexif()
    except Exception:
        return True  # unreadable metadata: decline rather than guess


def _exif_orientation(data: bytes) -> int:
    """Read the orientation tag with Pillow, which *is* the parity reference.

    Header-only, measured at ~11 us against a per-image budget in the
    milliseconds. Using Pillow here rather than a hand-rolled APP1 parser is
    deliberate: it makes orientation parity true by construction instead of by
    a second implementation that has to be kept in agreement.
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            value = image.getexif().get(0x0112)
    except Exception:
        return 1
    return value if isinstance(value, int) and 1 <= value <= 8 else 1


def _eligible(data: bytes, image_mode: str | None, nvimgcodec, pool) -> Any:
    """Header-only admission. Returns a CodeStream, or None to use Pillow.

    Ordered; the first failing condition decides. Every rejection is a
    correctness or safety statement, not a heuristic -- see the table in
    docs/features/multimodal_inputs.md.
    """
    # E2: only one output semantics exists, which is also why this module needs
    # no compatibility key. RGBA/None targets need mode conversion and
    # background compositing that we do not reproduce.
    if image_mode != "RGB":
        _count("pillow:image_mode")
        return None
    # E3: JPEG only. Every other codec reaches nvImageCodec through a CPU
    # plugin (no hardware win) while carrying real divergence: PNG/WebP alpha
    # and palette-tRNS differ from Pillow by up to 243/8-bit, and a palette
    # tRNS header reports num_channels=3, giving the predicate no signal at all.
    if len(data) < 4 or not data.startswith(_SOI):
        _count("pillow:not_jpeg")
        return None
    # E4: truncation. THE safety-critical row. A truncated JPEG parses cleanly
    # as a CodeStream and nvImageCodec decodes it to grey filler, while Pillow
    # raises OSError -- so without this check a cut-off upload silently reaches
    # the model instead of failing the request.
    if not data.endswith(_EOI):
        _count("pillow:truncated")
        return None
    try:
        stream = nvimgcodec.CodeStream(data)
        codec = str(stream.codec_name).lower()
        width = int(stream.width)
        height = int(stream.height)
        channels = int(stream.num_channels)
        precision = int(stream.precision)
    except Exception:
        _count("pillow:unreadable_header")
        return None
    if codec != "jpeg":  # E6: a JPEG SOI prefixing some other container.
        _count("pillow:not_jpeg")
        return None
    if width <= 0 or height <= 0:
        _count("pillow:bad_dimensions")
        return None
    # E8: 4-channel CMYK/YCCK and 2-channel LA are declined. The one CMYK
    # fixture measured clean, but the YCCK and inverted-Adobe variants are not
    # covered by any corpus, and V3 says a tie goes to Pillow.
    if channels not in (1, 3):
        _count("pillow:channels")
        return None
    # E9: >8-bit would be silently rescaled with allow_any_depth=False.
    if precision != 8:
        _count("pillow:precision")
        return None
    if width * height < MIN_GPU_PIXELS:
        # Deployment says images this small are not worth the accelerator; see
        # MIN_GPU_PIXELS.
        _count("pillow:below_min_gpu_pixels")
        return None
    # E10: preserve the existing pixel-limit error verbatim by leaving it to
    # the Pillow path rather than raising a second, differently-worded one.
    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    if max_pixels > 0 and width * height > max_pixels:
        _count("pillow:pixel_limit")
        return None
    # Keep the accelerated domain inside what the startup reservation covers.
    if width * height > MAX_ELIGIBLE_PIXELS:
        _count("pillow:over_reserved_domain")
        return None
    # E11: an image larger than the whole pool can never be leased; declining
    # here is what stops a resource bound from becoming a per-request ceiling.
    if width * height * 3 > pool.total_bytes:
        _count("pillow:over_pool")
        return None
    # E12: preserve the Pillow-only EXIF ImageID cache identity.
    if _has_image_id(data):
        _count("pillow:image_id")
        return None
    return stream


def _acquire_slot() -> Any:
    """Borrow a slot, or return None. Never waits."""
    global _CREATED, _ACTIVE
    with _LOCK:
        if _CLOSED or _DISABLED:
            return None
        generation = _GENERATION
        if _FREE:
            _ACTIVE += 1
            return _FREE.popleft()
        if _CREATED >= _NUM_DECODERS:
            return None
        # Reserve the quota slot under the lock, then build outside it: slot
        # construction is ~0.2 ms and must not be held against other threads.
        _CREATED += 1
        _ACTIVE += 1
    try:
        return _Slot(generation)
    except Exception:
        with _LOCK:
            _CREATED -= 1
            _ACTIVE -= 1
        logger.warning_once(
            "nvImageCodec decoder could not be created; using Pillow.",
            exc_info=True,
        )
        return None


def _release_slot(slot: Any) -> None:
    """Return a slot, unless its generation has been retired.

    A borrow can outlive shutdown(). Appending such a slot unconditionally lets
    it rejoin a *later* generation's pool, so a fresh borrower would receive a
    decoder built against retired state while the quota counter reads zero.
    Dropping it here is what makes shutdown/reconfigure safe.
    """
    global _ACTIVE
    with _LOCK:
        if getattr(slot, "generation", None) != _GENERATION or _CLOSED:
            return
        _ACTIVE -= 1
        _FREE.append(slot)


def decode_batch(
    datas: Sequence[bytes],
    image_mode: str | None = "RGB",
) -> list[np.ndarray | None]:
    """Decode what is safely decodable on the GPU; ``None`` means "use Pillow".

    Always returns exactly ``len(datas)`` entries, in input order. Never raises.
    """
    results: list[np.ndarray | None] = [None] * len(datas)
    if not datas:
        return results

    # R8/R9/R6 gate. All plain reads; no lock, so this is safe in a forked child.
    if os.getpid() != _PID or _CLOSED or _DISABLED:
        _count("pillow:not_available")
        return results

    from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

    pool = get_mm_gpu_ipc_pool()
    if pool is None:
        _count("pillow:no_gpu_pool")
        return results

    try:
        nvimgcodec = _load_nvimgcodec()
    except Exception:
        _disable("the nvidia-nvimgcodec package is not importable")
        return results

    admitted: list[tuple[int, Any]] = []
    raster_bytes = 0
    for index, data in enumerate(datas):
        stream = _eligible(data, image_mode, nvimgcodec, pool)
        if stream is not None:
            admitted.append((index, stream))
            raster_bytes += int(stream.width) * int(stream.height) * 3
    if not admitted:
        return results

    slot = _acquire_slot()
    if slot is None:
        # Every decoder is busy. Do NOT simply shed: at the request concurrency
        # where this feature matters, shedding turns it off (measured 0.4% of
        # images reaching the GPU at 16 concurrent callers). Instead park
        # briefly so a thread that owns a slot can adopt this work into its own
        # native call. That is what makes achieved batch width grow with load,
        # which is the whole point -- an NVJPG-equipped GPU decodes a width-4
        # batch 3.5x faster per image than four width-1 batches.
        return _await_leader(datas, admitted, results, raster_bytes)
    started = time.monotonic()
    try:
        _decode_on_slot(slot, pool, datas, admitted, results, raster_bytes)
    finally:
        # Time the whole interval the slot is held, not just decoder.decode().
        # nvImageCodec's decode is asynchronous: the device-to-host copy and its
        # synchronization happen later, in Image.cpu() inside _to_array. What a
        # parked caller waits for is the next moment this slot can adopt again,
        # which is full occupancy -- timing only the async submit would size the
        # window from a fraction of it.
        _observe_decode(time.monotonic() - started)
        _release_slot(slot)
    return results


def _decode_on_slot(
    slot: Any,
    pool: Any,
    datas: Sequence[bytes],
    admitted: list[tuple[int, Any]],
    results: list[np.ndarray | None],
    raster_bytes: int,
) -> None:
    """Decode this caller's work plus any parked work, in one native call."""
    # Adopt parked entries first, so the native call is as wide as the current
    # load allows. Bounded by COALESCE_WIDTH; the leader never waits for them.
    adopted = _adopt_parked(COALESCE_WIDTH - len(admitted))
    streams = [stream for _, stream in admitted] + [w.stream for w in adopted]
    if not streams:
        return
    total_bytes = raster_bytes + sum(w.raster_bytes for w in adopted)
    try:
        lease = pool.try_acquire(total_bytes)  # Still never waits.
        if lease is None:
            _count("pillow:no_gpu_bytes")
            _fail_parked(adopted)
            return
        import torch

        with lease, torch.cuda.stream(slot.stream):
            # Chunk to COALESCE_WIDTH. Without this a plural request of N
            # images issues a single width-N native call, and measured
            # throughput collapses past the engine count (A100: 4.01x at
            # width 5, 1.13x at width 8). COALESCE_WIDTH bounds adopted
            # waiters; it must bound this too.
            decoded = []
            for start in range(0, len(streams), COALESCE_WIDTH):
                part = streams[start : start + COALESCE_WIDTH]
                # Retain a buffer only for width-1 calls. Each retained buffer
                # holds a full raster that the startup reservation must cover,
                # and serving counters read native_width_1 for every image, so
                # this keeps the win while bounding retention at one raster per
                # slot rather than COALESCE_WIDTH of them.
                reuse = len(part) == 1 and len(slot.reusable) == 1
                if reuse:
                    out = slot.decoder.decode(
                        part,
                        images=slot.reusable,
                        params=slot.params,
                        cuda_stream=slot.stream.cuda_stream,
                    )
                else:
                    out = slot.decoder.decode(
                        part, params=slot.params, cuda_stream=slot.stream.cuda_stream
                    )
                    if len(part) == 1 and out and out[0] is not None:
                        slot.reusable = [out[0]]
                decoded.extend(out)
                _count_width(len(part))
            if len(decoded) != len(streams):
                _count("pillow:result_count")
                _fail_parked(adopted)
                return
            mine, theirs = decoded[: len(admitted)], decoded[len(admitted) :]
            # Everything below copies to the host. Drop the batch reference now
            # so the only live device handles are the per-image ones consumed
            # in the loops, shortening how long the GPU lease is pinned.
            decoded = None
            for (index, stream), image in zip(admitted, mine):
                if image is None:
                    _count("pillow:native_miss")
                    continue
                array = _to_array(image, stream, datas[index])
                if array is None:
                    continue
                results[index] = array
                _count("gpu")
            for waiter, image in zip(adopted, theirs):
                if image is not None:
                    waiter.result = _to_array(image, waiter.stream, waiter.data)
                if waiter.result is not None:
                    _count("gpu")
                else:
                    _count("pillow:native_miss")
                waiter.done.set()
    except Exception:
        # A native failure is never fatal: this caller and every adopted caller
        # fall back to Pillow. Nothing is retried, so one bad stream cannot
        # amplify into work for anybody.
        _count("pillow:native_error")
        for index, _ in admitted:
            results[index] = None
        _fail_parked(adopted)
        logger.warning_once("nvImageCodec decode failed; using Pillow.", exc_info=True)


def _await_leader(
    datas: Sequence[bytes],
    admitted: list[tuple[int, Any]],
    results: list[np.ndarray | None],
    raster_bytes: int,
) -> list[np.ndarray | None]:
    """Park this caller's work so a slot-owning thread can adopt it.

    Only single-image callers park: a multi-image request already forms its own
    wide native call and has nothing to gain. Waiting is bounded and shedding
    stays legal, so this adds no way to block indefinitely.
    """
    if len(admitted) != 1:
        _count("pillow:no_slot")
        return results
    index, stream = admitted[0]
    waiter = _Waiter(
        data=datas[index], stream=stream, raster_bytes=raster_bytes, done=Event()
    )
    parked_bytes = len(waiter.data)
    global _PARKED_BYTES
    with _LOCK:
        if (
            _CLOSED
            or _DISABLED
            or len(_PARKED) >= _MAX_PARKED
            or _PARKED_BYTES + parked_bytes > MAX_PARKED_BYTES
        ):
            _count("pillow:no_slot")
            return results
        _PARKED.append(waiter)
        _PARKED_BYTES += parked_bytes
    if waiter.done.wait(timeout=_park_timeout()):
        results[index] = waiter.result
        return results
    # Timed out. Withdraw -- unless a leader already claimed this entry, in
    # which case its decode is already in flight and we must let it finish
    # rather than race it. That wait is bounded by one native call.
    with _LOCK:
        if waiter in _PARKED:
            _PARKED.remove(waiter)
            _PARKED_BYTES -= len(waiter.data)
            _count("pillow:coalesce_timeout")
            return results
    waiter.done.wait(timeout=_CLAIMED_GRACE_SECONDS)
    results[index] = waiter.result
    if waiter.result is None:
        _count("pillow:coalesce_timeout")
    return results


def _observe_decode(seconds: float) -> None:
    """Track how long a slot stays held, to size the park window.

    Plain assignment under no lock: this is a hint for a timeout, so a torn
    update costs nothing and a lock on the hot path would cost more than it.
    """
    global _DECODE_EWMA_SECONDS
    _DECODE_EWMA_SECONDS = (1.0 - _DECODE_EWMA_ALPHA) * _DECODE_EWMA_SECONDS + (
        _DECODE_EWMA_ALPHA * seconds
    )


def _park_timeout() -> float:
    """Park long enough to reach the next adoption point, and no longer.

    Twice the observed slot occupancy: one to cover the call already in flight
    when we parked, and one for the leader that starts after it.
    """
    return min(
        MAX_COALESCE_WAIT_SECONDS,
        max(COALESCE_WAIT_SECONDS, 2.0 * _DECODE_EWMA_SECONDS),
    )


def _adopt_parked(capacity: int) -> list[_Waiter]:
    global _PARKED_BYTES
    if capacity <= 0:
        return []
    with _LOCK:
        taken = [_PARKED.popleft() for _ in range(min(capacity, len(_PARKED)))]
        _PARKED_BYTES -= sum(len(w.data) for w in taken)
    return taken


def _fail_parked(adopted: list[_Waiter]) -> None:
    for waiter in adopted:
        waiter.result = None
        waiter.done.set()


def _to_array(image: Any, stream: Any, data: bytes) -> np.ndarray | None:
    """Copy one decoded image to a host array, applying EXIF orientation."""
    try:
        host = np.asarray(image.cpu())
        planar = _OUTPUT_LAYOUT == "chw"
        height, width = int(stream.height), int(stream.width)
        expected = (3, height, width) if planar else (height, width, 3)
        if host.shape != expected or host.dtype != np.uint8:
            _count("pillow:unexpected_raster")
            return None
        # No defensive copy. Image.cpu() allocates a *fresh* host image and
        # transfers ownership; numpy keeps it alive through array.base, so this
        # does not alias a buffer a later decode can reuse. Verified by decoding
        # 8 further images through the same decoder and asserting an earlier
        # array is byte-identical (tests/multimodal/test_image_decoders_cuda.py
        # ::test_result_does_not_alias_decoder_memory). Copying here would cost
        # a full raster memcpy per image -- 6.2 MB at 1080p, 24.9 MB at 4K -- on
        # the frontend, which is most of what the array output contract saves.
        array = host
        orientation = _exif_orientation(data)
        if orientation != 1:
            ops = _ORIENTATION_OPS_CHW if planar else _ORIENTATION_OPS
            array = np.ascontiguousarray(ops[orientation](array))
        return array
    except Exception:
        _count("pillow:conversion_error")
        return None


def _disable(reason: str) -> None:
    global _DISABLED
    with _LOCK:
        if _DISABLED:
            return
        _DISABLED = True
    logger.warning("Disabling the nvImageCodec image backend: %s", reason)




def decoder_gpu_memory_bytes(num_decoders: int) -> int:
    """Per-process device footprint, for the startup KV-cache reservation."""
    return (
        num_decoders * (DECODER_WORKSPACE_BYTES + DECODER_RETAINED_RASTER_BYTES)
        + CUDA_CONTEXT_BYTES
    )
