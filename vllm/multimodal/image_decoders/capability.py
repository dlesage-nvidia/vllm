# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decide, per deployment, whether the GPU decoder may bypass PIL.

Returning a raw array instead of a ``PIL.Image`` skips the mode/EXIF
normalization in ``multimodal/parse.py``, which is applied only to PIL items.
A broad type alias permitting ``np.ndarray`` does not prove that *this* model's
image processor treats an array and a PIL image identically, so the bypass is
gated on evidence rather than on the type alias.

The probe runs once at startup against the real processor and requires
bit-exact agreement with the PIL result before allowing a bypass. Anything
unexpected -- an exception, a missing processor, a shape mismatch, a single
differing element -- falls back to ``"pil"``, which is always correct.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

# Deliberately awkward, non-square sizes: a square probe would hide a
# transposed-layout bug, and a patch-aligned one would hide a padding bug.
_PROBE_SHAPES = ((64, 48), (97, 65))

# Processor kwargs that change how a raw array is *interpreted*, as opposed to
# how it is transformed. The startup probe certifies a layout against the
# server-level configuration only; a request that overrides one of these is
# asking the processor to read the array differently than was certified, so the
# certification no longer holds for that request and it must take the PIL path.
LAYOUT_SENSITIVE_PROCESSOR_KWARGS = frozenset(
    {"input_data_format", "data_format", "do_convert_rgb"}
)


def request_invalidates_probe(processor_kwargs: Mapping[str, Any] | None) -> bool:
    """True when a request's processor kwargs void the startup certification."""
    if not processor_kwargs:
        return False
    return not LAYOUT_SENSITIVE_PROCESSOR_KWARGS.isdisjoint(processor_kwargs)


def _reference_and_candidates(width: int, height: int):
    from PIL import Image

    rs = np.random.RandomState(width * 7919 + height)
    hwc = rs.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return (
        Image.fromarray(hwc),
        {"hwc": hwc, "chw": np.ascontiguousarray(hwc.transpose(2, 0, 1))},
    )


def _call_kwargs(processor: Any, configured: Mapping[str, Any] | None) -> dict:
    """Keep only configured kwargs this processor's call signature accepts.

    The probe must exercise the processor the way production will call it. A
    deployment that pins ``input_data_format`` changes how a raw array is
    interpreted, so probing without it would certify a layout that production
    then feeds through a different code path. Unknown keys are dropped rather
    than forwarded, because an unexpected keyword would raise and demote an
    otherwise-good deployment to PIL.
    """
    if not configured:
        return {}
    import inspect

    accepted: set[str] = set()
    for name in ("preprocess", "__call__"):
        method = getattr(processor, name, None)
        if method is None:
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            # **kwargs accepts anything, so in production every configured key
            # really does reach this processor. Filtering to the names visible
            # in the signature would probe a *different* call than the one
            # production makes, and could certify a layout that the dropped
            # kwarg would have invalidated.
            return dict(configured)
        accepted.update(parameters)
    return {k: v for k, v in configured.items() if k in accepted}


def _outputs_match(
    processor: Any, reference: Any, candidate: Any, extra: Mapping[str, Any]
) -> bool:
    expected = processor(images=[reference], return_tensors="np", **extra)
    actual = processor(images=[candidate], return_tensors="np", **extra)
    if set(expected.keys()) != set(actual.keys()):
        return False
    for key in expected:
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        if left.shape != right.shape or not np.array_equal(left, right):
            return False
    return True


def _resolve_image_processor(
    processor: Any, processor_kwargs: Mapping[str, Any] | None = None
) -> Any:
    """Find the HF image processor behind whatever object we were handed.

    Callers pass vLLM's ``BaseMultiModalProcessor``, an HF ``ProcessorMixin``,
    or an image processor directly. Only the last is callable with ``images=``,
    so unwrap before probing -- otherwise the probe silently fails and every
    deployment falls back to PIL.
    """
    candidates = [processor]
    info = getattr(processor, "info", None)
    if info is not None:
        getter = getattr(info, "get_hf_processor", None)
        if callable(getter):
            # Ask for the processor the deployment actually configured; a
            # kwarg such as `size` or `do_resize` can change which array
            # layouts round-trip identically. Fall back to the bare call so a
            # rejected kwarg demotes the probe's fidelity, not the feature.
            for attempt in ((processor_kwargs or {}), {}):
                try:
                    candidates.append(getter(**attempt))
                    break
                except Exception:
                    logger.debug(
                        "nvImageCodec: get_hf_processor(**%s) failed.",
                        sorted(attempt), exc_info=True,
                    )

    for candidate in candidates:
        if candidate is None:
            continue
        inner = getattr(candidate, "image_processor", None)
        if inner is not None and callable(inner):
            return inner
        # An image processor handed in directly.
        if callable(candidate) and type(candidate).__name__.endswith("ImageProcessor"):
            return candidate
    return None



def _processor_device(processor_kwargs: Mapping[str, Any] | None) -> str | None:
    """The accelerator the processor itself runs on, if any.

    vLLM folds `--mm-processor-device` into `mm_processor_kwargs["device"]`, and
    permits an accelerator there only on an encode-only instance. Device output
    is offered only in that configuration: everywhere else the processor is on
    CPU and a device tensor would just be copied down again.
    """
    device = (processor_kwargs or {}).get("device")
    if not isinstance(device, str) or device in ("", "cpu", "auto"):
        return None
    return device


def _device_output_matches(processor, device: str, extra: Mapping[str, Any]) -> bool:
    """Does device input give the same answer as host input, on that device?

    The reference is deliberately the HOST-ARRAY path on the same device, not
    PIL on CPU. A deployment that asked for an accelerator processor has already
    accepted torchvision-on-device resampling (measured max|d| 0.0471 against
    PIL/CPU); what must be proven here is only that where the input started adds
    nothing further.
    """
    import torch

    for width, height in _PROBE_SHAPES:
        rs = np.random.RandomState(width * 7919 + height)
        hwc = rs.randint(0, 256, (height, width, 3), dtype=np.uint8)
        chw = torch.as_tensor(hwc, device=device).permute(2, 0, 1).contiguous()
        host = processor(images=[hwc], return_tensors="pt", **extra)
        dev = processor(images=[chw], return_tensors="pt", **extra)
        if set(host.keys()) != set(dev.keys()):
            return False
        for key in host:
            a, b = host[key], dev[key]
            a = a.cpu() if hasattr(a, "cpu") else torch.as_tensor(np.asarray(a))
            b = b.cpu() if hasattr(b, "cpu") else torch.as_tensor(np.asarray(b))
            if a.shape != b.shape or not torch.equal(a, b):
                return False
    return True


def probe_output_layout(
    processor: Any, processor_kwargs: Mapping[str, Any] | None = None
) -> str:
    """Return the fastest host layout this processor accepts bit-exactly.

    Preference order is ``chw`` then ``hwc``: CHW is measurably cheaper through
    the processor because it skips a transpose the processor would otherwise
    perform. ``pil`` is returned whenever a bypass cannot be proven safe.
    """
    image_processor = _resolve_image_processor(processor, processor_kwargs)
    if image_processor is None or not callable(image_processor):
        logger.info("nvImageCodec: no callable image processor; using PIL output.")
        return "pil"

    extra = _call_kwargs(image_processor, processor_kwargs)

    device = _processor_device(processor_kwargs)
    if device is not None:
        # Preferred when available: the decoded image never leaves the device.
        try:
            if _device_output_matches(image_processor, device, extra):
                logger.info(
                    "nvImageCodec: device output verified identical to host "
                    "input on %s for %s; keeping decoded images on the "
                    "accelerator.",
                    device, type(image_processor).__name__,
                )
                return "device"
            logger.info(
                "nvImageCodec: %s did not treat a device tensor identically to "
                "a host array; falling back to a host layout.",
                type(image_processor).__name__,
            )
        except Exception:
            logger.debug(
                "nvImageCodec: device-output probe raised for %s.",
                type(image_processor).__name__, exc_info=True,
            )

    probes = [_reference_and_candidates(width, height) for width, height in _PROBE_SHAPES]

    for layout in ("chw", "hwc"):
        try:
            if all(
                _outputs_match(image_processor, reference, candidates[layout], extra)
                for reference, candidates in probes
            ):
                logger.info(
                    "nvImageCodec: %s output verified bit-exact against PIL for %s; "
                    "bypassing PIL materialization.",
                    layout.upper(),
                    type(image_processor).__name__,
                )
                return layout
        except Exception:
            logger.debug(
                "nvImageCodec: %s probe raised for %s; trying next layout.",
                layout, type(image_processor).__name__, exc_info=True,
            )

    logger.info(
        "nvImageCodec: %s did not accept a raw array identically; using PIL output. "
        "GPU decoding still applies, but the PIL round trip is retained.",
        type(image_processor).__name__,
    )
    return "pil"


def _resolve_smart_resize(image_processor: Any):
    """Find smart_resize on the processor's own module, not a fixed import path.

    `transformers.models.qwen2_vl.image_processing_qwen2_vl` is an internal
    location upstream is free to move or rename, and when it moves this feature
    turns off rather than breaking loudly. Walking the processor class's MRO
    follows whatever module the processor actually lives in, so a family that
    grows its own module keeps working with no change here, and a family with
    no smart_resize is declined instead of being served a neighbour's copy.

    Returns (function, origin) or (None, tried) for the diagnostic.
    """
    import sys

    tried = []
    for cls in type(image_processor).__mro__:
        module = sys.modules.get(cls.__module__)
        if module is None:
            continue
        fn = getattr(module, "smart_resize", None)
        if callable(fn):
            return fn, cls.__module__
        tried.append(cls.__module__)
    # No fallback to a known module on purpose. A processor whose MRO exports no
    # smart_resize is not a smart_resize family, and applying another family's
    # sizing rule would silently produce targets the real processor never
    # chose. `from ... import smart_resize` in a processor module binds it as a
    # module attribute, so the walk above already covers re-exporting importers.
    return None, ", ".join(tried[:4]) or "no modules resolved"


def processor_resize_target(
    processor: Any, processor_kwargs: Mapping[str, Any] | None = None
):
    """A function giving a safe INTERMEDIATE size to shrink to on the device.

    Not the processor's final target. smart_resize is not idempotent: for a 4K
    source it returns 1008x560 (564,480 px, below the 589,824 floor), and
    feeding that back returns 1036x588 -- it oscillates between the two and has
    no fixed point. Pre-resizing to the processor's own answer therefore makes
    it resize a SECOND time, upscaling what we just downscaled. Measured cost of
    that mistake: e2e throughput fell 17% at 4K even though the frontend got
    4.1x faster in isolation.

    So shrink only to roughly twice the pixel budget, aspect preserved and
    factor-aligned, and let the processor perform its ordinary single resize
    from there. The host copy still drops about 7x at 4K (24.9 MB -> 3.5 MB),
    and the processor's own decision is left untouched.

    The returned function verifies per image that the processor would choose the
    same final size from the intermediate as from the source, and declines
    (returns None) when it would not, so this can never change what the model
    sees -- only how much data crosses the bus to produce it.
    """
    image_processor = _resolve_image_processor(processor, processor_kwargs)
    if image_processor is None:
        logger.warning(
            "nvImageCodec: resize target unavailable -- no image processor "
            "resolved from %s. Accelerator resizing is OFF.",
            type(processor).__name__,
        )
        return None
    smart_resize, origin = _resolve_smart_resize(image_processor)
    if smart_resize is None:
        logger.warning(
            "nvImageCodec: resize target unavailable -- no smart_resize found on "
            "%s or its bases (searched %s). Accelerator resizing is OFF.",
            type(image_processor).__name__, origin,
        )
        return None

    cfg = {**(getattr(image_processor, "__dict__", {}) or {}),
           **dict(processor_kwargs or {})}
    factor = int(getattr(image_processor, "patch_size", 14)) * int(
        getattr(image_processor, "merge_size", 2)
    )
    min_pixels = cfg.get("min_pixels")
    max_pixels = cfg.get("max_pixels")
    if not isinstance(min_pixels, int) or not isinstance(max_pixels, int):
        logger.warning(
            "nvImageCodec: resize target unavailable -- need integer min_pixels "
            "and max_pixels; got %r / %r from %s (factor=%d, smart_resize from "
            "%s). Accelerator resizing is OFF.",
            min_pixels, max_pixels, type(image_processor).__name__, factor,
            origin,
        )
        return None

    logger.info(
        "nvImageCodec: resize target derived from %s via smart_resize in %s "
        "(factor=%d, budget %d-%d px).",
        type(image_processor).__name__, origin, factor, min_pixels, max_pixels,
    )

    import math

    # Candidate shrink levels, smallest transfer first. 1.0 is the processor's
    # own target, which is ideal when it is a fixed point of smart_resize -- it
    # is for 16:9 sources, where the target lands exactly on the pixel budget.
    # It is NOT a fixed point in general (a 5000x3333 source gives 928x608, and
    # refeeding that gives 960x640), so larger intermediates are tried next and
    # anything that would change the processor's answer is refused.
    HEADROOMS = (1.0, 1.25, 1.5, 2.0, 3.0)

    def target(width: int, height: int):
        if width <= 0 or height <= 0:
            return None
        direct = smart_resize(height, width, factor=factor,
                              min_pixels=min_pixels, max_pixels=max_pixels)
        dh, dw = direct
        for headroom in HEADROOMS:
            if headroom == 1.0:
                cw, ch = int(dw), int(dh)
            else:
                scale = math.sqrt(headroom * max_pixels / float(width * height))
                if scale >= 1.0:
                    return None            # already small; nothing to save
                cw = max(factor, int(round(width * scale / factor)) * factor)
                ch = max(factor, int(round(height * scale / factor)) * factor)
            if cw * ch >= width * height:
                continue                   # would not shrink
            if smart_resize(ch, cw, factor=factor,
                            min_pixels=min_pixels, max_pixels=max_pixels) == direct:
                return int(cw), int(ch)
        logger.warning_once(
            "nvImageCodec: accelerator resize declined for %dx%d -- no shrink "
            "level leaves the processor target at %s (factor=%d).",
            width, height, direct, factor,
        )
        return None

    return target
