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
