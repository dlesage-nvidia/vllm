# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias, cast

from vllm.logger import init_logger

logger = init_logger(__name__)

VIDEO_RESIZE_SENSITIVE_PROCESSOR_KWARGS = frozenset(
    {
        "do_resize",
        "do_sample_frames",
        "fps",
        "max_pixels",
        "max_frames",
        "merge_size",
        "min_pixels",
        "min_frames",
        "num_frames",
        "patch_size",
        "resample",
        "size",
        "temporal_patch_size",
        "video_max_pixels",
        "video_metadata",
        "video_min_pixels",
    }
)

VideoResizeTarget: TypeAlias = Callable[[int, int, int], tuple[int, int] | None]


def request_invalidates_video_resize_target(
    configured: Mapping[str, Any] | None,
    requested: Mapping[str, object] | None,
) -> set[str]:
    """Return request overrides that invalidate the startup resize target."""
    if not requested:
        return set()
    configured = configured or {}
    return {
        name
        for name in VIDEO_RESIZE_SENSITIVE_PROCESSOR_KWARGS & requested.keys()
        if name not in configured or requested[name] != configured[name]
    }


def _resolve_video_processor(
    processor: Any,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Find the HF video processor behind a vLLM processor wrapper."""
    candidates = [processor]
    info = getattr(processor, "info", None)
    if info is not None:
        for getter_name in ("get_video_processor", "get_hf_processor"):
            getter = getattr(info, getter_name, None)
            if not callable(getter):
                continue
            for attempt in (processor_kwargs or {}, {}):
                try:
                    candidates.append(getter(**attempt))
                    break
                except Exception:
                    logger.debug(
                        "PyNvVideoCodec: %s(**%s) failed while resolving the "
                        "video processor.",
                        getter_name,
                        sorted(attempt),
                        exc_info=True,
                    )

    for candidate in candidates:
        if candidate is None:
            continue
        inner = getattr(candidate, "video_processor", None)
        if inner is not None and callable(inner):
            return inner
        if callable(candidate) and type(candidate).__name__.endswith("VideoProcessor"):
            return candidate
    return None


def _resolve_smart_resize(video_processor: Any):
    """Resolve ``smart_resize`` from the processor's own module hierarchy."""
    import sys

    tried = []
    for cls in type(video_processor).__mro__:
        module = sys.modules.get(cls.__module__)
        if module is None:
            continue
        fn = getattr(module, "smart_resize", None)
        if callable(fn):
            return fn, cls.__module__
        tried.append(cls.__module__)
    return None, ", ".join(tried[:4]) or "no modules resolved"


def _size_value(size: Any, *names: str) -> Any:
    for name in names:
        if isinstance(size, Mapping) and name in size:
            return size[name]
        value = getattr(size, name, None)
        if value is not None:
            return value
    return None


def _processor_configuration(
    processor: Any,
    processor_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    configured: dict[str, Any] = {}
    info = getattr(processor, "info", None)
    ctx = getattr(info, "ctx", None)
    get_merged = getattr(ctx, "get_merged_mm_kwargs", None)
    if callable(get_merged):
        try:
            configured.update(get_merged({}))
        except Exception:
            logger.debug(
                "PyNvVideoCodec: could not read merged processor kwargs.",
                exc_info=True,
            )
    configured.update(processor_kwargs or {})
    return configured


def processor_video_resize_target(
    processor: Any,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> VideoResizeTarget | None:
    """Return a safe intermediate GPU-resize target for decoded video frames.

    The returned callable accepts ``(width, height, num_frames)``. It only
    returns a smaller size when the processor's own ``smart_resize`` chooses
    the same final geometry from that intermediate as it does from the source.
    """
    video_processor = _resolve_video_processor(processor, processor_kwargs)
    if video_processor is None:
        logger.warning(
            "PyNvVideoCodec: gpu_resize requires a resolvable video processor; got %s.",
            type(processor).__name__,
        )
        return None

    smart_resize, origin = _resolve_smart_resize(video_processor)
    if smart_resize is None:
        logger.warning(
            "PyNvVideoCodec: no smart_resize found for %s (searched %s).",
            type(video_processor).__name__,
            origin,
        )
        return None

    configured = _processor_configuration(processor, processor_kwargs)
    if (
        configured.get("do_resize", getattr(video_processor, "do_resize", True))
        is False
    ):
        logger.warning(
            "PyNvVideoCodec: gpu_resize is incompatible with do_resize=False."
        )
        return None

    size = getattr(video_processor, "size", None)
    size_override = configured.get("size")
    min_pixels = _size_value(size_override, "shortest_edge", "min_pixels")
    max_pixels = _size_value(size_override, "longest_edge", "max_pixels")
    if min_pixels is None:
        min_pixels = configured.get("min_pixels")
    if max_pixels is None:
        max_pixels = configured.get("max_pixels")
    if min_pixels is None:
        min_pixels = _size_value(size, "shortest_edge", "min_pixels")
    if max_pixels is None:
        max_pixels = _size_value(size, "longest_edge", "max_pixels")

    patch_size = configured.get(
        "patch_size", getattr(video_processor, "patch_size", None)
    )
    merge_size = configured.get(
        "merge_size", getattr(video_processor, "merge_size", None)
    )
    temporal_factor = configured.get(
        "temporal_patch_size",
        getattr(video_processor, "temporal_patch_size", None),
    )
    if type(video_processor).__name__ == "Cosmos3EdgeVideoProcessor":
        # Cosmos3 Edge overrides Qwen3-VL preprocessing and calls the shared
        # smart_resize with temporal_factor=1 for its unmerged video frames.
        temporal_factor = 1

    sizing_values = (min_pixels, max_pixels, patch_size, merge_size, temporal_factor)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in sizing_values
    ):
        logger.warning(
            "PyNvVideoCodec: gpu_resize needs integer video sizing parameters; "
            "got min_pixels=%r, max_pixels=%r, patch_size=%r, merge_size=%r, "
            "temporal_factor=%r from %s.",
            min_pixels,
            max_pixels,
            patch_size,
            merge_size,
            temporal_factor,
            type(video_processor).__name__,
        )
        return None

    min_pixels = cast(int, min_pixels)
    max_pixels = cast(int, max_pixels)
    patch_size = cast(int, patch_size)
    merge_size = cast(int, merge_size)
    temporal_factor = cast(int, temporal_factor)
    factor = patch_size * merge_size
    if min_pixels <= 0 or max_pixels <= 0 or factor <= 0 or temporal_factor <= 0:
        logger.warning(
            "PyNvVideoCodec: gpu_resize video sizing parameters must be positive."
        )
        return None

    logger.info(
        "PyNvVideoCodec: resize target derived from %s via smart_resize in %s "
        "(factor=%d, temporal_factor=%d, budget=%d-%d pixels).",
        type(video_processor).__name__,
        origin,
        factor,
        temporal_factor,
        min_pixels,
        max_pixels,
    )

    import math

    headrooms = (1.0, 1.25, 1.5, 2.0, 3.0)

    def target(width: int, height: int, num_frames: int):
        if width <= 0 or height <= 0 or num_frames < temporal_factor:
            return None

        resize_kwargs = {
            "num_frames": num_frames,
            "factor": factor,
            "temporal_factor": temporal_factor,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }
        direct_h, direct_w = smart_resize(
            height=height,
            width=width,
            **resize_kwargs,
        )
        direct = int(direct_h), int(direct_w)

        for headroom in headrooms:
            if headroom == 1.0:
                candidate_w, candidate_h = direct[1], direct[0]
            else:
                scale = math.sqrt(
                    headroom * max_pixels / float(num_frames * width * height)
                )
                if scale >= 1.0:
                    return None
                candidate_w = max(factor, int(round(width * scale / factor)) * factor)
                candidate_h = max(factor, int(round(height * scale / factor)) * factor)

            if candidate_w * candidate_h >= width * height:
                continue
            repeated = smart_resize(
                height=candidate_h,
                width=candidate_w,
                **resize_kwargs,
            )
            if tuple(map(int, repeated)) == direct:
                return candidate_w, candidate_h

        logger.warning_once(
            "PyNvVideoCodec: gpu_resize declined %dx%dx%d because no smaller "
            "intermediate preserves the processor target %s.",
            width,
            height,
            num_frames,
            direct,
        )
        return None

    return target
