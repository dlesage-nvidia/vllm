# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

# CV-CUDA's bicubic resize needs a scratch buffer that grows with the frame
# count, source height and target width. CV-CUDA keeps every scratch buffer it
# allocates until the process exits, and reuses one only for requests no larger
# than it. So each resize is chunked to need at most this much, and the first
# one reserves exactly this much: one buffer, allocated in the startup memory
# profile, serves every resize.
_CVCUDA_SCRATCH_BYTES = 512 * 1024 * 1024
_cvcuda_scratch_reserved: set[int] = set()

# Frames in host memory are uploaded on a stream of their own. A copy from
# pageable memory first waits for its stream, so on the model's stream it would
# hold the engine thread until the GPU finishes every step already queued.
# Not aux_stream(): model layers queue their own work on that one.
_upload_streams: dict[int, torch.cuda.Stream] = {}  # device index -> stream
_default_resize_streams: dict[int, torch.cuda.Stream] = {}


def _upload_stream(device_index: int) -> torch.cuda.Stream:
    if device_index not in _upload_streams:
        _upload_streams[device_index] = torch.cuda.Stream(device_index)
    return _upload_streams[device_index]


class _CudaArrayOnStream:
    """CUDA Array Interface v3 view of a torch tensor written on *stream*.

    CV-CUDA orders its work after *stream*. Given a torch tensor directly
    (interface v2, which names no stream) it calls cudaDeviceSynchronize
    instead. The view keeps no reference to the tensor, so CV-CUDA's object
    cache holds no torch memory: torch reuses it in stream order, after the
    queued CV-CUDA work.
    """

    def __init__(self, tensor: torch.Tensor, stream: int) -> None:
        self.__cuda_array_interface__ = {
            **tensor.__cuda_array_interface__,
            "version": 3,
            "stream": stream,
        }


def cvcuda_cpu_uint8_scratch_bytes(
    height: int, width: int, target_height: int, target_width: int, num_frames: int = 1
) -> int:
    """Conservative aligned coefficient and uint8 intermediate workspace bound."""
    if min(height, width, target_height, target_width) <= 0 or num_frames < 0:
        raise ValueError("Resize extents must be positive and frame count nonnegative")
    if num_frames == 0 or (height, width) == (target_height, target_width):
        return 0
    cursor = 0
    for source, target in ((width, target_width), (height, target_height)):
        if source == target:
            continue
        support = (2 * source + target - 1) // target if source > target else 2
        coefficients = target * (2 * support + 1)
        for count, scalar_bytes in (
            (target, 8),
            (target, 8),
            (coefficients, 8),
            (coefficients, 2),
            (target, 8),
            (1, 4),
        ):
            cursor = (cursor + 7) // 8 * 8 + count * scalar_bytes
    if height != target_height and width != target_width:
        cursor = (cursor + 7) // 8 * 8 + num_frames * height * target_width * 3
    return (cursor + 7) // 8 * 8


def _cpu_uint8_reservation_shape() -> tuple[int, int]:
    lower, upper = 16, _CVCUDA_SCRATCH_BYTES // (3 * 4096)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if (
            cvcuda_cpu_uint8_scratch_bytes(middle, 16, 16, 4096)
            <= _CVCUDA_SCRATCH_BYTES
        ):
            lower = middle
        else:
            upper = middle - 1
    return lower, cvcuda_cpu_uint8_scratch_bytes(lower, 16, 16, 4096)


def cvcuda_bicubic_resize(
    frames: torch.Tensor,
    size: tuple[int, int],
    device: torch.device,
    max_chunk_bytes: int = 256 * 1024 * 1024,
    quantize_intermediate: bool = False,
) -> torch.Tensor:
    """GPU counterpart of `_bicubic_resize_and_normalize` with
    `do_cpu_normalize=False`.

    Resizes uint8 NHWC RGB *frames* to *size* `(H, W)` on *device* with
    CV-CUDA and returns uint8 NCHW pixels; the cast to float, rescaling and
    normalization stay with the model's input norm. Frames are uploaded in
    chunks of at most *max_chunk_bytes*, from host memory on a separate stream,
    so the caller waits for the copy but not for work queued on the current
    stream. The resize is queued on the current stream, after the upload.

    With *quantize_intermediate*, use public v0.19 FP32 TvResize separately
    for each axis, followed by selective exact integer GPU correction. This
    preserves TorchVision CPU uint8 coefficients and per-axis rounding/clipping
    without requiring a private CV-CUDA operator mode. Tensor temporaries use a
    separate 384 MiB budget; `max_chunk_bytes` retains its soft frame-chunk limit.
    """
    if quantize_intermediate:
        from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
            cvcuda_bicubic_resize as resize_uint8,
        )

        if (
            frames.ndim != 4
            or frames.dtype != torch.uint8
            or min(frames.shape[:3]) < 1
            or frames.shape[-1] != 3
            or max_chunk_bytes < 1
        ):
            raise ValueError(
                "TorchVision CPU uint8 resize requires nonempty RGB frames"
            )
        frame_bytes = frames.shape[1] * frames.shape[2] * 3
        return resize_uint8(
            frames,
            size,
            device,
            max_temporary_bytes=384 * 1024 * 1024,
            preferred_frame_step=max(1, max_chunk_bytes // frame_bytes),
        )

    import cvcuda

    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    torch_stream = torch.cuda.current_stream(device)
    if torch_stream.cuda_stream == 0:
        device_index = torch_stream.device_index
        if device_index not in _default_resize_streams:
            _default_resize_streams[device_index] = torch.cuda.Stream(device_index)
        resize_stream = _default_resize_streams[device_index]
        resize_stream.wait_stream(torch_stream)
        if frames.is_cuda:
            frames.record_stream(resize_stream)
        with torch.cuda.stream(resize_stream):
            output = cvcuda_bicubic_resize(
                frames,
                size,
                device,
                max_chunk_bytes=max_chunk_bytes,
                quantize_intermediate=quantize_intermediate,
            )
        torch_stream.wait_stream(resize_stream)
        output.record_stream(torch_stream)
        return output
    stream = cvcuda.as_stream(torch_stream)
    # CUDA Array Interface v3 numbers the legacy default stream 1.
    stream_id = torch_stream.cuda_stream or 1

    def tvresize(dst: torch.Tensor, src: torch.Tensor) -> None:
        # The destination's shape sets the output size.
        cvcuda.tvresize_into(
            cvcuda.as_tensor(_CudaArrayOnStream(dst, stream_id), "NHWC"),
            cvcuda.as_tensor(_CudaArrayOnStream(src, stream_id), "NHWC"),
            interp=cvcuda.Interp.CUBIC,
            antialias=True,
            stream=stream,
        )

    reservation_key = torch_stream.device_index
    if reservation_key not in _cvcuda_scratch_reserved:
        # One frame from 16 columns to 4096 needs exactly the cap.
        rows = -(-_CVCUDA_SCRATCH_BYTES // (12 * 4096))
        tvresize(
            torch.empty((1, 16, 4096, 3), dtype=torch.uint8, device=device),
            torch.zeros((1, rows, 16, 3), dtype=torch.uint8, device=device),
        )
        _cvcuda_scratch_reserved.add(reservation_key)

    num_frames, height, width, _ = frames.shape
    out = torch.empty((num_frames, 3, *size), dtype=torch.uint8, device=device)
    upload = (
        _upload_stream(torch_stream.device_index)
        if frames.device.type == "cpu"
        else None
    )
    resize = (height, width) != tuple(size)
    step = max(1, max_chunk_bytes // (height * width * 3))
    if resize:
        # CV-CUDA 0.19's scratch: 12 bytes per frame, source row and target
        # column, plus 136 bytes per target row and column. Only frames of
        # extreme aspect ratio need more than the cap on their own.
        frame_scratch = 12 * height * size[1]
        filter_scratch = 136 * (size[0] + size[1])
        scratch_limit = _CVCUDA_SCRATCH_BYTES
        if filter_scratch + frame_scratch > scratch_limit:
            raise ValueError("A source frame exceeds the CV-CUDA scratch memory budget")
        step = (
            max(1, min(step, (scratch_limit - filter_scratch) // frame_scratch))
            if frame_scratch
            else step
        )
    for start in range(0, num_frames, step):
        chunk = frames[start : start + step]
        if upload is None:
            src = chunk.to(device, non_blocking=True)
        else:
            # Pinning the frames first would let the copy run asynchronously,
            # but its host copy costs more on the engine's single torch thread.
            with torch.cuda.stream(upload), gpu_sync_allowed():
                src = chunk.to(device, non_blocking=True)
            torch_stream.wait_stream(upload)
            # src was allocated on the upload stream: keep its memory until the
            # work queued on the current stream has read it.
            src.record_stream(torch_stream)
        if resize:
            resized = src.new_empty((src.shape[0], *size, 3))
            tvresize(resized, src)
            src = resized
        cvcuda.reformat_into(
            cvcuda.as_tensor(
                _CudaArrayOnStream(out[start : start + step], stream_id), "NCHW"
            ),
            cvcuda.as_tensor(_CudaArrayOnStream(src, stream_id), "NHWC"),
            stream=stream,
        )
    return out
