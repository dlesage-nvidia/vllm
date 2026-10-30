# Experimental pure CV-CUDA image preprocessing

Nano Nemotron dynamic-resolution images can opt into GPU preprocessing using
shipping CV-CUDA operators, without an application CUDA kernel or CV-CUDA changes:

```bash
vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --trust-remote-code \
  --mm-processor-kwargs '{"pure_cvcuda_preprocessing": true}'
```

The default remains **false**. This option requires CUDA, CV-CUDA 0.19.0,
dynamic-resolution images, `mm_device_do_normalize=True`, and an eager multimodal
encoder (`compile_mm_encoder=False`, `cudagraph_mm_encoder=False`). Decoder settings
are unchanged. CUDA builds install CV-CUDA 0.19.0 automatically, using
`cvcuda-cu12` for CUDA 12 and `cvcuda-cu13` for CUDA 13. Installs using `--no-deps`
must provide the matching package separately. The measured configuration uses
CUDA 13 and BF16. Other configurations require qualification.

The native CPU processor retains geometry, ordering and token budgeting, but sends
raw RGB uint8 images to the model. The GPU path is CV-CUDA uint8-to-FP32 conversion,
FP32 bicubic antialiased resize, FP32 rescale and normalization, and NHWC-to-NCHW
reformat. The native model-dtype cast and patch packing remain. Already-normalized
images bypass native input normalization exactly once; native images and videos
still use their original normalization path. Video, audio, static tiling and
precomputed embeddings are unchanged.

## Numerical behavior

This is **not a byte-exact replacement** for current-main uint8 resizing, nor for
the pre-device-normalization FP32 implementation. It intentionally avoids uint8
intermediate rounding/clipping and retains bicubic overshoot outside [0,255], as
the original FP32 implementation did. Do not clamp the floating-point resize
output to force integer-image semantics. Smaller RMS differences from the pre-B
reference do not prove equal generated answers or task accuracy. No numerical or
generated-output equality gate is relaxed.

Only per-device/per-stream normalization constants are cached. Timed input pixels and outputs are never
cached. Pinned upload buffers and CV-CUDA buffer wrappers are retained until their
CUDA stream completes. Image extents are restricted to 1 through 16384; unsupported
configurations fail explicitly rather than switching numerical paths.

## Validation

With the optional dependency and a compatible model GPU installed, run:

```bash
VLLM_GPU_SYNC_CHECK=error .venv/bin/python -m pytest -v \
  tests/models/multimodal/test_nano_nemotron_vl.py -k pure_cvcuda
```
