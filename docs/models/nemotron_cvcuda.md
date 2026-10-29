<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Experimental Nemotron CV-CUDA vision

This opt-in image path extends [PR #57924](https://github.com/vllm-project/vllm/pull/57924)
at its tested head, `c8cfbb10f6c8d97a99117807cb0f26053e117928`. It is a
**fork-local draft for human review**, not an upstream-ready replacement for that
PR. The default processor and model path remain unchanged.

## Enable and ablate

Set `mm_processor_kwargs={"cvcuda_vision": True}` with device-side multimodal
normalization enabled. The equivalent CLI option is:

```text
--mm-processor-kwargs '{"cvcuda_vision": true}'
```

The image path also enables scalar residual/next-LayerNorm fusion and exact BF16
GELU lookup. Their independent ablation options are:

```text
--mm-processor-kwargs '{"cvcuda_vision": true, "cvcuda_residual_norm": false, "cvcuda_exact_gelu": false}'
--mm-processor-kwargs '{"cvcuda_vision": true, "cvcuda_exact_gelu": false}'
```

The first selects the portable preprocessing/identity/position control; the
second keeps residual/norm fusion but uses native GELU. Leaving `cvcuda_vision`
unset selects unmodified PR-B behavior.

Use eager inference, with multimodal encoder compilation and CUDA Graphs
disabled. First-use JIT builds require a CUDA toolkit and C++ compiler; the wheel
includes the CUDA-C++ sources. JIT compilation, coefficient qualification, weight
checks and GELU table creation must finish during untimed warmup.

The numerical runtime is deliberately pinned to CV-CUDA `0.19.0` and PyTorch git
revision `cf30153c4c131c8164ee7798e5022d810682e2cb` (tested with
`2.13.0+cu130`). Other runtime identities are rejected, not silently substituted.
These pins must be requalified before widening support; this is not yet a
drop-in feature for arbitrary released vLLM installations.

## Numerical contract

The reference is the unmodified PR head, not the earlier relaxed experimental
attention implementation. Preserve its uint8 width-then-height antialiased
bicubic resize, intermediate rounding/clamping, packed patch order, image sizes,
token budget and device affine normalization. Finite BF16 pixels and complete
connectors must match every raw byte, **including signed-zero bits**, without a
tolerance substitute.

Scalar residual/norm preserves BF16 residual rounding and the native width-1280
LayerNorm reduction order. It covers all 63 pairs in the 32-block RADIO encoder.
Exact GELU uses a 128 KiB bit-pattern table produced and checked against native
GELU outside timing, covering all 65,280 finite BF16 codes. Timed GELU is a
CUDA-C++ lookup, not framework-kernel delegation. Native attention and both
native linear layers remain unchanged.

## Included changes

- Owner-retaining Pillow Arrow RGBX views and borrowed CPU multimodal collation
  avoid redundant CPU packing/staging copies, without changing native tiling.
  Fragmented Pillow rows use exact, read-only RGBX serialization instead of an
  unavailable Arrow view. Inputs larger than the qualified 8192-pixel extent
  use native PR preprocessing with a warning, including large server-profile
  dummy images; the resize contract is not widened. Encoder leads remain enabled.
  One-tensor image groups use native multimodal fields for RPC transport while
  retaining zero-copy raw-pixel collation; no serializer registry is modified.
- Grouped uploads use shipping CV-CUDA `convertto`, `tvresize_into` and
  `reformat_into` operators. Separate compatibility kernels retain PR-B uint8
  rounding. **No CV-CUDA operator or library changes are included.**
- Final rounding, patch packing and input affine normalization are fused.
  Scratch retirement is fenced; timed input pixels and outputs are not cached.
- Identity scales are elided only after inspecting loaded weights. Native
  positional interpolation is reused per geometry within one forward.
- Portable scalar residual/next-LayerNorm and exact BF16 GELU are added
  sequentially after qualification of each parent path.

There are no GB300-specific attention/matrix kernels, relaxed attention adapters
or CUDA Graphs. Attention metadata reuse measured as a no-op and is excluded.
Architecture-neutral CUDA code does not imply full qualification on every GPU.

## GB300 validation and performance

The qualified run uses the actual patched PR, not the legacy D harness alone:
dynamic-resolution BF16 RADIO vision, B32 homogeneous 1920x1080 decoded RGB
images, natural and synthetic datasets, and one PyTorch intra-op CPU thread.
Each dataset has six untimed warmup
cycles followed by **96 complete same-process measured cycles**, balanced over
all six B/control/treatment permutations. No measured samples were removed.
Each iteration recomputes preprocessing and connector outputs.

The control includes the portable preprocessing/identity/position changes and
residual/norm fusion. The treatment adds only exact GELU to that control.

| Dataset | PR-B images/s | Norm control images/s | Norm + GELU images/s | Treatment total / encoder ms |
| --- | ---: | ---: | ---: | ---: |
| Natural | 32.3388 | 56.6582 | 58.1659 | 550.1501 / 505.5691 |
| Synthetic | 32.6054 | 56.0635 | 58.1080 | 550.6991 / 505.7499 |

Ratios use paired, position-stratified bootstrap 95% confidence intervals:

| Dataset | Treatment / PR-B FPS ratio | GELU-only incremental FPS ratio |
| --- | ---: | ---: |
| Natural | 1.798645 [1.795868, 1.801499] | 1.026612 [1.020731, 1.034805] |
| Synthetic | 1.782158 [1.778365, 1.786016] | 1.036467 [1.021228, 1.063783] |

The preceding residual/norm lead pooled every pair from complete 36- and
96-cycle trials: **132 pairs per dataset**, +4.31% natural
(95% CI +3.07-5.09%) and +3.68% synthetic (+1.15-5.15%). Do not attribute the
overall 1.78-1.80x gain to GELU or norm alone.

Qualification covers 30 adversarial norm cases, every finite BF16 GELU code,
all 63 real norm/32 attention/32 GELU boundaries, and natural **and** synthetic
B32 pixel/full-connector raw-byte parity against PR-B. Independent CPU replay
compared all **175,472,640 BF16 connector elements per dataset**, with zero
mismatched bytes or signed-zero mismatches.

Peak live Torch allocation was 18,174,239,232 bytes for the norm control and
18,843,525,632 bytes for norm + GELU: +669,286,400 bytes (about 0.62 GiB).
Peak reserved allocation was 32,994,492,416 bytes in the shared three-model
process. This is **not an isolated deployment footprint**, and the allocation
difference must not be attributed entirely to the 128 KiB LUT.

### Frozen source identities

The applied experimental update and CUDA-C++ bundles are identified by SHA256:

| Artifact | SHA256 |
| --- | --- |
| Full qualified experimental patch | `62f4a9a4a4c1957e1b0cfd46e9026af79f3112aea9512dee284cda1c18ee4f2b` |
| Norm-only CUDA-C++ bundle | `2491ee5e27e452e588b7f1906029e2e53871978a8c148b5f3f73937f3ab00a2d` |
| Norm + GELU CUDA-C++ bundle | `c79358406706a531468fde333751e416b2e9fa3b397f78d2f21f90615f53c688` |
| Tested GELU provenance | `8baaa491e4d0cae016ccd81d37cf8e4f52017cb513c10fa936a47018b1cd253f` |

The original patch hash describes the frozen experimental patch, not this review
branch. Repository preparation formats the C++/CUDA source, sorts Python imports,
adds type annotations and uses the equivalent unbounded `functools.cache`
spelling. The original experimental source artifacts remain untouched.

All nine C++/CUDA files retain equal lexical tokens after adjacent-string
normalization and identical preprocessor directives. The Python comparison
permits only those import/cache/type annotation differences. These source checks
are not generated-device-code proof
or a fresh GPU revalidation. The formatted norm + GELU bundle is
`49f125602f909025e614d80cd5e3c2ddf826a0630a4c2ad20d611e596c166868`;
the benchmark results refer to the frozen tested bundle, not a new measurement of
the formatted bytes. Exact tested and publication file hashes are available in
[the source identity manifest](nemotron_cvcuda_sources.json).

## H100 status and remaining gates

H100 standalone norm passed all 30 adversarial cases, and standalone GELU passed
all 65,280 finite BF16 codes using the same source bundles as GB300. An unchanged
H100-native B anchor completed, and portable-baseline B32 in-process pixel and
connector byte parity passed on both datasets.

Full-model H100 benchmarking was stopped after 36 natural and 12 synthetic
baseline cycles when the driver reported software thermal slowdown, 115 C HBM
temperature and 345 MHz SM clocks. That condition was recorded before the
full-model trial. The norm/GELU full-model H100 stages did **not** run, and
independent CPU connector replay for the partial baseline remains outstanding.
There is **no qualified nominal H100 FPS result**. Resume only on healthy,
exclusively leased hardware; do not infer a cooling diagnosis from the readings.

Before upstream submission or promotion, obtain human line-by-line review,
complete H100 full-model qualification, public `LLM`/server entrypoint regression
tests, decoder/model evaluation, upstream CI and isolated-memory review. This
AI-assisted fork-local draft does not certify those gates as passed.

Ragged/heterogeneous batches, videos, graph execution, other models and other
runtime versions are not qualified. Startup setup includes CPU/GPU synchronization;
steady-state `VLLM_GPU_SYNC_CHECK=error` validation remains outstanding.

Measurements cover **decoded-image-to-connector only**: not image decoding, full
tokenization, decoder serving or TTFT. They are not a matched serving comparison
with cross-run original-vLLM-PR numbers.
