# PyNvVideoCodec early GPU resize evidence

This evidence compares the literal head of personal-fork PR #1,
`fc52204ce7e0203456ceca030b90283dde28232a` (tree
`ae2af5c1d60f346efbb8a2375f46663b95835802`), with the stacked GPU-resize
candidate `3a64f5325f8e27581461c983902e23c52d906989` (tree
`a4ff4b35a27b1a59fb42628acf8db403cd1ec44f`). The candidate has exactly one
commit on top of PR #1.

## Publication workload

- Model: `Qwen/Qwen3-VL-2B-Instruct`, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`
- Input: one real 1920x1080 H.264 clip per request, sampled to 32 frames; the
  eight workload paths are hard links to the same clip
- Processor budget: maximum 1024x576 (589,824 pixels) per sampled frame, or
  `max_pixels=18,874,368` for 32 frames
- BF16, TP1, output length 32, two retained PyNvVideoCodec decoders
- Request concurrency C8/C16/C32; warmup/measured requests 24/64, 48/128, and
  96/256 respectively
- Six paired repetitions with balanced endpoint and concurrency order, a fresh
  server per endpoint, persistent non-streaming HTTP/1.1 connections, exactly
  C prewarmed connections, and no retries
- The only treatment option is `gpu_resize: true`
- Every GPU run acquired `/usr/local/bin/gpulock` for the whole machine before
  executing GPU work

## End-to-end throughput

<!-- markdownlint-disable MD013 MD060 -->

RTX PRO 6000 Blackwell Server Edition:

| Concurrency | PR #1 req/s, median [min, max] | GPU resize req/s, median [min, max] | Paired change, median [min, max] |
|---:|---:|---:|---:|
| 8 | 1.7710 [1.7630, 1.7787] | 1.9069 [1.9035, 1.9143] | +7.84% [+7.20%, +8.42%] |
| 16 | 1.7813 [1.7719, 1.7963] | 1.9225 [1.9159, 1.9239] | +7.91% [+6.70%, +8.56%] |
| 32 | 1.7864 [1.7841, 1.7970] | 1.9210 [1.9151, 1.9271] | +7.67% [+6.65%, +7.91%] |

A100 80GB PCIe:

| Concurrency | PR #1 req/s, median [min, max] | GPU resize req/s, median [min, max] | Paired change, median [min, max] |
|---:|---:|---:|---:|
| 8 | 0.8807 [0.8756, 0.8816] | 0.8899 [0.8790, 0.8944] | +1.00% [+0.38%, +1.64%] |
| 16 | 0.8820 [0.8786, 0.8867] | 0.8785 [0.8728, 0.8854] | -0.43% [-0.67%, -0.15%] |
| 32 | 0.8845 [0.8833, 0.8861] | 0.8756 [0.8696, 0.8781] | -0.98% [-1.71%, -0.83%] |

<!-- markdownlint-enable MD013 MD060 -->

The RTX improvement does not generalize uniformly to A100: the A100 result is
a small gain at C8 and a small, consistent regression at C16/C32. This is why
the feature remains opt-in rather than becoming the default path.

Every treatment cell on both GPUs reported exactly 19,712 `gpu_resized` and
19,712 `resize_cvcuda` frames. In the A100 timing matrix, all 3,696 paired
responses had identical configuration, request identity, and prompt-token IDs.
Completion tokens and text differed for all pairs, so this matrix establishes
input parity and timing, not output equivalence; deterministic output quality
is covered separately below.

The first A100 attempt filled the remote filesystem while writing the eighth
cell's result. Its incomplete JSON was discarded and its logs were retained.
The seven completed cells were losslessly compressed, then re-read,
payload-hash checked, and fully revalidated before the runner resumed the five
missing cells. No completed cell was rerun or silently accepted from its
manifest alone. `a100/manifest.json` records the recovery history and both
runner hashes.

## Output and image-quality checks

- The real-CUDA THWC/TCHW suite passed four tests on each GPU. Two cases forced
  CV-CUDA HQResize and bounded mean absolute pixel difference below 2 versus
  PIL; two forced the fp16 Torch bicubic fallback and checked layout, dtype,
  and observability counters.
- A deterministic RTX model-level audit used a Pexels cat clip and a traffic
  clip, 32 frames, output length 64. All warmup and measured completion-token
  sequences matched exactly between PR #1 and GPU resize. Measured C1
  throughput was 0.7718 versus 0.9354 requests/s.
- The concurrent publication matrix is a timing test, not an exact-output
  test. Input configuration, request identity, and prompt-token IDs must match;
  completion differences are classified separately. Same-commit control
  comparisons quantify the workload's inherent generation nondeterminism.

## Transfer and memory behavior

For the 1080p publication workload, sampled-raster device-to-host traffic falls
from 189.8 MiB to 54.0 MiB per request: 3.52x less, a 71.6% reduction.

The supplementary RTX 4K run reduces the transfer from 759.4 MiB to 54.0 MiB
per request: 14.06x less, a 92.9% reduction. A 50 ms per-process GPU-memory
sample at matched 4K C1 measured frontend API-server peaks of 1,594 MiB for
PR #1 and 1,618 MiB with GPU resize, a 24 MiB increase. The implementation
conservatively leases twice the raw sampled-frame bytes because decoded inputs
and resized outputs may overlap.

The A100 publication runs' whole-device monitor measured a 45,376 MiB peak for
PR #1 and 45,880 MiB for GPU resize in every repetition, a 504 MiB increase.
This includes the model, server processes, decoder, resize operator, and any
allocator or scratch state, and is distinct from the targeted RTX
frontend-process measurement above.

## Local validation

The exact candidate source passed 227 focused tests with four expected skips
and passed pre-commit on all 13 changed files. See `local-validation.txt` for
the command, skip reasons, and hook scope.

## Files

- `rtx/summary.json`: all six RTX pairs and aggregate intervals
- `rtx/summary-table.md`: rendered RTX table
- `rtx/manifest.json`: commands, commits, trees, environment, per-cell metrics,
  counters, hashes, and lock provenance
- `rtx/token-parity.json`: paired request/output audit
- `rtx/repeatability-audit.json`: same-commit versus cross-variant completion
  comparison
- `rtx/gpu-tests.log`: real-CUDA focused test output
- `a100/`: corresponding A100 artifacts
- `quality/*.json.gz`: losslessly compressed raw deterministic-quality, 4K
  smoke, and memory-run results
- `quality/*.csv`: per-process memory samples
- `support/`: frozen benchmark runner, harness, GPU monitor, and lock wrappers
- `SHA256SUMS`: hashes for every other file in this evidence tree

All paths and timestamps inside the artifacts are provenance records from the
benchmark machines. The evidence commit itself makes the published contents
immutable.
