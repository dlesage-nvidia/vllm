# PR #1 C32 throughput with CUDA MPS off and on

This refresh compares the exact pull-request base
`bc8abf31fef015339473f6071eda0de0305dd9b2` (tree
`09423356278c6c4bd871ccda98499474fad78bdd`) with pull-request head
`fc52204ce7e0203456ceca030b90283dde28232a` (tree
`ae2af5c1d60f346efbb8a2375f46663b95835802`). Both endpoints use
PyNvVideoCodec. This is the primary stacked-base comparison that isolates the
three commits in the draft PR.

## Qwen3-VL-2B results

| GPU | MPS | Base req/s, median [min, max] | PR req/s, median [min, max] | Paired PR change, median [min, max] |
|---|---|---:|---:|---:|
| A100 80 GB PCIe | off | 0.8454 [0.8353, 0.8483] | 0.8853 [0.8797, 0.8873] | +4.75% [+3.98%, +6.13%] |
| A100 80 GB PCIe | on | 0.8525 [0.8488, 0.8588] | 1.1933 [1.1930, 1.1940] | +39.98% [+38.98%, +40.66%] |
| RTX PRO 6000 Blackwell Server Edition | off | 0.9467 [0.9250, 0.9662] | 1.7789 [1.7707, 1.7850] | +88.10% [+83.25%, +92.53%] |
| RTX PRO 6000 Blackwell Server Edition | on | 0.9303 [0.8911, 0.9562] | 2.1211 [2.0723, 2.2292] | +129.77% [+123.16%, +138.17%] |

| GPU | Revision | MPS-off req/s, median [min, max] | MPS-on req/s, median [min, max] | Paired MPS change, median [min, max] |
|---|---|---:|---:|---:|
| A100 80 GB PCIe | base | 0.8454 [0.8353, 0.8483] | 0.8525 [0.8488, 0.8588] | +1.10% [+0.48%, +1.92%] |
| A100 80 GB PCIe | PR | 0.8853 [0.8797, 0.8873] | 1.1933 [1.1930, 1.1940] | +34.76% [+34.56%, +35.67%] |
| RTX PRO 6000 Blackwell Server Edition | base | 0.9467 [0.9250, 0.9662] | 0.9303 [0.8911, 0.9562] | -2.32% [-4.69%, +0.76%] |
| RTX PRO 6000 Blackwell Server Edition | PR | 1.7789 [1.7707, 1.7850] | 2.1211 [2.0723, 2.2292] | +19.66% [+16.62%, +24.88%] |

The MPS-off result agrees with the previous C32 result. The prior paired PR
changes were +4.18% on A100 and +85.13% on RTX. The new values are +4.75% and
+88.10%. On RTX, the PR-head median is nearly unchanged (1.7789 versus 1.7764
req/s); the new base median is lower and has a wider range, which raises the
paired percentage.

## Qwen2.5-VL-3B results

| GPU | MPS | Base req/s, median [min, max] | PR req/s, median [min, max] | Paired PR change, median [min, max] |
|---|---|---:|---:|---:|
| A100 80 GB PCIe | off | 0.6334 [0.6311, 0.6347] | 0.6330 [0.6280, 0.6338] | +0.01% [-1.06%, +0.28%] |
| A100 80 GB PCIe | on | 0.9250 [0.9244, 0.9252] | 0.9250 [0.9248, 0.9254] | +0.03% [-0.03%, +0.07%] |
| RTX PRO 6000 Blackwell Server Edition | off | 1.1458 [1.1378, 1.1484] | 1.1484 [1.1446, 1.1507] | +0.22% [-0.05%, +1.11%] |
| RTX PRO 6000 Blackwell Server Edition | on | 1.4842 [1.4832, 1.4849] | 1.4850 [1.4733, 1.4861] | +0.05% [-0.78%, +0.20%] |

| GPU | Revision | MPS-off req/s, median [min, max] | MPS-on req/s, median [min, max] | Paired MPS change, median [min, max] |
|---|---|---:|---:|---:|
| A100 80 GB PCIe | base | 0.6334 [0.6311, 0.6347] | 0.9250 [0.9244, 0.9252] | +46.00% [+45.72%, +46.48%] |
| A100 80 GB PCIe | PR | 0.6330 [0.6280, 0.6338] | 0.9250 [0.9248, 0.9254] | +46.15% [+45.93%, +47.35%] |
| RTX PRO 6000 Blackwell Server Edition | base | 1.1458 [1.1378, 1.1484] | 1.4842 [1.4832, 1.4849] | +29.57% [+29.21%, +30.42%] |
| RTX PRO 6000 Blackwell Server Edition | PR | 1.1484 [1.1446, 1.1507] | 1.4850 [1.4733, 1.4861] | +29.06% [+28.31%, +29.83%] |

The PR-vs-base result is neutral on both GPUs with MPS off and on. That agrees
with the previous C32 conclusion (A100 -0.06%, RTX -0.15% for the earlier
upstream-to-head comparison), while this refresh uses the literal PR base and
therefore isolates this PR. The refreshed A100 MPS-off range is wider because
repetition 1 was -1.06%; its paired median remains +0.01%.

## Workload and execution

- Models:
  - `Qwen/Qwen3-VL-2B-Instruct` at revision
    `89644892e4d85e24eaac8bacfd4f463576704203`
  - `Qwen/Qwen2.5-VL-3B-Instruct` at revision
    `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Input: one real 1920x1080 H.264 clip per request, sampled to 32 frames. Eight
  hard-linked paths prevent media-cache reuse from replacing decode work.
- **vLLM pixel budget: 1024x576, or 589,824 pixels, per sampled frame.**
  Qwen3-VL receives `max_pixels=18,874,368` for the 32-frame video;
  Qwen2.5-VL uses `max_pixels=589,824` with per-frame semantics.
- BF16, TP1, 32 output tokens, two PyNvVideoCodec hardware decoders.
- C32 only: 96 warmup requests followed by 256 measured requests.
- Six paired repetitions in balanced order, with a fresh server for every
  cell.
- Persistent non-streaming HTTP/1.1 connections, exactly 32 prewarmed
  connections, and no retries.
- MPS off explicitly clears MPS variables and uses an empty pipe setting. MPS
  on uses a fresh private daemon per cell, verifies 100% default active-thread
  allocation and a live server/client, then stops and archives that daemon.
- The only authority for machine exclusivity was the `gpulock` lease protocol.

Run the machine-specific launch script on the corresponding prepared host:

```bash
bash scripts/launch_a100.sh
bash scripts/launch_rtx.sh
```

The launch scripts invoke `run_matrix.py`, which contains the exact schedule,
commit checks, server arguments, MPS lifecycle, parity checks, and result
summarization. Each manifest records the full command for all 24 cells.

## Validation and telemetry

- A100 Qwen3-VL: `passed_exact` output parity.
- RTX Qwen3-VL: exact input and prompt-token parity; concurrent completion
  output was not bit-exact (`passed_input_only`).
- RTX Qwen2.5-VL: exact input and prompt-token parity; concurrent completion
  output was not bit-exact (`passed_input_only`).
- A100 Qwen2.5-VL: exact input and prompt-token parity; concurrent completion
  output was not bit-exact (`passed_input_only`).

GPU utilization, memory-controller utilization, memory use, and NVDEC
utilization were polled with a configured 0.2-second interval for every cell.
CPU utilization was not captured. `gpu-telemetry-summary.json` reports the
measured-request window only, including actual sample counts and maximum gaps;
each manifest records hashes for the full raw monitor and sample files.

All 96 benchmark cells passed. All 48 MPS-on cells recorded a live MPS
server/client and 100% default active-thread allocation. The telemetry
summarizer found no measured-window sample errors.

## Environment

| GPU | Driver | Compute capability | CUDA toolkit | Python | PyNvVideoCodec | PyTorch | Transformers |
|---|---|---:|---|---|---|---|---|
| A100 80 GB PCIe | 565.57.01 | 8.0 | 12.9 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |
| RTX PRO 6000 Blackwell Server Edition | 595.91.07 | 12.0 | 13.2 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |

The Transformers source package was frozen at SHA-256
`39591d428561f5a29479229b49643bfe2ebcf433b7b3c086f5064da5fef2f259`
(2,566 files, 47,133,316 bytes).

## Published files

Each GPU/model directory contains:

- `manifest.json`: exact commits, trees, configuration, balanced schedule,
  commands, throughput, MPS lifecycle evidence, and artifact hashes.
- `paired-throughput.csv`: all full-precision paired values.
- `summary.json` and `summary-table.md`: aggregate throughput results.
- `parity-audit.json`: all base/head and MPS-off/on parity comparisons.
- `gpu-telemetry-summary.json`: measured-window GPU and NVDEC statistics for
  every cell and aggregate statistics for each endpoint/MPS mode.

`scripts/` contains the exact matrix runner, model harnesses, monitor, launch
commands, and telemetry summarizer used for this refresh. `SHA256SUMS` covers
all published files.
