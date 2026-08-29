# Qwen2.5-VL-3B PyNvVideoCodec throughput

This compares upstream vLLM commit
`d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302` (tree
`9cc26997991af6f8f38150c9631d482d18b1bd2c`) with pull-request head
`fc52204ce7e0203456ceca030b90283dde28232a` (tree
`ae2af5c1d60f346efbb8a2375f46663b95835802`). Both endpoints use
PyNvVideoCodec, THWC output, and fused device-side normalization.

The PR is stacked on earlier image/media changes, so this is a cumulative
upstream-to-head comparison. It does not isolate the direct pinned-host-copy
commit.

## Results

| GPU | Concurrency | Upstream req/s, median [min, max] | PR req/s, median [min, max] | Paired PR-vs-upstream change, median [min, max] |
|---|---:|---:|---:|---:|
| A100 80 GB PCIe | 8 | 0.6290 [0.6270, 0.6339] | 0.6304 [0.6258, 0.6324] | -0.19% [-0.59%, +0.34%] |
| A100 80 GB PCIe | 16 | 0.6295 [0.6262, 0.6339] | 0.6316 [0.6270, 0.6341] | +0.05% [-0.36%, +0.79%] |
| A100 80 GB PCIe | 32 | 0.6321 [0.6278, 0.6357] | 0.6329 [0.6271, 0.6337] | -0.06% [-0.44%, +0.76%] |
| RTX PRO 6000 Blackwell Server Edition | 8 | 1.1490 [1.1320, 1.1564] | 1.1441 [1.1347, 1.1485] | -0.67% [-1.84%, +1.45%] |
| RTX PRO 6000 Blackwell Server Edition | 16 | 1.1469 [1.1420, 1.1512] | 1.1460 [1.1426, 1.1533] | +0.08% [-0.51%, +0.50%] |
| RTX PRO 6000 Blackwell Server Edition | 32 | 1.1464 [1.1406, 1.1537] | 1.1474 [1.1419, 1.1499] | -0.15% [-0.59%, +0.69%] |

Each interval is the range across six paired repetitions. The paired change is
computed within each repetition before taking the median.

There is no material end-to-end throughput change. Every paired range crosses
zero, and the median changes are between -0.67% and +0.08%. This is consistent
with Qwen2.5-VL already using fused device-side normalization: the PR leaves
that path unchanged. No direct-copy benefit is measurable in this cumulative
upstream-to-head comparison.

## Workload

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, revision
  `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Input: one real 1920x1080 H.264 clip per request, sampled to 32 frames; eight
  hard-linked paths prevent media-cache reuse from standing in for decode work
- **vLLM pixel budget: default 1024x576 (589,824 pixels) per sampled frame**
- Prompt: `Describe this video concisely and factually.`
- 11,551 prompt tokens, including 11,520 video embedding tokens
- BF16, tensor parallelism 1, output length 32
- `--max-model-len 32768`, `--max-num-batched-tokens 12288`,
  `--max-num-seqs 32`
- Two PyNvVideoCodec hardware decoders
- Concurrency 8, 16, and 32
- Warmup/measured requests: 24/64, 48/128, and 96/256 respectively
- Persistent non-streaming HTTP/1.1 connections, exactly C prewarmed
  connections, no retries
- Fresh server for each endpoint; six paired repetitions with balanced endpoint
  and concurrency order

Qwen2.5-VL interprets `max_pixels` per frame. The 12,288 batched-token limit is
higher than the Qwen3-VL benchmark's 9,216 because this workload contains
11,520 video embedding tokens.

## Execution

The full matrix command on each machine was:

```bash
/usr/local/bin/gpulock acquire agent-pynvv-qwen25vl-bench 21600 \
  'Qwen2.5-VL upstream/head C8/C16/C32 paired benchmark'

./scripts/run_locked.sh \
  <python> ./scripts/run_qwen25vl_matrix.py \
  --source-root <clean-vllm-checkout-containing-both-commits> \
  --python <python> \
  --transformers-root <transformers-5.14.1-source> \
  --hf-hub-cache <huggingface-cache> \
  --corpus <video-directory> \
  --harness ./scripts/benchmark_qwen25vl_e2e_persistent.py \
  --results <new-result-directory>
```

`run_locked.sh` renews the lease and releases it on exit. The matrix driver
checks out the exact commit for every endpoint, verifies a clean tree, launches
a fresh vLLM server, and records the complete server command and provenance in
each result.

The server configuration common to both endpoints was:

```bash
python -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --served-model-name qwen2.5-vl-video-throughput \
  --host 127.0.0.1 --port 18600 \
  --dtype bfloat16 --seed 0 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-batched-tokens 12288 --max-num-seqs 32 \
  --api-server-count 1 \
  --limit-mm-per-prompt '{"image":0,"video":1}' \
  --allowed-local-media-path <video-directory> \
  --media-io-kwargs \
    '{"video":{"backend":"pynvvideocodec","hw_decoders":2,"max_frames":32,"min_frames":32,"video_backend":"qwen2_vl"}}' \
  --mm-processor-kwargs '{"max_pixels":589824}' \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching \
  --mm-ipc-gpu-memory-gb 2 --kv-cache-memory-bytes 42949672960 \
  --mm-device-do-normalize
```

## Output validation

All 14,784 timed and warmup requests completed successfully. Every base/head
pair had the same prompt token IDs, request settings, 32-token completion
length, finish reason, stop reason, video index, and request payload.

Concurrent completion text was not bit-exact: 4,598 of 7,392 base/head pairs
differed. The same caption variants also changed between repetitions of the
same commit. The output-variability audit compares all 15 repetition pairs at
each request coordinate. Base/head mismatch rates are comparable to
same-variant cross-repetition rates on both GPUs. The RTX mode-frequency
comparison has 4.44% total-variation distance, so a modest distribution shift
cannot be excluded. The timing result is valid, but the throughput matrix is
not an exact-output-parity test.

## Environment

| GPU | Driver | Compute capability | Python | PyNvVideoCodec | PyTorch | Transformers |
|---|---|---:|---|---|---|---|
| A100 80 GB PCIe | 565.57.01 | 8.0 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |
| RTX PRO 6000 Blackwell Server Edition | 595.91.07 | 12.0 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |

## Files

- `SHA256SUMS`: hashes for every published result and script below
- `combined-summary.json`: validated full-precision results and output counts
- `combined-summary.txt`: human-readable rendering of the same summary
- `paired-throughput.csv`: all 36 paired throughput comparisons
- `output-variability-audit.json`: base/head and same-variant output analysis
- `scripts/benchmark_qwen25vl_e2e_persistent.py`: persistent HTTP benchmark
  client and server lifecycle driver
- `scripts/run_qwen25vl_matrix.py`: frozen six-repetition matrix
- `scripts/summarize_qwen25vl_matrix.py`: result validator and summarizer
- `scripts/run_locked.sh`: GPU lease renewal and release wrapper

The raw result archives are not committed because they contain machine-local
provenance. Their SHA-256 values are:

- A100: `52a18828e9c585ecdbde2ecae47c85961feaadd91caeff070d9c8b3c7518dff6`
- RTX PRO 6000: `6ea86a6c6ec1e982a61d900e0c7a905a6d214f2c7f14e008a98b081770af82b5`
