# Qwen3-VL-2B PyNvVideoCodec throughput

The current C32-only base/head refresh, including controlled CUDA MPS-off and
MPS-on runs for Qwen3-VL and Qwen2.5-VL, is in
[`c32-mps-refresh/`](c32-mps-refresh/README.md). The results below are the
earlier C8/C16/C32 matrix.

The upstream-vs-PR Qwen2.5-VL-3B results are in
[`qwen2.5-vl/`](qwen2.5-vl/README.md).

This data compares the exact pull-request base
`bc8abf31fef015339473f6071eda0de0305dd9b2` (tree
`09423356278c6c4bd871ccda98499474fad78bdd`) with benchmark head
`30d917599b104423e452fa718890af01c4ff4d39` (tree
`66c4849eb21973b9ca391b7b0911968f4aa63dac`). Both versions use
PyNvVideoCodec. The submitted branch has the same performance-path files as
the benchmark head.

The base is the fork PR's stacked nvImageCodec branch, not upstream `main`.
This comparison isolates the three PyNvVideoCodec/Qwen3-VL commits in the
draft.

## Results

| GPU | Concurrency | Base req/s, median [min, max] | Head req/s, median [min, max] | Paired change, median [min, max] |
|---|---:|---:|---:|---:|
| A100 80 GB PCIe | 8 | 0.8303 [0.8271, 0.8430] | 0.8787 [0.8753, 0.8823] | +5.97% [+3.84%, +6.68%] |
| A100 80 GB PCIe | 16 | 0.8408 [0.8347, 0.8476] | 0.8813 [0.8778, 0.8862] | +5.17% [+3.94%, +5.53%] |
| A100 80 GB PCIe | 32 | 0.8474 [0.8432, 0.8561] | 0.8832 [0.8808, 0.8849] | +4.18% [+3.17%, +4.91%] |
| RTX PRO 6000 Blackwell Server Edition | 8 | 0.9584 [0.9479, 0.9671] | 1.7473 [1.7259, 1.7637] | +82.08% [+80.26%, +85.16%] |
| RTX PRO 6000 Blackwell Server Edition | 16 | 0.9571 [0.9438, 0.9643] | 1.7644 [1.7536, 1.7709] | +84.43% [+82.63%, +87.63%] |
| RTX PRO 6000 Blackwell Server Edition | 32 | 0.9596 [0.9541, 0.9676] | 1.7764 [1.7682, 1.7871] | +85.13% [+82.73%, +87.07%] |

Each interval is the range across six paired repetitions.

## Workload

- Model: `Qwen/Qwen3-VL-2B-Instruct`, revision
  `89644892e4d85e24eaac8bacfd4f463576704203`
- Input: real 1920x1080 H.264 video, sampled to 32 frames
- **vLLM pixel budget: 1024x576 per sampled frame** (`589,824` pixels), or
  `max_pixels=18,874,368` for 32 frames
- Prompt: `Describe this video concisely and factually.`
- BF16, tensor parallelism 1, output length 32
- Two PyNvVideoCodec hardware decoders
- Concurrency 8, 16, and 32
- Warmup/measured requests: 24/64, 48/128, and 96/256 respectively
- Persistent non-streaming HTTP/1.1 connections, no retries
- Fresh server for each endpoint; six paired repetitions with balanced order

The base used THWC and host normalization. The head used explicit TCHW and
vLLM device normalization. The head also copied decoded frames directly into
pinned host memory.

## Server and request

Run the command once per variant. Set the two shell variables to one of these
pairs first:

- Base: `MEDIA_IO_KWARGS='{"video":{"backend":"pynvvideocodec","hw_decoders":2,"max_frames":32,"min_frames":32,"video_backend":"qwen3_vl"}}'`
  and `NORMALIZE_OPTION=--no-mm-device-do-normalize`.
- Head: `MEDIA_IO_KWARGS='{"video":{"backend":"pynvvideocodec","hw_decoders":2,"max_frames":32,"min_frames":32,"output_layout":"tchw","video_backend":"qwen3_vl"}}'`
  and `NORMALIZE_OPTION=--mm-device-do-normalize`.

```bash
.venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --served-model-name qwen3-vl-video-throughput \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --seed 0 --tensor-parallel-size 1 \
  --max-model-len 32768 --max-num-batched-tokens 9216 --max-num-seqs 32 \
  --api-server-count 1 \
  --limit-mm-per-prompt '{"image":0,"video":1}' \
  --allowed-local-media-path <video-dir> \
  --media-io-kwargs "$MEDIA_IO_KWARGS" \
  --mm-processor-kwargs '{"max_pixels":18874368}' \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching \
  --mm-ipc-gpu-memory-gb 2 --kv-cache-memory-bytes 42949672960 \
  "$NORMALIZE_OPTION"
```

Each request was an OpenAI-compatible `POST /v1/chat/completions` with this
body, substituting one of eight hard-linked video paths for `<video-file>`:

```json
{
  "model": "qwen3-vl-video-throughput",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "video_url", "video_url": {"url": "file://<video-file>"}},
      {"type": "text", "text": "Describe this video concisely and factually."}
    ]
  }],
  "max_completion_tokens": 32,
  "ignore_eos": true,
  "n": 1,
  "seed": 0,
  "temperature": 0.0,
  "top_p": 1.0,
  "return_token_ids": true
}
```

## Output comparison

The RTX timing run completed without failed requests. Prompt token IDs and
request settings matched. Completion output differed in 1,087 of 3,696 paired
responses. In every mismatch, only the first character changed from uppercase
to lowercase (`A woman...` to `a woman...`), token ID 32 to 64 at position
zero. There were no other text differences. The timing run therefore does not
establish exact completion parity.

The frozen RTX timing audit predates this follow-up and retains its original
`timing_valid_completion_mismatch_2b_unresolved` status. The separate case
audit and 2B model evaluation below add the later evidence without rewriting
the timing audit.

The A100 run had exact output parity across all 3,696 paired responses.

## Model evaluation

A separate Qwen3-VL-2B test passed vLLM's standard 16-frame ndarray-video
top-20-logprob comparison with Hugging Face:

```bash
git apply <evidence-dir>/qwen3-vl-2b-eval.patch
QWEN3_VL_2B_SNAPSHOT=<snapshot-dir>/89644892e4d85e24eaac8bacfd4f463576704203 \
  .venv/bin/python -m pytest -vv -s --tb=short \
  tests/models/multimodal/generation/test_qwen3_vl_2b_eval.py::test_qwen3_vl_2b_video_hf_top20_logprobs
git apply -R <evidence-dir>/qwen3-vl-2b-eval.patch
```

Result: **1 passed in 42.44 seconds**. The test pinned model revision
`89644892e4d85e24eaac8bacfd4f463576704203`. It validates Qwen3-VL processing
and model behavior, not PyNvVideoCodec decoding. The tracked Python source was
at submitted head `fc52204ce7e0203456ceca030b90283dde28232a`; existing
precompiled vLLM native extensions were reused. The evaluation environment used
PyTorch `2.13.0+cu129` and Transformers `5.15.1`.
The published replay patch parameterizes the snapshot path; the executed
selector used the same snapshot path as a literal.
The complete pytest log has SHA-256
`e8eabfa7b5f884bc6768b4c07a54312f0dc67809b36c20fbba9925acdf0746c2`.

## Environment

| GPU | Driver | Compute capability | Python | PyNvVideoCodec | PyTorch | Transformers |
|---|---|---:|---|---|---|---|
| A100 80 GB PCIe | 565.57.01 | 8.0 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |
| RTX PRO 6000 Blackwell Server Edition | 595.91.07 | 12.0 | 3.12.14 | 2.0.4 | 2.13.0+cu129 | 5.14.1 |

## Files

- `paired-throughput.csv`: 36 paired comparisons at full precision (18 per GPU)
- `a100-audit-v5.json`: sanitized A100 audit
- `rtx-pro-6000-audit-v5.json`: sanitized RTX audit
- `qwen3-vl-2b-eval.patch`: portable replay selector for the 2B model evaluation
- `qwen3-vl-2b-eval-result.txt`: evaluation exit status and timestamps
- `rtx-completion-case-audit.json`: classification of the RTX text mismatches

SHA-256:

- `a100-audit-v5.json`: `cb49688cf024464cfd5004f24e5cde53eeb4bcf84bd615ea13c760e7a7dd2279`
- `rtx-pro-6000-audit-v5.json`: `8f7e0a66126122c3ee96828031f7e668329ac030c73ded2ab5673680e97d7523`
- `paired-throughput.csv`: `bbbdcdd7a950388e6f90fc4b1a1600d1baed242ca1cc9d67fdaf39a812feb352`
- `qwen3-vl-2b-eval.patch`: `523440f8ab3e6a13420e2a22ddad74520c67b424998c9800ed48df75b101b875`
- `qwen3-vl-2b-eval-result.txt`: `4b0f37f33d4bfa9fc5cbe27d23907d6254c58a26fa2b4dabd00fbb6875809333`
- `rtx-completion-case-audit.json`: `acbcf80bc0ea2b6c79a0742b3ff33a8f6466450673d2ac55713977b0bf1a801a`
