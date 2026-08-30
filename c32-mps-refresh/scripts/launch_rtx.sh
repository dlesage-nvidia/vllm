#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

root=/home/ubuntu/work/vllm-pynvvideocodec-pr1-c32-mps-20260830
runner="$root/run_matrix.py"
source_root=/home/ubuntu/work/vllm-pynv-gpu-resize-20260829
python=/home/ubuntu/work/vllm-nvimagecodec-bench/vllm/.venv/bin/python
transformers_root=/home/ubuntu/work/vllm-pynv-e2e-transformers-5.14.1-20260827
traffic=/home/ubuntu/work/pynv-video-gpu-resize-results-20260829/quality/traffic-1080p.mp4
corpus="$root/corpus"
holder=agent-vllm-pr1-c32-mps-rtx

mkdir -p "$root/results-v2" "$root/tmp"
export TMPDIR="$root/tmp"
export PYTHONUNBUFFERED=1

python3 "$runner" \
  --model qwen25 \
  --source-root "$source_root" \
  --python "$python" \
  --transformers-root "$transformers_root" \
  --hf-hub-cache /home/ubuntu/work/hf-cache-qwen25vl-20260829 \
  --traffic-video "$traffic" \
  --corpus "$corpus" \
  --results "$root/results-v2/qwen25" \
  --holder "$holder" \
  --gpu-label RTX \
  --lease-seconds 28800 \
  --renew-seconds 1800

python3 "$runner" \
  --model qwen3 \
  --source-root "$source_root" \
  --python "$python" \
  --transformers-root "$transformers_root" \
  --hf-hub-cache /ephemeral/cache/huggingface/hub \
  --traffic-video "$traffic" \
  --corpus "$corpus" \
  --results "$root/results-v2/qwen3" \
  --holder "$holder" \
  --gpu-label RTX \
  --lease-seconds 28800 \
  --renew-seconds 1800
