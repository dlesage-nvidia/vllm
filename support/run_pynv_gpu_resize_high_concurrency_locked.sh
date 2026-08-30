#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

set -euo pipefail

gpu_label=${1:?usage: $0 GPU_LABEL RUN_NAME}
run_name=${2:?usage: $0 GPU_LABEL RUN_NAME}

lock_holder=agent-pynv-video-gpu-resize
lock_tool=/usr/local/bin/gpulock
lease_seconds=14400
source_root=/home/ubuntu/work/vllm-pynv-gpu-resize-20260829
python_bin=/home/ubuntu/work/vllm-nvimagecodec-bench/vllm/.venv/bin/python
support_root=/home/ubuntu/work/pynv_video_base_head_publication_freeze_v11
runner=/home/ubuntu/work/run_pynv_gpu_resize_high_concurrency_matrix.py
transformers_root=/home/ubuntu/work/vllm-pynv-e2e-transformers-5.14.1-20260827
artifact_root=/home/ubuntu/work/pynv-video-gpu-resize-results-20260829
traffic_video=$artifact_root/quality/traffic-1080p.mp4
corpus=$artifact_root/pr1-traffic-corpus
results=$artifact_root/$run_name
test_log=$artifact_root/.$run_name-gpu-tests.log
runner_log=$artifact_root/.$run_name-runner.log
python_cache=$artifact_root/.python-cache-$run_name
bytecode_quarantine=$artifact_root/.source-bytecode-quarantine-$run_name
expected_head=3a64f5325f8e27581461c983902e23c52d906989
runner_sha256=b2fc03fd3451b8ef33a6b11fde8d8745cf262db2524c951c74c90f44542c5f17

renewer_pid=
workload_pid=

cleanup_workload() {
    set +e
    if [[ -n "$workload_pid" ]]; then
        kill -INT -- "-$workload_pid" 2>/dev/null
        for _ in {1..60}; do
            kill -0 "$workload_pid" 2>/dev/null || break
            sleep 1
        done
        kill -TERM -- "-$workload_pid" 2>/dev/null
        wait "$workload_pid" 2>/dev/null
        workload_pid=
    fi
    pkill -TERM -f \
        'benchmark_pynvvideocodec_e2e_persistent.py.*--port 18700' \
        2>/dev/null
    pkill -TERM -f \
        'vllm.entrypoints.cli.main serve .*--port 18700' \
        2>/dev/null
}

cleanup() {
    set +e
    cleanup_workload
    if [[ -n "$renewer_pid" ]]; then
        kill "$renewer_pid" 2>/dev/null
        wait "$renewer_pid" 2>/dev/null
        renewer_pid=
    fi
    if [[ -e "$source_root/.git" ]]; then
        git -C "$source_root" -c advice.detachedHead=false checkout \
            --detach "$expected_head" >/dev/null 2>&1
    fi
    "$lock_tool" release "$lock_holder"
}

on_signal() {
    cleanup_workload
    exit 130
}

until "$lock_tool" acquire "$lock_holder" "$lease_seconds" \
    "PR1-vs-GPU-resize C8/C16/C32 six-pair matrix on $gpu_label"; do
    sleep 60
done
trap cleanup EXIT
trap on_signal INT TERM HUP

main_pid=$$
(
    while sleep 3600; do
        if ! "$lock_tool" renew "$lock_holder" "$lease_seconds"; then
            kill -TERM "$main_pid"
            exit 1
        fi
    done
) &
renewer_pid=$!

[[ -x "$python_bin" ]]
[[ -r "$traffic_video" ]]
[[ -d "$transformers_root" ]]
[[ -r "$support_root/benchmark_pynvvideocodec_e2e_persistent.py" ]]
[[ -r "$support_root/run_with_gpu_monitor_refined.py" ]]
[[ "$(sha256sum "$runner" | cut -d' ' -f1)" == "$runner_sha256" ]]
[[ ! -e "$results" ]]
[[ ! -e "$test_log" ]]
[[ ! -e "$runner_log" ]]
[[ ! -e "$python_cache" ]]
[[ ! -e "$bytecode_quarantine" ]]
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$expected_head" ]]
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=all)" ]]

mapfile -d '' cache_dirs < <(
    find "$source_root/vllm" -type d -name __pycache__ -print0
)
if ((${#cache_dirs[@]})); then
    mkdir -p "$bytecode_quarantine"
    for cache_dir in "${cache_dirs[@]}"; do
        relative_path=${cache_dir#"$source_root/"}
        destination=$bytecode_quarantine/$relative_path
        mkdir -p "$(dirname "$destination")"
        mv -- "$cache_dir" "$destination"
    done
fi
if find "$source_root/vllm" -type f -name '*.py[co]' -print -quit | grep -q .; then
    echo "benchmark source still contains Python bytecode after quarantine" >&2
    exit 1
fi

export PYTHONPATH=$source_root
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=$python_cache
export PYNV_GPU_LOCK_HOLDER=$lock_holder

setsid "$python_bin" -m pytest -q \
    "$source_root/tests/multimodal/test_video.py::test_pynvvideocodec_gpu_resize_uses_cvcuda_and_matches_pil" \
    "$source_root/tests/multimodal/test_video.py::test_pynvvc_gpu_resize_uses_torch_fallback" \
    -s >"$test_log" 2>&1 &
workload_pid=$!
wait "$workload_pid"
workload_pid=

setsid timeout --signal=INT --kill-after=180s 18000s \
    "$python_bin" "$runner" \
    --source-root "$source_root" \
    --python "$python_bin" \
    --transformers-root "$transformers_root" \
    --harness "$support_root/benchmark_pynvvideocodec_e2e_persistent.py" \
    --monitor "$support_root/run_with_gpu_monitor_refined.py" \
    --traffic-video "$traffic_video" \
    --corpus "$corpus" \
    --results "$results" \
    --gpu-label "$gpu_label" \
    --port 18700 >"$runner_log" 2>&1 &
workload_pid=$!
wait "$workload_pid"
workload_pid=

mv "$test_log" "$results/gpu-tests.log"
mv "$runner_log" "$results/runner.log"
sha256sum "$results/gpu-tests.log" "$results/runner.log" \
    >"$results/wrapper-artifacts.sha256"
cat "$results/summary-table.md"
