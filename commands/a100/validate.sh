#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact vLLM source checkout}"
: "${ASSET_ROOT:?set ASSET_ROOT to the immutable A100 harness bundle}"

PYTHON="${SOURCE_ROOT}/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ASSET_ROOT}"

"${PYTHON}" -m pytest -q -p no:cacheprovider \
    "${ASSET_ROOT}/test_persistent_http_harness.py" \
    "${ASSET_ROOT}/test_refined_gpu_guards.py" \
    "${ASSET_ROOT}/test_runtime_tree_manifest.py"

"${PYTHON}" "${ASSET_ROOT}/test_persistent_three_arm_campaign.py" \
    --runner "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_matrix.py" \
    --harness "${ASSET_ROOT}/benchmark_pynvvideocodec_e2e_persistent.py" \
    --pilot-runner "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_pilots.py" \
    --pixel-preflight "${ASSET_ROOT}/preflight_pynv_persistent_three_arm_pixel_parity.py" \
    --monitor "${ASSET_ROOT}/run_with_gpu_monitor_refined.py" \
    --campaign-contract "${ASSET_ROOT}/CAMPAIGN_CONTRACT.json"
