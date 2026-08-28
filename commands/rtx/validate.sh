#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact vLLM source checkout}"
: "${ASSET_ROOT:?set ASSET_ROOT to the immutable RTX harness bundle}"

PYTHON="${SOURCE_ROOT}/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ASSET_ROOT}"

"${PYTHON}" -m pytest -q \
    "${ASSET_ROOT}/test_persistent_http_harness.py" \
    "${ASSET_ROOT}/test_refined_gpu_guards.py" \
    "${ASSET_ROOT}/test_runtime_tree_manifest.py" \
    "${ASSET_ROOT}/test_rtx_endpoint_runner_contract.py"
