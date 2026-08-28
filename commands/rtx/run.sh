#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact vLLM source checkout}"
: "${ASSET_ROOT:?set ASSET_ROOT to the immutable RTX harness bundle}"
: "${RUNTIME_MANIFEST_ROOT:?set RUNTIME_MANIFEST_ROOT to the captured runtime manifests}"
: "${TRANSFORMERS_ROOT:?set TRANSFORMERS_ROOT to the frozen Transformers overlay}"
: "${HF_SNAPSHOT_ROOT:?set HF_SNAPSHOT_ROOT to the exact model snapshot}"
: "${CORPUS_ROOT:?set CORPUS_ROOT to the eight-hardlink video corpus}"
: "${RESULT_ROOT:?set RESULT_ROOT to a path that does not exist}"
: "${CONFLICTING_CONTROLLER_ROOT_1:?set the first known foreign controller root}"
: "${CONFLICTING_CONTROLLER_ROOT_2:?set the second known foreign controller root}"

PYTHON="${SOURCE_ROOT}/.venv/bin/python"
HARNESS_SHA256="d6da18d1fd77df44476a66aadfb7767174906ce58b4da9b972b38d052255bcf6"
DRIVER_SHA256="de1573ea884a11dce770b4fba2c322abe454c90c129836fbd576086551382b91"
FREEZE_MANIFEST_SHA256="11915ea2ded0fcbaf73fcf4cfa5ba9acd8a81efb449ec4ba3ff9212840f64e76"

fail() {
    echo "$*" >&2
    exit 2
}

require_exact_directory() {
    local variable_name="$1"
    local expected_basename="$2"
    local value="${!variable_name}"
    [[ "${value}" == /* ]] || fail "${variable_name} must be absolute"
    [[ -d "${value}" && ! -L "${value}" ]] || \
        fail "${variable_name} must be a real directory"
    [[ "$(realpath -e -- "${value}")" == "${value}" ]] || \
        fail "${variable_name} must be canonical"
    [[ "$(basename -- "${value}")" == "${expected_basename}" ]] || \
        fail "${variable_name} basename changed"
}

paths_overlap() {
    local left="$1"
    local right="$2"
    [[ "${left}" == "${right}" || "${left}" == "${right}/"* || "${right}" == "${left}/"* ]]
}

check_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    [[ -f "${path}" && ! -L "${path}" ]] || fail "missing real frozen file: ${path}"
    actual="$(sha256sum -- "${path}")"
    actual="${actual%% *}"
    [[ "${actual}" == "${expected}" ]] || fail "frozen file SHA-256 changed: ${path}"
}

require_exact_directory SOURCE_ROOT vllm-pynv-highc-rtx-20260828-v1
require_exact_directory ASSET_ROOT pynv-rtx-publication-freeze-v4
require_exact_directory RUNTIME_MANIFEST_ROOT pynv-runtime-manifests-v3
require_exact_directory TRANSFORMERS_ROOT vllm-pynv-e2e-transformers-5.14.1-20260827
require_exact_directory HF_SNAPSHOT_ROOT 89644892e4d85e24eaac8bacfd4f463576704203
require_exact_directory CORPUS_ROOT vllm-pynv-e2e-corpus-20260827
require_exact_directory CONFLICTING_CONTROLLER_ROOT_1 "$(basename -- "${CONFLICTING_CONTROLLER_ROOT_1}")"
require_exact_directory CONFLICTING_CONTROLLER_ROOT_2 "$(basename -- "${CONFLICTING_CONTROLLER_ROOT_2}")"

controller_basenames="$(basename -- "${CONFLICTING_CONTROLLER_ROOT_1}"):$(basename -- "${CONFLICTING_CONTROLLER_ROOT_2}")"
case "${controller_basenames}" in
    rtxbench:vllm-nvimagecodec-bench|vllm-nvimagecodec-bench:rtxbench) ;;
    *) fail "controller-root basenames changed" ;;
esac

[[ "${RESULT_ROOT}" == /* ]] || fail "RESULT_ROOT must be absolute"
[[ "$(basename -- "${RESULT_ROOT}")" == "vllm-pynv-qwen3vl-rtx-final-20260828-v2" ]] || \
    fail "RESULT_ROOT basename changed"
[[ ! -e "${RESULT_ROOT}" ]] || fail "RESULT_ROOT must not exist"
result_parent="$(dirname -- "${RESULT_ROOT}")"
[[ -d "${result_parent}" && ! -L "${result_parent}" ]] || \
    fail "RESULT_ROOT parent must be a real directory"
result_parent="$(realpath -e -- "${result_parent}")"
[[ "${RESULT_ROOT}" == "${result_parent}/vllm-pynv-qwen3vl-rtx-final-20260828-v2" ]] || \
    fail "RESULT_ROOT must be canonical and directly below its parent"

protected_roots=(
    "${SOURCE_ROOT}"
    "${ASSET_ROOT}"
    "${RUNTIME_MANIFEST_ROOT}"
    "${TRANSFORMERS_ROOT}"
    "${HF_SNAPSHOT_ROOT}"
    "${CORPUS_ROOT}"
    "${CONFLICTING_CONTROLLER_ROOT_1}"
    "${CONFLICTING_CONTROLLER_ROOT_2}"
)
for protected_root in "${protected_roots[@]}"; do
    paths_overlap "${RESULT_ROOT}" "${protected_root}" && \
        fail "RESULT_ROOT overlaps a protected input or controller root"
done
paths_overlap "${CONFLICTING_CONTROLLER_ROOT_1}" "${CONFLICTING_CONTROLLER_ROOT_2}" && \
    fail "controller roots must be disjoint"

check_sha256 "${ASSET_ROOT}/ARTIFACT_MANIFEST.json" "${FREEZE_MANIFEST_SHA256}"
check_sha256 "${ASSET_ROOT}/benchmark_pynvvideocodec_e2e_persistent.py" "${HARNESS_SHA256}"
check_sha256 "${ASSET_ROOT}/capture_runtime_tree_manifest.py" "d4edac7bc314aba8ceedc799b9d9b1c64ac880d340dff70b949db50066f1981a"
check_sha256 "${ASSET_ROOT}/preflight_pynv_endpoint_pixel_parity.py" "e4cb333cd47f3015ccf3aa510e3f6c26364cc4947b63d89053decc0f8156addb"
check_sha256 "${ASSET_ROOT}/pynv_gpu_guard.py" "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
check_sha256 "${ASSET_ROOT}/run_pynv_endpoint_high_concurrency_matrix_refined.py" "${DRIVER_SHA256}"
check_sha256 "${ASSET_ROOT}/run_pynv_endpoint_persistent_preflight.py" "924f7a6cf445678bc872ff5aa4d4cedfaac8db69ccf03fe7869ef0ba74b08636"
check_sha256 "${ASSET_ROOT}/run_with_gpu_monitor_refined.py" "239bcbbd0e635a8b44e46588142f336a8879067750aeee0d649faa8e62e950bc"
check_sha256 "${ASSET_ROOT}/test_persistent_http_harness.py" "9dadac7a651efb770557d98372acea90fe58db0a8ecf46a4a65adc8497c1a26b"
check_sha256 "${ASSET_ROOT}/test_refined_gpu_guards.py" "3cb113577694b1bd1a1fa02023827fc05a4331279ad6a16dc9339f49127c7d0b"
check_sha256 "${ASSET_ROOT}/test_rtx_endpoint_runner_contract.py" "296a42c744aea2830b5263125c9c41ef12ce81e9b93b13af963766942947ed11"
check_sha256 "${ASSET_ROOT}/test_runtime_tree_manifest.py" "d0ab0fcf324f6bc1042610a0ac5a970fe5fcf3fe31927aa904d0bb0ea76e0366"
check_sha256 "${ASSET_ROOT}/wait_for_exclusive_gpu_refined.py" "0a7119e7d0c40e3274ea9846db0b4e7213e7c1beeb4ddecce3a18d9641c5b02e"

"${PYTHON}" "${ASSET_ROOT}/run_pynv_endpoint_persistent_preflight.py" \
    --root "${SOURCE_ROOT}" \
    --assets "${ASSET_ROOT}" \
    --results "${RESULT_ROOT}/preflight" \
    --corpus "${CORPUS_ROOT}" \
    --transformers-root "${TRANSFORMERS_ROOT}" \
    --runtime-manifest-tool "${ASSET_ROOT}/capture_runtime_tree_manifest.py" \
    --runtime-manifest-test "${ASSET_ROOT}/test_runtime_tree_manifest.py" \
    --transformers-overlay-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/transformers-overlay.jsonl" \
    --transformers-overlay-manifest-summary "${RUNTIME_MANIFEST_ROOT}/transformers-overlay.summary.json" \
    --transformers-package-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/transformers.jsonl" \
    --transformers-package-manifest-summary "${RUNTIME_MANIFEST_ROOT}/transformers.summary.json" \
    --hf-snapshot-root "${HF_SNAPSHOT_ROOT}" \
    --hf-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/hf-snapshot.jsonl" \
    --hf-manifest-summary "${RUNTIME_MANIFEST_ROOT}/hf-snapshot.summary.json" \
    --harness-sha256 "${HARNESS_SHA256}" \
    --expected-driver-sha256 "${DRIVER_SHA256}" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_1}" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_2}"

"${PYTHON}" "${ASSET_ROOT}/run_pynv_endpoint_high_concurrency_matrix_refined.py" \
    --root "${SOURCE_ROOT}" \
    --results "${RESULT_ROOT}/matrix" \
    --python "${PYTHON}" \
    --transformers-root "${TRANSFORMERS_ROOT}" \
    --corpus "${CORPUS_ROOT}" \
    --harness "${ASSET_ROOT}/benchmark_pynvvideocodec_e2e_persistent.py" \
    --monitor "${ASSET_ROOT}/run_with_gpu_monitor_refined.py" \
    --idle-gate "${ASSET_ROOT}/wait_for_exclusive_gpu_refined.py" \
    --guard-helper "${ASSET_ROOT}/pynv_gpu_guard.py" \
    --runtime-manifest-tool "${ASSET_ROOT}/capture_runtime_tree_manifest.py" \
    --runtime-manifest-test "${ASSET_ROOT}/test_runtime_tree_manifest.py" \
    --transformers-overlay-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/transformers-overlay.jsonl" \
    --transformers-overlay-manifest-summary "${RUNTIME_MANIFEST_ROOT}/transformers-overlay.summary.json" \
    --transformers-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/transformers.jsonl" \
    --transformers-manifest-summary "${RUNTIME_MANIFEST_ROOT}/transformers.summary.json" \
    --hf-snapshot-root "${HF_SNAPSHOT_ROOT}" \
    --hf-manifest-jsonl "${RUNTIME_MANIFEST_ROOT}/hf-snapshot.jsonl" \
    --hf-manifest-summary "${RUNTIME_MANIFEST_ROOT}/hf-snapshot.summary.json" \
    --expected-harness-sha256 "${HARNESS_SHA256}" \
    --preflight-summary "${RESULT_ROOT}/preflight/pilot-summary.json" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_1}" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_2}" \
    --port 18600 \
    --idle-seconds 30 \
    --idle-timeout 1800 \
    --max-attempts 20
