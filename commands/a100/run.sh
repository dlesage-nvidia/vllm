#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${SOURCE_ROOT:?set SOURCE_ROOT to the exact vLLM source checkout}"
: "${ASSET_ROOT:?set ASSET_ROOT to the immutable A100 harness bundle}"
: "${RUNTIME_MANIFEST_ROOT:?set RUNTIME_MANIFEST_ROOT to the captured runtime manifests}"
: "${TRANSFORMERS_ROOT:?set TRANSFORMERS_ROOT to the frozen Transformers overlay}"
: "${HF_SNAPSHOT_ROOT:?set HF_SNAPSHOT_ROOT to the exact model snapshot}"
: "${CORPUS_ROOT:?set CORPUS_ROOT to the eight-hardlink video corpus}"
: "${PREFLIGHT_RESULT_ROOT:?set PREFLIGHT_RESULT_ROOT to a path that does not exist}"
: "${RESULT_ROOT:?set RESULT_ROOT to the timed-matrix path that does not exist}"
: "${CONFLICTING_CONTROLLER_ROOT_1:?set the first known foreign controller root}"
: "${CONFLICTING_CONTROLLER_ROOT_2:?set the second known foreign controller root}"

PYTHON="${SOURCE_ROOT}/.venv/bin/python"
HARNESS_SHA256="71adcc9ddb99e65e51d9531ed40728b8261f0f763c2fd1d89c2610a58fa3aa2b"
DRIVER_SHA256="a18499e1d2276d6e46b89cc62bf39b156af33d36b89591d921192ac160fd76f0"
FREEZE_MANIFEST_SHA256="6edbaf2489feaef386619edb797c524b852ad01db63022c7f5d2c3a8c4a2f7e8"

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

require_expected_controller_root() {
    local variable_name="$1"
    local expected_basename="$2"
    local value="${!variable_name}"
    local parent
    [[ "${value}" == /* ]] || fail "${variable_name} must be absolute"
    [[ "$(basename -- "${value}")" == "${expected_basename}" ]] || \
        fail "${variable_name} basename changed"
    parent="$(dirname -- "${value}")"
    [[ -d "${parent}" && ! -L "${parent}" ]] || \
        fail "${variable_name} parent must be a real directory"
    parent="$(realpath -e -- "${parent}")"
    [[ "${value}" == "${parent}/${expected_basename}" ]] || \
        fail "${variable_name} must be canonical and directly below its parent"
    if [[ -e "${value}" ]]; then
        [[ -d "${value}" && ! -L "${value}" ]] || \
            fail "${variable_name} must be absent or a real directory"
        [[ "$(realpath -e -- "${value}")" == "${value}" ]] || \
            fail "${variable_name} must be canonical"
    fi
}

require_fresh_result_root() {
    local variable_name="$1"
    local expected_basename="$2"
    local value="${!variable_name}"
    local parent
    [[ "${value}" == /* ]] || fail "${variable_name} must be absolute"
    [[ "$(basename -- "${value}")" == "${expected_basename}" ]] || \
        fail "${variable_name} basename changed"
    [[ ! -e "${value}" ]] || fail "${variable_name} must not exist"
    parent="$(dirname -- "${value}")"
    [[ -d "${parent}" && ! -L "${parent}" ]] || \
        fail "${variable_name} parent must be a real directory"
    parent="$(realpath -e -- "${parent}")"
    [[ "${value}" == "${parent}/${expected_basename}" ]] || \
        fail "${variable_name} must be canonical and directly below its parent"
    [[ "$(basename -- "${parent}")" == "vllm-pynv-qwen3vl-e2e-20260828-v1" ]] || \
        fail "${variable_name} parent basename changed"
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

validate_source_endpoint() {
    local commit
    local tree
    local status
    commit="$(git -C "${SOURCE_ROOT}" rev-parse --verify "HEAD^{commit}" 2>/dev/null)" || \
        fail "SOURCE_ROOT is not a readable Git checkout"
    tree="$(git -C "${SOURCE_ROOT}" rev-parse --verify "HEAD^{tree}" 2>/dev/null)" || \
        fail "SOURCE_ROOT tree cannot be resolved"
    case "${commit}:${tree}" in
        d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302:9cc26997991af6f8f38150c9631d482d18b1bd2c|\
        bc8abf31fef015339473f6071eda0de0305dd9b2:09423356278c6c4bd871ccda98499474fad78bdd|\
        30d917599b104423e452fa718890af01c4ff4d39:66c4849eb21973b9ca391b7b0911968f4aa63dac) ;;
        *) fail "SOURCE_ROOT is not at an exact campaign endpoint commit and tree" ;;
    esac
    status="$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" || \
        fail "SOURCE_ROOT status cannot be read"
    [[ -z "${status}" ]] || fail "SOURCE_ROOT must be clean"
}

validate_asset_inventory() {
    local asset
    local name
    local -a actual_assets
    local -A expected_assets=(
        [ARTIFACT_MANIFEST.json]=1
        [CAMPAIGN_CONTRACT.json]=1
        [SHARED_V3_ARTIFACT_MANIFEST.json]=1
        [VALIDATION.json]=1
        [benchmark_pynvvideocodec_e2e_persistent.py]=1
        [capture_runtime_tree_manifest.py]=1
        [preflight_pynv_persistent_three_arm_pixel_parity.py]=1
        [pynv_gpu_guard.py]=1
        [run_pynv_endpoint_high_concurrency_matrix_refined.py]=1
        [run_pynv_persistent_three_arm_high_concurrency_matrix.py]=1
        [run_pynv_persistent_three_arm_high_concurrency_pilots.py]=1
        [run_with_gpu_monitor_refined.py]=1
        [test_persistent_http_harness.py]=1
        [test_persistent_three_arm_campaign.py]=1
        [test_refined_gpu_guards.py]=1
        [test_runtime_tree_manifest.py]=1
        [wait_for_exclusive_gpu_refined.py]=1
    )
    [[ "$(stat -c '%a' -- "${ASSET_ROOT}")" == "555" ]] || \
        fail "ASSET_ROOT mode must be 0555"
    mapfile -d '' actual_assets < <(
        find "${ASSET_ROOT}" -mindepth 1 -maxdepth 1 -print0
    )
    [[ "${#actual_assets[@]}" -eq "${#expected_assets[@]}" ]] || \
        fail "ASSET_ROOT must contain exactly the 17 frozen files"
    for asset in "${actual_assets[@]}"; do
        name="$(basename -- "${asset}")"
        [[ -n "${expected_assets[${name}]+present}" ]] || \
            fail "unexpected ASSET_ROOT entry: ${name}"
        [[ -f "${asset}" && ! -L "${asset}" ]] || \
            fail "ASSET_ROOT entry must be a real regular file: ${name}"
        [[ "$(stat -c '%a' -- "${asset}")" == "444" ]] || \
            fail "ASSET_ROOT file mode must be 0444: ${name}"
    done
}

require_exact_directory SOURCE_ROOT vllm-pynv-a100-publication-3arm-20260828-v1
require_exact_directory ASSET_ROOT pynv_a100_publication_freeze_v4
require_exact_directory RUNTIME_MANIFEST_ROOT pynv-runtime-manifests-v3
require_exact_directory TRANSFORMERS_ROOT vllm-pynv-e2e-transformers-5.14.1-20260827
require_exact_directory HF_SNAPSHOT_ROOT 89644892e4d85e24eaac8bacfd4f463576704203
require_exact_directory CORPUS_ROOT vllm-pynv-e2e-corpus-20260827
require_expected_controller_root CONFLICTING_CONTROLLER_ROOT_1 a100bench
require_expected_controller_root CONFLICTING_CONTROLLER_ROOT_2 vllm-nvimagecodec-bench

controller_basenames="$(basename -- "${CONFLICTING_CONTROLLER_ROOT_1}"):$(basename -- "${CONFLICTING_CONTROLLER_ROOT_2}")"
case "${controller_basenames}" in
    a100bench:vllm-nvimagecodec-bench|vllm-nvimagecodec-bench:a100bench) ;;
    *) fail "controller-root basenames changed" ;;
esac

require_fresh_result_root \
    PREFLIGHT_RESULT_ROOT a100-publication-preflight-v4
require_fresh_result_root \
    RESULT_ROOT a100-publication-matrix-c8c16c32-r6-v4
[[ "$(dirname -- "${PREFLIGHT_RESULT_ROOT}")" == "$(dirname -- "${RESULT_ROOT}")" ]] || \
    fail "preflight and matrix roots must share the frozen campaign parent"
paths_overlap "${PREFLIGHT_RESULT_ROOT}" "${RESULT_ROOT}" && \
    fail "preflight and matrix roots must be disjoint"

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
for ((left_index = 0; left_index < ${#protected_roots[@]}; left_index++)); do
    for ((right_index = left_index + 1; right_index < ${#protected_roots[@]}; right_index++)); do
        paths_overlap \
            "${protected_roots[left_index]}" \
            "${protected_roots[right_index]}" && \
            fail "protected input and controller roots must be pairwise disjoint"
    done
done
for protected_root in "${protected_roots[@]}"; do
    paths_overlap "${PREFLIGHT_RESULT_ROOT}" "${protected_root}" && \
        fail "PREFLIGHT_RESULT_ROOT overlaps a protected input or controller root"
    paths_overlap "${RESULT_ROOT}" "${protected_root}" && \
        fail "RESULT_ROOT overlaps a protected input or controller root"
done
paths_overlap "${CONFLICTING_CONTROLLER_ROOT_1}" "${CONFLICTING_CONTROLLER_ROOT_2}" && \
    fail "controller roots must be disjoint"

check_sha256 "${ASSET_ROOT}/ARTIFACT_MANIFEST.json" "${FREEZE_MANIFEST_SHA256}"
check_sha256 "${ASSET_ROOT}/CAMPAIGN_CONTRACT.json" "82659a3caad4282094ca7d893816d1f34af9d0067c54062b88c386232ab224d0"
check_sha256 "${ASSET_ROOT}/SHARED_V3_ARTIFACT_MANIFEST.json" "930da0b3ca9fe2712653c25bdcad1c4a65d8a0de3ba31324596bd9e88e5a9b60"
check_sha256 "${ASSET_ROOT}/VALIDATION.json" "e84e905836191eead19131dc2ef5e1419ae800a6d76f26099105e07a793c7e4f"
check_sha256 "${ASSET_ROOT}/benchmark_pynvvideocodec_e2e_persistent.py" "${HARNESS_SHA256}"
check_sha256 "${ASSET_ROOT}/capture_runtime_tree_manifest.py" "d4edac7bc314aba8ceedc799b9d9b1c64ac880d340dff70b949db50066f1981a"
check_sha256 "${ASSET_ROOT}/preflight_pynv_persistent_three_arm_pixel_parity.py" "af4cadebcfada425997baf3f773b0f81d4103e981d87222cc355bc884db0a2d8"
check_sha256 "${ASSET_ROOT}/pynv_gpu_guard.py" "4a2910fee2810afdb42f2a74611808bc692482df2992a9ac0cd0c8dd0a1104fb"
check_sha256 "${ASSET_ROOT}/run_pynv_endpoint_high_concurrency_matrix_refined.py" "7045f370bcdac85e82249f77193dcd0339cb4de3944d0084731013c3eeb93f57"
check_sha256 "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_matrix.py" "${DRIVER_SHA256}"
check_sha256 "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_pilots.py" "77aa45391de3a3436168827025ac6c5b7812bcc977e121d99fd244a5159f68b0"
check_sha256 "${ASSET_ROOT}/run_with_gpu_monitor_refined.py" "239bcbbd0e635a8b44e46588142f336a8879067750aeee0d649faa8e62e950bc"
check_sha256 "${ASSET_ROOT}/test_persistent_http_harness.py" "9dadac7a651efb770557d98372acea90fe58db0a8ecf46a4a65adc8497c1a26b"
check_sha256 "${ASSET_ROOT}/test_persistent_three_arm_campaign.py" "de5ed3502ea7ae1be34fa3b2b3c68c2e321227ff55349aaf0635fcd4f585221e"
check_sha256 "${ASSET_ROOT}/test_refined_gpu_guards.py" "3cb113577694b1bd1a1fa02023827fc05a4331279ad6a16dc9339f49127c7d0b"
check_sha256 "${ASSET_ROOT}/test_runtime_tree_manifest.py" "d0ab0fcf324f6bc1042610a0ac5a970fe5fcf3fe31927aa904d0bb0ea76e0366"
check_sha256 "${ASSET_ROOT}/wait_for_exclusive_gpu_refined.py" "0a7119e7d0c40e3274ea9846db0b4e7213e7c1beeb4ddecce3a18d9641c5b02e"
validate_asset_inventory
validate_source_endpoint

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

"${PYTHON}" "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_pilots.py" \
    --root "${SOURCE_ROOT}" \
    --assets "${ASSET_ROOT}" \
    --results "${PREFLIGHT_RESULT_ROOT}" \
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

validate_source_endpoint

"${PYTHON}" "${ASSET_ROOT}/run_pynv_persistent_three_arm_high_concurrency_matrix.py" \
    --root "${SOURCE_ROOT}" \
    --results "${RESULT_ROOT}" \
    --python "${PYTHON}" \
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
    --corpus "${CORPUS_ROOT}" \
    --harness "${ASSET_ROOT}/benchmark_pynvvideocodec_e2e_persistent.py" \
    --monitor "${ASSET_ROOT}/run_with_gpu_monitor_refined.py" \
    --idle-gate "${ASSET_ROOT}/wait_for_exclusive_gpu_refined.py" \
    --guard-helper "${ASSET_ROOT}/pynv_gpu_guard.py" \
    --expected-harness-sha256 "${HARNESS_SHA256}" \
    --preflight-summary "${PREFLIGHT_RESULT_ROOT}/pilot-summary.json" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_1}" \
    --conflicting-controller-root "${CONFLICTING_CONTROLLER_ROOT_2}" \
    --port 18600 \
    --idle-seconds 30 \
    --idle-timeout 1800 \
    --max-attempts 20
