#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

holder=agent-pynvv-qwen25vl-bench
lease_seconds=21600
renew_seconds=1800
main_pid=$$
renew_pid=

cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "${renew_pid}" ]]; then
        kill "${renew_pid}" 2>/dev/null || true
        wait "${renew_pid}" 2>/dev/null || true
    fi
    /usr/local/bin/gpulock release "${holder}" || status=1
    exit "${status}"
}
trap cleanup EXIT INT TERM

/usr/local/bin/gpulock renew "${holder}" "${lease_seconds}"
(
    while sleep "${renew_seconds}"; do
        if ! /usr/local/bin/gpulock renew "${holder}" "${lease_seconds}"; then
            kill -TERM "${main_pid}"
            exit 1
        fi
    done
) &
renew_pid=$!

"$@"
