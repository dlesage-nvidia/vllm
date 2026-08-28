<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# RTX command replay

`validate.sh` is the exact focused validation command. `run.sh` records the
preflight and timed-matrix argv, including all pinned hashes and non-default
options. Paths are supplied as environment variables so this public copy does
not disclose the collector's machine layout. Before executing either frozen
runner, `run.sh` enforces the exact root basenames, canonical real directories,
disjoint controller/result namespaces, the result basename, the freeze-manifest
hash, and every frozen program/test hash.

The collection used these root basenames:

- `SOURCE_ROOT`: `vllm-pynv-highc-rtx-20260828-v1`
- `ASSET_ROOT`: `pynv-rtx-publication-freeze-v4`
- `RUNTIME_MANIFEST_ROOT`: `pynv-runtime-manifests-v3`
- `TRANSFORMERS_ROOT`: `vllm-pynv-e2e-transformers-5.14.1-20260827`
- `CORPUS_ROOT`: `vllm-pynv-e2e-corpus-20260827`
- `RESULT_ROOT`: `vllm-pynv-qwen3vl-rtx-final-20260828-v2`

`HF_SNAPSHOT_ROOT` is the cached snapshot at revision
`89644892e4d85e24eaac8bacfd4f463576704203`. The two controller-root variables
identify the known, disjoint image-benchmark workspaces. They scope the guard's
exact `./screen.sh` controller match; the guard independently rejects live
benchmark entry points, vLLM processes, external GPU processes, and non-idle
telemetry throughout the 30-second quiet gate and every measured cell.
