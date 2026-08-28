<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# A100 command replay

`validate.sh` is the exact focused validation command. `run.sh` records the
preflight and timed-matrix argv, including all pinned hashes and non-default
options. Paths are supplied as environment variables so this public copy does
not disclose the collector's machine layout. Before executing either frozen
runner, `run.sh` enforces the exact input-root basenames, canonical real
directories, disjoint controller/result namespaces, the two fresh result-root
basenames, the freeze-manifest hash, and every frozen program/test hash.

The collection used these root basenames:

- `SOURCE_ROOT`: `vllm-pynv-a100-publication-3arm-20260828-v1`
- `ASSET_ROOT`: `pynv_a100_publication_freeze_v5`
- `RUNTIME_MANIFEST_ROOT`: `pynv-runtime-manifests-v3`
- `TRANSFORMERS_ROOT`: `vllm-pynv-e2e-transformers-5.14.1-20260827`
- `CORPUS_ROOT`: `vllm-pynv-e2e-corpus-20260827`
- Result parent: `vllm-pynv-qwen3vl-e2e-20260828-v1`
- `PREFLIGHT_RESULT_ROOT`: `a100-publication-preflight-v5`
- `RESULT_ROOT`: `a100-publication-matrix-c8c16c32-r6-v5`

`HF_SNAPSHOT_ROOT` is the cached snapshot at revision
`89644892e4d85e24eaac8bacfd4f463576704203`. The two controller-root variables
identify the known, disjoint `a100bench` and `vllm-nvimagecodec-bench`
workspaces; a root may be absent, but its canonical parent and exact basename
are checked. Validation exercises the live process guard and therefore must
pass on a clean host before collection. The guard independently rejects live
benchmark entry points, vLLM processes, external GPU processes, and non-idle
telemetry throughout the 1200-second ingress gate, the 30-second per-cell
gates, and every measured cell.
