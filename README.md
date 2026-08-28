# Qwen3-VL 2B PyNvVideoCodec E2E benchmark

This tree is the publication boundary for the benchmark evidence cited by
[`dlesage-nvidia/vllm#1`](https://github.com/dlesage-nvidia/vllm/pull/1).
It is intentionally incomplete until the final, independently validated
benchmark bundles and sanitized results are inserted and inventoried.

## Compared revisions

| Role | Commit | Git tree | Meaning |
|---|---|---|---|
| `upstream` | `d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302` | `9cc26997991af6f8f38150c9631d482d18b1bd2c` | upstream stack anchor |
| `pr-base` | `bc8abf31fef015339473f6071eda0de0305dd9b2` | `09423356278c6c4bd871ccda98499474fad78bdd` | pull-request base |
| `pr-head` | `30d917599b104423e452fa718890af01c4ff4d39` | `66c4849eb21973b9ca391b7b0911968f4aa63dac` | pull-request head |

`pr-base` to `pr-head` is the primary comparison because it isolates this
pull request. `upstream` to `pr-head` is cumulative: `upstream` to `pr-base`
contains related image-backend and shared media/IPC changes. RTX measures
`upstream` and `pr-head`; A100 measures all three roles. Every endpoint
explicitly uses PyNvVideoCodec.

## Fixed workload

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Model revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Prompt | `Describe this video concisely and factually.` |
| Input | H.264, 1920x1080, 914 frames, 30.498 s, SHA-256 `b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d` |
| Sampling | 32 frames |
| vLLM pixel budget | 1024x576 per sampled frame; `max_pixels=18,874,368` total |
| Serving | BF16, TP1, output length 32, two hardware decoders |
| Client | non-streaming pooled HTTP/1.1, exactly C prewarmed slots, zero request retries |
| Concurrency | C8, C16, C32 |
| Warmup/measured requests | 24/64, 48/128, 96/256 |
| Repetitions | six fresh servers per revision in a balanced committed order |

The eight workload filenames are hard links to one clip, not eight different
videos. The clip is not distributed here. Its exact derivation was not
retained, so the byte hash above is the input identity; this repository does
not claim a reconstruction recipe.

## Artifact layout

- `harness/{shared,rtx,a100}`: exact frozen benchmark programs.
- `tests/{shared,rtx,a100}`: exact tests frozen with those programs.
- `manifests/{freezes,runtime}`: source-bundle and runtime-tree manifests.
- `commands/{rtx,a100}`: machine-recorded exact validation and run commands.
- `results/public`: deterministic sanitized measurements only.
- `tools/public_tree.py`: fail-closed inventory and publication checker.

`manifests/staging-slots.json` is the only slot registry. Empty arrays and null
digests are deliberate placeholders, not benchmark values. After files are
copied into one target, inventory that slot with the SHA-256 of the immutable
source manifest that authorized the copy:

```bash
python3 tools/public_tree.py inventory-slot \
  --slot SLOT_ID --source-manifest SOURCE_MANIFEST
```

The tool computes the source-manifest hash, file sizes, file hashes, and the
slot-tree hash. It does not accept caller-supplied hashes. Git trees were
resolved from the exact local object database with
`set-role-tree --role ROLE --repo REPOSITORY`. Final publication is fail-closed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/publication -v
python3 tools/public_tree.py generate
python3 tools/public_tree.py check
sha256sum --check SHA256SUMS
```

Until every slot is ready, only the explicit staging check is expected to pass:

```bash
python3 tools/public_tree.py check --allow-pending
```

## Publication boundary

Raw roots are private because they contain machine paths and identities, GPU
identifiers, process IDs, timestamps, full logs, responses, and generated
text. Only positive-schema sanitizer output belongs in `results/public`.
The checker rejects undeclared files, symlinks, raw-log extensions, private
absolute paths, common credential forms, unsafe JSON metadata keys, mismatched
slot inventories, and stale generated manifests.

The final result summary will carry all six accepted measurements per cell,
dispersion, same-repetition paired deltas, geometric ratios and confidence
intervals, token-parity status, and allowlisted hardware/software provenance.
No numeric result or immutable publication tag is claimed by this staging
tree.
