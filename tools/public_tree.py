#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed inventory and publication checks for this artifact tree."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable

SCHEMA_EXPERIMENT = "pynv-qwen3vl-public-artifact-experiment-v1"
SCHEMA_SLOTS = "pynv-public-artifact-slots-v1"
SCHEMA_TREE = "pynv-public-tree-manifest-v1"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILE_BYTES = 256 * 1024 * 1024

EXPERIMENT_PATH = "manifests/experiment.json"
SLOTS_PATH = "manifests/staging-slots.json"
TREE_MANIFEST_PATH = "manifests/public-tree.json"
CHECKSUMS_PATH = "SHA256SUMS"

STATIC_FILES = {
    "LICENSE",
    "README.md",
    EXPERIMENT_PATH,
    SLOTS_PATH,
    "tests/publication/test_public_tree.py",
    "tools/public_tree.py",
}
GENERATED_FILES = {TREE_MANIFEST_PATH, CHECKSUMS_PATH}

EXPECTED_SLOTS = (
    ("shared-harness", "frozen-harness", "harness/shared"),
    ("rtx-harness", "frozen-harness", "harness/rtx"),
    ("a100-harness", "frozen-harness", "harness/a100"),
    ("shared-tests", "frozen-tests", "tests/shared"),
    ("rtx-tests", "frozen-tests", "tests/rtx"),
    ("a100-tests", "frozen-tests", "tests/a100"),
    (
        "shared-freeze-manifests",
        "freeze-manifests",
        "manifests/freezes/shared",
    ),
    ("rtx-freeze-manifests", "freeze-manifests", "manifests/freezes/rtx"),
    (
        "a100-freeze-manifests",
        "freeze-manifests",
        "manifests/freezes/a100",
    ),
    ("rtx-runtime-manifests", "runtime-manifests", "manifests/runtime/rtx"),
    (
        "a100-runtime-manifests",
        "runtime-manifests",
        "manifests/runtime/a100",
    ),
    ("rtx-commands", "exact-commands", "commands/rtx"),
    ("a100-commands", "exact-commands", "commands/a100"),
    ("public-results", "sanitized-results", "results/public"),
)

EXPECTED_COMMITS = {
    "upstream": "d1e5e66ee30ba4bc020ac8e14b05e7a8c41b9302",
    "pr-base": "bc8abf31fef015339473f6071eda0de0305dd9b2",
    "pr-head": "30d917599b104423e452fa718890af01c4ff4d39",
}
EXPECTED_HEAD_TREE = "66c4849eb21973b9ca391b7b0911968f4aa63dac"
EXPECTED_WORKLOAD = {
    "clip": {
        "duration_seconds": 30.498,
        "encoded_frames": 914,
        "height": 1080,
        "sha256": "b5816375c491528f23799b1d1d67100355d1d43730db4898d480e4edb5065a5d",
        "width": 1920,
    },
    "concurrency": [8, 16, 32],
    "decoder_backend": "pynvvideocodec",
    "max_pixels_total": 18_874_368,
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
    "output_tokens": 32,
    "pixel_budget_per_frame": {"height": 576, "width": 1024},
    "prompt": "Describe this video concisely and factually.",
    "sampled_frames": 32,
}

ALLOWED_SUFFIXES = {
    "frozen-harness": {".json", ".py"},
    "frozen-tests": {".json", ".py"},
    "freeze-manifests": {".json"},
    "runtime-manifests": {".json", ".jsonl"},
    "exact-commands": {".json", ".md", ".sh", ".txt"},
    "sanitized-results": {".csv", ".json", ".jsonl", ".md"},
}

UNSAFE_JSON_KEYS = {
    "branch",
    "full_server_log",
    "generated_text",
    "gpu_uuid",
    "host",
    "hostname",
    "pci_bus_id",
    "pid",
    "pids",
    "raw_response",
    "server_log",
    "started_at",
    "finished_at",
    "user",
    "username",
}


class ArtifactError(RuntimeError):
    """A publication contract, integrity, or safety failure."""


def _fail(message: str) -> None:
    raise ArtifactError(message)


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail(f"JSON object has duplicate key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON number {value!r} is forbidden")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicates_rejected,
            parse_constant=_reject_constant,
        )
    except ArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"{path}: invalid JSON: {exc}") from exc


def _load_json_bytes(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates_rejected,
            parse_constant=_reject_constant,
        )
    except ArtifactError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"{label}: invalid JSON: {exc}") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot encode canonical JSON: {exc}") from exc


def _pretty_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot encode JSON: {exc}") from exc


def _strict_keys(value: Any, required: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{label}: expected object")
    expected = set(required)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        _fail(f"{label}: missing keys: {', '.join(missing)}")
    if unknown:
        _fail(f"{label}: unknown keys: {', '.join(unknown)}")
    return value


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label}: expected non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{label}: unsafe relative path {value!r}")
    if any(part.startswith(".") for part in path.parts):
        _fail(f"{label}: hidden path component is forbidden: {value!r}")
    return path.as_posix()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{label}: expected lowercase SHA-256")
    return value


def _sha1(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or SHA1_RE.fullmatch(value) is None:
        _fail(f"{label}: expected lowercase Git object ID")
    return value


def _validate_experiment(root: Path, *, allow_pending: bool) -> dict[str, Any]:
    path = root / EXPERIMENT_PATH
    value = _strict_keys(
        _load_json(path),
        {"comparison", "pull_request", "roles", "schema", "workload"},
        EXPERIMENT_PATH,
    )
    if value["schema"] != SCHEMA_EXPERIMENT:
        _fail(f"{EXPERIMENT_PATH}: unexpected schema")
    if value["pull_request"] != "dlesage-nvidia/vllm#1":
        _fail(f"{EXPERIMENT_PATH}: unexpected pull request")
    expected_comparison = {
        "a100_roles": ["upstream", "pr-base", "pr-head"],
        "primary": ["pr-base", "pr-head"],
        "rtx_roles": ["upstream", "pr-head"],
        "secondary_cumulative": ["upstream", "pr-head"],
    }
    if value["comparison"] != expected_comparison:
        _fail(f"{EXPERIMENT_PATH}: comparison roles or ordering changed")
    if value["workload"] != EXPECTED_WORKLOAD:
        _fail(f"{EXPERIMENT_PATH}: fixed workload changed")

    roles = value["roles"]
    if not isinstance(roles, dict) or set(roles) != set(EXPECTED_COMMITS):
        _fail(f"{EXPERIMENT_PATH}: roles must be upstream, pr-base, and pr-head")
    for role, commit in EXPECTED_COMMITS.items():
        record = _strict_keys(roles[role], {"commit", "tree"}, f"roles.{role}")
        if record["commit"] != commit:
            _fail(f"{EXPERIMENT_PATH}: {role} commit changed")
        tree = _sha1(record["tree"], f"roles.{role}.tree", nullable=True)
        if role == "pr-head" and tree != EXPECTED_HEAD_TREE:
            _fail(f"{EXPERIMENT_PATH}: pr-head tree changed")
        if not allow_pending and tree is None:
            _fail(f"{EXPERIMENT_PATH}: unresolved machine-generated tree for {role}")
    return value


def _validate_inventory_entry(value: Any, label: str) -> dict[str, Any]:
    entry = _strict_keys(value, {"bytes", "path", "sha256"}, label)
    entry["path"] = _safe_relative(entry["path"], f"{label}.path")
    if (
        isinstance(entry["bytes"], bool)
        or not isinstance(entry["bytes"], int)
        or entry["bytes"] < 0
    ):
        _fail(f"{label}.bytes: expected non-negative integer")
    _sha256(entry["sha256"], f"{label}.sha256")
    return entry


def _slot_tree_sha256(files: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json(files))


def _validate_kind_minimum(slot: dict[str, Any]) -> None:
    paths = [entry["path"] for entry in slot["files"]]
    basenames = {PurePosixPath(path).name for path in paths}
    kind = slot["kind"]
    if kind == "frozen-harness" and not any(path.endswith(".py") for path in paths):
        _fail(f"slot {slot['id']}: frozen harness has no Python program")
    if kind == "frozen-tests" and not any(
        PurePosixPath(path).name.startswith("test_") and path.endswith(".py")
        for path in paths
    ):
        _fail(f"slot {slot['id']}: frozen tests have no test_*.py file")
    if kind == "freeze-manifests" and "ARTIFACT_MANIFEST.json" not in basenames:
        _fail(f"slot {slot['id']}: missing ARTIFACT_MANIFEST.json")
    if kind == "runtime-manifests":
        if not any(path.endswith(".summary.json") for path in paths):
            _fail(f"slot {slot['id']}: missing runtime summary manifest")
        if not any(path.endswith(".jsonl") for path in paths):
            _fail(f"slot {slot['id']}: missing runtime file manifest")
    if kind == "exact-commands" and "commands.json" not in basenames:
        _fail(f"slot {slot['id']}: missing machine-recorded commands.json")
    if kind == "sanitized-results":
        required = {
            "collection-summary.json",
            "provenance.json",
            "public-manifest.json",
            "requests.jsonl.gz",
            "token-parity.json",
        }
        missing = sorted(required - basenames)
        if missing:
            _fail(f"slot {slot['id']}: missing sanitized results: {', '.join(missing)}")


def _validate_slot_doc(
    value: Any, *, allow_pending: bool
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    document = _strict_keys(value, {"schema", "slots"}, SLOTS_PATH)
    if document["schema"] != SCHEMA_SLOTS:
        _fail(f"{SLOTS_PATH}: unexpected schema")
    if not isinstance(document["slots"], list):
        _fail(f"{SLOTS_PATH}.slots: expected array")
    if len(document["slots"]) != len(EXPECTED_SLOTS):
        _fail(f"{SLOTS_PATH}: slot count changed")

    slots: dict[str, dict[str, Any]] = {}
    expected_by_id = {item[0]: item for item in EXPECTED_SLOTS}
    for index, raw in enumerate(document["slots"]):
        slot = _strict_keys(
            raw,
            {
                "files",
                "id",
                "kind",
                "source_manifest_sha256",
                "status",
                "target",
                "tree_sha256",
            },
            f"slots[{index}]",
        )
        slot_id = slot["id"]
        if not isinstance(slot_id, str) or slot_id not in expected_by_id:
            _fail(f"slots[{index}].id: unexpected slot {slot_id!r}")
        if slot_id in slots:
            _fail(f"slots[{index}].id: duplicate slot {slot_id!r}")
        _, expected_kind, expected_target = expected_by_id[slot_id]
        if slot["kind"] != expected_kind or slot["target"] != expected_target:
            _fail(f"slot {slot_id}: kind or target changed")
        if slot["status"] not in {"pending", "ready"}:
            _fail(f"slot {slot_id}: status must be pending or ready")
        if not isinstance(slot["files"], list):
            _fail(f"slot {slot_id}: files must be an array")
        slot["files"] = [
            _validate_inventory_entry(entry, f"slot {slot_id}.files[{entry_index}]")
            for entry_index, entry in enumerate(slot["files"])
        ]
        paths = [entry["path"] for entry in slot["files"]]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            _fail(f"slot {slot_id}: file paths must be unique and sorted")

        if slot["status"] == "pending":
            if slot["files"] or slot["source_manifest_sha256"] is not None:
                _fail(f"slot {slot_id}: pending slot contains final inventory data")
            if slot["tree_sha256"] is not None:
                _fail(f"slot {slot_id}: pending slot contains a tree hash")
            if not allow_pending:
                _fail(f"slot {slot_id}: final machine-generated inventory is pending")
        else:
            if not slot["files"]:
                _fail(f"slot {slot_id}: ready slot has no files")
            _sha256(
                slot["source_manifest_sha256"],
                f"slot {slot_id}.source_manifest_sha256",
            )
            _sha256(slot["tree_sha256"], f"slot {slot_id}.tree_sha256")
            if slot["tree_sha256"] != _slot_tree_sha256(slot["files"]):
                _fail(f"slot {slot_id}: tree hash does not match inventory")
            _validate_kind_minimum(slot)
        slots[slot_id] = slot
    if set(slots) != set(expected_by_id):
        _fail(f"{SLOTS_PATH}: missing expected slot")
    return document, slots


def _walk_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    if root.is_symlink() or not root.is_dir():
        _fail(f"artifact root is not a real directory: {root}")
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        names.sort()
        filenames.sort()
        if base == root and ".git" in names:
            metadata = root / ".git"
            mode = metadata.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail("root Git metadata is not a real directory")
            names.remove(".git")
        for name in names:
            path = base / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative, "tree directory")
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                _fail(f"tree contains symlink or non-directory node: {relative}")
            directories.add(relative)
        for name in filenames:
            path = base / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            _safe_relative(relative, "tree file")
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                _fail(f"tree contains symlink or non-regular file: {relative}")
            if path.stat().st_size > MAX_FILE_BYTES:
                _fail(f"tree file exceeds publication size cap: {relative}")
            files.add(relative)
    return files, directories


def _allowed_directories(slots: dict[str, dict[str, Any]]) -> set[str]:
    fixed = {
        "commands",
        "harness",
        "manifests",
        "manifests/freezes",
        "manifests/runtime",
        "results",
        "tests",
        "tests/publication",
        "tools",
    }
    allowed = set(fixed)
    for slot in slots.values():
        target = PurePosixPath(slot["target"])
        allowed.add(target.as_posix())
        for entry in slot["files"]:
            parent = target / PurePosixPath(entry["path"]).parent
            while parent != target.parent:
                allowed.add(parent.as_posix())
                if parent == target:
                    break
                parent = parent.parent
    return allowed


def _required_directories() -> set[str]:
    required = {
        "commands",
        "harness",
        "manifests",
        "manifests/freezes",
        "manifests/runtime",
        "results",
        "tests",
        "tests/publication",
        "tools",
    }
    required.update(item[2] for item in EXPECTED_SLOTS)
    return required


def _allowed_suffix(kind: str, relative: str) -> bool:
    path = PurePosixPath(relative)
    if kind == "sanitized-results" and relative.endswith(".jsonl.gz"):
        return True
    if path.name == "SHA256SUMS" and kind == "sanitized-results":
        return True
    return path.suffix in ALLOWED_SUFFIXES[kind]


def _scan_json_keys(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower()
            timestamp_key = (
                normalized_key in {"captured_utc", "timestamp", "timestamps"}
                or normalized_key.endswith("_time_ns")
                or normalized_key.endswith("_timestamp")
                or normalized_key.endswith("_timestamps")
            )
            if normalized_key in UNSAFE_JSON_KEYS or timestamp_key:
                documented_contract_key = isinstance(child, str) and (
                    (label.endswith(".evidence_contract") and key == "raw_response")
                    or (label.endswith(".provenance_contract") and key == "server_log")
                )
                if not documented_contract_key:
                    _fail(f"{label}: private or raw JSON key is forbidden: {key!r}")
            _scan_json_keys(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_json_keys(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label}: non-finite number is forbidden")


def _scan_text(raw: bytes, label: str) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"{label}: publication text is not UTF-8: {exc}") from exc

    private_roots = "|".join(
        ("data", "ephemeral", "home", "mnt", "root", "tmp", "var/tmp", "workspace")
    )
    posix_pattern = re.compile(
        rf"(?<![A-Za-z0-9])/(?:{private_roots})(?:/|(?=[\s\"']))",
        re.IGNORECASE,
    )
    backslash = chr(92)
    windows_pattern = re.compile(
        r"(?<![A-Za-z0-9])[A-Za-z]:" + re.escape(backslash) + r"[^\s]+"
    )
    unc_pattern = re.compile(
        "(?<!"
        + re.escape(backslash)
        + ")"
        + re.escape(backslash * 2)
        + r"[A-Za-z0-9._-]+"
        + re.escape(backslash)
        + r"[A-Za-z0-9$._-]+(?:"
        + re.escape(backslash)
        + r"[^\s]+)?"
    )
    checks = (
        (posix_pattern, "private absolute POSIX path"),
        (windows_pattern, "absolute Windows path"),
        (unc_pattern, "absolute UNC path"),
        (re.compile("hf" + r"_[A-Za-z0-9]{20,}"), "Hugging Face token"),
        (re.compile("gh" + r"[pousr]_[A-Za-z0-9]{20,}"), "GitHub token"),
        (re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"), "GitHub token"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
        (
            re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
            "private key",
        ),
    )
    for pattern, description in checks:
        if pattern.search(text):
            _fail(f"{label}: {description} is forbidden")


def _scan_file(path: Path, relative: str) -> None:
    raw = path.read_bytes()
    if relative.endswith(".jsonl.gz"):
        if len(raw) < 10 or raw[:3] != b"\x1f\x8b\x08":
            _fail(f"{relative}: expected gzip data")
        if raw[4:8] != b"\x00\x00\x00\x00":
            _fail(f"{relative}: gzip timestamp must be zero")
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise ArtifactError(f"{relative}: invalid gzip data: {exc}") from exc
        if len(raw) > MAX_FILE_BYTES:
            _fail(f"{relative}: decompressed publication data exceeds size cap")
        _scan_text(raw, f"{relative} (decompressed)")
        for index, line in enumerate(raw.splitlines(), start=1):
            if line.strip():
                _scan_json_keys(
                    _load_json_bytes(line, f"{relative}:{index}"),
                    f"{relative}:{index}",
                )
        return

    if relative.startswith("harness/"):
        # The frozen CLI help uses these two generic examples. They identify no
        # machine or corpus and must not force a change to already frozen bytes.
        for name in ("clip-0.mp4", "clip-1.mp4"):
            generic_example = ("/" + f"data/{name}").encode("utf-8")
            raw = raw.replace(generic_example, name.encode("utf-8"))
    if relative == "tests/shared/test_refined_gpu_guards.py":
        # These exact strings are synthetic process-table fixtures in the frozen
        # guard tests. Keep the exception narrower than the temporary-root prefix.
        synthetic_root = ("/" + "tmp/").encode("utf-8")
        synthetic_examples = (
            b'f"' + synthetic_root + b'{tagged_controller}"',
            b'"' + synthetic_root + b'status.py"',
            b'"' + synthetic_root + b'unrelated"',
            b'"' + synthetic_root + b'unexpected.py"',
            b'"' + synthetic_root + b'unrelated.py"',
            b'"' + synthetic_root + b'not-nvidia-cuda-mps-server-wrapper"',
        )
        for example in synthetic_examples:
            raw = raw.replace(example, b'"synthetic-process-path"')
    _scan_text(raw, relative)
    if relative.endswith(".json"):
        _scan_json_keys(_load_json_bytes(raw, relative), relative)
    elif relative.endswith(".jsonl"):
        for index, line in enumerate(raw.splitlines(), start=1):
            if line.strip():
                _scan_json_keys(
                    _load_json_bytes(line, f"{relative}:{index}"),
                    f"{relative}:{index}",
                )


def _expected_dynamic_files(
    slots: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for slot in slots.values():
        target = PurePosixPath(slot["target"])
        for entry in slot["files"]:
            relative = (target / entry["path"]).as_posix()
            if relative in expected:
                _fail(f"slot inventories overlap at {relative}")
            if not _allowed_suffix(slot["kind"], entry["path"]):
                _fail(f"slot {slot['id']}: forbidden file type: {entry['path']}")
            expected[relative] = entry
    return expected


def _manifest_entries(root: Path, files: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": (root / relative).stat().st_size,
            "path": relative,
            "sha256": _sha256_file(root / relative),
        }
        for relative in sorted(files)
    ]


def _expected_generated(
    root: Path, *, state: str, publication_files: set[str]
) -> tuple[bytes, bytes]:
    entries = _manifest_entries(root, publication_files)
    manifest = {
        "experiment_sha256": _sha256_file(root / EXPERIMENT_PATH),
        "files": entries,
        "schema": SCHEMA_TREE,
        "slot_registry_sha256": _sha256_file(root / SLOTS_PATH),
        "state": state,
        "tree_sha256": _sha256_bytes(_canonical_json(entries)),
    }
    checksums = "".join(
        f"{entry['sha256']}  {entry['path']}\n" for entry in entries
    ).encode("utf-8")
    return _pretty_json(manifest), checksums


def validate_tree(
    root: Path,
    *,
    allow_pending: bool,
    verify_generated: bool = True,
    slot_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    _validate_experiment(root, allow_pending=allow_pending)
    source_slots = (
        slot_document if slot_document is not None else _load_json(root / SLOTS_PATH)
    )
    _, slots = _validate_slot_doc(source_slots, allow_pending=allow_pending)
    actual_files, actual_directories = _walk_tree(root)
    expected_dynamic = _expected_dynamic_files(slots)
    expected_files = STATIC_FILES | set(expected_dynamic)
    allowed_files = expected_files | GENERATED_FILES
    unexpected_files = sorted(actual_files - allowed_files)
    missing_files = sorted(STATIC_FILES - actual_files)
    if unexpected_files:
        _fail(f"undeclared files: {', '.join(unexpected_files)}")
    if missing_files:
        _fail(f"missing static files: {', '.join(missing_files)}")
    unexpected_directories = sorted(actual_directories - _allowed_directories(slots))
    if unexpected_directories:
        _fail(f"undeclared directories: {', '.join(unexpected_directories)}")
    missing_directories = sorted(_required_directories() - actual_directories)
    if missing_directories:
        _fail(f"missing required directories: {', '.join(missing_directories)}")

    for relative, entry in expected_dynamic.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            _fail(f"inventoried file missing or unsafe: {relative}")
        if path.stat().st_size != entry["bytes"]:
            _fail(f"inventoried byte count changed: {relative}")
        if _sha256_file(path) != entry["sha256"]:
            _fail(f"inventoried digest changed: {relative}")

    for relative in sorted(expected_files):
        _scan_file(root / relative, relative)

    pending = any(slot["status"] == "pending" for slot in slots.values())
    unresolved_tree = any(
        record["tree"] is None
        for record in _load_json(root / EXPERIMENT_PATH)["roles"].values()
    )
    state = "staging" if pending or unresolved_tree else "publication"
    expected_manifest, expected_sums = _expected_generated(
        root, state=state, publication_files=expected_files
    )
    present_generated = GENERATED_FILES & actual_files
    if verify_generated:
        if present_generated and present_generated != GENERATED_FILES:
            _fail(
                "generated manifest and SHA256SUMS must either both exist or both be absent"
            )
        if present_generated:
            if (root / TREE_MANIFEST_PATH).read_bytes() != expected_manifest:
                _fail(f"stale or invalid generated manifest: {TREE_MANIFEST_PATH}")
            if (root / CHECKSUMS_PATH).read_bytes() != expected_sums:
                _fail(f"stale or invalid generated checksums: {CHECKSUMS_PATH}")
        elif not allow_pending:
            _fail("publication manifest and SHA256SUMS have not been generated")

    return {
        "files": len(expected_files),
        "slots": len(slots),
        "state": state,
        "tree_sha256": _sha256_bytes(
            _canonical_json(_manifest_entries(root, expected_files))
        ),
    }


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def generate(root: Path, *, allow_pending: bool) -> dict[str, Any]:
    result = validate_tree(
        root,
        allow_pending=allow_pending,
        verify_generated=False,
    )
    slots = _validate_slot_doc(
        _load_json(root / SLOTS_PATH), allow_pending=allow_pending
    )[1]
    files = STATIC_FILES | set(_expected_dynamic_files(slots))
    manifest, checksums = _expected_generated(
        root,
        state=result["state"],
        publication_files=files,
    )
    _atomic_write(root / TREE_MANIFEST_PATH, manifest)
    _atomic_write(root / CHECKSUMS_PATH, checksums)
    return validate_tree(root, allow_pending=allow_pending, verify_generated=True)


def _slot_files(root: Path, target: str) -> list[dict[str, Any]]:
    target_path = root / target
    if target_path.is_symlink() or not target_path.is_dir():
        _fail(f"slot target is not a real directory: {target}")
    entries: list[dict[str, Any]] = []
    for path in sorted(target_path.rglob("*")):
        relative = path.relative_to(target_path).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            _fail(f"slot target contains a symlink: {target}/{relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            _fail(f"slot target contains a non-regular file: {target}/{relative}")
        _safe_relative(relative, "slot file")
        if path.stat().st_size > MAX_FILE_BYTES:
            _fail(f"slot file exceeds publication size cap: {target}/{relative}")
        entries.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def inventory_slot(
    root: Path,
    slot_id: str,
    source_manifest: Path,
    *,
    replace: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    source_manifest = source_manifest.resolve(strict=True)
    if source_manifest.is_symlink() or not source_manifest.is_file():
        _fail("--source-manifest must name a real regular file")
    if source_manifest.stat().st_size > MAX_FILE_BYTES:
        _fail("--source-manifest exceeds the publication size cap")
    source_manifest_sha256 = _sha256_file(source_manifest)
    document = _load_json(root / SLOTS_PATH)
    _, slots = _validate_slot_doc(document, allow_pending=True)
    if slot_id not in slots:
        _fail(f"unknown slot: {slot_id}")
    slot = slots[slot_id]
    if slot["status"] == "ready" and not replace:
        _fail(f"slot {slot_id} is already ready; use --replace to re-inventory")
    entries = _slot_files(root, slot["target"])
    if not entries:
        _fail(f"slot {slot_id} target is empty")
    updated = json.loads(json.dumps(document))
    updated_slot = next(item for item in updated["slots"] if item["id"] == slot_id)
    updated_slot.update(
        {
            "files": entries,
            "source_manifest_sha256": source_manifest_sha256,
            "status": "ready",
            "tree_sha256": _slot_tree_sha256(entries),
        }
    )
    validate_tree(
        root,
        allow_pending=True,
        verify_generated=False,
        slot_document=updated,
    )
    _atomic_write(root / SLOTS_PATH, _pretty_json(updated))
    return updated_slot


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        _fail(f"git verification failed: {' '.join(arguments)}")
    output = completed.stdout.strip()
    if "\n" in output or "\r" in output:
        _fail("git verification returned multiple lines")
    return output


def set_role_tree(root: Path, role: str, repo: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    if role not in {"upstream", "pr-base"}:
        _fail("--role must be upstream or pr-base; pr-head is immutable")
    repo = repo.resolve(strict=True)
    if not repo.is_dir():
        _fail("--repo must name a directory")
    experiment = _load_json(root / EXPERIMENT_PATH)
    _validate_experiment(root, allow_pending=True)
    if experiment["roles"][role]["tree"] is not None:
        _fail(f"tree for {role} is already set")
    commit = EXPECTED_COMMITS[role]
    resolved_commit = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved_commit != commit:
        _fail(f"verified Git object does not resolve to the exact {role} commit")
    tree = _run_git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}")
    _sha1(tree, "verified Git tree")
    experiment["roles"][role]["tree"] = tree
    _atomic_write(root / EXPERIMENT_PATH, _pretty_json(experiment))
    _validate_experiment(root, allow_pending=True)
    return experiment["roles"][role]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="artifact root (defaults to this tool's parent tree)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate the artifact tree")
    check.add_argument("--allow-pending", action="store_true")

    generate_parser = subparsers.add_parser(
        "generate", help="generate and validate deterministic publication manifests"
    )
    generate_parser.add_argument("--allow-pending", action="store_true")

    inventory = subparsers.add_parser(
        "inventory-slot", help="machine-inventory one populated staging slot"
    )
    inventory.add_argument("--slot", required=True)
    inventory.add_argument("--source-manifest", required=True, type=Path)
    inventory.add_argument("--replace", action="store_true")

    role_tree = subparsers.add_parser(
        "set-role-tree", help="record a machine-verified unresolved Git tree"
    )
    role_tree.add_argument("--role", required=True, choices=("upstream", "pr-base"))
    role_tree.add_argument("--repo", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            result = validate_tree(args.root, allow_pending=args.allow_pending)
        elif args.command == "generate":
            result = generate(args.root, allow_pending=args.allow_pending)
        elif args.command == "inventory-slot":
            result = inventory_slot(
                args.root,
                args.slot,
                args.source_manifest,
                replace=args.replace,
            )
        elif args.command == "set-role-tree":
            result = set_role_tree(args.root, args.role, args.repo)
        else:  # pragma: no cover - argparse makes this unreachable
            _fail(f"unsupported command: {args.command}")
    except ArtifactError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
