# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping

TRANSFORMERS_PACKAGE_BASENAME = "transformers"
TRANSFORMERS_ANCHOR = Path("__init__.py")


def resolve_hf_snapshot(
    *, model: str, revision: str, explicit_root: Path | None
) -> tuple[Path, dict[str, Any]]:
    from huggingface_hub import snapshot_download

    resolved_download = (
        Path(
            snapshot_download(
                repo_id=model,
                revision=revision,
                local_files_only=True,
            )
        )
        .expanduser()
        .resolve(strict=True)
    )
    if explicit_root is not None:
        validated_root = explicit_root.expanduser().resolve(strict=True)
        if resolved_download != validated_root:
            raise RuntimeError(
                "offline snapshot_download resolved a different cache tree than "
                f"--root: {resolved_download} != {validated_root}"
            )
    else:
        validated_root = resolved_download
    return validated_root, {
        "model": model,
        "revision": revision,
        "local_files_only": True,
        "snapshot_download_resolved_root": str(resolved_download),
        "explicit_root": None if explicit_root is None else str(validated_root),
        "resolved_root_exact": resolved_download == validated_root,
    }


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def publish_exclusive(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def stable_file_sha256(path: Path) -> tuple[int, str, tuple[int, ...]]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"resolved path is not a regular file: {path}")
        sha256 = sha256_stream(stream)
        after = os.fstat(stream.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"file changed while hashing: {path}")
    return before.st_size, sha256, before_identity


def stable_file_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"evidence path is not a regular file: {path}")
        value = stream.read()
        after = os.fstat(stream.fileno())
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"evidence file changed while reading: {path}")
    return value


def regular_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            if not path.is_file():
                raise RuntimeError(f"broken or non-file symlink: {path}")
            files.append(path)
            continue
        raise RuntimeError(f"special filesystem entry is not permitted: {path}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def capture(root: Path, *, kind: str) -> tuple[bytes, dict[str, Any]]:
    cached_content: dict[Path, tuple[int, str, tuple[int, ...]]] = {}
    encoded_entries: list[bytes] = []
    total_bytes = 0
    symlink_count = 0
    for path in regular_files(root):
        relative_path = path.relative_to(root).as_posix()
        logical_before = path.lstat()
        symlink_target_basename = None
        symlink_target = None
        if path.is_symlink():
            symlink_count += 1
            symlink_target = os.readlink(path)
            symlink_target_basename = Path(symlink_target).name
        resolved = path.resolve(strict=True)
        cached = cached_content.get(resolved)
        current_stat = resolved.stat()
        current_identity = (
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_size,
            current_stat.st_mtime_ns,
        )
        if cached is None or cached[2] != current_identity:
            cached = stable_file_sha256(resolved)
            cached_content[resolved] = cached
        size, sha256, unused_identity = cached
        del unused_identity
        logical_after = path.lstat()
        logical_before_identity = (
            logical_before.st_dev,
            logical_before.st_ino,
            logical_before.st_mode,
            logical_before.st_size,
            logical_before.st_mtime_ns,
        )
        logical_after_identity = (
            logical_after.st_dev,
            logical_after.st_ino,
            logical_after.st_mode,
            logical_after.st_size,
            logical_after.st_mtime_ns,
        )
        if logical_before_identity != logical_after_identity:
            raise RuntimeError(f"logical path changed while hashing: {path}")
        if symlink_target is not None and os.readlink(path) != symlink_target:
            raise RuntimeError(f"symlink changed while hashing: {path}")
        if path.resolve(strict=True) != resolved:
            raise RuntimeError(f"resolved path changed while hashing: {path}")
        entry: dict[str, Any] = {
            "bytes": size,
            "path": relative_path,
            "sha256": sha256,
        }
        if kind == "hf-snapshot":
            entry["symlink_target_basename"] = symlink_target_basename
        encoded_entries.append(canonical_json_bytes(entry))
        total_bytes += size
    manifest = b"".join(encoded_entries)
    return manifest, {
        "regular_file_count": len(encoded_entries),
        "logical_total_bytes": total_bytes,
        "unique_resolved_file_count": len(cached_content),
        "symlink_count": symlink_count,
        "manifest_bytes": len(manifest),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def build_summary(
    *,
    root: Path,
    kind: str,
    model: str | None,
    revision: str | None,
    output_jsonl: Path,
    aggregate: Mapping[str, Any],
    captured_utc: str,
    root_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "vllm-runtime-tree-manifest-v3",
        "status": "passed",
        "kind": kind,
        "captured_utc": captured_utc,
        "root": str(root),
        "resolved_root_basename": root.name,
        "model": model,
        "revision": revision,
        "root_contract": dict(root_contract),
        "entry_schema": (
            ["bytes", "path", "sha256", "symlink_target_basename"]
            if kind == "hf-snapshot"
            else ["bytes", "path", "sha256"]
        ),
        "method": (
            "sorted POSIX-relative snapshot path; SHA256 of actual resolved bytes "
            "for every regular file, including every safetensors symlink target; "
            "canonical compact sorted-key UTF-8 JSON plus newline per entry"
            if kind == "hf-snapshot"
            else "sorted POSIX-relative regular-file path; SHA256 of actual bytes; "
            "canonical compact sorted-key UTF-8 JSON plus newline per entry"
        ),
        "manifest": {"path": str(output_jsonl), **aggregate},
    }


def validate_existing(
    *,
    root: Path,
    kind: str,
    model: str | None,
    revision: str | None,
    output_jsonl: Path,
    output_summary: Path,
    root_contract: Mapping[str, Any],
) -> dict[str, Any]:
    for path in (output_jsonl, output_summary):
        if not path.exists():
            raise FileNotFoundError(path)
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"evidence path must be a direct regular file: {path}")
    stored_manifest = stable_file_bytes(output_jsonl)
    stored_summary_bytes = stable_file_bytes(output_summary)
    stored_summary = json.loads(stored_summary_bytes)
    if not isinstance(stored_summary, dict):
        raise RuntimeError("summary must be a JSON object")
    if canonical_json_bytes(stored_summary) != stored_summary_bytes:
        raise RuntimeError("summary is not canonical compact sorted-key JSON")
    captured_utc = stored_summary.get("captured_utc")
    if not isinstance(captured_utc, str):
        raise RuntimeError("summary lacks a string captured_utc")
    captured_time = datetime.fromisoformat(captured_utc)
    if captured_time.tzinfo is None or captured_time.utcoffset() is None:
        raise RuntimeError("captured_utc must include a UTC offset")
    derived_manifest, aggregate = capture(root, kind=kind)
    expected_summary = build_summary(
        root=root,
        kind=kind,
        model=model,
        revision=revision,
        output_jsonl=output_jsonl,
        aggregate=aggregate,
        captured_utc=captured_utc,
        root_contract=root_contract,
    )
    if stored_summary != expected_summary:
        raise RuntimeError("stored summary does not match the derived tree manifest")
    if stored_manifest != derived_manifest:
        raise RuntimeError("stored JSONL does not match the current resolved bytes")
    return {
        "status": "passed",
        "kind": kind,
        "root": str(root),
        "manifest_path": str(output_jsonl),
        "manifest_sha256": aggregate["manifest_sha256"],
        "regular_file_count": aggregate["regular_file_count"],
        "logical_total_bytes": aggregate["logical_total_bytes"],
        "summary_path": str(output_summary),
        "summary_sha256": hashlib.sha256(stored_summary_bytes).hexdigest(),
        "actual_resolved_bytes_rehashed": True,
    }


def parse_sha256(value: str, *, option: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{option} must be exactly 64 hexadecimal characters")
    return normalized


def validate_root_contract(
    *,
    root: Path,
    kind: str,
    expected_root_basename: str | None,
    anchor_relative_path: str | None,
    anchor_sha256: str | None,
) -> dict[str, Any]:
    if kind == "transformers":
        if expected_root_basename not in (None, TRANSFORMERS_PACKAGE_BASENAME):
            raise ValueError(
                "kind=transformers has the fixed expected root basename "
                f"{TRANSFORMERS_PACKAGE_BASENAME!r}"
            )
        expected_root_basename = TRANSFORMERS_PACKAGE_BASENAME
        if anchor_relative_path not in (None, TRANSFORMERS_ANCHOR.as_posix()):
            raise ValueError(
                "kind=transformers has the fixed anchor relative path "
                f"{TRANSFORMERS_ANCHOR.as_posix()!r}"
            )
        anchor_relative_path = TRANSFORMERS_ANCHOR.as_posix()
        if anchor_sha256 is None:
            raise ValueError("--anchor-sha256 is required for kind=transformers")
    elif kind == "transformers-overlay":
        if expected_root_basename is None:
            raise ValueError(
                "--expected-root-basename is required for kind=transformers-overlay"
            )
        if anchor_relative_path is None or anchor_sha256 is None:
            raise ValueError(
                "--anchor-relative-path and --anchor-sha256 are required for "
                "kind=transformers-overlay"
            )
    elif any(
        value is not None
        for value in (expected_root_basename, anchor_relative_path, anchor_sha256)
    ):
        raise ValueError(
            "root basename and anchor options only apply to transformers manifests"
        )
    else:
        return {"type": "hf-snapshot-revision-basename"}

    assert expected_root_basename is not None
    assert anchor_relative_path is not None
    assert anchor_sha256 is not None
    if root.name != expected_root_basename:
        raise RuntimeError(
            f"resolved root basename mismatch: {root.name!r} != "
            f"{expected_root_basename!r}"
        )
    relative_anchor = Path(anchor_relative_path)
    if (
        relative_anchor.is_absolute()
        or relative_anchor.as_posix() in ("", ".")
        or ".." in relative_anchor.parts
    ):
        raise ValueError("anchor relative path must stay within the manifest root")
    anchor = root / relative_anchor
    try:
        anchor_mode = anchor.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"required manifest anchor is absent: {anchor}") from error
    if not stat.S_ISREG(anchor_mode):
        raise RuntimeError(
            f"manifest anchor must be a direct regular file, not a symlink: {anchor}"
        )
    expected_anchor_sha256 = parse_sha256(anchor_sha256, option="--anchor-sha256")
    unused_size, actual_anchor_sha256, unused_identity = stable_file_sha256(anchor)
    del unused_size, unused_identity
    if actual_anchor_sha256 != expected_anchor_sha256:
        raise RuntimeError(
            f"manifest anchor SHA256 mismatch: {actual_anchor_sha256} != "
            f"{expected_anchor_sha256}"
        )
    return {
        "type": "basename-and-anchor-sha256",
        "expected_root_basename": expected_root_basename,
        "anchor_relative_path": relative_anchor.as_posix(),
        "anchor_sha256": expected_anchor_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("transformers", "transformers-overlay", "hf-snapshot"),
        required=True,
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--expected-root-basename")
    parser.add_argument("--anchor-relative-path")
    parser.add_argument("--anchor-sha256")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="rehash the source tree and validate existing evidence without writing",
    )
    args = parser.parse_args()
    hf_resolution: dict[str, Any] | None = None
    if args.kind == "hf-snapshot":
        if not args.model or not args.revision:
            parser.error("--model and --revision are required for hf-snapshot")
        args.root, hf_resolution = resolve_hf_snapshot(
            model=args.model,
            revision=args.revision,
            explicit_root=args.root,
        )
    elif args.root is None:
        parser.error("--root is required for transformers")
    assert args.root is not None
    root = args.root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    root_contract = validate_root_contract(
        root=root,
        kind=args.kind,
        expected_root_basename=args.expected_root_basename,
        anchor_relative_path=args.anchor_relative_path,
        anchor_sha256=args.anchor_sha256,
    )
    if hf_resolution is not None:
        root_contract = {**root_contract, "offline_resolution": hf_resolution}
    if (
        args.kind == "hf-snapshot"
        and args.revision is not None
        and len(args.revision) == 40
        and all(character in "0123456789abcdef" for character in args.revision)
        and root.name != args.revision
    ):
        raise RuntimeError(
            "resolved Hugging Face snapshot directory does not match the exact "
            f"requested commit: {root.name} != {args.revision}"
        )
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_summary = args.output_summary.expanduser().resolve()
    if output_jsonl == output_summary:
        raise ValueError("manifest and summary outputs must differ")
    if args.validate_existing:
        report = validate_existing(
            root=root,
            kind=args.kind,
            model=args.model,
            revision=args.revision,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            root_contract=root_contract,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if output_jsonl.exists() or output_summary.exists():
        raise FileExistsError("refusing to overwrite runtime manifest evidence")
    manifest, aggregate = capture(root, kind=args.kind)
    summary = build_summary(
        root=root,
        kind=args.kind,
        model=args.model,
        revision=args.revision,
        output_jsonl=output_jsonl,
        aggregate=aggregate,
        captured_utc=datetime.now(timezone.utc).isoformat(),
        root_contract=root_contract,
    )
    publish_exclusive(output_jsonl, manifest)
    publish_exclusive(output_summary, canonical_json_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
