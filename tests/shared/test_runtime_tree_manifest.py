# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TOOL = Path(__file__).with_name("capture_runtime_tree_manifest.py")
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def run_tool(
    *arguments: object,
    check: bool = True,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *(str(item) for item in arguments)],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def fake_hf_environment(tmp_path: Path, resolved_snapshot: Path) -> dict[str, str]:
    module_root = tmp_path / "fake-huggingface-hub"
    module_root.mkdir(exist_ok=True)
    (module_root / "huggingface_hub.py").write_text(
        "import os\n"
        "def snapshot_download(*, repo_id, revision, local_files_only):\n"
        "    assert repo_id and revision and local_files_only is True\n"
        "    return os.environ['FAKE_HF_RESOLVED_SNAPSHOT']\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(module_root)
    environment["FAKE_HF_RESOLVED_SNAPSHOT"] = str(resolved_snapshot)
    return environment


def test_canonical_transformers_capture_and_validation(tmp_path: Path) -> None:
    root = tmp_path / "transformers"
    (root / "nested").mkdir(parents=True)
    anchor = b'# package anchor\n__version__ = "fixture"\n'
    (root / "__init__.py").write_bytes(anchor)
    (root / "z.txt").write_bytes(b"z\n")
    (root / "nested" / "\N{LATIN SMALL LETTER E WITH ACUTE}.json").write_bytes(
        b'{"fixture":true}\n'
    )
    jsonl = tmp_path / "transformers.jsonl"
    summary = tmp_path / "transformers.summary.json"
    run_tool(
        "--kind",
        "transformers",
        "--root",
        root,
        "--anchor-sha256",
        hashlib.sha256(anchor).hexdigest(),
        "--output-jsonl",
        jsonl,
        "--output-summary",
        summary,
    )
    raw = jsonl.read_bytes()
    entries = [json.loads(line) for line in raw.splitlines()]
    assert [entry["path"] for entry in entries] == [
        "__init__.py",
        "nested/\N{LATIN SMALL LETTER E WITH ACUTE}.json",
        "z.txt",
    ]
    assert b"nested/\xc3\xa9.json" in raw
    assert b"\\u00e9" not in raw
    aggregate = json.loads(summary.read_text())["manifest"]
    assert aggregate["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert aggregate["regular_file_count"] == 3
    assert aggregate["logical_total_bytes"] == sum(entry["bytes"] for entry in entries)
    validation = run_tool(
        "--kind",
        "transformers",
        "--root",
        root,
        "--anchor-sha256",
        hashlib.sha256(anchor).hexdigest(),
        "--output-jsonl",
        jsonl,
        "--output-summary",
        summary,
        "--validate-existing",
    )
    assert json.loads(validation.stdout)["actual_resolved_bytes_rehashed"] is True


def test_hf_snapshot_hashes_actual_symlink_bytes(tmp_path: Path) -> None:
    snapshot = tmp_path / REVISION
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    blob = b"actual safetensors fixture bytes\x00\xff"
    (blobs / "blob-content-id").write_bytes(blob)
    (snapshot / "model.safetensors").symlink_to("../blobs/blob-content-id")
    (snapshot / "config.json").write_bytes(b"{}\n")
    jsonl = tmp_path / "hf.jsonl"
    summary = tmp_path / "hf.summary.json"
    arguments = (
        "--kind",
        "hf-snapshot",
        "--root",
        snapshot,
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--output-jsonl",
        jsonl,
        "--output-summary",
        summary,
    )
    environment = fake_hf_environment(tmp_path, snapshot)
    run_tool(*arguments, environment=environment)
    entries = {
        entry["path"]: entry
        for entry in (json.loads(line) for line in jsonl.read_bytes().splitlines())
    }
    safetensors = entries["model.safetensors"]
    assert safetensors["symlink_target_basename"] == "blob-content-id"
    assert safetensors["bytes"] == len(blob)
    assert safetensors["sha256"] == hashlib.sha256(blob).hexdigest()
    assert entries["config.json"]["symlink_target_basename"] is None
    run_tool(*arguments, "--validate-existing", environment=environment)
    summary_record = json.loads(summary.read_text())
    resolution = summary_record["root_contract"]["offline_resolution"]
    assert resolution["local_files_only"] is True
    assert Path(resolution["snapshot_download_resolved_root"]) == snapshot
    assert resolution["resolved_root_exact"] is True


def test_hf_explicit_root_rejects_another_offline_cache(tmp_path: Path) -> None:
    requested = tmp_path / "requested" / REVISION
    resolved_elsewhere = tmp_path / "other-cache" / REVISION
    requested.mkdir(parents=True)
    resolved_elsewhere.mkdir(parents=True)
    (requested / "config.json").write_text("{}\n")
    (resolved_elsewhere / "config.json").write_text("{}\n")
    completed = run_tool(
        "--kind",
        "hf-snapshot",
        "--root",
        requested,
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--output-jsonl",
        tmp_path / "wrong-cache.jsonl",
        "--output-summary",
        tmp_path / "wrong-cache.summary.json",
        environment=fake_hf_environment(tmp_path, resolved_elsewhere),
        check=False,
    )
    assert completed.returncode != 0
    assert "resolved a different cache tree" in completed.stderr


def test_validation_and_capture_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "transformers"
    root.mkdir()
    anchor = b"# exact package anchor\n"
    (root / "__init__.py").write_bytes(anchor)
    source = root / "source.py"
    source.write_bytes(b"before\n")
    jsonl = tmp_path / "tree.jsonl"
    summary = tmp_path / "tree.summary.json"
    arguments = (
        "--kind",
        "transformers",
        "--root",
        root,
        "--anchor-sha256",
        hashlib.sha256(anchor).hexdigest(),
        "--output-jsonl",
        jsonl,
        "--output-summary",
        summary,
    )
    run_tool(*arguments)
    assert run_tool(*arguments, check=False).returncode != 0
    source.write_bytes(b"after\n")
    assert run_tool(*arguments, "--validate-existing", check=False).returncode != 0
    source.write_bytes(b"before\n")
    run_tool(*arguments, "--validate-existing")
    with jsonl.open("ab") as stream:
        stream.write(b"\n")
    assert run_tool(*arguments, "--validate-existing", check=False).returncode != 0

    wrong_snapshot = tmp_path / "not-the-requested-revision"
    wrong_snapshot.mkdir()
    (wrong_snapshot / "config.json").write_bytes(b"{}\n")
    wrong_environment = fake_hf_environment(tmp_path, wrong_snapshot)
    assert (
        run_tool(
            "--kind",
            "hf-snapshot",
            "--root",
            wrong_snapshot,
            "--model",
            MODEL,
            "--revision",
            REVISION,
            "--output-jsonl",
            tmp_path / "wrong.jsonl",
            "--output-summary",
            tmp_path / "wrong.summary.json",
            environment=wrong_environment,
            check=False,
        ).returncode
        != 0
    )

    assert (
        run_tool(
            "--kind",
            "transformers",
            "--root",
            root,
            "--anchor-sha256",
            hashlib.sha256(anchor).hexdigest(),
            "--output-jsonl",
            tmp_path / "absent.jsonl",
            "--output-summary",
            tmp_path / "absent.summary.json",
            "--validate-existing",
            check=False,
        ).returncode
        != 0
    )


def test_transformers_root_contract_rejects_parent_and_wrong_anchor(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    package = overlay / "transformers"
    package.mkdir(parents=True)
    anchor = b"# exact imported package\n"
    (package / "__init__.py").write_bytes(anchor)
    anchor_sha256 = hashlib.sha256(anchor).hexdigest()

    parent_result = run_tool(
        "--kind",
        "transformers",
        "--root",
        overlay,
        "--anchor-sha256",
        anchor_sha256,
        "--output-jsonl",
        tmp_path / "wrong-parent.jsonl",
        "--output-summary",
        tmp_path / "wrong-parent.summary.json",
        check=False,
    )
    assert parent_result.returncode != 0
    assert "root basename mismatch" in parent_result.stderr

    wrong_anchor = run_tool(
        "--kind",
        "transformers",
        "--root",
        package,
        "--anchor-sha256",
        "0" * 64,
        "--output-jsonl",
        tmp_path / "wrong-anchor.jsonl",
        "--output-summary",
        tmp_path / "wrong-anchor.summary.json",
        check=False,
    )
    assert wrong_anchor.returncode != 0
    assert "anchor SHA256 mismatch" in wrong_anchor.stderr

    overlay_jsonl = tmp_path / "overlay.jsonl"
    overlay_summary = tmp_path / "overlay.summary.json"
    run_tool(
        "--kind",
        "transformers-overlay",
        "--root",
        overlay,
        "--expected-root-basename",
        "overlay",
        "--anchor-relative-path",
        "transformers/__init__.py",
        "--anchor-sha256",
        anchor_sha256,
        "--output-jsonl",
        overlay_jsonl,
        "--output-summary",
        overlay_summary,
    )
    overlay_contract = json.loads(overlay_summary.read_text())["root_contract"]
    assert overlay_contract == {
        "type": "basename-and-anchor-sha256",
        "expected_root_basename": "overlay",
        "anchor_relative_path": "transformers/__init__.py",
        "anchor_sha256": anchor_sha256,
    }
