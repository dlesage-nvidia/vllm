# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "pynv_public_tree", SOURCE_ROOT / "tools" / "public_tree.py"
)
assert SPEC is not None and SPEC.loader is not None
public_tree = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(public_tree)


class PublicTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "artifact"
        shutil.copytree(
            SOURCE_ROOT,
            self.root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        slot_path = self.root / public_tree.SLOTS_PATH
        slot_document = json.loads(slot_path.read_text(encoding="utf-8"))
        for slot in slot_document["slots"]:
            target = self.root / str(slot["target"])
            for child in target.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            slot.update(
                {
                    "files": [],
                    "source_manifest_sha256": None,
                    "status": "pending",
                    "tree_sha256": None,
                }
            )
        slot_path.write_bytes(public_tree._pretty_json(slot_document))
        for generated in public_tree.GENERATED_FILES:
            (self.root / generated).unlink(missing_ok=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _populate_slot(self, slot: dict[str, object]) -> None:
        target = self.root / str(slot["target"])
        kind = slot["kind"]
        if kind == "frozen-harness":
            (target / "runner.py").write_text("print('frozen harness')\n")
        elif kind == "frozen-tests":
            (target / "test_contract.py").write_text("def test_contract(): pass\n")
        elif kind == "freeze-manifests":
            (target / "ARTIFACT_MANIFEST.json").write_text("{}\n")
        elif kind == "runtime-manifests":
            (target / "runtime.summary.json").write_text("{}\n")
            (target / "runtime.jsonl").write_text("{}\n")
        elif kind == "exact-commands":
            (target / "commands.json").write_text("{}\n")
        elif kind == "sanitized-results":
            for name in (
                "collection-summary.json",
                "provenance.json",
                "public-manifest.json",
                "token-parity.json",
            ):
                (target / name).write_text("{}\n")
            with (target / "requests.jsonl.gz").open("wb") as stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                    compressed.write(b"{}\n")
        else:  # pragma: no cover
            self.fail(f"unhandled fixture slot kind: {kind}")

    def _source_manifest(self, name: str) -> Path:
        path = Path(self.temporary.name) / f"source-{name}.json"
        path.write_text(json.dumps({"source": name}) + "\n")
        return path

    def _make_ready(self) -> None:
        slots = json.loads(
            (self.root / public_tree.SLOTS_PATH).read_text(encoding="utf-8")
        )["slots"]
        for index, slot in enumerate(slots, start=1):
            self._populate_slot(slot)
            public_tree.inventory_slot(
                self.root,
                str(slot["id"]),
                self._source_manifest(f"{index:02d}"),
            )

    def test_staging_passes_only_when_pending_is_explicitly_allowed(self) -> None:
        result = public_tree.validate_tree(self.root, allow_pending=True)
        self.assertEqual(result["state"], "staging")
        with self.assertRaisesRegex(public_tree.ArtifactError, "inventory is pending"):
            public_tree.validate_tree(self.root, allow_pending=False)

    def test_root_git_metadata_is_excluded_but_must_be_a_real_directory(self) -> None:
        metadata = self.root / ".git"
        (metadata / "objects").mkdir(parents=True)
        (metadata / "objects" / "private-machine-state").write_text("ignored\n")
        self.assertEqual(
            public_tree.validate_tree(self.root, allow_pending=True)["state"],
            "staging",
        )
        shutil.rmtree(metadata)
        metadata.symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "hidden path|symlink|Git metadata"
        ):
            public_tree.validate_tree(self.root, allow_pending=True)

    def test_semantic_roles_and_exact_head_identity_are_frozen(self) -> None:
        experiment_path = self.root / public_tree.EXPERIMENT_PATH
        experiment = json.loads(experiment_path.read_text())
        self.assertEqual(set(experiment["roles"]), {"upstream", "pr-base", "pr-head"})
        self.assertEqual(
            experiment["roles"]["pr-head"],
            {
                "commit": "30d917599b104423e452fa718890af01c4ff4d39",
                "tree": "66c4849eb21973b9ca391b7b0911968f4aa63dac",
            },
        )
        experiment["roles"]["pr-head"]["tree"] = "0" * 40
        experiment_path.write_text(json.dumps(experiment))
        with self.assertRaisesRegex(public_tree.ArtifactError, "pr-head tree changed"):
            public_tree.validate_tree(self.root, allow_pending=True)

    def test_inventory_is_computed_and_detects_later_mutation(self) -> None:
        target = self.root / "harness/shared"
        payload = target / "runner.py"
        payload.write_text("print('one')\n")
        source_manifest = self._source_manifest("shared-harness")
        slot = public_tree.inventory_slot(self.root, "shared-harness", source_manifest)
        self.assertEqual(slot["status"], "ready")
        self.assertEqual(slot["files"][0]["bytes"], len(payload.read_bytes()))
        self.assertEqual(slot["files"][0]["sha256"], public_tree._sha256_file(payload))
        self.assertEqual(
            slot["source_manifest_sha256"],
            public_tree._sha256_file(source_manifest),
        )
        payload.write_text("print('two')\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "digest changed"):
            public_tree.validate_tree(self.root, allow_pending=True)

    def test_complete_tree_generates_deterministic_manifest(self) -> None:
        self._make_ready()
        first = public_tree.generate(self.root, allow_pending=False)
        manifest = (self.root / public_tree.TREE_MANIFEST_PATH).read_bytes()
        sums = (self.root / public_tree.CHECKSUMS_PATH).read_bytes()
        second = public_tree.generate(self.root, allow_pending=False)
        self.assertEqual(first, second)
        self.assertEqual(
            manifest, (self.root / public_tree.TREE_MANIFEST_PATH).read_bytes()
        )
        self.assertEqual(sums, (self.root / public_tree.CHECKSUMS_PATH).read_bytes())
        self.assertEqual(
            public_tree.validate_tree(self.root, allow_pending=False)["state"],
            "publication",
        )

    def test_generated_manifest_becomes_stale_after_any_change(self) -> None:
        self._make_ready()
        public_tree.generate(self.root, allow_pending=False)
        (self.root / "harness/shared/runner.py").write_text("print('changed')\n")
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "byte count changed|digest changed"
        ):
            public_tree.validate_tree(self.root, allow_pending=False)

    def test_undeclared_file_symlink_and_raw_log_fail_closed(self) -> None:
        extra = self.root / "surprise.txt"
        extra.write_text("unexpected\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "undeclared files"):
            public_tree.validate_tree(self.root, allow_pending=True)
        extra.unlink()

        link = self.root / "commands/rtx/link"
        link.symlink_to(self.root / "README.md")
        with self.assertRaisesRegex(public_tree.ArtifactError, "symlink"):
            public_tree.validate_tree(self.root, allow_pending=True)
        link.unlink()

        raw_log = self.root / "commands/rtx/server.log"
        raw_log.write_text("raw\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "undeclared files"):
            public_tree.validate_tree(self.root, allow_pending=True)
        (self.root / "commands/rtx/commands.json").write_text("{}\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "forbidden file type"):
            public_tree.inventory_slot(
                self.root, "rtx-commands", self._source_manifest("raw-log")
            )

    def test_missing_required_slot_directory_is_rejected(self) -> None:
        (self.root / "harness/a100").rmdir()
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "missing required directories"
        ):
            public_tree.validate_tree(self.root, allow_pending=True)

    def test_private_path_and_secret_are_rejected_before_inventory_write(self) -> None:
        target = self.root / "commands/rtx/commands.json"
        private_path = "/" + "home/private/run"
        target.write_text(json.dumps({"command": private_path}) + "\n")
        before = (self.root / public_tree.SLOTS_PATH).read_bytes()
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "private absolute POSIX"
        ):
            public_tree.inventory_slot(
                self.root, "rtx-commands", self._source_manifest("rtx-commands")
            )
        self.assertEqual(before, (self.root / public_tree.SLOTS_PATH).read_bytes())

        target.write_text(json.dumps({"token": "hf" + "_" + "A" * 24}) + "\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "Hugging Face token"):
            public_tree.inventory_slot(
                self.root, "rtx-commands", self._source_manifest("rtx-commands-2")
            )

    def test_only_frozen_generic_harness_paths_are_allowlisted(self) -> None:
        target = self.root / "harness/shared/runner.py"
        generic_root = "/" + "data/"
        target.write_text(f"{generic_root}clip-0.mp4 {generic_root}clip-1.mp4\n")
        public_tree._scan_file(target, "harness/shared/runner.py")

        target.write_text(f"{generic_root}private-clip.mp4\n")
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "private absolute POSIX"
        ):
            public_tree._scan_file(target, "harness/shared/runner.py")

    def test_only_exact_synthetic_guard_paths_are_allowlisted(self) -> None:
        target = self.root / "tests/shared/test_refined_gpu_guards.py"
        synthetic_root = "/" + "tmp/"
        target.write_text(
            f'f"{synthetic_root}{{tagged_controller}}"\n'
            f'"{synthetic_root}status.py"\n'
            f'"{synthetic_root}unrelated"\n'
            f'"{synthetic_root}unexpected.py"\n'
            f'"{synthetic_root}unrelated.py"\n'
            f'"{synthetic_root}not-nvidia-cuda-mps-server-wrapper"\n'
        )
        public_tree._scan_file(target, "tests/shared/test_refined_gpu_guards.py")

        target.write_text(f'"{synthetic_root}private-process.py"\n')
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "private absolute POSIX"
        ):
            public_tree._scan_file(target, "tests/shared/test_refined_gpu_guards.py")

    def test_only_exact_a100_private_root_negative_assertion_is_allowlisted(
        self,
    ) -> None:
        target = self.root / "tests/a100/test_persistent_three_arm_campaign.py"
        home_root = "/" + "home/"
        temporary_root = "/" + "tmp/"
        target.write_text(
            f'assert "{home_root}" not in contract_bytes and '
            f'"{temporary_root}" not in contract_bytes\n'
        )
        public_tree._scan_file(
            target, "tests/a100/test_persistent_three_arm_campaign.py"
        )

        target.write_text(f'leaked = "{home_root}private/run"\n')
        with self.assertRaisesRegex(
            public_tree.ArtifactError, "private absolute POSIX"
        ):
            public_tree._scan_file(
                target, "tests/a100/test_persistent_three_arm_campaign.py"
            )

    def test_unc_scanner_rejects_paths_without_matching_escaped_newlines(self) -> None:
        target = self.root / "commands/rtx/commands.txt"
        backslash = chr(92)
        target.write_text(f"value={backslash}{backslash}n') {backslash}n\n")
        public_tree._scan_file(target, "commands/rtx/commands.txt")

        target.write_text(
            backslash * 2 + "private-server" + backslash + "private-share\n"
        )
        with self.assertRaisesRegex(public_tree.ArtifactError, "absolute UNC"):
            public_tree._scan_file(target, "commands/rtx/commands.txt")

    def test_unknown_slot_and_registry_field_are_rejected(self) -> None:
        with self.assertRaisesRegex(public_tree.ArtifactError, "unknown slot"):
            public_tree.inventory_slot(
                self.root, "unknown", self._source_manifest("unknown")
            )
        path = self.root / public_tree.SLOTS_PATH
        value = json.loads(path.read_text())
        value["future"] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(public_tree.ArtifactError, "unknown keys"):
            public_tree.validate_tree(self.root, allow_pending=True)

    def test_unsafe_json_metadata_key_is_rejected(self) -> None:
        target = self.root / "commands/rtx/commands.json"
        target.write_text(json.dumps({"hostname": "private-machine"}) + "\n")
        with self.assertRaisesRegex(public_tree.ArtifactError, "JSON key"):
            public_tree.inventory_slot(
                self.root, "rtx-commands", self._source_manifest("unsafe-key")
            )

        for key in ("captured_utc", "started_monotonic_time_ns"):
            with self.assertRaisesRegex(public_tree.ArtifactError, "JSON key"):
                public_tree._scan_json_keys({key: "private-time"}, "public")

    def test_contract_descriptions_do_not_disable_raw_key_guard(self) -> None:
        public_tree._scan_json_keys(
            {
                "evidence_contract": {"raw_response": "validated then discarded"},
                "provenance_contract": {"server_log": "hash-bound only"},
            },
            "freeze-manifest",
        )
        with self.assertRaisesRegex(public_tree.ArtifactError, "raw JSON key"):
            public_tree._scan_json_keys(
                {"raw_response": "generated caption"}, "public-result"
            )
        with self.assertRaisesRegex(public_tree.ArtifactError, "raw JSON key"):
            public_tree._scan_json_keys(
                {"evidence_contract": {"raw_response": {"content": "caption"}}},
                "freeze-manifest",
            )


if __name__ == "__main__":
    unittest.main()
