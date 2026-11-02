"""Source-only safety checks for the fresh-process saved LoRA verifier."""

from __future__ import annotations

from unittest import TestCase

from scripts.verify_saved_open_weight_adapter import build_parser, run


class SavedOpenWeightReloadCLITests(TestCase):
    def _args(self, *extra):
        return build_parser().parse_args([
            "--snapshot", "/tmp/snapshot-does-not-exist",
            "--adapter-dir", "/tmp/adapter-does-not-exist",
            "--model-id", "offline-test-model",
            "--revision", "a" * 40,
            *extra,
        ])

    def test_explicit_permission_required_before_any_model_load(self):
        with self.assertRaisesRegex(PermissionError, "allow-in-place-reload"):
            run(self._args())

    def test_wrong_revision_rejected_before_local_model_load(self):
        with self.assertRaisesRegex(ValueError, "40-hex"):
            run(self._args("--allow-in-place-reload", "--revision", "branch-main"))

    def test_tiny_or_duplicate_invalid_seed_rejected_before_local_model_load(self):
        with self.assertRaisesRegex(ValueError, "uint32"):
            run(self._args("--allow-in-place-reload", "--seed", "-1"))
        with self.assertRaisesRegex(ValueError, "heldout-pairs"):
            run(self._args("--allow-in-place-reload", "--heldout-pairs", "0"))
