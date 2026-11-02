"""Safe appraisal-head safetensors custody and independent-process readback."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None


def _fixture():
    import torch
    from pleasure_experiment.open_weight_adapter import OpenWeightBinding
    from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
    from pleasure_experiment.open_weight_local_model import LocalModelManifest

    class ToyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)
            self.output = torch.nn.Linear(2, 8)
            with torch.no_grad():
                self.embedding.weight.zero_()
                self.embedding.weight[3] = torch.tensor([1.0, 0.0])
                self.embedding.weight[4] = torch.tensor([0.0, 1.0])
                self.output.weight.zero_()
                self.output.bias.zero_()
                self.output.bias[3] = 0.02

        def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
            hidden = self.embedding(input_ids)
            return SimpleNamespace(
                logits=self.output(hidden),
                hidden_states=(hidden,) if output_hidden_states else None,
            )

    model = ToyLM()
    probe = TorchHiddenAppraisal(
        model, action_token_ids=(3, 4), hidden_size=2,
        learning_rate=1.0, allow_training=True,
        max_delta_l2=1.0,
    )
    binding = OpenWeightBinding(
        model_id="tiny/stub", model_revision="abc123", runtime_instance_id="source-pid",
        checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
    )
    snapshot = LocalModelManifest(
        snapshot_directory="/tmp/verified-by-caller", files=(),
        checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
        snapshot_sha256="c" * 64, total_bytes=100,
    )
    return torch, model, probe, binding, snapshot


@unittest.skipUnless(HAS_TORCH and HAS_SAFETENSORS, "optional torch/safetensors missing")
class AppraisalPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.location = Path(self.tmp.name) / "appraisal-v1"
        self.torch, self.model, self.probe, self.binding, self.snapshot = _fixture()

    def _train(self):
        for i in range(8):
            choice = self.probe.score((1,), partition="TRAIN")
            self.probe.observe_chosen(
                choice, measured_quality=1.0,
                evidence_id=sha256(f"trained-chosen-{i}".encode()).hexdigest(),
            )

    def _save(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            save_versioned_appraisal_head,
        )
        return save_versioned_appraisal_head(
            self.probe, binding=self.binding, snapshot_manifest=self.snapshot,
            destination_directory=self.location, allow_persistent_export=True,
            expected_head_sha256=self.probe.head_digest,
        )

    def test_unbound_training_preview_prevents_persistence(self):
        preview = self.probe.score((1,), partition="TRAIN", defer_choice=True)
        with self.assertRaisesRegex(ValueError, "pending"):
            self._save()
        self.assertFalse(self.location.exists())
        self.probe.abandon_unbound(preview)

    def _fresh(self, *, runtime="fresh-pid"):
        from pleasure_experiment.open_weight_adapter import OpenWeightBinding
        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
        torch, model, _, _, _ = _fixture()
        probe = TorchHiddenAppraisal(
            model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=False,
        )
        binding = OpenWeightBinding(
            model_id=self.binding.model_id,
            model_revision=self.binding.model_revision,
            runtime_instance_id=runtime,
            checkpoint_sha256=self.binding.checkpoint_sha256,
            tokenizer_sha256=self.binding.tokenizer_sha256,
        )
        return torch, model, probe, binding

    def test_fresh_head_import_keeps_base_exact_and_changes_fixed_eval(self):
        from pleasure_experiment.open_weight_adapter import _named_digest
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )

        self._train()
        expected = self.probe.score((1,), gain=10).prob_a
        saved = self._save()
        torch, model, fresh, binding = self._fresh()
        initial = fresh.score((1,), gain=10).prob_a
        model_before = _named_digest(torch, tuple(model.named_parameters()))
        with self.assertRaises(PermissionError):
            load_versioned_appraisal_head(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location,
            )
        loaded = load_versioned_appraisal_head(
            fresh, binding=binding, snapshot_manifest=self.snapshot,
            source_directory=self.location, allow_in_place_mutation=True,
        )
        self.assertEqual(fresh.head_digest, saved.head_weights_sha256)
        self.assertEqual(loaded.loaded_runtime_instance_id, "fresh-pid")
        self.assertEqual(loaded.binding.runtime_instance_id, "source-pid")
        self.assertEqual(fresh.score((1,), gain=10).prob_a, expected)
        self.assertNotEqual(initial, expected)
        self.assertEqual(_named_digest(torch, tuple(model.named_parameters())), model_before)
        self.assertEqual(set(x.name for x in self.location.iterdir()),
                         {"head.safetensors", "manifest.json"})

    def test_calibrated_head_schema_is_persisted_and_rejected_by_legacy(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import (
            FEATURE_SCHEMA as CALIBRATED_SCHEMA, TorchCalibratedAppraisal,
        )
        from pleasure_experiment.open_weight_appraisal_persistence import (
            FEATURE_SCHEMA as LEGACY_SCHEMA,
            load_versioned_appraisal_head, save_versioned_appraisal_head,
        )

        calibrated = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.1, allow_training=True,
        )
        decision = calibrated.score((1,), partition="TRAIN", defer_choice=True)
        calibrated.observe_chosen(
            calibrated.bind_choice(decision, "A"), measured_quality=0.95,
            evidence_id=sha256(b"calibrated-persistence-training").hexdigest(),
        )
        saved = save_versioned_appraisal_head(
            calibrated, binding=self.binding, snapshot_manifest=self.snapshot,
            destination_directory=self.location, allow_persistent_export=True,
        )
        self.assertEqual(json.loads((self.location / "manifest.json").read_text())
                         ["feature_schema"], CALIBRATED_SCHEMA)
        self.assertNotEqual(CALIBRATED_SCHEMA, LEGACY_SCHEMA)

        _, fresh_model, _, fresh_binding = self._fresh()
        fresh = TorchCalibratedAppraisal(
            fresh_model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=False,
        )
        imported = load_versioned_appraisal_head(
            fresh, binding=fresh_binding, snapshot_manifest=self.snapshot,
            source_directory=self.location, allow_in_place_mutation=True,
        )
        self.assertEqual(imported.head_weights_sha256, saved.head_weights_sha256)
        self.assertEqual(fresh.head_digest, calibrated.head_digest)
        self.assertEqual(fresh.score((1,), gain=4).prob_a,
                         calibrated.score((1,), gain=4).prob_a)

        _, _, legacy, legacy_binding = self._fresh()
        legacy_before = legacy.head_digest
        with self.assertRaisesRegex(ValueError, "feature schema"):
            load_versioned_appraisal_head(
                legacy, binding=legacy_binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )
        self.assertEqual(legacy.head_digest, legacy_before)

    def test_legacy_head_schema_is_rejected_by_calibrated_target(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        from pleasure_experiment.open_weight_appraisal_persistence import load_versioned_appraisal_head

        self._train()
        self._save()
        _, model, _, binding = self._fresh()
        calibrated = TorchCalibratedAppraisal(
            model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=False,
        )
        before = calibrated.head_digest
        with self.assertRaisesRegex(ValueError, "feature schema"):
            load_versioned_appraisal_head(
                calibrated, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )
        self.assertEqual(calibrated.head_digest, before)

    def test_strict_export_gates_and_no_overwrite(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            save_versioned_appraisal_head,
        )
        with self.assertRaises(PermissionError):
            save_versioned_appraisal_head(
                self.probe, binding=self.binding,
                snapshot_manifest=self.snapshot, destination_directory=self.location,
            )
        self._save()
        with self.assertRaises(FileExistsError):
            self._save()
        with self.assertRaises(ValueError):
            save_versioned_appraisal_head(
                self.probe, binding=self.binding, snapshot_manifest=self.snapshot,
                destination_directory=Path(self.tmp.name) / "invalid",
                expected_head_sha256="0" * 64, allow_persistent_export=True,
            )

    def test_refuses_head_storage_aliased_to_frozen_model(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            save_versioned_appraisal_head,
        )
        # A malicious caller could attach the appraisal's weight tensor to a
        # frozen-model parameter; loading it must never write through aliasing.
        with self.torch.no_grad():
            self.probe._head.weight = self.torch.nn.Parameter(
                self.model.embedding.weight[:1], requires_grad=True,
            )
        with self.assertRaisesRegex(ValueError, "shares tensor storage"):
            save_versioned_appraisal_head(
                self.probe, binding=self.binding, snapshot_manifest=self.snapshot,
                destination_directory=self.location, allow_persistent_export=True,
            )

    def test_import_rejects_wrong_frozen_model_without_mutation(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )
        self._train()
        self._save()
        torch, model, fresh, binding = self._fresh()
        before = fresh.head_digest
        with torch.no_grad():
            model.output.bias[3].add_(0.1)
        with self.assertRaisesRegex(ValueError, "fresh model differs"):
            load_versioned_appraisal_head(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )
        self.assertEqual(fresh.head_digest, before)

    def test_rejects_linked_source_entries(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )
        self._save()
        _, _, fresh, binding = self._fresh()
        head = self.location / "head.safetensors"
        raw = head.read_bytes()
        copy = Path(self.tmp.name) / "data-copy.safetensors"
        copy.write_bytes(raw)
        head.unlink()
        try:
            head.symlink_to(copy)
        except (OSError, NotImplementedError):
            self.skipTest("host cannot create local symlinks")
        with self.assertRaises(ValueError):
            load_versioned_appraisal_head(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )

    def test_corrupt_extra_symlink_and_manifest_identity_are_rejected(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )
        self._train()
        self._save()
        _, _, fresh, binding = self._fresh()
        initial = fresh.head_digest

        marker = self.location / "surprise.txt"
        marker.write_text("unexpected", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_versioned_appraisal_head(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )
        marker.unlink()
        manifest_path = self.location / "manifest.json"
        original = manifest_path.read_bytes()
        for field, wrong in (
            ("feature_schema", "different-hidden-extraction"),
            ("action_token_ids", [4, 3]),
            ("hidden_size", 3),
            ("head_weights_sha256", "0" * 64),
        ):
            payload = json.loads(original)
            payload[field] = wrong
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_versioned_appraisal_head(
                    fresh, binding=binding, snapshot_manifest=self.snapshot,
                    source_directory=self.location, allow_in_place_mutation=True,
                )
            self.assertEqual(initial, fresh.head_digest)
        manifest_path.write_bytes(original)
        with (self.location / "head.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaises(ValueError):
            load_versioned_appraisal_head(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.location, allow_in_place_mutation=True,
            )
        self.assertEqual(initial, fresh.head_digest)

    def test_late_file_readback_failure_rolls_back_after_successful_copy(self):
        from pleasure_experiment.open_weight_appraisal_persistence import (
            load_versioned_appraisal_head,
        )
        from pleasure_experiment import open_weight_appraisal_persistence as persistence

        self._train()
        self._save()
        _, _, fresh, binding = self._fresh()
        original_digest = fresh.head_digest
        original_sha = persistence._sha_file
        calls = []

        def fail_late(*args, **kwargs):
            calls.append(1)
            if len(calls) == 4:
                raise RuntimeError("injected postcopy file readback failure")
            return original_sha(*args, **kwargs)

        with mock.patch.object(persistence, "_sha_file", side_effect=fail_late):
            with self.assertRaisesRegex(RuntimeError, "postcopy"):
                load_versioned_appraisal_head(
                    fresh, binding=binding, snapshot_manifest=self.snapshot,
                    source_directory=self.location, allow_in_place_mutation=True,
                )
        self.assertEqual(fresh.head_digest, original_digest)

    def test_fresh_python_process_imports_head_and_reproduces_probability(self):
        self._train()
        saved = self._save()
        expected = self.probe.score((1,), gain=10).prob_a
        # Entire new interpreter, newly allocated head and model. Only the
        # versioned safetensors path is transferred, never a Python object.
        script = r'''
import json, os, sys
from tests.test_open_weight_appraisal_persistence import _fixture
from pleasure_experiment.open_weight_adapter import OpenWeightBinding, _named_digest
from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal
from pleasure_experiment.open_weight_appraisal_persistence import load_versioned_appraisal_head
torch, model, _, binding, snapshot = _fixture()
fresh = TorchHiddenAppraisal(model, action_token_ids=(3, 4), hidden_size=2,
                            learning_rate=0.0, allow_training=False)
model_before = _named_digest(torch, tuple(model.named_parameters()))
new_binding = OpenWeightBinding(model_id=binding.model_id,
    model_revision=binding.model_revision, runtime_instance_id="independent-process-pid",
    checkpoint_sha256=binding.checkpoint_sha256,
    tokenizer_sha256=binding.tokenizer_sha256)
receipt = load_versioned_appraisal_head(fresh, binding=new_binding,
    snapshot_manifest=snapshot, source_directory=sys.argv[1], allow_in_place_mutation=True)
print(json.dumps({"pid": os.getpid(), "head": fresh.head_digest,
                  "source_runtime": receipt.binding.runtime_instance_id,
                  "fresh_runtime": receipt.loaded_runtime_instance_id,
                  "prob_a": fresh.score((1,), gain=10).prob_a,
                  "base_unchanged": _named_digest(torch, tuple(model.named_parameters())) == model_before}))
'''
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.location)],
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env, text=True, capture_output=True, check=True, timeout=60,
        )
        report = json.loads(result.stdout)
        self.assertNotEqual(report["pid"], os.getpid())
        self.assertEqual(report["head"], saved.head_weights_sha256)
        self.assertEqual(report["prob_a"], expected)
        self.assertTrue(report["base_unchanged"])
        self.assertEqual(report["fresh_runtime"], "independent-process-pid")


if __name__ == "__main__":
    unittest.main()
