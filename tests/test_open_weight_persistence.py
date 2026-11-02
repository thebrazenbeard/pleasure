"""Actual safetensors LoRA persistence, identity, and tampering regressions."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pleasure_experiment.open_weight_adapter import OpenWeightBinding
from pleasure_experiment.open_weight_local_model import (
    LocalModelManifest, PreparedLocalLoRA,
)
from pleasure_experiment.open_weight_persistence import (
    FORMAT_VERSION, load_versioned_lora_adapter, save_versioned_lora_adapter,
)

try:
    import torch
    import safetensors.torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "optional PyTorch+safetensors unavailable")
class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.directory = Path(self.work.name)
        self.target = self.directory / "adapter-v001"
        self.binding = OpenWeightBinding(
            model_id="tiny-lora", model_revision="a" * 40,
            runtime_instance_id="offline-probe",
            checkpoint_sha256="b" * 64, tokenizer_sha256="c" * 64,
        )
        self.snapshot = LocalModelManifest(
            snapshot_directory=str(self.directory), files=(),
            checkpoint_sha256=self.binding.checkpoint_sha256,
            tokenizer_sha256=self.binding.tokenizer_sha256,
            snapshot_sha256="d" * 64, total_bytes=0,
        )

    def _prepared(self, *, runtime_instance_id=None):
        torch.manual_seed(17)
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2, bias=False)
        model.block = torch.nn.Module()
        model.block.lora_A = torch.nn.ModuleDict({
            "default": torch.nn.Linear(2, 2, bias=False),
        })
        model.block.lora_B = torch.nn.ModuleDict({
            "default": torch.nn.Linear(2, 2, bias=False),
        })
        names = ("block.lora_A.default.weight", "block.lora_B.default.weight")
        for name, param in model.named_parameters():
            param.requires_grad_(name in names)
        return PreparedLocalLoRA(
            model=model, binding=(
                self.binding if runtime_instance_id is None else
                replace(self.binding, runtime_instance_id=runtime_instance_id)
            ), action_token_ids=(7, 8),
            trainable_parameter_names=names, rank=2, alpha=4,
        )

    def _save(self, prepared):
        return save_versioned_lora_adapter(
            prepared, snapshot_manifest=self.snapshot,
            destination_directory=self.target,
            allow_persistent_export=True,
        )

    def test_versioned_export_and_independent_model_reload(self):
        trained = self._prepared()
        fresh = self._prepared(runtime_instance_id="fresh-model-process-002")
        with torch.no_grad():
            trained.model.block.lora_A["default"].weight.add_(0.2)
            trained.model.block.lora_B["default"].weight.sub_(0.3)
        before = fresh.model.block.lora_A["default"].weight.detach().clone()
        unchanged_backbone = fresh.model.backbone.weight.detach().clone()
        exported = self._save(trained)
        self.assertEqual(exported.format_version, FORMAT_VERSION)
        self.assertEqual(exported.tensor_count, 2)
        self.assertNotEqual(exported.trainable_weights_sha256, "0" * 64)
        self.assertEqual({p.name for p in self.target.iterdir()},
                         {"adapter.safetensors", "manifest.json"})
        with self.assertRaises(PermissionError):
            load_versioned_lora_adapter(
                fresh, snapshot_manifest=self.snapshot,
                source_directory=self.target,
            )
        reloaded = load_versioned_lora_adapter(
            fresh, snapshot_manifest=self.snapshot,
            source_directory=self.target, allow_in_place_mutation=True,
        )
        self.assertEqual(reloaded.trainable_weights_sha256,
                         exported.trainable_weights_sha256)
        self.assertEqual(reloaded.binding.runtime_instance_id, "offline-probe")
        self.assertEqual(reloaded.loaded_runtime_instance_id,
                         "fresh-model-process-002")
        self.assertFalse(torch.equal(
            before, fresh.model.block.lora_A["default"].weight,
        ))
        self.assertTrue(torch.equal(
            trained.model.block.lora_A["default"].weight,
            fresh.model.block.lora_A["default"].weight,
        ))
        self.assertTrue(torch.equal(
            trained.model.block.lora_B["default"].weight,
            fresh.model.block.lora_B["default"].weight,
        ))
        self.assertTrue(torch.equal(
            unchanged_backbone, fresh.model.backbone.weight,
        ))
        self.assertFalse(reloaded.externally_authenticated)
        self.assertFalse(reloaded.hosted_model_modified)

    def test_export_refuses_overwrite_and_unapproved_persistence(self):
        prepared = self._prepared()
        with self.assertRaises(PermissionError):
            save_versioned_lora_adapter(
                prepared, snapshot_manifest=self.snapshot,
                destination_directory=self.target,
            )
        self.assertFalse(self.target.exists())
        with self.assertRaisesRegex(ValueError, "learned model tensor readback"):
            save_versioned_lora_adapter(
                prepared, snapshot_manifest=self.snapshot,
                destination_directory=self.target,
                expected_trainable_sha256="0" * 64,
                allow_persistent_export=True,
            )
        self.assertFalse(self.target.exists())
        self._save(prepared)
        with self.assertRaises(FileExistsError):
            self._save(prepared)
        with self.assertRaises(ValueError):
            save_versioned_lora_adapter(
                prepared, snapshot_manifest=self.snapshot,
                destination_directory=self.directory / ".." / "escape",
                allow_persistent_export=True,
            )

    def test_rejects_mismatched_model_and_snapshot_without_mutation(self):
        trained = self._prepared()
        self._save(trained)
        fresh = self._prepared()
        baseline = fresh.model.block.lora_A["default"].weight.detach().clone()
        mismatched = replace(
            fresh, binding=replace(fresh.binding, model_id="other-model"),
        )
        with self.assertRaises(ValueError):
            load_versioned_lora_adapter(
                mismatched, snapshot_manifest=self.snapshot,
                source_directory=self.target, allow_in_place_mutation=True,
            )
        self.assertTrue(torch.equal(
            baseline, fresh.model.block.lora_A["default"].weight,
        ))
        with self.assertRaises(ValueError):
            load_versioned_lora_adapter(
                fresh, snapshot_manifest=replace(
                    self.snapshot, snapshot_sha256="e" * 64,
                ),
                source_directory=self.target, allow_in_place_mutation=True,
            )

    def test_rejects_corrupt_safetensors_and_unknown_files(self):
        self._save(self._prepared())
        target = self._prepared()
        weight_before = target.model.block.lora_A["default"].weight.detach().clone()
        with (self.target / "adapter.safetensors").open("ab") as file:
            file.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "differs from"):
            load_versioned_lora_adapter(
                target, snapshot_manifest=self.snapshot,
                source_directory=self.target, allow_in_place_mutation=True,
            )
        self.assertTrue(torch.equal(weight_before,
                                    target.model.block.lora_A["default"].weight))
        (self.target / "unknown.py").write_text("pass\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only two"):
            load_versioned_lora_adapter(
                target, snapshot_manifest=self.snapshot,
                source_directory=self.target, allow_in_place_mutation=True,
            )

    def test_rejects_manifest_tampering_and_wrong_tensor_metadata(self):
        self._save(self._prepared())
        target = self._prepared()
        baseline = target.model.block.lora_A["default"].weight.detach().clone()
        manifest = self.target / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["tensors"][0]["shape"] = [999]
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            load_versioned_lora_adapter(
                target, snapshot_manifest=self.snapshot,
                source_directory=self.target, allow_in_place_mutation=True,
            )
        self.assertTrue(torch.equal(baseline,
                                    target.model.block.lora_A["default"].weight))
        data["tensors"][0]["shape"] = [2, 2]
        data["format_version"] = 99
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "format version"):
            load_versioned_lora_adapter(
                target, snapshot_manifest=self.snapshot,
                source_directory=self.target, allow_in_place_mutation=True,
            )

    def test_rejects_directory_links(self):
        self._save(self._prepared())
        linked = self.directory / "adapter-link"
        try:
            linked.symlink_to(self.target, target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("filesystem does not permit creating symlinks")
        with self.assertRaisesRegex(ValueError, "link or junction"):
            load_versioned_lora_adapter(
                self._prepared(), snapshot_manifest=self.snapshot,
                source_directory=linked, allow_in_place_mutation=True,
            )

    def test_rejects_trainable_storage_aliased_to_frozen_backbone(self):
        prepared = self._prepared()
        shared = prepared.model.block.lora_A["default"].weight
        prepared.model.backbone.weight = torch.nn.Parameter(
            shared.detach(), requires_grad=False,
        )
        with self.assertRaisesRegex(ValueError, "shares storage"):
            self._save(prepared)
        self.assertFalse(self.target.exists())

    def test_late_file_change_aborts_and_restores_exact_live_parameters(self):
        self._save(self._prepared())
        prepared = self._prepared()
        baseline = tuple((name, p.detach().clone()) for name, p in
                         prepared.model.named_parameters())
        import pleasure_experiment.open_weight_persistence as persistence

        original = persistence._sha_file
        calls = 0

        def changing_file(path, *, max_bytes):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated external file change after copy")
            return original(path, max_bytes=max_bytes)

        with patch.object(persistence, "_sha_file", side_effect=changing_file):
            with self.assertRaisesRegex(RuntimeError, "file change"):
                load_versioned_lora_adapter(
                    prepared, snapshot_manifest=self.snapshot,
                    source_directory=self.target, allow_in_place_mutation=True,
                )
        for name, param in prepared.model.named_parameters():
            self.assertTrue(torch.equal(param, dict(baseline)[name]))


if __name__ == "__main__":
    unittest.main()
