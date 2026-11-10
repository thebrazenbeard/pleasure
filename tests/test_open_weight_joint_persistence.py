"""Safe joint PEFT LoRA/head persistence, replay ledger, and restart integrity."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


HAS_OPTIONAL = all(importlib.util.find_spec(name) is not None
                   for name in ("torch", "peft", "transformers", "safetensors"))

PINNED_SNAPSHOT = Path("/tmp/pleasure-smollm2-safe-v3")
PINNED_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
PINNED_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


@unittest.skipUnless(HAS_OPTIONAL, "optional local Torch/PEFT/safetensors unavailable")
class JointPersistenceTests(unittest.TestCase):
    def setUp(self):
        from tests.test_open_weight_joint_appraisal import _toy_joint
        self.torch, self.prepared, self.probe = _toy_joint()
        self.binding = self.prepared.binding
        from pleasure_experiment.open_weight_local_model import LocalModelManifest
        self.snapshot = LocalModelManifest(
            snapshot_directory="/tmp/toy-not-a-real-checkpoint", files=(),
            checkpoint_sha256=self.binding.checkpoint_sha256,
            tokenizer_sha256=self.binding.tokenizer_sha256,
            snapshot_sha256="c" * 64, total_bytes=100,
        )
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.directory = Path(self.work.name)
        self.path = self.directory / "joint-lora-head-v001"

    @staticmethod
    def _update(probe, i, *, quality=0.8):
        preview = probe.score((1, 2, 5 + i % 5), partition="TRAIN",
                              defer_choice=True, episode_id=f"toy-episode-{i}")
        bound = probe.bind_choice(preview, "A" if i % 2 == 0 else "B")
        return probe.observe_chosen(
            bound, measured_quality=quality,
            evidence_id=sha256(f"persisted-chosen-{i}".encode()).hexdigest(),
        )

    def _fresh(self, *, runtime_id="fresh-runtime", train_lora=True):
        from tests.test_open_weight_joint_appraisal import _toy_joint
        _, prepared, probe = _toy_joint(train_lora=train_lora)
        source = replace(prepared, binding=replace(prepared.binding,
                                                   runtime_instance_id=runtime_id))
        probe._prepared = source
        return probe, source.binding

    def _save(self, *, probe=None, path=None):
        from pleasure_experiment.open_weight_joint_persistence import (
            save_versioned_joint_appraisal,
        )
        current = probe if probe is not None else self.probe
        return save_versioned_joint_appraisal(
            current, binding=current._prepared.binding,
            snapshot_manifest=self.snapshot,
            destination_directory=self.path if path is None else path,
            expected_joint_sha256=current.joint_digest,
            allow_persistent_export=True,
        )

    def _load(self, probe, binding, *, path=None):
        from pleasure_experiment.open_weight_joint_persistence import (
            load_versioned_joint_appraisal,
        )
        return load_versioned_joint_appraisal(
            probe, binding=binding, snapshot_manifest=self.snapshot,
            source_directory=self.path if path is None else path,
            allow_in_place_mutation=True,
        )

    def test_train_save_new_model_reload_continue_and_resave(self):
        before = self.probe.joint_digest
        frozen = self.probe.base_digest
        self._update(self.probe, 0)
        self.assertNotEqual(before, self.probe.joint_digest)
        saved = self._save()
        self.assertEqual(saved.updates_at_export, 1)
        self.assertEqual(set(path.name for path in self.path.iterdir()),
                         {"weights.safetensors", "manifest.json"})
        manifest = json.loads((self.path / "manifest.json").read_text())
        self.assertEqual(manifest["optimizer_state_format"],
                         "stateless_sgd_v1_no_momentum")
        self.assertEqual(manifest["optimizer_spec"]["train_lora"], True)
        self.assertEqual(len(manifest["seen_evidence_ids"]), 1)

        fresh, fresh_binding = self._fresh()
        self.assertEqual(fresh.base_digest, frozen)
        initial_state = fresh.joint_digest
        artifact = self._load(fresh, fresh_binding)
        self.assertEqual(fresh.joint_digest, saved.joint_weights_sha256)
        self.assertEqual(fresh.head_digest, saved.head_weights_sha256)
        self.assertEqual(fresh.lora_digest, saved.lora_weights_sha256)
        self.assertEqual(fresh.updates, 1)
        self.assertEqual(artifact.loaded_runtime_instance_id, "fresh-runtime")
        self.assertNotEqual(initial_state, fresh.joint_digest)
        self.assertEqual(fresh.export_training_state(), self.probe.export_training_state())
        for learner in (self.probe, fresh):
            prediction = learner.score((1, 2, 5), gain=12.0,
                                       partition="EVAL", episode_id="matching-eval")
            self.assertTrue(0 < prediction.prob_a < 1)
            if learner is self.probe:
                expected = prediction
            else:
                self.assertEqual(prediction.prob_a, expected.prob_a)
                self.assertEqual(prediction.predicted_quality_a,
                                 expected.predicted_quality_a)

        # Replaying a consumed TRAIN episode is forbidden after process reload.
        with self.assertRaisesRegex(ValueError, "replay"):
            fresh.score((1, 2, 5), partition="TRAIN", defer_choice=True,
                        episode_id="toy-episode-0")
        second = self._update(fresh, 1)
        self.assertEqual(second.update_index, 2)
        self.assertNotEqual(fresh.joint_digest, saved.joint_weights_sha256)
        self.assertEqual(fresh.base_digest, frozen)
        next_path = self.directory / "joint-lora-head-v002"
        newer = self._save(probe=fresh, path=next_path)
        self.assertEqual(newer.updates_at_export, 2)
        self.assertNotEqual(newer.joint_weights_sha256, saved.joint_weights_sha256)
        third, binding3 = self._fresh(runtime_id="third-runtime")
        self._load(third, binding3, path=next_path)
        self.assertEqual(third.updates, 2)
        self.assertEqual(third.joint_digest, newer.joint_weights_sha256)
        with self.assertRaisesRegex(ValueError, "replay"):
            third.score((1, 2, 6), partition="TRAIN", defer_choice=True,
                        episode_id="toy-episode-1")

    def test_permission_and_no_overwrite_and_safe_destination(self):
        from pleasure_experiment.open_weight_joint_persistence import (
            save_versioned_joint_appraisal, load_versioned_joint_appraisal,
        )
        with self.assertRaises(PermissionError):
            save_versioned_joint_appraisal(
                self.probe, binding=self.binding,
                snapshot_manifest=self.snapshot, destination_directory=self.path,
            )
        self.assertFalse(self.path.exists())
        self._save()
        with self.assertRaises(FileExistsError):
            self._save()
        fresh, binding = self._fresh()
        with self.assertRaises(PermissionError):
            load_versioned_joint_appraisal(
                fresh, binding=binding, snapshot_manifest=self.snapshot,
                source_directory=self.path,
            )
        with self.assertRaises(ValueError):
            self._save(path=self.directory / ".." / "escape")

    def test_head_only_control_and_wrong_training_recipe_fail_closed(self):
        from tests.test_open_weight_joint_appraisal import _toy_joint
        _, head_only_prepared, head_only = _toy_joint(train_lora=False)
        location = self.directory / "head-only-control"
        self._save(probe=head_only, path=location)
        saved = json.loads((location / "manifest.json").read_text())
        self.assertEqual(saved["optimizer_spec"]["train_lora"], False)
        joint, binding = self._fresh(train_lora=True)
        original = joint.joint_digest
        with self.assertRaisesRegex(ValueError, "optimizer_spec"):
            self._load(joint, binding, path=location)
        self.assertEqual(joint.joint_digest, original)

    def test_import_rejects_rollback_of_live_trained_ledger(self):
        self._update(self.probe, 0)
        self._save()
        self._update(self.probe, 1)
        current_sha = self.probe.joint_digest
        current_ledger = self.probe.export_training_state()
        with self.assertRaisesRegex(ValueError, "pristine TRAIN ledger"):
            self._load(self.probe, self.binding)
        self.assertEqual(self.probe.joint_digest, current_sha)
        self.assertEqual(self.probe.export_training_state(), current_ledger)

    def test_wrong_model_snapshot_and_manifest_schema_rejected(self):
        self._save()
        fresh, binding = self._fresh()
        original = fresh.joint_digest
        from pleasure_experiment.open_weight_joint_persistence import (
            load_versioned_joint_appraisal,
        )
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            load_versioned_joint_appraisal(
                fresh, binding=binding,
                snapshot_manifest=replace(self.snapshot, snapshot_sha256="d" * 64),
                source_directory=self.path, allow_in_place_mutation=True,
            )
        manifest = self.path / "manifest.json"
        data = json.loads(manifest.read_text())
        for field, wrong in (
            ("feature_schema", "wrong-feature-transform"),
            ("head_weights_sha256", "0" * 64),
            ("lora_config", {"rank": 7, "alpha": 4,
                             "target_modules": ["c_attn"]}),
            ("action_token_ids", [4, 3]),
        ):
            altered = dict(data)
            altered[field] = wrong
            manifest.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises((ValueError, RuntimeError)):
                self._load(fresh, binding)
            self.assertEqual(fresh.joint_digest, original)
        manifest.write_text(json.dumps(data), encoding="utf-8")

    def test_corrupt_weights_unknown_entry_and_symlink_rejected(self):
        self._save()
        fresh, binding = self._fresh()
        original = fresh.joint_digest
        weights = self.path / "weights.safetensors"
        with weights.open("ab") as out:
            out.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "safetensors file digest mismatch"):
            self._load(fresh, binding)
        self.assertEqual(fresh.joint_digest, original)
        (self.path / "unknown.py").write_text("not allowed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self._load(fresh, binding)

    def test_replay_ledger_digest_tampering_rejected_before_import(self):
        self._update(self.probe, 0)
        self._save()
        fresh, binding = self._fresh()
        original = fresh.joint_digest
        manifest = self.path / "manifest.json"
        data = json.loads(manifest.read_text())
        data["seen_evidence_ids"][0] = "0" * 64
        manifest.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "replay ledger"):
            self._load(fresh, binding)
        self.assertEqual(fresh.joint_digest, original)

    def test_pending_training_prevents_export_and_no_artifact_left(self):
        preview = self.probe.score(
            (1, 2, 5), partition="TRAIN", defer_choice=True,
            episode_id="pending-before-export",
        )
        with self.assertRaisesRegex(ValueError, "pending"):
            self._save()
        self.assertFalse(self.path.exists())
        self.probe.abandon_unbound(preview)

    def test_atomic_publish_conflict_cleans_private_staging(self):
        from pleasure_experiment import open_weight_joint_persistence as persistence
        with mock.patch.object(persistence, "_atomic_no_replace",
                               side_effect=FileExistsError("injected concurrent target")):
            with self.assertRaisesRegex(FileExistsError, "concurrent target"):
                self._save()
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_changed_pretrained_base_rejects_import_before_mutation(self):
        self._save()
        fresh, binding = self._fresh()
        with self.torch.no_grad():
            name, parameter = next(
                (name, parameter) for name, parameter in
                fresh._prepared.model.named_parameters()
                if name not in fresh._prepared.trainable_parameter_names
            )
            parameter.view(-1)[0].add_(0.01)
        with self.assertRaisesRegex(RuntimeError, "frozen pretrained"):
            self._load(fresh, binding)

    def test_failure_after_import_rolls_back_tensors_and_ledger(self):
        from pleasure_experiment import open_weight_joint_persistence as persistence
        self._update(self.probe, 0)
        self._save()
        fresh, binding = self._fresh()
        before = fresh.joint_digest
        state = fresh.export_training_state()
        original = persistence._sha_file
        calls = []

        def faulty(*args, **kwargs):
            calls.append(1)
            # manifest upfront, safetensors upfront, then weights/manifest
            # after successful mutation; reject late post-copy readback.
            if len(calls) == 4:
                raise RuntimeError("injected postcopy failure")
            return original(*args, **kwargs)

        with mock.patch.object(persistence, "_sha_file", side_effect=faulty):
            with self.assertRaisesRegex(RuntimeError, "postcopy"):
                self._load(fresh, binding)
        self.assertEqual(fresh.joint_digest, before)
        self.assertEqual(fresh.export_training_state(), state)

    def test_separate_python_process_reloads_updates_and_resaves(self):
        self._update(self.probe, 0)
        initial = self._save()
        continuation = self.directory / "joint-lora-head-fresh-process-v002"
        source_root = str(Path(__file__).resolve().parents[1])
        worker = r'''
import json, sys
from dataclasses import replace
from hashlib import sha256
from tests.test_open_weight_joint_appraisal import _toy_joint
from pleasure_experiment.open_weight_local_model import LocalModelManifest
from pleasure_experiment.open_weight_joint_persistence import (load_versioned_joint_appraisal, save_versioned_joint_appraisal)
_, prepared, fresh = _toy_joint()
binding = replace(prepared.binding, runtime_instance_id="independent-child-process")
fresh._prepared = replace(prepared, binding=binding)
snapshot = LocalModelManifest(snapshot_directory="/tmp/toy-not-a-real-checkpoint", files=(), checkpoint_sha256=binding.checkpoint_sha256, tokenizer_sha256=binding.tokenizer_sha256, snapshot_sha256="c"*64, total_bytes=100)
receipt = load_versioned_joint_appraisal(fresh, binding=binding, snapshot_manifest=snapshot, source_directory=sys.argv[1], allow_in_place_mutation=True)
restored = fresh.joint_digest
replay_rejected = False
try:
    fresh.score((1,2,5), partition="TRAIN", defer_choice=True, episode_id="toy-episode-0")
except ValueError:
    replay_rejected = True
decision = fresh.score((1,2,6), partition="TRAIN", defer_choice=True, episode_id="toy-episode-1")
bound = fresh.bind_choice(decision,"B")
update = fresh.observe_chosen(bound, measured_quality=.7, evidence_id=sha256(b"continued-new-genuine-chosen-fixture").hexdigest())
new = save_versioned_joint_appraisal(fresh, binding=binding, snapshot_manifest=snapshot, destination_directory=sys.argv[2], allow_persistent_export=True)
print(json.dumps({"origin_runtime":receipt.binding.runtime_instance_id,"fresh_runtime":receipt.loaded_runtime_instance_id,"restored_sha":restored,"new_sha":new.joint_weights_sha256,"updates":new.updates_at_export,"new_update_index":update.update_index,"old_episode_rejected":replay_rejected}))
'''
        command = [sys.executable, "-I", "-B", "-c",
                   "import sys; sys.path.insert(0, " + repr(source_root) + "); " + worker,
                   str(self.path), str(continuation)]
        process = subprocess.run(command, cwd=source_root,
                                 capture_output=True, text=True, timeout=90)
        self.assertEqual(process.returncode, 0, process.stderr[-1000:])
        result = json.loads(process.stdout)
        self.assertEqual(result["origin_runtime"], "tiny-runtime")
        self.assertEqual(result["fresh_runtime"], "independent-child-process")
        self.assertEqual(result["restored_sha"], initial.joint_weights_sha256)
        self.assertNotEqual(result["new_sha"], result["restored_sha"])
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["new_update_index"], 2)
        self.assertTrue(result["old_episode_rejected"])
        self.assertTrue(continuation.is_dir())


@unittest.skipUnless(
    HAS_OPTIONAL and os.environ.get("PLEASURE_RUN_REAL_CPU_JOINT_PERSISTENCE_SMOKE") == "1"
    and PINNED_SNAPSHOT.is_dir(),
    "requires explicit opt-in, installed Torch/PEFT, and local pinned SmolLM2",
)
class RealPinnedJointPersistenceSmoke(unittest.TestCase):
    """TRAIN-only real 135M witness; never opens a held-out payoff generator."""

    def test_one_update_save_independent_process_continue_and_resave(self):
        import torch
        from pleasure_experiment.open_weight_local_model import (
            load_verified_local_causal_model, prepare_explicit_lora,
        )
        from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal
        from pleasure_experiment.open_weight_joint_persistence import (
            save_versioned_joint_appraisal,
        )

        torch.set_num_threads(2)
        loaded = load_verified_local_causal_model(
            PINNED_SNAPSHOT, model_id=PINNED_MODEL_ID,
            model_revision=PINNED_REVISION,
            runtime_instance_id="pinned-persistence-train-process",
        )
        prepared = prepare_explicit_lora(
            loaded, target_modules=("q_proj", "v_proj"), rank=2,
            alpha=4, max_trainable_parameters=1_000_000,
        )
        probe = TorchJointLoRAAppraisal(
            prepared, hidden_size=prepared.model.config.hidden_size,
            learning_rate=0.02, allow_training=True, max_updates=4,
        )
        prompt_ids = tuple(loaded.tokenizer.encode(
            "Two options have uncertain benefits. A: 0.14 and 0.82. "
            "B: 0.77 and 0.33. Choose the better one: ",
            add_special_tokens=False,
        ))
        frozen = probe.base_digest
        lora_before, head_before = probe.lora_digest, probe.head_digest
        preview = probe.score(prompt_ids, partition="TRAIN", defer_choice=True,
                              episode_id="pinned-real-episode-0")
        bound = probe.bind_choice(preview, "A")
        update = probe.observe_chosen(
            bound, measured_quality=0.8,
            evidence_id=sha256(b"pinned-real-TRAIN-receipt-0").hexdigest(),
        )
        self.assertEqual(update.update_index, 1)
        self.assertNotEqual(probe.lora_digest, lora_before)
        self.assertNotEqual(probe.head_digest, head_before)
        self.assertEqual(probe.base_digest, frozen)
        observed = probe.score(
            prompt_ids, partition="EVAL", episode_id="pinned-no-payoff-check",
        )
        with tempfile.TemporaryDirectory(prefix="pinned-joint-persistence-") as root:
            first_path = Path(root) / "version-001"
            second_path = Path(root) / "version-002"
            receipt = save_versioned_joint_appraisal(
                probe, binding=prepared.binding,
                snapshot_manifest=loaded.manifest,
                destination_directory=first_path,
                expected_joint_sha256=probe.joint_digest,
                allow_persistent_export=True,
            )
            worker = r'''
import json
import sys
from hashlib import sha256
import torch
from pleasure_experiment.open_weight_local_model import load_verified_local_causal_model, prepare_explicit_lora
from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal
from pleasure_experiment.open_weight_joint_persistence import load_versioned_joint_appraisal, save_versioned_joint_appraisal

torch.set_num_threads(2)
loaded = load_verified_local_causal_model(
    sys.argv[1], model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    model_revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    runtime_instance_id="pinned-persistence-independent-import",
)
prepared = prepare_explicit_lora(
    loaded, target_modules=("q_proj", "v_proj"), rank=2, alpha=4,
    max_trainable_parameters=1000000,
)
probe = TorchJointLoRAAppraisal(
    prepared, hidden_size=prepared.model.config.hidden_size,
    learning_rate=.02, allow_training=True, max_updates=4,
)
frozen_sha = probe.base_digest
receipt = load_versioned_joint_appraisal(
    probe, binding=prepared.binding, snapshot_manifest=loaded.manifest,
    source_directory=sys.argv[2], allow_in_place_mutation=True,
)
ids = tuple(json.loads(sys.argv[4]))
check = probe.score(ids, partition="EVAL", episode_id="independent-no-payoff")
assert check.prob_a == float(sys.argv[5]), ("evaluation probability drift",check.prob_a,sys.argv[5])
assert probe.joint_digest == sys.argv[6]
assert probe.head_digest == sys.argv[7]
assert probe.lora_digest == sys.argv[8]
replay_blocked = False
try:
    probe.score(ids, partition="TRAIN", defer_choice=True, episode_id="pinned-real-episode-0")
except ValueError:
    replay_blocked = True
assert replay_blocked
preview = probe.score(ids, partition="TRAIN", defer_choice=True, episode_id="pinned-real-episode-1")
bound = probe.bind_choice(preview, "B")
update = probe.observe_chosen(
    bound, measured_quality=.7,
    evidence_id=sha256(b"pinned-real-TRAIN-receipt-1").hexdigest(),
)
assert update.update_index == 2
assert probe.base_digest == frozen_sha
new = save_versioned_joint_appraisal(
    probe, binding=prepared.binding, snapshot_manifest=loaded.manifest,
    destination_directory=sys.argv[3], allow_persistent_export=True,
)
print("PINNED_JOINT_CHILD="+json.dumps({
    "first_joint_sha":receipt.joint_weights_sha256,
    "next_joint_sha":new.joint_weights_sha256,
    "first_updates":receipt.updates_at_export,
    "next_updates":new.updates_at_export,
    "replay_blocked":replay_blocked,
    "eval_prob_a":check.prob_a,
    "frozen_sha":probe.base_digest,
    "initial_runtime":receipt.binding.runtime_instance_id,
    "loaded_runtime":receipt.loaded_runtime_instance_id,
},sort_keys=True))
'''
            source_root = str(Path(__file__).resolve().parents[1])
            command = [
                sys.executable, "-I", "-B", "-c",
                "import sys; sys.path.insert(0, " + repr(source_root) + "); " + worker,
                str(PINNED_SNAPSHOT), str(first_path), str(second_path),
                json.dumps(prompt_ids), repr(observed.prob_a),
                receipt.joint_weights_sha256, receipt.head_weights_sha256,
                receipt.lora_weights_sha256,
            ]
            process = subprocess.run(command, cwd=source_root, capture_output=True,
                                     text=True, timeout=180)
            self.assertEqual(process.returncode, 0, process.stderr[-1500:])
            child_line = next(
                (line[len("PINNED_JOINT_CHILD="):]
                 for line in process.stdout.splitlines()
                 if line.startswith("PINNED_JOINT_CHILD=")), None,
            )
            self.assertIsNotNone(child_line, process.stdout[-500:])
            result = json.loads(child_line)
            self.assertEqual(result["first_joint_sha"], receipt.joint_weights_sha256)
            self.assertNotEqual(result["first_joint_sha"], result["next_joint_sha"])
            self.assertEqual(result["first_updates"], 1)
            self.assertEqual(result["next_updates"], 2)
            self.assertTrue(result["replay_blocked"])
            self.assertEqual(result["eval_prob_a"], observed.prob_a)
            self.assertEqual(result["frozen_sha"], frozen)
            self.assertEqual(result["initial_runtime"], "pinned-persistence-train-process")
            self.assertEqual(result["loaded_runtime"],
                             "pinned-persistence-independent-import")
            self.assertTrue(second_path.is_dir())
            print("PINNED_JOINT_PERSISTENCE="+json.dumps({
                "joint_sha_v1": receipt.joint_weights_sha256,
                "joint_sha_v2": result["next_joint_sha"],
                "base_sha_unchanged": frozen,
                "evaluation_prob_a_exact": result["eval_prob_a"],
                "updates_v1": 1, "updates_v2": 2,
                "replayed_episode_rejected": True,
                "independent_python_process": True,
                "train_only_no_payoff_generator": True,
                "hosted_model_modified": False,
            }, sort_keys=True))
