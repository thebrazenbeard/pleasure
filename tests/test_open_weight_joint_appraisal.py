"""Joint PEFT LoRA + quality head: genuine gradients and safe restart state."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from types import SimpleNamespace
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PEFT = importlib.util.find_spec("peft") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def _toy_joint(*, train_lora: bool = True, allow_training: bool = True):
    """Tiny local GPT2/PEFT fixture, no downloaded weights or remote code."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from pleasure_experiment.open_weight_adapter import OpenWeightBinding
    from pleasure_experiment.open_weight_local_model import (
        LoadedLocalCausalModel, LocalModelManifest, prepare_explicit_lora,
    )
    from pleasure_experiment.open_weight_joint_appraisal import TorchJointLoRAAppraisal

    config = GPT2Config(vocab_size=32, n_positions=32, n_ctx=32,
                        n_embd=16, n_layer=1, n_head=2,
                        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(112233)
        model = GPT2LMHeadModel(config)
    binding = OpenWeightBinding(
        model_id="tiny/test", model_revision="fixed",
        runtime_instance_id="tiny-runtime", checkpoint_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
    )
    record = LoadedLocalCausalModel(
        model=model, tokenizer=SimpleNamespace(), binding=binding,
        action_token_ids=(3, 4),
        manifest=LocalModelManifest(
            snapshot_directory="/tmp/toy-test-only", files=(),
            checkpoint_sha256="a" * 64, tokenizer_sha256="b" * 64,
            snapshot_sha256="c" * 64, total_bytes=100,
        ),
        action_prefix="Answer:", action_suffixes=(" A", " B"),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(445566)
        prepared = prepare_explicit_lora(
            record, target_modules=("c_attn",), rank=2, alpha=4,
        )
    probe = TorchJointLoRAAppraisal(
        prepared, hidden_size=16, learning_rate=.05,
        allow_training=allow_training, max_updates=6, train_lora=train_lora,
    )
    return torch, prepared, probe


@unittest.skipUnless(HAS_TORCH and HAS_PEFT and HAS_TRANSFORMERS,
                     "optional PyTorch/PEFT/transformers missing")
class JointAppraisalTests(unittest.TestCase):
    def _update(self, probe, index: int, *, quality=.9):
        episode = f"tiny-episode-{index}"
        decision = probe.score((1, 2, 5 + index % 5), partition="TRAIN",
                               gain=0.0, defer_choice=True, episode_id=episode)
        bound = probe.bind_choice(decision, "A" if index % 2 == 0 else "B")
        return probe.observe_chosen(
            bound, measured_quality=quality,
            evidence_id=sha256(f"toy-chosen-{index}".encode()).hexdigest(),
        )

    def test_joint_chosen_gradient_reaches_lora_and_head_without_base_change(self):
        _, _, probe = _toy_joint()
        before_l = probe.lora_digest
        before_h = probe.head_digest
        base = probe.base_digest
        update = self._update(probe, 0)
        self.assertGreater(update.lora_gradient_tensors, 0)
        self.assertGreater(update.lora_nonzero_gradient_tensors, 0)
        self.assertTrue(update.head_changed)
        self.assertEqual(probe.base_digest, base)
        self.assertEqual(update.frozen_base_digest, base)
        self.assertNotEqual(probe.head_digest, before_h)
        if probe.lora_digest == before_l:
            self._update(probe, 1)
        self.assertNotEqual(probe.lora_digest, before_l)

    def test_head_only_control_does_not_update_lora(self):
        _, _, probe = _toy_joint(train_lora=False)
        before = probe.lora_digest
        update = self._update(probe, 0)
        self.assertEqual(update.lora_nonzero_gradient_tensors, 0)
        self.assertEqual(probe.lora_digest, before)
        self.assertTrue(update.head_changed)

    def test_permission_evaluation_and_replay_gates(self):
        _, _, frozen = _toy_joint(allow_training=False)
        with self.assertRaises(PermissionError):
            frozen.score((1,), episode_id="not-authorized", partition="TRAIN")
        eval_decision = frozen.score((1,), episode_id="test-eval", partition="EVAL")
        self.assertEqual(eval_decision.partition, "EVAL")
        self.assertTrue(0 < eval_decision.prob_a < 1)
        with self.assertRaises(ValueError):
            frozen.bind_choice(eval_decision, "A")

        _, _, trained = _toy_joint()
        preview = trained.score((1, 2), partition="TRAIN", defer_choice=True,
                                episode_id="pending-episode")
        with self.assertRaises(ValueError):
            trained.score((3,), partition="TRAIN", defer_choice=True,
                          episode_id="other-episode")
        with self.assertRaises(ValueError):
            trained.observe_chosen(preview, measured_quality=.8,
                                   evidence_id="a" * 64)
        bound = trained.bind_choice(preview, "B")
        trained.observe_chosen(bound, measured_quality=.8, evidence_id="a" * 64)
        with self.assertRaises(ValueError):
            trained.score((1, 2), partition="TRAIN", defer_choice=True,
                          episode_id="pending-episode")

    def test_export_import_reproduces_weights_and_replay_state(self):
        _, _, trained = _toy_joint()
        self._update(trained, 0)
        tensors = trained.export_tensors()
        state = trained.export_training_state()
        imported_sha = trained.joint_digest
        _, _, fresh = _toy_joint()
        fresh.import_tensors(tensors, expected_digest=imported_sha,
                             updates_at_export=state["updates"],
                             seen_evidence_ids=state["seen_evidence_ids"],
                             seen_episode_ids=state["seen_episode_ids"])
        self.assertEqual(fresh.joint_digest, imported_sha)
        self.assertEqual(fresh.export_training_state(), state)
        with self.assertRaises(ValueError):
            fresh.score((1, 2), partition="TRAIN", defer_choice=True,
                        episode_id="tiny-episode-0")
        restored_decision = fresh.score((1, 2, 5), partition="EVAL",
                                        episode_id="fixed-eval", gain=12.0)
        live_decision = trained.score((1, 2, 5), partition="EVAL",
                                      episode_id="fixed-eval", gain=12.0)
        self.assertEqual(restored_decision.prob_a, live_decision.prob_a)
        self.assertEqual(restored_decision.predicted_quality_a,
                         live_decision.predicted_quality_a)
        next_update = self._update(fresh, 1)
        self.assertEqual(next_update.update_index, 2)

    def test_import_corruption_keeps_original_tensors(self):
        _, _, probe = _toy_joint()
        original = probe.joint_digest
        tensor_map = probe.export_tensors()
        head = tensor_map["head.weight"]
        head.add_(.1)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            probe.import_tensors(tensor_map, expected_digest=original)
        self.assertEqual(probe.joint_digest, original)
        with self.assertRaisesRegex(ValueError, "ledger"):
            probe.import_tensors(probe.export_tensors(), expected_digest=original,
                                 updates_at_export=1, seen_evidence_ids=(),
                                 seen_episode_ids=())

    def test_frozen_parameter_mutation_fails_closed(self):
        torch, _, probe = _toy_joint()
        # Deliberate hostile mutation: even a small immutable-base drift must
        # invalidate the source-local session before a new decision is issued.
        frozen = probe._frozen[0][1]
        with torch.no_grad():
            frozen.flatten()[0].add_(.01)
        with self.assertRaisesRegex(RuntimeError, "frozen pretrained"):
            probe.score((1, 2), partition="EVAL", episode_id="mutation-check")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            probe.score((1, 2), partition="EVAL", episode_id="mutated-session")

    def test_pending_train_decision_blocks_state_export_and_import(self):
        _, _, probe = _toy_joint()
        tensors = probe.export_tensors()
        digest = probe.joint_digest
        preview = probe.score((1, 2), partition="TRAIN", defer_choice=True,
                              episode_id="pending-exact")
        with self.assertRaisesRegex(ValueError, "pending"):
            probe.export_tensors()
        with self.assertRaisesRegex(ValueError, "pending"):
            probe.import_tensors(tensors, expected_digest=digest)
        probe.abandon_unbound(preview)
        self.assertEqual(probe.joint_digest, digest)
