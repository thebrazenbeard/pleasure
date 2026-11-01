"""Source-only tests of a hidden-state-derived, trainable appraisal head."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from types import SimpleNamespace
from unittest import TestCase, skipUnless, mock


def _evidence(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


@skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
class HiddenAppraisalTests(TestCase):
    def setUp(self):
        import torch

        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal

        class TinyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 2)
                self.out = torch.nn.Linear(2, 8)
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([1.0, 0.0])
                    self.embedding.weight[4] = torch.tensor([0.0, 1.0])
                    self.out.weight.zero_()
                    self.out.bias.zero_()
                    self.out.bias[3] = 0.02

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                hidden = self.embedding(input_ids)
                return SimpleNamespace(
                    logits=self.out(hidden),
                    hidden_states=(hidden,) if output_hidden_states else None,
                )

        self.torch = torch
        self.model = TinyCausalLM()
        self.probe = TorchHiddenAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=1.0, allow_training=True,
            max_updates=32, max_delta_l2=1.0,
        )

    def test_chosen_feedback_changes_head_but_not_language_model(self):
        from pleasure_experiment.open_weight_appraisal import compare_appraisal_ablations

        base = tuple(p.detach().clone() for p in self.model.parameters())
        before = self.probe.head_digest
        initial = compare_appraisal_ablations(self.probe, (1,), gain=10)
        self.assertEqual(initial["off"].chosen_action, "A")
        self.assertEqual(initial["on"].chosen_action, "A")
        self.assertAlmostEqual(initial["on"].predicted_quality_a, 0.5)

        for n in range(10):
            choice = self.probe.score((1,), gain=0, partition="TRAIN")
            self.assertEqual(choice.chosen_action, "A")
            receipt = self.probe.observe_chosen(
                choice, measured_quality=1.0,
                evidence_id=_evidence(f"chosen-positive-{n}"),
            )
            self.assertTrue(receipt.optimizer_step_executed)
            self.assertNotEqual(receipt.after_digest, receipt.before_digest)
        self.assertNotEqual(self.probe.head_digest, before)
        ablations = compare_appraisal_ablations(self.probe, (1,), gain=10)
        self.assertEqual(ablations["off"].chosen_action, "A")
        self.assertEqual(ablations["on"].chosen_action, "A")
        self.assertEqual(ablations["inverted"].chosen_action, "B")
        self.assertGreater(ablations["on"].prob_a, ablations["off"].prob_a)
        self.assertLess(ablations["inverted"].prob_a, ablations["off"].prob_a)
        for old, new in zip(base, self.model.parameters()):
            self.assertTrue(self.torch.equal(old, new))
        self.assertTrue(self.model.training)

    def test_eval_is_read_only_and_feedback_requires_training_decision(self):
        initial = self.probe.head_digest
        test = self.probe.score((1,), gain=6, partition="EVAL")
        self.assertEqual(initial, self.probe.head_digest)
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            self.probe.observe_chosen(
                test, measured_quality=1.0, evidence_id=_evidence("eval-leak"),
            )
        train = self.probe.score((1,), gain=0, partition="TRAIN")
        with self.assertRaisesRegex(ValueError, "pending"):
            self.probe.score((1,), gain=0, partition="TRAIN")
        with self.assertRaises(ValueError):
            self.probe.observe_chosen(
                train, measured_quality=float("nan"), evidence_id=_evidence("nan"),
            )
        self.assertEqual(self.probe.head_digest, initial)
        self.probe.observe_chosen(
            train, measured_quality=0.9, evidence_id=_evidence("good"),
        )
        with self.assertRaises(ValueError):
            self.probe.observe_chosen(
                train, measured_quality=0.9, evidence_id=_evidence("good"),
            )

    def test_zero_rate_frozen_head_and_snapshot_round_trip(self):
        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal

        frozen = TorchHiddenAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=True,
        )
        before = frozen.head_digest
        decision = frozen.score((1,), partition="TRAIN")
        result = frozen.observe_chosen(
            decision, measured_quality=1.0, evidence_id=_evidence("frozen"),
        )
        self.assertFalse(result.optimizer_step_executed)
        self.assertEqual(frozen.head_digest, before)

        choice = self.probe.score((1,), partition="TRAIN")
        self.probe.observe_chosen(
            choice, measured_quality=1.0, evidence_id=_evidence("persist"),
        )
        snapshot = self.probe.export_tensors()
        fresh = TorchHiddenAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=True,
        )
        self.assertNotEqual(fresh.head_digest, self.probe.head_digest)
        fresh.import_tensors(snapshot, expected_digest=self.probe.head_digest)
        self.assertEqual(fresh.head_digest, self.probe.head_digest)
        self.assertEqual(
            fresh.score((1,), gain=7).prob_a,
            self.probe.score((1,), gain=7).prob_a,
        )
        snapshot["weight"].zero_()
        self.assertEqual(fresh.head_digest, self.probe.head_digest)

    def test_bad_state_cannot_partially_replace_head(self):
        choice = self.probe.score((1,), partition="TRAIN")
        initial = self.probe.head_digest
        with self.assertRaisesRegex(ValueError, "pending"):
            self.probe.import_tensors(self.probe.export_tensors(), expected_digest=initial)
        self.probe.observe_chosen(
            choice, measured_quality=0.7, evidence_id=_evidence("resolve"),
        )
        current = self.probe.head_digest
        state = self.probe.export_tensors()
        state["weight"] = self.torch.zeros((3, 2))
        with self.assertRaises(ValueError):
            self.probe.import_tensors(state, expected_digest=current)
        self.assertEqual(self.probe.head_digest, current)
        with self.assertRaises(ValueError):
            self.probe.import_tensors(self.probe.export_tensors(), expected_digest="0" * 64)
        self.assertEqual(self.probe.head_digest, current)

    def test_late_import_readback_failure_rolls_back_preceding_copies(self):
        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal

        choice = self.probe.score((1,), partition="TRAIN")
        self.probe.observe_chosen(
            choice, measured_quality=0.95, evidence_id=_evidence("to-import"),
        )
        target = TorchHiddenAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=2,
            learning_rate=0.0, allow_training=False,
        )
        original = target.head_digest
        with mock.patch.object(target, "_digest", side_effect=RuntimeError("late hash fault")):
            with self.assertRaisesRegex(RuntimeError, "late hash fault"):
                target.import_tensors(
                    self.probe.export_tensors(), expected_digest=self.probe.head_digest,
                )
        self.assertEqual(target.head_digest, original)

    def test_training_gate_and_replay_errors(self):
        from pleasure_experiment.open_weight_appraisal import TorchHiddenAppraisal

        with self.assertRaises(PermissionError):
            TorchHiddenAppraisal(
                self.model, action_token_ids=(3, 4), hidden_size=2,
                learning_rate=0.1, allow_training=False,
            ).score((1,), partition="TRAIN")
        decision = self.probe.score((1,), partition="TRAIN")
        with self.assertRaises(ValueError):
            self.probe.observe_chosen(
                decision, measured_quality=0.5, evidence_id="not-a-sha",
            )
        self.probe.observe_chosen(
            decision, measured_quality=0.8, evidence_id=_evidence("once"),
        )
        another = self.probe.score((1,), partition="TRAIN")
        with self.assertRaisesRegex(ValueError, "replayed"):
            self.probe.observe_chosen(
                another, measured_quality=0.8, evidence_id=_evidence("once"),
            )
