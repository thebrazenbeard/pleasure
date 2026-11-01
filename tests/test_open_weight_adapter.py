"""No-download checks for optional caller-supplied PyTorch policy training."""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from hashlib import sha256
from types import MethodType, SimpleNamespace
from unittest import TestCase, skipUnless

from pleasure_experiment.open_weight_adapter import (
    EVIDENCE_CLASS,
    OpenWeightBinding,
    OpenWeightOutcome,
    TorchChosenOutcomeAdapter,
)


def digest(value: str) -> str:
    return sha256(value.encode("ascii")).hexdigest()


BINDING = OpenWeightBinding(
    "toy-open-policy", "local-r1", "runtime-001", digest("checkpoint"), digest("tokenizer")
)
RUBRIC = digest("selected-only-rubric")


class OptionalDependencyTests(TestCase):
    def test_binding_rejects_missing_provenance_identity(self):
        with self.assertRaises(ValueError):
            replace(BINDING, checkpoint_sha256="unverified")
        with self.assertRaises(ValueError):
            replace(BINDING, tokenizer_sha256="F" * 64)
        with self.assertRaises(ValueError):
            replace(BINDING, model_id="../bad model")

    def test_outcome_rejects_nonfinite_and_wrong_choice(self):
        o = OpenWeightOutcome(
            evidence_id=digest("measurement"), decision_digest=digest("choice"),
            episode_id="episode-1", chosen_action="B", evaluator_id="source-oracle",
            rubric_digest=RUBRIC, measured_quality=0.9,
        )
        self.assertEqual(o.measured_quality, 0.9)
        for invalid in (float("nan"), float("inf"), -0.01, 1.01, True, 10 ** 1000):
            with self.subTest(invalid=str(invalid)[:20]), self.assertRaises(ValueError):
                replace(o, measured_quality=invalid)
        with self.assertRaises(ValueError):
            replace(o, chosen_action="C")
        with self.assertRaises(ValueError):
            replace(o, decision_digest="abc")

    def test_explicit_in_place_authorization_before_any_optional_import(self):
        with self.assertRaises(PermissionError):
            TorchChosenOutcomeAdapter(
                object(), binding=BINDING, action_token_ids=(0, 1),
                trainable_parameter_names=("adapter.weight",), learning_rate=0.1,
                expected_evaluator_id="source-oracle", expected_rubric_digest=RUBRIC,
            )

    def test_reject_bad_actions_and_parameter_name_sets(self):
        for action_ids in ((0, 0), (0, -1), (True, 1), (0,), [0, 1]):
            with self.subTest(action_ids=action_ids), self.assertRaises(ValueError):
                TorchChosenOutcomeAdapter(
                    object(), binding=BINDING, action_token_ids=action_ids,
                    trainable_parameter_names=("adapter.weight",), learning_rate=0.1,
                    expected_evaluator_id="source-oracle", expected_rubric_digest=RUBRIC,
                    allow_in_place_mutation=True,
                )
        with self.assertRaises(ValueError):
            TorchChosenOutcomeAdapter(
                object(), binding=BINDING, action_token_ids=(0, 1),
                trainable_parameter_names=(), learning_rate=0.1,
                expected_evaluator_id="source-oracle", expected_rubric_digest=RUBRIC,
                allow_in_place_mutation=True,
            )

    def test_missing_torch_fails_closed_without_claiming_optimizer(self):
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("PyTorch installed; real-tensor tests cover it")
        with self.assertRaisesRegex(RuntimeError, "PyTorch unavailable"):
            TorchChosenOutcomeAdapter(
                object(), binding=BINDING, action_token_ids=(0, 1),
                trainable_parameter_names=("adapter.weight",), learning_rate=0.1,
                expected_evaluator_id="source-oracle", expected_rubric_digest=RUBRIC,
                allow_in_place_mutation=True,
            )


@skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
class RealTorchOptimizerTests(TestCase):
    def make_adapter(
        self, *, learning_rate: float = 0.5, max_delta_l2: float = 1.0,
        binding: OpenWeightBinding = BINDING,
        initial_gradient: tuple[float, float] | None = None,
    ):
        import torch

        class TinyCausalLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Embedding(8, 2)
                self.adapter_logits = torch.nn.Parameter(torch.zeros(2))
                self.register_buffer("frozen_counter", torch.zeros(1, dtype=torch.int64))
                self.config = SimpleNamespace(use_cache=True)

            def forward(self, *, input_ids):
                # A genuine trainable next-token policy tensor, not a reward
                # head. Backbone is present but deliberately frozen and has
                # no influence on this synthetic two-token fixture.
                batch, length = input_ids.shape
                self.backbone(input_ids)
                return self.adapter_logits.view(1, 1, 2).expand(batch, length, 2)

        model = TinyCausalLanguageModel()
        if initial_gradient is not None:
            model.adapter_logits.grad = torch.tensor(initial_gradient, dtype=torch.float32)
        adapter = TorchChosenOutcomeAdapter(
            model, binding=binding, action_token_ids=(0, 1),
            trainable_parameter_names=("adapter_logits",),
            learning_rate=learning_rate, expected_evaluator_id="source-oracle",
            expected_rubric_digest=RUBRIC, allow_in_place_mutation=True,
            max_delta_l2=max_delta_l2,
        )
        return adapter, model

    def feedback(self, choice, quality: float, *, evidence: str = "feedback"):
        return OpenWeightOutcome(
            evidence_id=digest(evidence), decision_digest=choice.decision_digest,
            episode_id=choice.episode_id, chosen_action=choice.chosen_action,
            evaluator_id="source-oracle", rubric_digest=RUBRIC,
            measured_quality=quality,
        )

    def test_measured_negative_outcome_changes_actual_policy_weights_and_next_action(self):
        adapter, model = self.make_adapter()
        before = adapter.weights_digest
        assert model.backbone.weight.requires_grad is False
        c1 = adapter.infer(
            input_ids=(1, 2), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        self.assertEqual(c1.chosen_action, "B")
        update = adapter.observe(self.feedback(c1, 0.0))
        self.assertTrue(update.optimizer_step_executed)
        self.assertTrue(update.optimizer_applied)
        self.assertTrue(update.update_detected)
        self.assertNotEqual(before, adapter.weights_digest)
        self.assertGreater(update.gradient_norm, 0)
        self.assertGreater(update.delta_l2, 0)
        c2 = adapter.infer(
            input_ids=(1, 2), offer_nonce="offer-2", episode_id="eval-1",
            source_partition="EVAL",
        )
        self.assertEqual(c2.chosen_action, "A")
        restored = adapter.restore()
        self.assertTrue(restored.closed)
        self.assertEqual(restored.restored_digest, before)
        self.assertTrue(model.backbone.weight.requires_grad)
        self.assertEqual(model.training, True)
        with self.assertRaises(RuntimeError):
            adapter.infer(
                input_ids=(1,), offer_nonce="later", episode_id="closed",
                source_partition="TRAIN",
            )

    def test_zero_lr_observes_outcome_without_parameter_update(self):
        adapter, _ = self.make_adapter(learning_rate=0.0)
        baseline = adapter.weights_digest
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        update = adapter.observe(self.feedback(choice, 0.0))
        self.assertFalse(update.optimizer_step_executed)
        self.assertFalse(update.optimizer_applied)
        self.assertEqual(update.step_index, 1)
        self.assertEqual(adapter.weights_digest, baseline)
        self.assertTrue(adapter.restore().closed)

    def test_read_only_eval_cannot_be_trained_and_wrong_outcome_rejected(self):
        adapter, _ = self.make_adapter()
        baseline = adapter.weights_digest
        evaluated = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="eval-1",
            source_partition="EVAL",
        )
        with self.assertRaises(RuntimeError):
            adapter.observe(self.feedback(evaluated, 1.0))
        self.assertEqual(adapter.weights_digest, baseline)
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-2", episode_id="train-1",
            source_partition="TRAIN",
        )
        with self.assertRaises(ValueError):
            adapter.observe(replace(self.feedback(choice, 1.0), chosen_action="A"))
        self.assertEqual(adapter.weights_digest, baseline)
        adapter.restore()

    def test_delta_cap_failure_rolls_back_and_closes(self):
        adapter, _ = self.make_adapter(max_delta_l2=0.000001)
        baseline = adapter.weights_digest
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        with self.assertRaisesRegex(RuntimeError, "parameter-change budget"):
            adapter.observe(self.feedback(choice, 1.0))
        self.assertEqual(adapter.weights_digest, baseline)
        self.assertTrue(adapter.restore().closed)

    def test_external_trainable_tensor_mutation_between_choice_and_feedback_rejected(self):
        import torch
        adapter, model = self.make_adapter()
        baseline = adapter.weights_digest
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        with torch.no_grad():
            model.adapter_logits[0] += 0.25
        with self.assertRaisesRegex(RuntimeError, "changed after action"):
            adapter.observe(self.feedback(choice, 1.0))
        restored = adapter.restore()
        self.assertEqual(restored.restored_digest, baseline)

    def test_external_mutation_between_episodes_is_not_miscredited_to_online_sgd(self):
        import torch
        adapter, model = self.make_adapter()
        baseline = adapter.weights_digest
        with torch.no_grad():
            model.adapter_logits[1] += 0.5
        with self.assertRaisesRegex(RuntimeError, "outside observed optimizer steps"):
            adapter.infer(
                input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
                source_partition="TRAIN",
            )
        self.assertEqual(adapter.restore().restored_digest, baseline)

    def test_frozen_nonfloat_buffer_mutation_is_detected(self):
        adapter, model = self.make_adapter()
        frozen_baseline = adapter.frozen_backbone_digest
        model.frozen_counter[0] = 1
        with self.assertRaisesRegex(RuntimeError, "frozen base model"):
            adapter.infer(
                input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
                source_partition="TRAIN",
            )
        restored = adapter.restore()
        self.assertEqual(restored.frozen_backbone_digest, frozen_baseline)
        self.assertEqual(int(model.frozen_counter[0].item()), 0)

    def test_optimizer_membership_substitution_cannot_train_backbone(self):
        adapter, model = self.make_adapter()
        baseline = adapter.weights_digest
        adapter._optimizer.param_groups[0]["params"] = [model.backbone.weight]
        with self.assertRaisesRegex(RuntimeError, "optimizer parameter membership"):
            adapter.infer(
                input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
                source_partition="TRAIN",
            )
        self.assertEqual(adapter.restore().restored_digest, baseline)

    def test_external_binding_and_returned_choice_alias_cannot_rewrite_target(self):
        mutable_binding = replace(BINDING)
        adapter, _ = self.make_adapter(binding=mutable_binding)
        object.__setattr__(mutable_binding, "model_id", "forged-target")
        self.assertEqual(adapter.binding.model_id, "toy-open-policy")
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="eval-1",
            source_partition="EVAL",
        )
        object.__setattr__(choice.binding, "model_id", "forged-receipt")
        self.assertEqual(adapter.binding.model_id, "toy-open-policy")
        next_choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-2", episode_id="eval-2",
            source_partition="EVAL",
        )
        self.assertEqual(next_choice.binding.model_id, "toy-open-policy")
        adapter.restore()

    def test_failed_gradient_forward_restores_all_owned_model_state(self):
        import torch
        adapter, model = self.make_adapter()
        baseline = adapter.weights_digest
        frozen_baseline = adapter.frozen_backbone_digest
        original_forward = model.forward

        def fail_after_mutating_buffer(self, *, input_ids):
            if torch.is_grad_enabled():
                with torch.no_grad():
                    self.frozen_counter[0] += 5
                raise RuntimeError("synthetic backward-path failure")
            return original_forward(input_ids=input_ids)

        model.forward = MethodType(fail_after_mutating_buffer, model)
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic backward-path failure"):
            adapter.observe(self.feedback(choice, 1.0))
        self.assertEqual(adapter.weights_digest, baseline)
        self.assertEqual(adapter.frozen_backbone_digest, frozen_baseline)
        self.assertEqual(int(model.frozen_counter[0].item()), 0)
        self.assertTrue(model.training)
        self.assertTrue(model.backbone.weight.requires_grad)
        self.assertTrue(model.config.use_cache)
        with self.assertRaises(RuntimeError):
            adapter.infer(
                input_ids=(1,), offer_nonce="offer-2", episode_id="closed",
                source_partition="TRAIN",
            )
        self.assertEqual(adapter.restore().restored_digest, baseline)

    def test_frozen_backbone_parameter_drift_never_returns_successful_restore(self):
        import torch
        adapter, model = self.make_adapter()
        with torch.no_grad():
            model.backbone.weight[0, 0] += 1.0
        with self.assertRaisesRegex(RuntimeError, "baseline rollback unverified"):
            adapter.restore()
        self.assertTrue(model.training)
        self.assertTrue(model.backbone.weight.requires_grad)
        self.assertTrue(model.config.use_cache)
        with self.assertRaises(RuntimeError):
            adapter.infer(
                input_ids=(1,), offer_nonce="later", episode_id="closed",
                source_partition="TRAIN",
            )

    def test_restore_preserves_preexisting_selected_gradient_snapshot(self):
        adapter, model = self.make_adapter(initial_gradient=(3.0, 4.0))
        choice = adapter.infer(
            input_ids=(1,), offer_nonce="offer-1", episode_id="train-1",
            source_partition="TRAIN",
        )
        adapter.observe(self.feedback(choice, 1.0))
        self.assertTrue(adapter.restore().closed)
        self.assertEqual(model.adapter_logits.grad.tolist(), [3.0, 4.0])
