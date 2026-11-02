"""Optional real-autograd integration test across the complete chosen-outcome loop.

The fixture is randomly initialized, tiny and source-local. Passing it does
not prove that a pretrained LLM checkpoint or hosted conversational model was
trained, nor that any model experienced pleasure.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
from unittest import TestCase, skipUnless

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding,
    TorchChosenOutcomeAdapter,
)
from pleasure_experiment.open_weight_trial import OpenWeightTrialRunner
from pleasure_experiment.outcome_arena import OutcomeArena


def _sha(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


@skipUnless(importlib.util.find_spec("torch") is not None, "optional PyTorch unavailable")
class GenuineTensorClosedLoopTests(TestCase):
    def _run(self, learning_rate: float):
        import torch

        class TinyCausalLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Embedding(8, 3)
                self.policy = torch.nn.Linear(3, 2)
                with torch.no_grad():
                    self.backbone.weight.copy_(torch.tensor([
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [1.0, 1.0, 1.0],
                    ]))
                    self.policy.weight.zero_()
                    self.policy.bias.zero_()

            def forward(self, *, input_ids):
                # Both policy tokens arise from a real differentiable neural
                # forward pass involving a fixed embedding and trainable head.
                return self.policy(self.backbone(input_ids))

        seed = 12345
        evaluator_id = "source-local-open-weight-oracle"
        rubric = OutcomeArena(seed=seed, evaluator_id=evaluator_id).rubric_digest
        binding = OpenWeightBinding(
            model_id="tiny-torch-causal-fixture",
            model_revision="generated-test-fixture-v1",
            runtime_instance_id="source-test-instance",
            checkpoint_sha256=_sha("synthetic-checkpoint-not-an-external-manifest"),
            tokenizer_sha256=_sha("synthetic-tokenizer-not-an-external-manifest"),
        )
        model = TinyCausalLM()
        adapter = TorchChosenOutcomeAdapter(
            model,
            binding=binding,
            action_token_ids=(0, 1),
            trainable_parameter_names=("policy.weight", "policy.bias"),
            learning_rate=learning_rate,
            expected_evaluator_id=evaluator_id,
            expected_rubric_digest=rubric,
            allow_in_place_mutation=True,
            max_grad_norm=1.0,
            max_delta_l2=0.5,
        )
        runner = OpenWeightTrialRunner(
            adapter=adapter,
            encode_offer=lambda offer: (1 + offer.pair_index % 5, 1 + offer.side),
            seed=seed,
            training_pairs=4,
            heldout_pairs=4,
            evaluator_id=evaluator_id,
        )
        report = runner.run()
        return report, model

    def test_chosen_outcome_updates_actual_tensors_and_future_inference(self):
        learned, learned_model = self._run(learning_rate=0.4)
        frozen, frozen_model = self._run(learning_rate=0.0)

        self.assertEqual(learned.initial_weights_digest, frozen.initial_weights_digest)
        self.assertEqual(learned.frozen_backbone_digest, frozen.frozen_backbone_digest)
        self.assertNotEqual(learned.learned_weights_digest, learned.initial_weights_digest)
        self.assertEqual(frozen.learned_weights_digest, frozen.initial_weights_digest)
        self.assertGreater(learned.observed_weight_changes, 0)
        self.assertEqual(frozen.observed_weight_changes, 0)
        self.assertEqual(learned.optimizer_steps_reported, 4)
        self.assertEqual(frozen.optimizer_steps_reported, 0)
        self.assertEqual(len(learned.training), 4)
        self.assertEqual(len(learned.heldout), 8)
        self.assertEqual(learned.heldout_primary.episodes, 4)
        self.assertTrue(learned.restored)
        self.assertTrue(frozen.restored)
        self.assertEqual(learned.restored_weights_digest, learned.initial_weights_digest)
        self.assertEqual(frozen.restored_weights_digest, frozen.initial_weights_digest)
        self.assertTrue(learned_model.backbone.weight.requires_grad)
        self.assertTrue(frozen_model.backbone.weight.requires_grad)
        # Heldout inference genuinely consumed the updated neural tensors.
        self.assertTrue(any(
            abs(a.choice.prob_b - b.choice.prob_b) > 1e-7
            for a, b in zip(learned.heldout, frozen.heldout)
        ))
        self.assertFalse(learned.hosted_model_modified)
        self.assertFalse(learned.externally_authenticated)
        self.assertFalse(learned.real_llm_weight_change_established)
