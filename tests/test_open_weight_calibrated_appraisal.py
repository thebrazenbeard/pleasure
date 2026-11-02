"""Calibrated, relative hidden-state appraisal on chosen-only bandit feedback."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
from types import SimpleNamespace
import unittest


def evidence(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


class SourceContractTests(unittest.TestCase):
    def test_feature_schema_identifies_a_separate_calibrated_architecture(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import (
            FEATURE_SCHEMA, TorchCalibratedAppraisal,
        )
        self.assertEqual(FEATURE_SCHEMA,
                         "action-centered-unit-difference-bounded-logit-2-v1")
        self.assertEqual(TorchCalibratedAppraisal.feature_schema, FEATURE_SCHEMA)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "optional PyTorch unavailable")
class CalibratedAppraisalTests(unittest.TestCase):
    def setUp(self):
        import torch
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal

        class LargeMagnitudeActionLM(torch.nn.Module):
            def __init__(self, scale=1000.0):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 3)
                self.out = torch.nn.Linear(3, 8)
                with torch.no_grad():
                    self.embedding.weight.zero_()
                    self.embedding.weight[3] = torch.tensor([scale, 0.0, 0.0])
                    self.embedding.weight[4] = torch.tensor([0.0, scale, 0.0])
                    self.out.weight.zero_()
                    self.out.bias.zero_()
                    self.out.bias[3] = 0.2  # Greedy LM prefers A.

            def forward(self, *, input_ids, output_hidden_states=False, use_cache=False):
                states = self.embedding(input_ids)
                return SimpleNamespace(logits=self.out(states),
                                       hidden_states=(states,) if output_hidden_states else None)

        self.torch = torch
        self.model = LargeMagnitudeActionLM()
        self.LM = LargeMagnitudeActionLM
        self.probe = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, max_updates=64, allow_training=True,
        )

    def _train(self, probe, count=24):
        for i in range(count):
            choice = probe.score((1,), partition="TRAIN", gain=0.0,
                                 defer_choice=True, episode_id=f"train-{i}")
            self.assertEqual(choice.chosen_action, "UNBOUND")
            sampled = "A" if i % 2 == 0 else "B"
            bound = probe.bind_choice(choice, sampled)
            update = probe.observe_chosen(
                bound, measured_quality=0.9 if sampled == "A" else 0.1,
                evidence_id=evidence(f"chosen-{i}"),
            )
            self.assertTrue(update.optimizer_step_executed)

    def test_unit_relative_features_train_discrimination_without_saturation(self):
        base = tuple(p.detach().clone() for p in self.model.parameters())
        initial = self.probe.score((1,), gain=8)
        self.assertEqual((initial.predicted_quality_a, initial.predicted_quality_b), (0.5, 0.5))
        self._train(self.probe)
        on = self.probe.score((1,), gain=8)
        off = self.probe.score((1,), gain=0)
        inverted = self.probe.score((1,), gain=-8)
        self.assertGreater(on.predicted_quality_a, 0.55)
        self.assertLess(on.predicted_quality_b, 0.45)
        self.assertLess(on.predicted_quality_a, 0.89)
        self.assertGreater(on.predicted_quality_b, 0.11)
        self.assertGreater(on.prob_a, off.prob_a)
        self.assertLess(inverted.prob_a, off.prob_a)
        self.assertTrue(all(self.torch.equal(before, after)
                            for before, after in zip(base, self.model.parameters())))

    def test_relative_training_is_invariant_to_extreme_hidden_magnitude(self):
        other = self.LM(scale=1.0)
        other_probe = type(self.probe)(
            other, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, max_updates=64, allow_training=True,
        )
        self._train(self.probe)
        self._train(other_probe)
        self.assertEqual(self.probe.head_digest, other_probe.head_digest)
        self.assertEqual(self.probe.score((1,), gain=8).prob_a,
                         other_probe.score((1,), gain=8).prob_a)

    def test_explicit_l2_regularization_reduces_head_weight_norm(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal

        unregularized = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, max_updates=64, allow_training=True,
            l2_penalty=0.0,
        )
        regularized = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, max_updates=64, allow_training=True,
            l2_penalty=0.6,
        )
        self._train(unregularized, count=32)
        self._train(regularized, count=32)

        def norm(probe):
            return float(self.torch.linalg.vector_norm(self.torch.cat([
                value.reshape(-1) for value in probe.export_tensors().values()
            ])))

        self.assertLess(norm(regularized), norm(unregularized))

    def test_saved_head_clone_replays_probabilities_but_replay_feedback_rejected(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        self._train(self.probe, count=4)
        digest = self.probe.head_digest
        fresh = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.0, allow_training=False,
        )
        fresh.import_tensors(self.probe.export_tensors(), expected_digest=digest)
        self.assertEqual(fresh.head_digest, digest)
        self.assertEqual(fresh.score((1,), gain=8).prob_a,
                         self.probe.score((1,), gain=8).prob_a)
        with self.assertRaises(PermissionError):
            fresh.score((1,), partition="TRAIN")
        with self.assertRaises(ValueError):
            self.probe.observe_chosen(self.probe.score((1,)), measured_quality=0.5,
                                      evidence_id=evidence("eval-replay"))

    def test_wrong_partition_or_prompt_rejected_when_using_cached_features(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache

        cache = FrozenActionFeatureCache(
            self.model, action_token_ids=(3, 4), checkpoint_sha256="a" * 64,
            tokenizer_sha256="b" * 64, max_entries=8,
        )
        cached_probe = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, allow_training=True, feature_cache=cache,
        )
        bundle = cache.get_features((1,), partition="EVAL", episode_id="eval-1")
        evaluated = cached_probe.score((1,), partition="EVAL", gain=2,
                                       episode_id="eval-1", features=bundle)
        self.assertEqual(evaluated.input_ids_sha256, bundle.input_ids_sha256)
        with self.assertRaises(ValueError):
            cached_probe.score((1,), partition="TRAIN", gain=0,
                               episode_id="train-1", features=bundle, defer_choice=True)
        with self.assertRaises(ValueError):
            cached_probe.score((2,), partition="EVAL", gain=0,
                               episode_id="eval-1", features=bundle)
        with self.assertRaises(ValueError):
            cached_probe.score((1,), partition="EVAL", gain=0,
                               episode_id="wrong", features=bundle)

    def test_forged_or_mutated_cache_bundle_cannot_change_reward_predictions(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache

        cache = FrozenActionFeatureCache(
            self.model, action_token_ids=(3, 4), checkpoint_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
        )
        probe = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, allow_training=True, feature_cache=cache,
        )
        bundle = cache.get_features((1,), partition="EVAL", episode_id="eval-1")
        # A bundle's frozen dataclass prevents attribute reassignment, not
        # mutation of the underlying tensor storage or metadata replacement.
        tampered = replace(bundle, feature_a=bundle.feature_a.clone())
        tampered.feature_a[0] += 10000.0
        for forged in (
            tampered,
            replace(bundle, logits=bundle.logits + self.torch.tensor([1000.0, -1000.0])),
            replace(bundle, frozen_model_sha256="f" * 64),
            replace(bundle, checkpoint_sha256="f" * 64),
            replace(bundle, tokenizer_sha256="f" * 64),
            replace(bundle, partition="TRAIN", episode_id="train-1"),
        ):
            with self.subTest(field=(forged.partition, forged.frozen_model_sha256,
                                     forged.checkpoint_sha256, forged.tokenizer_sha256)):
                with self.assertRaises(ValueError):
                    probe.score((1,), gain=0, partition=forged.partition,
                                defer_choice=forged.partition == "TRAIN",
                                episode_id=forged.episode_id, features=forged)
        self.assertEqual(probe.updates, 0)

    def test_changed_frozen_model_invalidates_cached_scores(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache

        cache = FrozenActionFeatureCache(
            self.model, action_token_ids=(3, 4), checkpoint_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
        )
        probe = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, feature_cache=cache,
        )
        issued = cache.get_features((1,), partition="EVAL", episode_id="eval-1")
        with self.torch.no_grad():
            self.model.embedding.weight[3, 0] += 0.5
        with self.assertRaises(RuntimeError):
            probe.score((1,), episode_id="eval-1", features=issued)

    def test_cached_features_remain_valid_within_verified_epoch(self):
        from pleasure_experiment.open_weight_calibrated_appraisal import TorchCalibratedAppraisal
        from pleasure_experiment.open_weight_feature_cache import FrozenActionFeatureCache

        cache = FrozenActionFeatureCache(
            self.model, action_token_ids=(3, 4), checkpoint_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
        )
        probe = TorchCalibratedAppraisal(
            self.model, action_token_ids=(3, 4), hidden_size=3,
            learning_rate=0.4, feature_cache=cache,
        )
        with cache.verified_epoch() as epoch:
            issued = epoch.get_features((1,), partition="EVAL", episode_id="eval-1")
            decision = probe.score((1,), episode_id="eval-1", features=issued)
            self.assertEqual(decision.input_ids_sha256, issued.input_ids_sha256)

