"""Integrated falsifiers for trained-parameter -> sigmoid -> recurrent choice."""

import math
import unittest

from pleasure_experiment.weight_delta_lab import (
    ModelBinding, OutcomeCase, TrainingPolicy,
    apply_delta, digest_weights, fit_delta, predict_appraisal,
)
from pleasure_experiment.weight_first_trial import (
    DecisionProbe, RecurrentPolicy, RecurrentState, run_weight_first_trial,
)


BINDING = ModelBinding("surrogate-linear", "fixed-r1", "instance-001", "two-vector-v1")
BASE = (0.0, 0.0)
TRAIN = (
    OutcomeCase("train-a", "e-train-a", "caller-labels", (1.0, 0.0), 1.0),
    OutcomeCase("train-b", "e-train-b", "caller-labels", (0.0, 1.0), 0.0),
)
HOLDOUT = (
    OutcomeCase("hold-a", "e-hold-a", "caller-labels", (0.8, 0.0), 1.0),
    OutcomeCase("hold-b", "e-hold-b", "caller-labels", (0.0, 0.8), 0.0),
)
PROBES = (
    DecisionProbe("probe-1", "e-probe-1", BINDING, (1.0, 0.0), (0.0, 1.0)),
    DecisionProbe("probe-2", "e-probe-2", BINDING, (0.8, 0.0), (0.0, 0.8)),
    DecisionProbe("probe-3", "e-probe-3", BINDING, (1.0, 0.0), (0.0, 1.0)),
)
POLICY = TrainingPolicy(learning_rate=0.4, max_delta_l2=0.8,
                        maximum_examples=16, maximum_steps=12)


def trial(**changes):
    arguments = dict(
        base_weights=BASE, binding=BINDING, training_cases=TRAIN,
        heldout_cases=HOLDOUT, probes=PROBES, training_policy=POLICY,
        recurrent_policy=RecurrentPolicy(retention=0.5, gain=0.6),
    )
    arguments.update(changes)
    return run_weight_first_trial(**arguments)


class WeightFirstIntegratedTrialTests(unittest.TestCase):
    def test_five_arms_use_actual_learned_weights_and_opaque_choices(self):
        receipt = trial()
        self.assertEqual(receipt.capability_tier, "SURROGATE_WEIGHT_UPDATE_LAB")
        self.assertEqual(receipt.phenomenology, "UNRESOLVED")
        self.assertEqual(receipt.authorized_effects, ())
        self.assertEqual(receipt.evidence_class, "SOURCE_ONLY_CALLER_LABELS_UNAUTHENTICATED")
        self.assertTrue(receipt.update_detected)
        self.assertTrue(receipt.categorical_choice_changed)
        self.assertEqual(receipt.effect_classification,
                         "WEIGHT_CHANGE_WITH_CATEGORICAL_DIFFERENCE")
        self.assertEqual(receipt.binding, BINDING)

        self.assertEqual(receipt.baseline.weights_digest, digest_weights(BASE))
        self.assertEqual(receipt.disabled.weights_digest, receipt.baseline.weights_digest)
        self.assertEqual(receipt.withdrawal.weights_digest, receipt.baseline.weights_digest)
        self.assertNotEqual(receipt.trained.weights_digest, receipt.baseline.weights_digest)
        self.assertNotEqual(receipt.shuffled.weights_digest, receipt.trained.weights_digest)
        self.assertEqual(receipt.baseline.steps, receipt.disabled.steps)
        self.assertEqual(receipt.baseline.final_state, receipt.disabled.final_state)
        self.assertEqual([s.action for s in receipt.baseline.steps], ["B", "B", "B"])
        self.assertEqual([s.action for s in receipt.disabled.steps], ["B", "B", "B"])
        self.assertEqual([s.action for s in receipt.trained.steps], ["A", "A", "A"])
        self.assertEqual([s.action for s in receipt.shuffled.steps], ["B", "B", "B"])
        self.assertTrue(all(s.tie for s in receipt.baseline.steps))
        self.assertEqual(receipt.withdrawal.initial_state, receipt.trained.final_state)
        self.assertNotEqual(receipt.withdrawal.final_state, receipt.baseline.final_state)
        self.assertLess(
            receipt.heldout_trained.mean_squared_error,
            receipt.heldout_baseline.mean_squared_error,
        )
        self.assertGreater(
            receipt.heldout_shuffled.mean_squared_error,
            receipt.heldout_baseline.mean_squared_error,
        )
        for result in (
            receipt.heldout_baseline, receipt.heldout_trained, receipt.heldout_shuffled,
        ):
            self.assertEqual(result.split_status, "DISJOINT_IDS_CHECKED")

    def test_appraisal_is_exact_same_sigmoid_on_fit_delta_coefficients(self):
        receipt = trial()
        delta = fit_delta(BASE, TRAIN, BINDING, POLICY, reserved_holdout=HOLDOUT)
        effective = apply_delta(BASE, delta, BINDING)
        self.assertEqual(receipt.trained.weights_digest, delta.effective_digest)
        self.assertEqual(receipt.trained_delta_digest, delta.delta_digest)
        for step, probe in zip(receipt.trained.steps, PROBES):
            self.assertEqual(step.appraisal_a, predict_appraisal(effective, probe.features_a))
            self.assertEqual(step.appraisal_b, predict_appraisal(effective, probe.features_b))
        expected = 1.0 / (1.0 + math.exp(-effective[0]))
        self.assertEqual(receipt.trained.steps[0].appraisal_a, expected)

    def test_disabling_weights_restores_base_even_with_recurrent_aftereffects(self):
        receipt = trial()
        self.assertEqual(receipt.withdrawal.weights_digest, receipt.base_digest)
        self.assertEqual(receipt.withdrawal.steps[0].appraisal_a, 0.5)
        self.assertEqual(receipt.withdrawal.steps[0].appraisal_b, 0.5)
        self.assertGreater(receipt.withdrawal.steps[0].state_a,
                           receipt.withdrawal.steps[0].state_b)
        reset = trial(recurrent_policy=RecurrentPolicy(retention=0.0, gain=0.6))
        self.assertEqual(reset.withdrawal.weights_digest, reset.base_digest)
        self.assertEqual(reset.withdrawal.final_state, RecurrentState())
        self.assertTrue(all(step.tie for step in reset.withdrawal.steps))
        self.assertEqual([step.action for step in reset.withdrawal.steps], ["B", "B", "B"])

    def test_zero_learning_rate_keeps_every_arm_on_base_weights(self):
        zero = TrainingPolicy(0.0, POLICY.max_delta_l2,
                              POLICY.maximum_examples, POLICY.maximum_steps)
        receipt = trial(training_policy=zero)
        self.assertFalse(receipt.update_detected)
        self.assertFalse(receipt.categorical_choice_changed)
        self.assertEqual(receipt.effect_classification, "NO_WEIGHT_CHANGE")
        for arm in (
            receipt.trained, receipt.shuffled, receipt.disabled, receipt.withdrawal,
        ):
            self.assertEqual(arm.weights_digest, receipt.base_digest)
            self.assertTrue(all(step.tie for step in arm.steps))
            self.assertEqual([step.action for step in arm.steps], ["B", "B", "B"])
        self.assertEqual(receipt.heldout_trained, receipt.heldout_baseline)

    def test_nonzero_learning_rate_can_still_yield_zero_gradient_and_no_weight_change(self):
        cancelling = (
            OutcomeCase("cancel-a", "e-cancel-a", "caller-labels", (1.0, 0.0), 1.0),
            OutcomeCase("cancel-b", "e-cancel-b", "caller-labels", (1.0, 0.0), 0.0),
        )
        receipt = trial(training_cases=cancelling)
        self.assertGreater(POLICY.learning_rate, 0.0)
        self.assertFalse(receipt.update_detected)
        self.assertFalse(receipt.categorical_choice_changed)
        self.assertEqual(receipt.effect_classification, "NO_WEIGHT_CHANGE")
        self.assertEqual(receipt.trained.weights_digest, receipt.base_digest)
        self.assertEqual(receipt.heldout_trained, receipt.heldout_baseline)
        self.assertEqual([step.action for step in receipt.trained.steps], ["B", "B", "B"])

    def test_changed_weights_without_categorical_choice_effect_are_not_overclaimed(self):
        nondiscriminating = DecisionProbe(
            "same-features", "e-same-features", BINDING, (1.0, 0.0), (1.0, 0.0)
        )
        receipt = trial(probes=(nondiscriminating,))
        self.assertTrue(receipt.update_detected)
        self.assertFalse(receipt.categorical_choice_changed)
        self.assertEqual(receipt.effect_classification,
                         "WEIGHT_CHANGE_NO_CATEGORICAL_DIFFERENCE")
        self.assertEqual(receipt.trained.steps[0].action, "B")

    def test_fresh_runs_are_exactly_deterministic(self):
        self.assertEqual(trial(), trial())

    def test_uninformative_shuffled_control_is_rejected(self):
        all_positive = (
            OutcomeCase("x", "episode-x", "caller-labels", (1.0, 0.0), 1.0),
            OutcomeCase("y", "episode-y", "caller-labels", (0.0, 1.0), 1.0),
        )
        with self.assertRaisesRegex(ValueError, "distinct training outcome labels"):
            trial(training_cases=all_positive)

    def test_probe_replays_and_training_overlap_are_rejected(self):
        duplicate = DecisionProbe("probe-new", PROBES[0].episode_id, BINDING,
                                  (1.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "duplicate probe or episode ID"):
            trial(probes=(PROBES[0], duplicate))
        overlap = DecisionProbe("probe-overlap", TRAIN[0].episode_id, BINDING,
                                (1.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "overlap training or holdout episodes"):
            trial(probes=(overlap,))
        with self.assertRaisesRegex(ValueError, "training and holdout overlap"):
            trial(heldout_cases=(TRAIN[0],))

    def test_cross_model_or_runtime_probes_are_rejected(self):
        other = ModelBinding("surrogate-linear", "fixed-r1", "instance-OTHER", "two-vector-v1")
        mismatched = DecisionProbe("probe-cross", "e-probe-cross", other,
                                   (1.0, 0.0), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "probe model binding mismatch"):
            trial(probes=(mismatched,))
        altered = DecisionProbe("probe-tampered", "e-probe-tampered", BINDING,
                                (1.0, 0.0), (0.0, 1.0))
        object.__setattr__(altered, "binding", other)
        with self.assertRaisesRegex(ValueError, "probe model binding mismatch"):
            trial(probes=(altered,))

    def test_bad_probe_feature_and_policy_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "A/B feature dimensionality mismatch"):
            DecisionProbe("bad", "episode-bad", BINDING, (1.0,), (0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "probe dimensionality mismatch"):
            trial(probes=(DecisionProbe("bad", "episode-bad", BINDING, (0.8,), (0.2,)),))
        for malformed in (True, -0.1, 1.1, math.nan, math.inf, "0.5"):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                RecurrentPolicy(retention=malformed)
        with self.assertRaisesRegex(ValueError, "gain must be positive"):
            RecurrentPolicy(gain=0.0)

    def test_corrupted_frozen_probe_is_revalidated_before_running(self):
        altered = DecisionProbe("original", "episode-original", BINDING,
                                (1.0, 0.0), (0.0, 1.0))
        object.__setattr__(altered, "features_a", (math.nan, 0.0))
        with self.assertRaisesRegex(ValueError, "features must be finite"):
            trial(probes=(altered,))

    def test_recurrent_state_stays_bounded_under_large_appraisal(self):
        receipt = trial(
            base_weights=(10.0, -10.0),
            recurrent_policy=RecurrentPolicy(retention=1.0, gain=1.0),
        )
        for arm in (receipt.baseline, receipt.trained, receipt.shuffled,
                    receipt.disabled, receipt.withdrawal):
            for step in arm.steps:
                self.assertLessEqual(abs(step.state_a), 1.0)
                self.assertLessEqual(abs(step.state_b), 1.0)


if __name__ == "__main__":
    unittest.main()
