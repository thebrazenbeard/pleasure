"""Behavioral checks for the isolated, synthetic weight-update laboratory."""

from dataclasses import replace
import math
import unittest

from pleasure_experiment.weight_delta_lab import (
    ModelBinding,
    OutcomeCase,
    TrainingPolicy,
    WeightDeltaSession,
    apply_delta,
    digest_weights,
    evaluate_heldout,
    fit_delta,
    single_outcome_step,
)


def binding(model_id="surrogate-linear-head"):
    return ModelBinding(
        model_id=model_id,
        model_revision="synthetic-revision-001",
        runtime_instance_id="isolated-lab-instance",
        feature_schema_id="two-feature-lab-v1",
    )


def sample(case_id, episode_id, features, value):
    return OutcomeCase(
        case_id=case_id,
        episode_id=episode_id,
        evaluator_id="lab-external-labels-unverified",
        features=features,
        observed_value=value,
    )


BASE = (0.0, 0.0)
TRAIN = (
    sample("train-positive", "episode-a", (1.0, 0.0), 1.0),
    sample("train-negative", "episode-b", (0.0, 1.0), 0.0),
)
HOLDOUT = (
    sample("hold-positive", "episode-c", (0.8, 0.0), 1.0),
    sample("hold-negative", "episode-d", (0.0, 0.8), 0.0),
)
POLICY = TrainingPolicy(
    learning_rate=0.4, max_delta_l2=0.8, maximum_examples=16, maximum_steps=12
)


class WeightDeltaLabTests(unittest.TestCase):
    def test_one_step_gradient_matches_independent_hand_calculation(self):
        # At weights (0, 0), sigmoid predicts 0.5 for both orthogonal cases.
        # The mean logistic gradients are (-0.25, +0.25); one 0.4 step must
        # therefore yield (+0.1, -0.1) without hitting the norm budget.
        one_step = TrainingPolicy(
            learning_rate=0.4, max_delta_l2=0.8, maximum_examples=16, maximum_steps=1
        )
        delta = fit_delta(BASE, TRAIN, binding(), one_step)
        self.assertAlmostEqual(delta.delta[0], 0.1, places=12)
        self.assertAlmostEqual(delta.delta[1], -0.1, places=12)

    def test_genuine_labels_update_parameters_and_improve_unseen_cases(self):
        original_digest = digest_weights(BASE)
        delta = fit_delta(BASE, TRAIN, binding(), POLICY, reserved_holdout=HOLDOUT)
        effective = apply_delta(BASE, delta, binding())
        self.assertNotEqual(effective, BASE)
        self.assertNotEqual(digest_weights(effective), original_digest)
        self.assertEqual(delta.base_digest, original_digest)
        self.assertEqual(delta.effective_digest, digest_weights(effective))
        self.assertEqual(delta.capability_tier, "SURROGATE_WEIGHT_UPDATE_LAB")
        self.assertEqual(delta.phenomenology, "UNRESOLVED")
        self.assertGreater(sum(x * x for x in delta.delta), 0)
        self.assertLessEqual(math.sqrt(sum(x * x for x in delta.delta)), POLICY.max_delta_l2 + 1e-12)

        baseline = evaluate_heldout(BASE, HOLDOUT)
        changed = evaluate_heldout(effective, HOLDOUT, training_cases=TRAIN)
        self.assertLess(changed.mean_squared_error, baseline.mean_squared_error)
        self.assertEqual(baseline.split_status, "UNVERIFIED_WITHOUT_TRAINING_SET")
        self.assertEqual(changed.split_status, "DISJOINT_IDS_CHECKED")
        self.assertEqual(BASE, (0.0, 0.0))  # Original checkpoint was not mutated.

    def test_shuffled_outcomes_are_an_adverse_control(self):
        genuine = fit_delta(BASE, TRAIN, binding(), POLICY)
        reversed_labels = (
            sample("rev-positive", "episode-e", (1.0, 0.0), 0.0),
            sample("rev-negative", "episode-f", (0.0, 1.0), 1.0),
        )
        shuffled = fit_delta(BASE, reversed_labels, binding(), POLICY)
        genuine_score = evaluate_heldout(apply_delta(BASE, genuine, binding()), HOLDOUT)
        shuffled_score = evaluate_heldout(apply_delta(BASE, shuffled, binding()), HOLDOUT)
        self.assertLess(genuine_score.mean_squared_error, shuffled_score.mean_squared_error)

    def test_zero_learning_rate_cannot_change_parameters(self):
        zero = TrainingPolicy(
            learning_rate=0.0, max_delta_l2=0.8, maximum_examples=16, maximum_steps=12
        )
        delta = fit_delta(BASE, TRAIN, binding(), zero)
        self.assertEqual(delta.delta, (0.0, 0.0))
        self.assertEqual(apply_delta(BASE, delta, binding()), BASE)

    def test_disable_restores_exact_base_even_after_exception(self):
        delta = fit_delta(BASE, TRAIN, binding(), POLICY)
        session = WeightDeltaSession(BASE, delta, binding())
        self.assertEqual(session.current_weights, BASE)
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            with session as effective:
                self.assertNotEqual(effective, BASE)
                self.assertTrue(session.active)
                raise RuntimeError("probe failed")
        self.assertFalse(session.active)
        self.assertEqual(session.current_weights, BASE)
        self.assertEqual(digest_weights(session.current_weights), delta.base_digest)
        self.assertNotEqual(session.enable(), BASE)
        self.assertEqual(session.disable(), BASE)

    def test_session_snapshot_fields_cannot_be_reassigned_or_deleted(self):
        delta = fit_delta(BASE, TRAIN, binding(), POLICY)
        session = WeightDeltaSession(BASE, delta, binding())
        for field in ("_base", "_effective", "_active"):
            with self.subTest(field=field), self.assertRaisesRegex(
                AttributeError, "validated weight session is immutable"
            ):
                setattr(session, field, (0.9, 0.9))
            with self.subTest(deleted=field), self.assertRaisesRegex(
                AttributeError, "validated weight session is immutable"
            ):
                delattr(session, field)
        with self.assertRaisesRegex(AttributeError, "validated weight session is immutable"):
            session.some_new_attribute = "injected"
        self.assertEqual(session.current_weights, BASE)
        self.assertEqual(session.enable(), apply_delta(BASE, delta, binding()))
        self.assertEqual(session.disable(), BASE)

    def test_model_identity_and_digest_swap_rejected(self):
        delta = fit_delta(BASE, TRAIN, binding(), POLICY)
        with self.assertRaisesRegex(ValueError, "model binding mismatch"):
            apply_delta(BASE, delta, binding("different-model"))
        with self.assertRaisesRegex(ValueError, "model binding mismatch"):
            WeightDeltaSession(BASE, delta, binding("different-model"))
        with self.assertRaisesRegex(ValueError, "base weights digest mismatch"):
            apply_delta((0.1, 0.0), delta, binding())
        tampered = fit_delta(BASE, TRAIN, binding(), POLICY)
        object.__setattr__(tampered, "delta", (0.8, 0.8))
        with self.assertRaisesRegex(ValueError, "delta digest mismatch"):
            apply_delta(BASE, tampered, binding())

    def test_apply_rechecks_policy_digest_and_bounded_delta(self):
        delta = fit_delta(BASE, TRAIN, binding(), POLICY)
        with self.assertRaisesRegex(ValueError, "policy digest mismatch"):
            apply_delta(BASE, replace(delta, policy_digest="tampered"), binding())
        too_large = (0.9, 0.0)
        forged = replace(
            delta,
            delta=too_large,
            delta_digest=digest_weights(too_large),
            effective_digest=digest_weights(too_large),
        )
        with self.assertRaisesRegex(ValueError, "delta exceeds policy max_delta_l2"):
            apply_delta(BASE, forged, binding())

    def test_pure_single_outcome_step_only_updates_chosen_features(self):
        changed = single_outcome_step(
            base_weights=BASE, current_weights=BASE, chosen_features=(0.0, 1.0),
            observed_score=0.0, policy=POLICY,
        )
        self.assertEqual(changed, (0.0, -0.2))
        self.assertEqual(BASE, (0.0, 0.0))
        zero = TrainingPolicy(0.0, POLICY.max_delta_l2,
                              POLICY.maximum_examples, POLICY.maximum_steps)
        self.assertEqual(single_outcome_step(
            base_weights=BASE, current_weights=BASE, chosen_features=(0.0, 1.0),
            observed_score=0.0, policy=zero,
        ), BASE)

    def test_single_outcome_step_rejects_overbudget_current_and_invalid_labels(self):
        with self.assertRaisesRegex(ValueError, "current weights exceed cumulative"):
            single_outcome_step(
                base_weights=BASE, current_weights=(1.0, 0.0),
                chosen_features=(1.0, 0.0), observed_score=0.0, policy=POLICY,
            )
        with self.assertRaisesRegex(ValueError, "single outcome feature dimensionality"):
            single_outcome_step(
                base_weights=BASE, current_weights=BASE, chosen_features=(1.0,),
                observed_score=0.0, policy=POLICY,
            )
        for bad in (math.nan, math.inf, True, -0.1, 1.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                single_outcome_step(
                    base_weights=BASE, current_weights=BASE,
                    chosen_features=(0.0, 1.0), observed_score=bad, policy=POLICY,
                )

    def test_effective_weights_remain_in_admitted_domain_at_boundary(self):
        nearly_saturated = (10.0, 0.0)
        outward_gradient = (
            sample("edge-positive", "edge-episode", (1.0, 1.0), 1.0),
        )
        delta = fit_delta(nearly_saturated, outward_gradient, binding(), POLICY)
        effective = apply_delta(nearly_saturated, delta, binding())
        self.assertEqual(effective[0], 10.0)
        self.assertGreater(effective[1], 0.0)
        self.assertEqual(digest_weights(effective), delta.effective_digest)
        self.assertLessEqual(
            math.sqrt(sum(value * value for value in delta.delta)), POLICY.max_delta_l2 + 1e-12
        )

    def test_holdout_split_status_requires_supplied_disjoint_training(self):
        result = evaluate_heldout(BASE, HOLDOUT)
        self.assertEqual(result.split_status, "UNVERIFIED_WITHOUT_TRAINING_SET")
        explicit_empty = evaluate_heldout(BASE, HOLDOUT, training_cases=())
        self.assertEqual(explicit_empty.split_status, "UNVERIFIED_WITHOUT_TRAINING_SET")
        checked = evaluate_heldout(BASE, HOLDOUT, training_cases=TRAIN)
        self.assertEqual(checked.split_status, "DISJOINT_IDS_CHECKED")
        with self.assertRaisesRegex(ValueError, "training and holdout overlap"):
            evaluate_heldout(BASE, (TRAIN[0],), training_cases=TRAIN)

    def test_train_holdout_overlap_and_replay_within_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "training and holdout overlap"):
            fit_delta(BASE, TRAIN, binding(), POLICY, reserved_holdout=(TRAIN[0],))
        repeated_episode = sample("new-case", TRAIN[0].episode_id, (0.2, 0.0), 0.3)
        with self.assertRaisesRegex(ValueError, "duplicate training episode_id"):
            fit_delta(BASE, (TRAIN[0], repeated_episode), binding(), POLICY)
        repeated_holdout_episode = sample(
            "hold-new-case", HOLDOUT[0].episode_id, (0.7, 0.0), 1.0
        )
        with self.assertRaisesRegex(ValueError, "duplicate holdout episode_id"):
            evaluate_heldout(BASE, (HOLDOUT[0], repeated_holdout_episode))

    def test_corrupt_numeric_inputs_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, True, "0.2", 1.1, -0.1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sample("bad", "episode", (1.0, 0.0), bad)
        for bad in (math.nan, math.inf, -math.inf, True, "0.2", 2.0):
            with self.subTest(feature=bad), self.assertRaises(ValueError):
                sample("bad", "episode", (bad, 0.0), 1.0)
        for bad in (math.nan, math.inf, True, "0.1", 11.0):
            with self.subTest(weight=bad), self.assertRaises(ValueError):
                digest_weights((bad, 0.0))

    def test_bad_training_budgets_and_shapes_rejected(self):
        for kwargs in (
            {"learning_rate": -0.1},
            {"max_delta_l2": 0.0},
            {"maximum_examples": 0},
            {"maximum_steps": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainingPolicy(**{**POLICY.__dict__, **kwargs})
        with self.assertRaisesRegex(ValueError, "feature dimensionality mismatch"):
            fit_delta(BASE, (sample("bad-shape", "episode", (0.2,), 1.0),), binding(), POLICY)
        with self.assertRaisesRegex(ValueError, "too many training cases"):
            policy = TrainingPolicy(0.1, 0.2, 1, 1)
            fit_delta(BASE, TRAIN, binding(), policy)


if __name__ == "__main__":
    unittest.main()
