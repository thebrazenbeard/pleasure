"""Adversarial tests of source-only selected-action, one-step online SGD."""

from dataclasses import replace
from hashlib import sha256
from math import hypot, inf, nan
from threading import Barrier, Thread
import unittest

from pleasure_experiment.online_weight_trial import (
    ChosenOutcome, OnlineRecurrentPolicy, OnlineWeightTrial,
)
from pleasure_experiment.weight_delta_lab import (
    ModelBinding, TrainingPolicy, digest_weights,
)
from pleasure_experiment.weight_first_trial import DecisionProbe, RecurrentState


MODEL = ModelBinding("online-tiny", "fixed-r1", "runtime-r1", "two-features-v1")
POLICY = TrainingPolicy(0.4, 0.8, 16, 12)
RUBRIC = sha256(b"source-only-rubric-v1").hexdigest()
EVALUATOR = "source-only-evaluator"
NO_RECURRENCE = OnlineRecurrentPolicy(retention=0.5, outcome_gain=0.0, decision_gain=0.0)


def probe(index, *, binding=MODEL, a=(1.0, 0.0), b=(0.0, 1.0)):
    return DecisionProbe(f"task-{index}", f"episode-{index}", binding, a, b)


def trial(*, base=(0.0, 0.0), policy=POLICY, recur=NO_RECURRENCE,
          holdout=("episode-eval",), session_nonce=None):
    return OnlineWeightTrial(
        base_weights=base, binding=MODEL, training_policy=policy,
        recurrent_policy=recur, expected_evaluator_id=EVALUATOR,
        expected_rubric_digest=RUBRIC, reserved_holdout_episode_ids=holdout,
        trial_session_nonce=session_nonce,
    )


def choose(engine, index, *, nonce=None, partition="TRAIN", given_probe=None):
    return engine.choose(
        given_probe if given_probe is not None else probe(index),
        offer_nonce=nonce or f"nonce-{index}", source_partition=partition,
    )


def outcome(choice, *, quality=0.0, evidence=None, **changes):
    fields = dict(
        evidence_id=evidence or f"measured-{choice.probe_id}",
        decision_digest=choice.decision_digest, probe_id=choice.probe_id,
        episode_id=choice.episode_id, binding=MODEL,
        chosen_action=choice.chosen_action, evaluator_id=EVALUATOR,
        rubric_digest=RUBRIC, measured_quality=quality,
    )
    fields.update(changes)
    return ChosenOutcome(**fields)


class OnlineWeightTrialTests(unittest.TestCase):
    def test_chosen_negative_B_causes_real_one_step_update_then_next_choice_A(self):
        engine = trial()
        first = choose(engine, 1)
        self.assertEqual(first.chosen_action, "B")
        self.assertEqual(first.appraisal_a, 0.5)
        self.assertEqual(first.appraisal_b, 0.5)
        self.assertEqual(first.source_partition, "TRAIN")
        self.assertEqual(first.weights_digest, digest_weights((0.0, 0.0)))
        self.assertEqual(engine.weights_readback(), (0.0, 0.0))  # No label admitted yet.
        change = engine.observe(outcome(first, quality=0.0))
        self.assertEqual(change.decision_digest, first.decision_digest)
        self.assertEqual(change.prediction_before, 0.5)
        self.assertEqual(change.measured_quality, 0.0)
        self.assertEqual(change.before_digest, first.weights_digest)
        self.assertTrue(change.update_detected)
        self.assertEqual(change.effect_classification, "WEIGHT_CHANGE_SOURCE_LOCAL")
        self.assertNotEqual(change.after_digest, change.before_digest)
        self.assertEqual(engine.weights_readback()[0], 0.0)
        self.assertLess(engine.weights_readback()[1], 0.0)
        second = choose(engine, 2)
        self.assertEqual(second.chosen_action, "A")
        self.assertEqual(second.weights_digest, change.after_digest)
        self.assertEqual(engine.step_index, 1)
        self.assertTrue(engine.pending)

    def test_positive_observed_B_reinforces_B_without_unobserved_A_label(self):
        engine = trial()
        first = choose(engine, 1)
        change = engine.observe(outcome(first, quality=1.0))
        self.assertEqual(engine.weights_readback()[0], 0.0)
        self.assertGreater(engine.weights_readback()[1], 0.0)
        self.assertEqual(choose(engine, 2).chosen_action, "B")
        self.assertEqual(change.chosen_action, "B")

    def test_zero_lr_and_recursion_only_ablations_are_separable(self):
        no_learning = TrainingPolicy(0.0, 0.8, 16, 12)
        frozen = trial(policy=no_learning, recur=NO_RECURRENCE)
        frozen_choice = choose(frozen, 1)
        receipt = frozen.observe(outcome(frozen_choice, quality=0.0))
        self.assertFalse(receipt.update_detected)
        self.assertEqual(receipt.effect_classification, "NO_WEIGHT_CHANGE")
        self.assertEqual(frozen.weights_readback(), (0.0, 0.0))
        self.assertEqual(choose(frozen, 2).chosen_action, "B")

        recurrent_only = trial(
            policy=no_learning,
            recur=OnlineRecurrentPolicy(retention=0.5, outcome_gain=0.6, decision_gain=0.5),
        )
        recurrence_choice = choose(recurrent_only, 1)
        recurrent_only.observe(outcome(recurrence_choice, quality=0.0))
        self.assertEqual(recurrent_only.weights_readback(), (0.0, 0.0))
        self.assertEqual(choose(recurrent_only, 2).chosen_action, "A")
        self.assertNotEqual(recurrent_only.state, RecurrentState())

    def test_outcome_absent_or_wrong_action_digest_episode_rejects_atomically(self):
        engine = trial()
        with self.assertRaisesRegex(RuntimeError, "no pending"):
            engine.observe(outcome(
                ChosenOutcome("dummy", "0" * 64, "task-dummy", "episode-dummy", MODEL,
                              "B", EVALUATOR, RUBRIC, 0.0)
            ))
        choice = choose(engine, 1)
        before = engine.weights_digest
        before_state = engine.state
        for wrong, error in (
            ({"chosen_action": "A"}, "pending chosen action"),
            ({"decision_digest": "0" * 64}, "pending chosen action"),
            ({"episode_id": "different-episode"}, "pending chosen action"),
            ({"probe_id": "different-task"}, "pending chosen action"),
            ({"binding": ModelBinding("wrong", "fixed-r1", "runtime-r1", "two-features-v1")},
             "outcome model binding"),
        ):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, error):
                engine.observe(outcome(choice, **wrong))
            self.assertTrue(engine.pending)
            self.assertEqual(engine.weights_digest, before)
            self.assertEqual(engine.state, before_state)
            self.assertEqual(engine.step_index, 0)
        good = engine.observe(outcome(choice))
        self.assertTrue(good.update_detected)
        with self.assertRaisesRegex(RuntimeError, "no pending"):
            engine.observe(outcome(choice))

    def test_swapped_evaluator_or_rubric_is_rejected_without_partial_state(self):
        engine = trial()
        choice = choose(engine, 1)
        for wrong, message in (
            ({"evaluator_id": "other-evaluator"}, "evaluator ID mismatch"),
            ({"rubric_digest": "a" * 64}, "rubric digest mismatch"),
        ):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, message):
                engine.observe(outcome(choice, **wrong))
            self.assertEqual(engine.weights_digest, choice.weights_digest)
            self.assertEqual(engine.state, RecurrentState())
            self.assertTrue(engine.pending)
            self.assertEqual(engine.step_index, 0)
        self.assertTrue(engine.observe(outcome(choice)).update_detected)

    def test_replay_episodes_evidence_ids_and_holdout_train_leakage_rejected(self):
        engine = trial()
        first = choose(engine, 1)
        engine.observe(outcome(first, evidence="evidence-1"))
        with self.assertRaisesRegex(ValueError, "duplicate online episode"):
            choose(engine, 1)
        with self.assertRaisesRegex(ValueError, "reserved holdout episode"):
            choose(engine, "eval")
        second = choose(engine, 2)
        with self.assertRaisesRegex(ValueError, "duplicate online outcome evidence ID"):
            engine.observe(outcome(second, evidence="evidence-1"))
        self.assertEqual(engine.step_index, 1)
        self.assertTrue(engine.pending)
        engine.observe(outcome(second, evidence="evidence-2"))

    def test_eval_is_readonly_same_policy_and_cannot_be_trained_on(self):
        engine = trial(holdout=("episode-eval", "episode-eval-2"))
        heldout = probe("eval")
        before = engine.weights_digest
        eval_one = engine.evaluate_choice(heldout, offer_nonce="eval-nonce")
        with self.assertRaisesRegex(ValueError, "duplicate evaluated episode ID"):
            engine.evaluate_choice(heldout, offer_nonce="eval-nonce")
        self.assertEqual(eval_one.source_partition, "EVAL")
        self.assertEqual(engine.weights_digest, before)
        self.assertEqual(engine.state, RecurrentState())
        self.assertEqual(engine.step_index, 0)
        self.assertFalse(engine.pending)
        with self.assertRaisesRegex(ValueError, "TRAIN source partition"):
            choose(engine, "eval", partition="EVAL")
        with self.assertRaisesRegex(ValueError, "EVAL source partition"):
            engine.evaluate_choice(heldout, offer_nonce="eval-nonce", source_partition="TRAIN")
        with self.assertRaisesRegex(ValueError, "evaluated episode cannot be trained on"):
            choose(engine, "eval")
        choice = choose(engine, 1)
        self.assertEqual(choice.appraisal_a, eval_one.appraisal_a)
        self.assertEqual(choice.appraisal_b, eval_one.appraisal_b)
        self.assertNotEqual(choice.decision_digest, eval_one.decision_digest)
        with self.assertRaisesRegex(RuntimeError, "training outcome pending"):
            engine.evaluate_choice(heldout, offer_nonce="eval-nonce")
        engine.observe(outcome(choice))
        eval_after = engine.evaluate_choice(probe("eval-2"), offer_nonce="eval-nonce-2")
        self.assertEqual(eval_after.weights_digest, engine.weights_digest)
        self.assertEqual(engine.step_index, 1)

    def test_unregistered_eval_or_train_then_eval_crossover_rejected_atomically(self):
        engine = trial(holdout=("episode-eval",))
        initial = engine.weights_digest
        with self.assertRaisesRegex(ValueError, "preregistered as holdout"):
            engine.evaluate_choice(probe("unregistered"), offer_nonce="nonce-unregistered")
        self.assertEqual(engine.weights_digest, initial)
        self.assertEqual(engine.step_index, 0)
        self.assertFalse(engine.pending)
        train_choice = choose(engine, 1)
        engine.observe(outcome(train_choice))
        weight_before_eval = engine.weights_digest
        with self.assertRaisesRegex(ValueError, "training episode cannot also be evaluated"):
            engine.evaluate_choice(probe(1), offer_nonce="nonce-eval-after-train")
        self.assertEqual(engine.weights_digest, weight_before_eval)
        self.assertEqual(engine.step_index, 1)

    def test_external_frozen_dto_mutation_cannot_rewrite_private_pending_choice(self):
        engine = trial()
        returned = choose(engine, 1)
        original_digest = returned.decision_digest
        original_action = returned.chosen_action
        external_binding = returned.binding
        object.__setattr__(returned, "decision_digest", "0" * 64)
        object.__setattr__(returned, "chosen_action", "A")
        object.__setattr__(external_binding, "runtime_instance_id", "attacker-runtime")
        object.__setattr__(engine.binding, "runtime_instance_id", "mutated-copy")
        object.__setattr__(engine.state, "b", 1.0)
        self.assertEqual(engine.binding, MODEL)
        self.assertEqual(engine.state, RecurrentState())
        self.assertEqual(engine.weights_digest, digest_weights((0.0, 0.0)))
        correct = ChosenOutcome(
            "evidence-authored", original_digest, "task-1", "episode-1", MODEL,
            original_action, EVALUATOR, RUBRIC, 0.0,
        )
        receipt = engine.observe(correct)
        self.assertTrue(receipt.update_detected)
        self.assertEqual(receipt.decision_digest, original_digest)

    def test_default_random_session_nonce_blocks_cross_instance_outcome_replay(self):
        first_engine = trial()
        second_engine = trial()
        first = choose(first_engine, 1, nonce="identical-offer-nonce")
        second = choose(second_engine, 1, nonce="identical-offer-nonce")
        self.assertEqual(first.weights_digest, second.weights_digest)
        self.assertEqual(first.binding, second.binding)
        self.assertEqual(first.chosen_action, second.chosen_action)
        self.assertEqual(first.session_nonce_source, "GENERATED_SOURCE_LOCAL")
        self.assertEqual(second.session_nonce_source, "GENERATED_SOURCE_LOCAL")
        self.assertNotEqual(first.session_nonce_digest, second.session_nonce_digest)
        self.assertNotEqual(first.decision_digest, second.decision_digest)
        with self.assertRaisesRegex(ValueError, "outcome does not match pending chosen action"):
            second_engine.observe(outcome(first))
        self.assertEqual(second_engine.weights_readback(), (0.0, 0.0))
        self.assertTrue(second_engine.pending)
        self.assertEqual(second_engine.step_index, 0)
        self.assertTrue(second_engine.observe(outcome(second)).update_detected)

    def test_identical_caller_session_nonce_is_explicitly_replay_equivalent(self):
        first_engine = trial(session_nonce="reproducible-session-for-source-test")
        second_engine = trial(session_nonce="reproducible-session-for-source-test")
        first = choose(first_engine, 1, nonce="same-offer-nonce")
        second = choose(second_engine, 1, nonce="same-offer-nonce")
        self.assertEqual(first.decision_digest, second.decision_digest)
        self.assertEqual(first.session_nonce_source, "CALLER_DECLARED_REPLAY_EQUIVALENT_IF_REUSED")
        self.assertEqual(second.session_nonce_source, first.session_nonce_source)
        # This is intentional for deterministic source tests; it proves why
        # caller-controlled nonce reuse is NOT a cross-run anti-replay claim.
        self.assertTrue(second_engine.observe(outcome(first)).update_detected)

    def test_chosen_state_update_decays_both_channels_and_impacts_only_chosen(self):
        recur = OnlineRecurrentPolicy(retention=0.5, outcome_gain=0.5, decision_gain=1.0)
        engine = trial(
            base=(0.0, 0.0), policy=TrainingPolicy(0.0, 0.8, 16, 12), recur=recur,
        )
        first = choose(engine, 1)
        self.assertEqual(first.chosen_action, "B")
        engine.observe(outcome(first, quality=1.0))
        self.assertEqual(engine.state, RecurrentState(0.0, 0.5))
        second = choose(engine, 2)
        self.assertEqual(second.chosen_action, "B")
        engine.observe(outcome(second, quality=0.0))
        self.assertEqual(engine.state, RecurrentState(0.0, -0.25))
        third = choose(engine, 3)
        self.assertEqual(third.chosen_action, "A")
        engine.observe(outcome(third, quality=1.0))
        self.assertEqual(engine.state, RecurrentState(0.5, -0.125))

    def test_weight_and_recurrent_state_restored_on_exception_and_further_io_closed(self):
        engine = trial(recur=OnlineRecurrentPolicy(0.5, 0.7, 0.5))
        with self.assertRaisesRegex(RuntimeError, "experimental failure"):
            with engine:
                choice = choose(engine, 1)
                engine.observe(outcome(choice, quality=0.0))
                self.assertNotEqual(engine.weights_readback(), (0.0, 0.0))
                self.assertNotEqual(engine.state, RecurrentState())
                raise RuntimeError("experimental failure")
        self.assertEqual(engine.weights_readback(), (0.0, 0.0))
        self.assertEqual(engine.state, RecurrentState())
        self.assertFalse(engine.pending)
        report = engine.restore()
        self.assertTrue(report.closed)
        self.assertEqual(report.restored_digest, digest_weights((0.0, 0.0)))
        with self.assertRaisesRegex(RuntimeError, "restored and closed"):
            choose(engine, 2)
        with self.assertRaisesRegex(RuntimeError, "restored and closed"):
            engine.evaluate_choice(probe("eval"), offer_nonce="eval-nonce")

    def test_budget_and_bad_outcome_values_rejected(self):
        limited = trial(policy=TrainingPolicy(0.4, 0.8, 1, 1))
        first = choose(limited, 1)
        limited.observe(outcome(first))
        with self.assertRaisesRegex(RuntimeError, "training budget exhausted"):
            choose(limited, 2)
        for invalid in (nan, inf, -0.1, 1.1, True, "0.5"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ChosenOutcome("bad", "a" * 64, "p", "e", MODEL, "A",
                              EVALUATOR, RUBRIC, invalid)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            OnlineWeightTrial(
                base_weights=(0.0, 0.0), binding=MODEL, training_policy=POLICY,
                expected_evaluator_id=EVALUATOR, expected_rubric_digest="BAD",
            )

    def test_atomic_concurrent_double_observe_only_one_update(self):
        engine = trial()
        first = choose(engine, 1)
        measured = outcome(first)
        barrier = Barrier(3)
        responses = []

        def worker():
            barrier.wait()
            try:
                responses.append(("ok", engine.observe(measured)))
            except RuntimeError:
                responses.append(("rejected", None))

        threads = [Thread(target=worker), Thread(target=worker)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.assertEqual([kind for kind, _ in responses].count("ok"), 1)
        self.assertEqual([kind for kind, _ in responses].count("rejected"), 1)
        self.assertEqual(engine.step_index, 1)
        self.assertEqual(engine.weights_readback(), (0.0, -0.2))

    def test_cumulative_update_norm_and_coordinate_domain_remain_bounded(self):
        small = TrainingPolicy(learning_rate=1.0, max_delta_l2=0.1,
                               maximum_examples=16, maximum_steps=12)
        engine = trial(policy=small, base=(10.0, 0.0), holdout=())
        for index in range(1, 7):
            action = choose(engine, index)
            engine.observe(outcome(action, quality=0.0))
            effective = engine.weights_readback()
            self.assertTrue(all(-10.0 <= x <= 10.0 for x in effective))
            self.assertLessEqual(hypot(effective[0] - 10.0, effective[1]), 0.1 + 1e-12)


if __name__ == "__main__":
    unittest.main()
