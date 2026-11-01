"""Black-box matched source-local online-feedback runner tests."""

from dataclasses import asdict, replace
from unittest.mock import patch
import math
import unittest

from pleasure_experiment.closed_loop_experiment import (
    ARM_NO_RECURRENCE,
    ARM_ONLINE,
    ARM_ZERO,
    ARM_ZERO_NO_RECURRENCE,
    ClosedLoopAborted,
    EVIDENCE_CLASS,
    EVAL_FAMILY,
    SCHEMA_VERSION,
    TRAIN_FAMILY,
    _validate_resolution,
    run_closed_loop_experiment,
)
from pleasure_experiment.online_weight_trial import OnlineWeightTrial
from pleasure_experiment.outcome_arena import OutcomeArena
from pleasure_experiment.weight_delta_lab import ModelBinding, digest_weights
from pleasure_experiment.weight_first_trial import RecurrentState


BINDING = ModelBinding("tiny-numeric-model", "revision-one", "lab-instance", "two-features")


def run(**overrides):
    arguments = dict(binding=BINDING, training_pairs=4, evaluation_pairs=4)
    arguments.update(overrides)
    return run_closed_loop_experiment(**arguments)


class ClosedLoopExperimentTests(unittest.TestCase):
    def test_four_factorial_arms_resolve_only_committed_training_actions(self):
        report = run()
        self.assertEqual(report.schema_version, SCHEMA_VERSION)
        self.assertEqual(report.evidence_class, EVIDENCE_CLASS)
        self.assertEqual(report.primary_heldout_selection_rule, "ONE_SIDE_PER_PAIR_INDEX_PARITY_V1")
        self.assertEqual(report.phenomenology, "UNRESOLVED")
        self.assertEqual(report.execution_target_kind, "SYNTHETIC_NUMERIC_SURROGATE")
        self.assertTrue(report.binding_is_caller_declared)
        self.assertFalse(report.hosted_model_modified)
        self.assertFalse(report.externally_authenticated)
        self.assertTrue(report.full_factorial_controls_completed)
        self.assertFalse(report.recurrence_effect_causally_isolated)
        self.assertEqual(report.authorized_effects, ())
        self.assertEqual(report.binding, BINDING)
        self.assertEqual(report.base_weights_digest, digest_weights((0.0, 0.0)))
        self.assertEqual(
            tuple(arm.arm_id for arm in report.arms),
            (ARM_ONLINE, ARM_ZERO, ARM_NO_RECURRENCE, ARM_ZERO_NO_RECURRENCE),
        )
        for arm in report.arms:
            self.assertEqual((arm.planned_training_episodes, arm.completed_training_episodes), (4, 4))
            self.assertEqual((arm.planned_heldout_episodes, arm.completed_heldout_episodes), (8, 8))
            self.assertEqual(arm.missing_outcome_count, 0)
            self.assertEqual(arm.prequential.episodes, 4)
            self.assertEqual(arm.heldout_independent_primary.episodes, 4)
            self.assertEqual(arm.heldout.episodes, 8)
            self.assertEqual((arm.independent_training_pairs, arm.independent_heldout_pairs), (4, 4))
            self.assertEqual(arm.prequential.chosen_a + arm.prequential.chosen_b, 4)
            self.assertEqual(arm.heldout.chosen_a + arm.heldout.chosen_b, 8)
            self.assertEqual(
                arm.heldout_independent_primary.chosen_a + arm.heldout_independent_primary.chosen_b, 4,
            )
            self.assertEqual(arm.restore.update_count, 4)
            self.assertFalse(arm.restore.cancelled_pending_decision)
            self.assertTrue(arm.restore.closed)
            self.assertEqual(arm.restore.restored_digest, report.base_weights_digest)
            self.assertEqual(arm.initial_weights_digest, report.base_weights_digest)
            self.assertEqual(digest_weights(arm.initial_weights), arm.initial_weights_digest)
            self.assertEqual(digest_weights(arm.learned_weights), arm.learned_weights_digest)
            self.assertEqual(arm.training_policy.maximum_steps, 4)
            self.assertEqual(arm.training_policy.maximum_examples, 4)
            self.assertEqual(
                arm.frozen_weight_state_reset_choice_differences,
                sum(
                    step.choice.chosen_action != (
                        "A" if step.choice.appraisal_a > step.choice.appraisal_b else "B"
                    ) for step in arm.heldout_episodes
                ),
            )
            self.assertEqual(
                arm.frozen_weight_state_reset_primary_choice_differences,
                sum(
                    step.choice.chosen_action != (
                        "A" if step.choice.appraisal_a > step.choice.appraisal_b else "B"
                    ) for step in arm.heldout_episodes
                    if step.offer.side == step.offer.pair_index % 2
                ),
            )

            train_ids = {step.offer.episode_id for step in arm.training_episodes}
            eval_ids = {step.offer.episode_id for step in arm.heldout_episodes}
            self.assertEqual(len(train_ids), 4)
            self.assertEqual(len(eval_ids), 8)
            self.assertTrue(train_ids.isdisjoint(eval_ids))
            self.assertEqual([step.offer.pair_index for step in arm.training_episodes], [0, 1, 2, 3])
            self.assertEqual([step.offer.side for step in arm.training_episodes], [0, 1, 0, 1])
            for index, step in enumerate(arm.training_episodes):
                self.assertEqual(step.phases, ("ISSUE", "CHOOSE", "COMMIT", "RESOLVE", "OBSERVE"))
                self.assertEqual(step.offer.partition, "TRAIN")
                self.assertEqual(step.offer.family_id, TRAIN_FAMILY)
                self.assertEqual(step.choice.source_partition, "TRAIN")
                self.assertEqual(step.choice.probe_id, step.offer.task_id)
                self.assertEqual(step.choice.episode_id, step.offer.episode_id)
                self.assertEqual(step.commitment.action, step.choice.chosen_action)
                self.assertEqual(step.outcome.action, step.choice.chosen_action)
                self.assertEqual(step.outcome.decision_digest, step.choice.decision_digest)
                self.assertEqual(step.update.decision_digest, step.choice.decision_digest)
                self.assertEqual(step.update.evidence_id, step.outcome.evidence_id)
                self.assertEqual(step.update.before_digest, step.choice.weights_digest)
                self.assertEqual(digest_weights(step.before_weights), step.update.before_digest)
                self.assertEqual(digest_weights(step.after_weights), step.update.after_digest)
                self.assertEqual(step.before_weights,
                                 arm.training_episodes[index - 1].after_weights
                                 if index > 0 else arm.initial_weights)
                self.assertEqual(step.after_weights,
                                 arm.training_episodes[index + 1].before_weights
                                 if index + 1 < 4 else arm.learned_weights)
                self.assertEqual(step.update.after_digest,
                                 arm.training_episodes[index + 1].choice.weights_digest
                                 if index + 1 < 4 else arm.learned_weights_digest)
                self.assertEqual(step.update.step_index, index + 1)
                self.assertNotIn("unchosen_quality", asdict(step.outcome))
            for step in arm.heldout_episodes:
                self.assertEqual(step.phases, ("ISSUE", "EVALUATE", "COMMIT", "RESOLVE"))
                self.assertEqual(step.offer.partition, "EVAL")
                self.assertEqual(step.offer.family_id, EVAL_FAMILY)
                self.assertEqual(step.choice.source_partition, "EVAL")
                self.assertEqual(step.choice.weights_digest, arm.learned_weights_digest)
                self.assertEqual(step.choice.step_index, 4)
                self.assertEqual(step.outcome.action, step.choice.chosen_action)
                self.assertEqual(step.outcome.decision_digest, step.choice.decision_digest)
                self.assertEqual(step.outcome.episode_id, step.offer.episode_id)

        # Same public environment allocations for all control arms, including
        # mirrored side assignments and a disjoint heldout family.
        for partition in ("training_episodes", "heldout_episodes"):
            first = tuple(asdict(step.offer) for step in getattr(report.arms[0], partition))
            for arm in report.arms[1:]:
                self.assertEqual(first, tuple(asdict(step.offer) for step in getattr(arm, partition)))

    def test_identical_fresh_seed_and_binding_reproduce_exact_report(self):
        one = run(seed=3456)
        two = run(seed=3456)
        self.assertEqual(one, two)
        self.assertNotEqual(one.train_schedule_digest, one.eval_schedule_digest)
        self.assertNotEqual(one.train_schedule_digest, run(seed=3457).train_schedule_digest)

    def test_default_fixture_reports_weight_effect_without_inventing_recurrence_benefit(self):
        # Registered fixed-seed smoke result, not an assertion of a broader
        # statistical gain or subjective pleasure. Recurrence-off matches the
        # ONLINE heldout chosen quality here and is BETTER prequentially.
        report = run(binding=BINDING, training_pairs=24, evaluation_pairs=24)
        online, zero, no_recurrence, zero_no_recurrence = report.arms
        self.assertGreater(online.weight_update_count, 0)
        self.assertEqual(zero.weight_update_count, 0)
        self.assertEqual(zero.training_policy.learning_rate, 0.0)
        self.assertEqual(zero.learned_weights_digest, report.base_weights_digest)
        self.assertNotEqual(online.learned_weights_digest, report.base_weights_digest)
        self.assertEqual(no_recurrence.final_recurrent_state, RecurrentState())
        self.assertEqual(no_recurrence.recurrent_policy.decision_gain, 0.0)
        self.assertGreater(no_recurrence.weight_update_count, 0)
        self.assertEqual(zero_no_recurrence.weight_update_count, 0)
        self.assertEqual(zero_no_recurrence.learned_weights_digest, report.base_weights_digest)
        self.assertEqual(zero_no_recurrence.final_recurrent_state, RecurrentState())
        self.assertEqual(no_recurrence.frozen_weight_state_reset_choice_differences, 0)
        self.assertEqual(zero_no_recurrence.frozen_weight_state_reset_choice_differences, 0)
        self.assertEqual(zero.frozen_weight_state_reset_primary_choice_differences, 24)
        self.assertAlmostEqual(online.heldout.mean_realized_quality, 0.589439, places=6)
        self.assertAlmostEqual(zero.heldout.mean_realized_quality, 0.509364, places=6)
        self.assertAlmostEqual(no_recurrence.heldout.mean_realized_quality, 0.589439, places=6)
        # Primary quality is one preregistered parity side per *independent*
        # heldout pair; the both-sides mean above is only a symmetry diagnostic.
        self.assertAlmostEqual(online.heldout_independent_primary.mean_realized_quality,
                               0.589439, places=6)
        self.assertAlmostEqual(zero.heldout_independent_primary.mean_realized_quality,
                               0.506786, places=6)
        self.assertAlmostEqual(no_recurrence.heldout_independent_primary.mean_realized_quality,
                               0.589439, places=6)
        self.assertAlmostEqual(zero_no_recurrence.heldout_independent_primary.mean_realized_quality,
                               0.511942, places=6)
        self.assertAlmostEqual(online.heldout_independent_primary.mean_chosen_prediction_squared_error,
                               0.006542, places=6)
        self.assertAlmostEqual(zero.heldout_independent_primary.mean_chosen_prediction_squared_error,
                               0.019086, places=6)
        self.assertAlmostEqual(no_recurrence.heldout_independent_primary.mean_chosen_prediction_squared_error,
                               0.006038, places=6)
        self.assertAlmostEqual(zero_no_recurrence.heldout_independent_primary.mean_chosen_prediction_squared_error,
                               0.013465, places=6)
        self.assertGreater(no_recurrence.prequential.mean_realized_quality,
                           online.prequential.mean_realized_quality)
        self.assertAlmostEqual(online.prequential.mean_realized_quality, 0.529856, places=6)
        self.assertAlmostEqual(zero.prequential.mean_realized_quality, 0.499450, places=6)
        self.assertAlmostEqual(no_recurrence.prequential.mean_realized_quality, 0.545892, places=6)
        self.assertAlmostEqual(online.heldout.mean_chosen_prediction_squared_error, 0.006542, places=6)
        self.assertAlmostEqual(zero.heldout.mean_chosen_prediction_squared_error, 0.016276, places=6)
        self.assertAlmostEqual(no_recurrence.heldout.mean_chosen_prediction_squared_error, 0.006038, places=6)
        # A different preregistered seed does NOT support a universal online
        # or recurrence benefit; report such null/adverse controls honestly.
        seed_seven = run(binding=BINDING, training_pairs=24, evaluation_pairs=24, seed=7)
        self.assertAlmostEqual(
            seed_seven.arms[0].heldout_independent_primary.mean_realized_quality,
            seed_seven.arms[1].heldout_independent_primary.mean_realized_quality,
        )
        self.assertLess(
            seed_seven.arms[0].heldout_independent_primary.mean_realized_quality,
            seed_seven.arms[2].heldout_independent_primary.mean_realized_quality,
        )

    def test_quality_metrics_are_observed_selected_only_and_finite(self):
        for seed in (1, 7, 12345, 99999):
            report = run(seed=seed)
            for arm in report.arms:
                for episodes, metrics in (
                    (arm.training_episodes, arm.prequential),
                    (arm.heldout_episodes, arm.heldout),
                    (tuple(step for step in arm.heldout_episodes
                           if step.offer.side == step.offer.pair_index % 2),
                     arm.heldout_independent_primary),
                ):
                    qualities = [step.outcome.measured_quality for step in episodes]
                    self.assertAlmostEqual(metrics.mean_realized_quality,
                                           math.fsum(qualities) / len(qualities))
                    self.assertTrue(math.isfinite(metrics.mean_chosen_prediction_squared_error))
                    self.assertGreaterEqual(metrics.mean_chosen_prediction_squared_error, 0.0)
                    self.assertLessEqual(metrics.mean_chosen_prediction_squared_error, 1.0)
                    for step in episodes:
                        self.assertEqual(step.choice.binding, BINDING)
                        self.assertEqual(step.outcome.action, step.choice.chosen_action)

    def test_runtime_order_is_choice_commit_resolve_observe_not_preloaded_training(self):
        events = []
        original_choose = OnlineWeightTrial.choose
        original_evaluate = OnlineWeightTrial.evaluate_choice
        original_commit = OutcomeArena.commit
        original_resolve = OutcomeArena.resolve
        original_observe = OnlineWeightTrial.observe

        def choose(model, probe, **keywords):
            events.append("CHOOSE")
            return original_choose(model, probe, **keywords)

        def evaluate(model, probe, **keywords):
            events.append("EVALUATE")
            return original_evaluate(model, probe, **keywords)

        def commit(arena, offer, *args, **keywords):
            events.append("COMMIT")
            return original_commit(arena, offer, *args, **keywords)

        def resolve(arena, ticket):
            events.append("RESOLVE")
            return original_resolve(arena, ticket)

        def observe(model, outcome):
            events.append("OBSERVE")
            return original_observe(model, outcome)

        with patch.object(OnlineWeightTrial, "choose", choose), \
             patch.object(OnlineWeightTrial, "evaluate_choice", evaluate), \
             patch.object(OnlineWeightTrial, "observe", observe), \
             patch.object(OutcomeArena, "commit", commit), \
             patch.object(OutcomeArena, "resolve", resolve):
            run(training_pairs=1, evaluation_pairs=1)
        self.assertEqual(
            events,
            (["CHOOSE", "COMMIT", "RESOLVE", "OBSERVE"] +
             ["EVALUATE", "COMMIT", "RESOLVE"] * 2) * 4,
        )

    def test_missing_outcome_fails_closed_and_accounts_for_denominator(self):
        restored = []
        observed = []
        original_restore = OnlineWeightTrial.restore
        original_observe = OnlineWeightTrial.observe

        def restore(model):
            receipt = original_restore(model)
            restored.append(receipt)
            return receipt

        def observe(model, outcome):
            observed.append(outcome)
            return original_observe(model, outcome)

        with patch.object(OutcomeArena, "resolve", return_value=None), \
             patch.object(OnlineWeightTrial, "restore", restore), \
             patch.object(OnlineWeightTrial, "observe", observe):
            with self.assertRaises(ClosedLoopAborted) as caught:
                run(training_pairs=2, evaluation_pairs=1)
        aborted = caught.exception
        self.assertEqual(aborted.arm_id, ARM_ONLINE)
        self.assertEqual(aborted.planned_episodes, 4)
        self.assertEqual(aborted.completed_episodes, 0)
        self.assertEqual(aborted.missing_outcomes, 1)
        self.assertEqual(aborted.aborted_episodes, 3)
        self.assertEqual(observed, [])
        self.assertEqual(len(restored), 1)
        self.assertTrue(restored[0].closed)
        self.assertTrue(restored[0].cancelled_pending_decision)
        self.assertEqual(restored[0].restored_digest, digest_weights((0.0, 0.0)))

    def test_substituted_action_outcome_rejected_before_training(self):
        original_resolve = OutcomeArena.resolve
        observed = []

        def substituted(arena, ticket):
            valid = original_resolve(arena, ticket)
            return replace(valid, action="B" if valid.action == "A" else "A")

        def reject_observe(_model, outcome):
            observed.append(outcome)
            raise AssertionError("substituted outcome must never reach optimizer")

        with patch.object(OutcomeArena, "resolve", substituted), \
             patch.object(OnlineWeightTrial, "observe", reject_observe):
            with self.assertRaises(ClosedLoopAborted) as caught:
                run(training_pairs=1, evaluation_pairs=1)
        self.assertIn("outcome/commitment binding mismatch", str(caught.exception))
        self.assertEqual(caught.exception.missing_outcomes, 1)
        self.assertEqual(observed, [])

    def test_hostile_ticket_alias_and_outcome_checksum_substitution_rejected(self):
        result = run(training_pairs=1, evaluation_pairs=1)
        step = result.arms[0].training_episodes[0]

        def validate(ticket, outcome):
            return _validate_resolution(
                step.offer, step.choice, ticket, outcome,
                result.evaluator_id_declared, result.rubric_digest_declared,
            )

        self.assertIsNone(validate(step.commitment, step.outcome))
        for field, malicious in (
            ("episode_id", "different-episode"),
            ("model_id", "different-model"),
            ("runtime_instance_id", "different-runtime"),
            ("commit_id", "0" * 64),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding mismatch"):
                validate(replace(step.commitment, **{field: malicious}), step.outcome)
        with self.assertRaisesRegex(ValueError, "evidence checksum mismatch"):
            validate(step.commitment, replace(step.outcome, evidence_id="0" * 64))

    def test_arbitrary_sensor_error_counts_missing_without_message_guessing(self):
        for failure in (RuntimeError("sensor offline"), TimeoutError("server lag")):
            with self.subTest(error=type(failure).__name__):
                with patch.object(OutcomeArena, "resolve", side_effect=failure):
                    with self.assertRaises(ClosedLoopAborted) as caught:
                        run(training_pairs=1, evaluation_pairs=1)
                self.assertEqual(caught.exception.planned_episodes, 3)
                self.assertEqual(caught.exception.completed_episodes, 0)
                self.assertEqual(caught.exception.missing_outcomes, 1)
                self.assertEqual(caught.exception.aborted_episodes, 2)

    def test_heldout_measurement_failure_counts_completed_training_without_scoring(self):
        original_resolve = OutcomeArena.resolve
        seen = []

        def fail_in_heldout(arena, ticket):
            partition = arena._issued[ticket.task_id].partition
            seen.append(partition)
            if partition == "EVAL":
                raise TimeoutError("readout unavailable")
            return original_resolve(arena, ticket)

        with patch.object(OutcomeArena, "resolve", fail_in_heldout):
            with self.assertRaises(ClosedLoopAborted) as caught:
                run(training_pairs=1, evaluation_pairs=1)
        self.assertEqual(seen, ["TRAIN", "EVAL"])
        self.assertEqual(caught.exception.planned_episodes, 3)
        self.assertEqual(caught.exception.completed_episodes, 1)
        self.assertEqual(caught.exception.missing_outcomes, 1)
        self.assertEqual(caught.exception.aborted_episodes, 1)

    def test_hosted_looking_caller_label_cannot_reclassify_toy_execution(self):
        binding = ModelBinding("gpt-6.1", "hosted", "chat", "pairs")
        result = run(binding=binding)
        self.assertEqual(result.binding.model_id, "gpt-6.1")
        self.assertTrue(result.binding_is_caller_declared)
        self.assertEqual(result.execution_target_kind, "SYNTHETIC_NUMERIC_SURROGATE")
        self.assertFalse(result.hosted_model_modified)
        self.assertFalse(result.externally_authenticated)

    def test_bounded_parameters_and_two_feature_identity(self):
        for invalid in (0, -1, True, 65, 1.0):
            with self.subTest(training_pairs=invalid), self.assertRaises(ValueError):
                run(training_pairs=invalid)
        for invalid in (0, True, 129, 1.5):
            with self.subTest(evaluation_pairs=invalid), self.assertRaises(ValueError):
                run(evaluation_pairs=invalid)
        for base in ((0.0,), (0.0, 0.0, 0.0), (math.nan, 0.0)):
            with self.subTest(base=base), self.assertRaises(ValueError):
                run(base_weights=base)
        with self.assertRaises(ValueError):
            run(learning_rate=-0.1)
        with self.assertRaises(ValueError):
            run(seed=2 ** 40)
        with self.assertRaises(TypeError):
            run(binding="hosted-model")


if __name__ == "__main__":
    unittest.main()
