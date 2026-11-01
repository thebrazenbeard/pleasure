"""End-to-end, actual numeric runner -> source-only evidence bridge checks."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import unittest

from pleasure_experiment.closed_loop_experiment import (
    ARM_NO_RECURRENCE, ARM_ONLINE, ARM_ZERO, run_closed_loop_experiment,
)
from pleasure_experiment.online_evidence_bridge import (
    _quality, build_online_evidence_bundle, build_online_evidence_journal,
)
from pleasure_experiment.online_learning_receipts import (
    OnlineEvidenceJournal, receipt_sha256,
)
from pleasure_experiment.weight_delta_lab import ModelBinding, digest_weights
from pleasure_experiment.weight_first_trial import RecurrentState


def actual_report():
    return run_closed_loop_experiment(
        binding=ModelBinding(
            model_id="evidence-bridge-tiny-surrogate",
            model_revision="fixed-r1",
            runtime_instance_id="study-1",
            feature_schema_id="two-feature-1",
        ),
        seed=1234, training_pairs=4, evaluation_pairs=3,
    )


class OnlineEvidenceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = actual_report()

    def bundle(self, report=None, arm_id=ARM_ONLINE):
        return build_online_evidence_bundle(
            report or self.report,
            arm_id=arm_id,
            session_id=f"bridge-source-{arm_id}",
            session_epoch_id="bridge-epoch-1",
        )

    def test_real_online_update_consumed_and_exact_readback_chain(self):
        for arm in self.report.arms:
            with self.subTest(arm=arm.arm_id):
                bundle = self.bundle(arm_id=arm.arm_id)
                journal = bundle.training_journal
                events = journal.receipts
                self.assertEqual(len(events), 3 * len(arm.training_episodes))
                self.assertEqual(len(bundle.heldout_evidence), len(arm.heldout_episodes))
                self.assertEqual(bundle.capability_tier, "SOURCE_ONLY_UNAUTHENTICATED")
                self.assertEqual(bundle.phenomenology, "UNRESOLVED")
                before = arm.initial_weights_digest
                for i, episode in enumerate(arm.training_episodes):
                    action, outcome, optimizer = events[3 * i:3 * (i + 1)]
                    self.assertEqual((action.kind, outcome.kind, optimizer.kind),
                                     ("ACTION_COMMITTED", "OUTCOME_OBSERVED", "OPTIMIZER_REPORTED"))
                    self.assertEqual(action.action_commit_id, episode.commitment.commit_id)
                    self.assertEqual(action.decision_digest, episode.choice.decision_digest)
                    self.assertEqual(action.payload.predicted_quality,
                                     episode.update.prediction_before)
                    self.assertEqual(action.payload.adapter_before_sha256, before)
                    self.assertEqual(outcome.event_id, episode.outcome.evidence_id)
                    self.assertEqual(outcome.payload.measured_quality,
                                     episode.update.measured_quality)
                    self.assertEqual(outcome.payload.rubric_sha256,
                                     self.report.rubric_digest_declared)
                    self.assertEqual(optimizer.payload.adapter_before_sha256,
                                     episode.update.before_digest)
                    self.assertEqual(optimizer.payload.adapter_after_sha256,
                                     episode.update.after_digest)
                    self.assertEqual(optimizer.payload.declared_optimizer_steps, 1)
                    self.assertIsNone(optimizer.payload.trainer_receipt_claim_sha256)
                    self.assertIsNone(optimizer.payload.gradient_summary_claim_sha256)
                    self.assertIsNone(action.payload.execution_receipt_claim_sha256)
                    self.assertIsNone(outcome.payload.custody_claim_sha256)
                    self.assertEqual(outcome.payload.evidence_class, "UNAUTHENTICATED_SOURCE_ONLY")
                    self.assertEqual(optimizer.previous_event_sha256, receipt_sha256(outcome))
                    before = episode.update.after_digest
                self.assertEqual(before, arm.learned_weights_digest)
                self.assertEqual(journal.head_sha256, receipt_sha256(events[-1]))

    def test_actual_heldout_choices_read_only_without_invented_update(self):
        bundle = self.bundle()
        arm = next(arm for arm in self.report.arms if arm.arm_id == ARM_ONLINE)
        self.assertTrue(bundle.heldout_evidence)
        for evidence, episode in zip(bundle.heldout_evidence, arm.heldout_episodes):
            self.assertEqual(evidence.update_status, "READ_ONLY_NO_UPDATE")
            self.assertEqual(evidence.schema, "PLEASURE_READ_ONLY_EVAL_SOURCE_V1")
            self.assertEqual(evidence.action_commit_id, episode.commitment.commit_id)
            self.assertEqual(evidence.decision_digest, episode.choice.decision_digest)
            self.assertEqual(evidence.learned_weights_sha256, arm.learned_weights_digest)
            self.assertEqual(evidence.measured_quality, episode.outcome.measured_quality)
            self.assertEqual(evidence.evaluator_id, episode.outcome.evaluator_id)
            self.assertEqual(evidence.rubric_sha256, episode.outcome.rubric_digest)
            self.assertEqual(evidence.capability_tier, "SOURCE_ONLY_UNAUTHENTICATED")
        self.assertTrue(all(item.kind != "ABORTED" for item in bundle.training_journal.receipts))
        self.assertEqual(len(bundle.training_journal.receipts), 3 * len(arm.training_episodes))

    def test_journal_export_import_after_actual_numeric_run(self):
        journal = self.bundle().training_journal
        encoded = journal.export_json()
        reconstructed = OnlineEvidenceJournal.import_json(encoded)
        self.assertEqual(reconstructed.export_json(), encoded)
        self.assertEqual(reconstructed.head_sha256, journal.head_sha256)
        self.assertEqual(reconstructed.receipts, journal.receipts)
        self.assertNotIn("private_message", encoded)
        self.assertNotIn("host_qualified", encoded)
        self.assertIn("SOURCE_ONLY_UNAUTHENTICATED", encoded)
        journal2 = build_online_evidence_journal(
            self.report, arm_id=ARM_ONLINE,
            session_id="bridge-source-ONLINE_FEEDBACK",
            session_epoch_id="bridge-epoch-1",
        )
        self.assertEqual(journal2.export_json(), encoded)

    def test_tampered_real_after_digest_rejected_even_when_self_consistent_flags(self):
        arm = self.report.arms[0]
        episode = arm.training_episodes[0]
        forged_update = replace(
            episode.update, after_digest="f" * 64,
            update_detected=True, effect_classification="WEIGHT_CHANGE_SOURCE_LOCAL",
        )
        forged_episode = replace(episode, update=forged_update)
        forged_arm = replace(arm, training_episodes=(forged_episode,) + arm.training_episodes[1:])
        forged_report = replace(self.report, arms=(forged_arm,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "before/after weight readback|pre-outcome decision|final learned"):
            self.bundle(forged_report)

    def test_tampered_action_commit_digest_rejected(self):
        arm = self.report.arms[0]
        episode = arm.training_episodes[0]
        tampered = replace(episode, commitment=replace(episode.commitment, commit_id="f" * 64))
        forged = replace(self.report, arms=(replace(
            arm, training_episodes=(tampered,) + arm.training_episodes[1:]),) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "arena commit digest mismatch"):
            self.bundle(forged)

    def test_wrong_chosen_action_and_swapped_outcome_rejected(self):
        arm = self.report.arms[0]
        first, second = arm.training_episodes[:2]
        wrong_action = replace(first, commitment=replace(first.commitment, action=(
            "B" if first.choice.chosen_action == "A" else "A"
        )))
        forged = replace(self.report, arms=(replace(
            arm, training_episodes=(wrong_action,) + arm.training_episodes[1:]),) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "arena commitment/action"):
            self.bundle(forged)
        swapped_outcome = replace(first, outcome=second.outcome)
        forged = replace(self.report, arms=(replace(
            arm, training_episodes=(swapped_outcome,) + arm.training_episodes[1:]),) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "arena resolved outcome|aggregated prequential"):
            self.bundle(forged)

    def test_same_evaluator_wrong_resolved_action_rejected(self):
        arm = self.report.arms[0]
        first = arm.training_episodes[0]
        opposite = "A" if first.outcome.action == "B" else "B"
        tampered = replace(first, outcome=replace(first.outcome, action=opposite))
        forged = replace(self.report, arms=(replace(
            arm, training_episodes=(tampered,) + arm.training_episodes[1:]),) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "arena resolved outcome"):
            self.bundle(forged)

    def test_forged_chosen_appraisal_and_matching_update_prediction_rejected(self):
        arm = self.report.arms[0]
        first = arm.training_episodes[0]
        selected_a = first.choice.chosen_action == "A"
        new_prediction = 0.123456789
        choice = replace(first.choice, **(
            {"appraisal_a": new_prediction} if selected_a else {"appraisal_b": new_prediction}
        ))
        changed = replace(first, choice=choice, update=replace(
            first.update, prediction_before=new_prediction,
        ))
        episodes = (changed,) + arm.training_episodes[1:]
        forged_arm = replace(arm, training_episodes=episodes, prequential=_quality(episodes))
        forged = replace(self.report, arms=(forged_arm,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "appraisal/score/action numeric replay"):
            self.bundle(forged)

    def test_forged_after_weights_digest_and_effect_flags_rejected_by_sgd_replay(self):
        arm = self.report.arms[0]
        first = arm.training_episodes[0]
        wrong_weights = (0.99, -0.99)
        wrong_digest = digest_weights(wrong_weights)
        update = replace(
            first.update, after_digest=wrong_digest,
            update_detected=True, effect_classification="WEIGHT_CHANGE_SOURCE_LOCAL",
        )
        changed = replace(first, after_weights=wrong_weights, update=update)
        forged_arm = replace(arm, training_episodes=(changed,) + arm.training_episodes[1:])
        forged = replace(self.report, arms=(forged_arm,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "actual numerical SGD after-weights replay"):
            self.bundle(forged)

    def test_forged_chosen_outcome_and_consistent_evidence_hash_rejected(self):
        zero_arm = next(arm for arm in self.report.arms if arm.arm_id == ARM_ZERO)
        first = zero_arm.training_episodes[0]
        forged_quality = 0.1 if first.outcome.measured_quality > 0.5 else 0.9
        token = ["outcome", first.commitment.commit_id, forged_quality,
                 first.outcome.rubric_digest]
        forged_evidence_id = sha256(
            json.dumps(token, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest()
        mutated = replace(first, outcome=replace(
            first.outcome, measured_quality=forged_quality, evidence_id=forged_evidence_id,
        ), update=replace(
            first.update, measured_quality=forged_quality, evidence_id=forged_evidence_id,
        ))
        episodes = (mutated,) + zero_arm.training_episodes[1:]
        forged_arm = replace(zero_arm, training_episodes=episodes,
                             prequential=_quality(episodes))
        forged = replace(self.report, arms=tuple(
            forged_arm if arm.arm_id == ARM_ZERO else arm for arm in self.report.arms
        ))
        with self.assertRaisesRegex(ValueError, "chosen-only outcome replay/readback mismatch"):
            self.bundle(forged, arm_id=ARM_ZERO)

    def test_forged_heldout_quality_metric_rejected(self):
        arm = self.report.arms[0]
        false_value = 0.99 if arm.heldout.mean_realized_quality != 0.99 else 0.01
        forged_arm = replace(arm, heldout=replace(
            arm.heldout, mean_realized_quality=false_value,
        ))
        forged = replace(self.report, arms=(forged_arm,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "aggregated prequential or heldout"):
            self.bundle(forged)

    def test_forged_recurrent_state_rejected(self):
        arm = self.report.arms[0]
        first = arm.training_episodes[0]
        forged_state = RecurrentState(0.123, -0.234)
        changed = replace(first, update=replace(
            first.update, recurrent_state_after=forged_state,
        ))
        forged = replace(self.report, arms=(replace(
            arm, training_episodes=(changed,) + arm.training_episodes[1:]),) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "recurrent-state replay"):
            self.bundle(forged)

    def test_wrong_evaluator_rubric_or_model_binding_rejected(self):
        arm = self.report.arms[0]
        episode = arm.training_episodes[0]
        cases = (
            ("evaluator", replace(episode, outcome=replace(episode.outcome,
                                                          evaluator_id="bogus-evaluator"))),
            ("rubric", replace(episode, outcome=replace(episode.outcome,
                                                      rubric_digest="a" * 64))),
            ("model", replace(episode, choice=replace(
                episode.choice, binding=replace(episode.choice.binding, model_id="wrong")))),
        )
        for label, malformed in cases:
            with self.subTest(case=label):
                forged = replace(self.report, arms=(replace(
                    arm, training_episodes=(malformed,) + arm.training_episodes[1:]),)
                    + self.report.arms[1:])
                with self.assertRaisesRegex(
                    ValueError, "binding mismatch|readback mismatch|arena resolved outcome"
                ):
                    self.bundle(forged)

    def test_missing_outcome_or_episode_not_silently_dropped(self):
        arm = self.report.arms[0]
        incomplete = replace(arm, training_episodes=arm.training_episodes[:-1])
        forged = replace(self.report, arms=(incomplete,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "incomplete or missing"):
            self.bundle(forged)
        incomplete = replace(arm, heldout_episodes=arm.heldout_episodes[:-1])
        forged = replace(self.report, arms=(incomplete,) + self.report.arms[1:])
        with self.assertRaisesRegex(ValueError, "incomplete or missing"):
            self.bundle(forged)

    def test_heldout_is_not_reflected_into_train_journal_and_tamper_rejected(self):
        arm = self.report.arms[0]
        eval_episode = arm.heldout_episodes[0]
        wrong = replace(eval_episode, choice=replace(
            eval_episode.choice, weights_digest=arm.initial_weights_digest))
        forged = replace(self.report, arms=(replace(
            arm, heldout_episodes=(wrong,) + arm.heldout_episodes[1:]),) + self.report.arms[1:])
        if arm.initial_weights_digest != arm.learned_weights_digest:
            with self.assertRaisesRegex(ValueError, "pre-outcome decision"):
                self.bundle(forged)
        bundle = self.bundle()
        self.assertTrue(all(e.payload.partition == "TRAIN"
                            for e in bundle.training_journal.receipts
                            if e.kind == "ACTION_COMMITTED"))

    def test_arm_control_semantics_actual_zero_and_disabled_recurrence(self):
        zero = self.bundle(arm_id=ARM_ZERO)
        updates = [e for e in zero.training_journal.receipts if e.kind == "OPTIMIZER_REPORTED"]
        self.assertTrue(updates)
        self.assertTrue(all(e.payload.arm == "zero" for e in updates))
        self.assertTrue(all(e.payload.declared_optimizer_steps == 1 for e in updates))
        self.assertTrue(all(e.payload.adapter_after_sha256 == e.payload.adapter_before_sha256
                            for e in updates))
        no_rec = self.bundle(arm_id=ARM_NO_RECURRENCE)
        self.assertTrue(all(e.payload.arm == "correct_outcome"
                            for e in no_rec.training_journal.receipts
                            if e.kind == "OPTIMIZER_REPORTED"))

    def test_source_boundary_promotion_rejected(self):
        with self.assertRaisesRegex(ValueError, "promoted host authority"):
            self.bundle(replace(self.report, externally_authenticated=True))
        with self.assertRaisesRegex(ValueError, "promoted host authority"):
            self.bundle(replace(self.report, hosted_model_modified=True))
        with self.assertRaisesRegex(ValueError, "unknown closed-loop arm"):
            self.bundle(arm_id="invented-arm")


if __name__ == "__main__":
    unittest.main()
