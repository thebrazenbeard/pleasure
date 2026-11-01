"""Framework-neutral control tests; these DO NOT train a language model."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from unittest import TestCase, mock
import json

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, OpenWeightChoice, OpenWeightUpdate, OpenWeightRestore,
)
from pleasure_experiment.open_weight_trial import (
    OpenWeightTrialRunner, OpenWeightTrialAborted, _input_digest,
)
from pleasure_experiment.outcome_arena import OutcomeArena


class _FakeTokenPolicy:
    """Synthetic behavioral double for trial ORDERING, never an LLM witness."""

    def __init__(self, *, frozen: bool = False, forged_choice: bool = False,
                 forged_update: bool = False, mutate_during_infer: str = "",
                 dishonest_restore: bool = False, forged_scores: bool = False,
                 forged_action: bool = False, force_tie: bool = False,
                 promote_claim: str = "", invalid_session_digest: bool = False) -> None:
        self._binding = OpenWeightBinding(
            "fake-token-policy", "fixed-r1", "test-instance",
            "1" * 64, "2" * 64,
        )
        self.base = "b" * 64
        self.digest = self.base
        self.count = 0
        self.pending_choice = None
        self.closed = False
        self.frozen = frozen
        self.forged_choice = forged_choice
        self.forged_update = forged_update
        self.mutate_during_infer = mutate_during_infer
        self.dishonest_restore = dishonest_restore
        self.forged_scores = forged_scores
        self.forged_action = forged_action
        self.force_tie = force_tie
        self.promote_claim = promote_claim
        self.invalid_session_digest = invalid_session_digest
        self.history = []

    @property
    def binding(self):
        return self._binding

    @property
    def weights_digest(self):
        return self.digest

    @property
    def frozen_backbone_digest(self):
        return "c" * 64

    @property
    def step_index(self):
        return self.count

    @property
    def pending(self):
        return self.pending_choice is not None

    def infer(self, *, input_ids, offer_nonce, episode_id, source_partition,
              score_bias=(0.0, 0.0)):
        if self.closed or self.pending:
            raise RuntimeError("fake policy not ready")
        pa = 0.5 if self.force_tie else (
            0.6 if input_ids[-1] % 2 == 0 else 0.4
        )
        pb = 1 - pa
        action = "A" if pa > pb else "B"
        digest = sha256(json.dumps(
            [self.digest, self.count, offer_nonce, episode_id,
             source_partition, input_ids, action],
            separators=(",", ":"),
        ).encode("ascii")).hexdigest()
        choice = OpenWeightChoice(
            binding=self.binding, episode_id=episode_id,
            source_partition=source_partition, chosen_action=action,
            decision_digest=digest, weights_digest=self.digest,
            frozen_backbone_digest="c" * 64,
            input_ids_digest=_input_digest(input_ids),
            offer_nonce_digest=sha256(offer_nonce.encode()).hexdigest(),
            session_nonce_digest="d" * 64,
            prob_a=pa, prob_b=pb, score_a=pa, score_b=pb,
            step_index=self.count,
        )
        if self.forged_choice and source_partition == "TRAIN":
            choice = replace(choice, offer_nonce_digest="e" * 64)
        if self.forged_scores and source_partition == "TRAIN":
            choice = replace(choice, score_a=999.0, score_b=-999.0)
        if self.forged_action and source_partition == "TRAIN":
            choice = replace(
                choice, chosen_action="B" if choice.chosen_action == "A" else "A"
            )
        if self.promote_claim == "choice" and source_partition == "TRAIN":
            choice = replace(
                choice, evidence_class="HOSTED_GPT_TRAINED",
                phenomenology="PLEASURE_PROVEN",
            )
        if self.invalid_session_digest and source_partition == "TRAIN":
            choice = replace(choice, session_nonce_digest="not-a-digest")
        if source_partition == "TRAIN":
            self.pending_choice = choice
        self.history.append(("infer", source_partition, episode_id, self.digest))
        if self.mutate_during_infer == source_partition:
            self.digest = sha256(f"improper-infer-{episode_id}".encode()).hexdigest()
        return choice

    def observe(self, outcome):
        if self.closed or self.pending_choice is None:
            raise RuntimeError("no pending action")
        choice = self.pending_choice
        if (
            outcome.episode_id != choice.episode_id
            or outcome.decision_digest != choice.decision_digest
            or outcome.chosen_action != choice.chosen_action
        ):
            raise ValueError("chosen outcome mismatch")
        before = self.digest
        if not self.frozen:
            self.digest = sha256(
                f"{before}:{outcome.measured_quality}:{self.count}".encode()
            ).hexdigest()
        self.count += 1
        self.pending_choice = None
        result = OpenWeightUpdate(
            decision_digest=outcome.decision_digest,
            evidence_id=outcome.evidence_id,
            chosen_action=outcome.chosen_action,
            measured_quality=outcome.measured_quality,
            prediction_before=(
                choice.prob_a if choice.chosen_action == "A" else choice.prob_b
            ),
            before_digest=before, after_digest=self.digest,
            frozen_backbone_digest="c" * 64,
            step_index=self.count,
            gradient_norm=0.1 if not self.frozen else 0.0,
            delta_l2=0.1 if not self.frozen else 0.0,
            optimizer_applied=not self.frozen,
            optimizer_step_executed=not self.frozen,
        )
        self.history.append(("observe", "TRAIN", outcome.episode_id, before))
        if self.forged_update:
            result = replace(result, after_digest="f" * 64)
        if self.promote_claim == "update":
            result = replace(
                result, evidence_class="HOSTED_GPT_TRAINED",
                phenomenology="PLEASURE_PROVEN",
            )
        return result

    def restore(self):
        self.history.append(("restore", "", "", self.digest))
        before = self.digest
        self.digest = self.base
        self.pending_choice = None
        self.closed = True
        result = OpenWeightRestore(
            before_digest=before, restored_digest=self.base,
            frozen_backbone_digest="c" * 64,
            optimizer_state_cleared=not self.dishonest_restore,
            cancelled_pending=False,
            closed=True,
        )
        return replace(
            result, evidence_class="HOSTED_GPT_TRAINED", phenomenology="PLEASURE_PROVEN"
        ) if self.promote_claim == "restore" else result


def _encode(offer):
    """Only offered public features can reach this deterministic fake encoder."""
    values = offer.features_a + offer.features_b
    return (13, 17, *(int(v * 100) + 30 for v in values))


class OpenWeightTrialRunnerTests(TestCase):
    def test_action_then_chosen_outcome_then_update_and_readonly_eval(self):
        fake = _FakeTokenPolicy()
        report = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            seed=7, training_pairs=3, heldout_pairs=4,
        ).run()
        self.assertEqual((len(report.training), len(report.heldout)), (3, 8))
        self.assertEqual(report.prequential.episodes, 3)
        self.assertEqual(report.heldout_primary.episodes, 4)
        self.assertEqual(report.heldout_mirrored_diagnostic.episodes, 8)
        self.assertEqual(report.observed_weight_changes, 3)
        self.assertEqual(report.optimizer_steps_reported, 3)
        self.assertNotEqual(report.initial_weights_digest, report.learned_weights_digest)
        self.assertEqual(report.restored_weights_digest, report.initial_weights_digest)
        self.assertTrue(report.restored)
        self.assertFalse(report.externally_authenticated)
        self.assertFalse(report.hosted_model_modified)
        self.assertFalse(report.reward_prediction_calibration_measured)
        self.assertFalse(report.real_llm_weight_change_established)
        self.assertEqual(report.backend_kind, "SOURCE_ONLY_TEST_DOUBLE_OR_UNKNOWN")
        self.assertEqual([entry[0] for entry in fake.history[:6]],
                         ["infer", "observe"] * 3)
        self.assertEqual(sum(entry[0] == "observe" for entry in fake.history), 3)
        self.assertEqual(sum(entry[1] == "EVAL" for entry in fake.history), 8)
        self.assertTrue(fake.closed)
        for episode in report.training:
            self.assertEqual(episode.outcome.action, episode.choice.chosen_action)
            self.assertEqual(episode.update.evidence_id, episode.outcome.evidence_id)
            self.assertEqual(episode.choice.weights_digest,
                             episode.update.before_digest)
        for episode in report.heldout:
            self.assertEqual(episode.choice.weights_digest, report.learned_weights_digest)

    def test_frozen_zero_learning_control_receives_feedback_without_weight_change(self):
        fake = _FakeTokenPolicy(frozen=True)
        result = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=3, heldout_pairs=2,
        ).run()
        self.assertEqual(result.observed_weight_changes, 0)
        self.assertEqual(result.optimizer_steps_reported, 0)
        self.assertEqual(result.learned_weights_digest, result.initial_weights_digest)
        self.assertEqual(len(result.training), 3)
        self.assertTrue(all(not e.update.optimizer_applied for e in result.training))

    def test_same_public_schedule_between_online_and_frozen_controls(self):
        online = OpenWeightTrialRunner(
            adapter=_FakeTokenPolicy(), encode_offer=_encode,
            seed=17, training_pairs=4, heldout_pairs=3,
        ).run()
        frozen = OpenWeightTrialRunner(
            adapter=_FakeTokenPolicy(frozen=True), encode_offer=_encode,
            seed=17, training_pairs=4, heldout_pairs=3,
        ).run()
        self.assertEqual(online.train_schedule_sha256, frozen.train_schedule_sha256)
        self.assertEqual(online.heldout_schedule_sha256, frozen.heldout_schedule_sha256)
        self.assertEqual([r.offer.episode_id for r in online.heldout],
                         [r.offer.episode_id for r in frozen.heldout])

    def test_forged_choice_binding_fails_closed_and_restores(self):
        fake = _FakeTokenPolicy(forged_choice=True)
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=2, heldout_pairs=2,
        )
        with self.assertRaisesRegex(OpenWeightTrialAborted, "VALIDATE_CHOICE") as caught:
            runner.run()
        self.assertTrue(caught.exception.restoration_succeeded)
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(fake.closed)

    def test_forged_scores_with_valid_probabilities_rejected_before_commit(self):
        fake = _FakeTokenPolicy(forged_scores=True)
        with self.assertRaisesRegex(
            OpenWeightTrialAborted, "score/probability readback mismatch"
        ) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(caught.exception.restoration_succeeded)
        self.assertTrue(fake.closed)

    def test_forged_action_disagrees_with_zero_bias_policy(self):
        fake = _FakeTokenPolicy(forged_action=True)
        with self.assertRaisesRegex(
            OpenWeightTrialAborted, "chosen action does not match"
        ) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(caught.exception.restoration_succeeded)

    def test_injected_hosted_phenomenology_claims_in_receipts_are_rejected(self):
        for stage in ("choice", "update", "restore"):
            with self.subTest(stage=stage):
                fake = _FakeTokenPolicy(promote_claim=stage)
                with self.assertRaises(OpenWeightTrialAborted):
                    OpenWeightTrialRunner(
                        adapter=fake, encode_offer=_encode,
                        training_pairs=1, heldout_pairs=1,
                    ).run()
                self.assertTrue(fake.closed)

    def test_malformed_session_nonce_digest_rejected(self):
        fake = _FakeTokenPolicy(invalid_session_digest=True)
        with self.assertRaisesRegex(
            OpenWeightTrialAborted, "session nonce digest"
        ) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(caught.exception.restoration_succeeded)

    def test_equal_action_probabilities_choose_b_tie_rule(self):
        fake = _FakeTokenPolicy(force_tie=True)
        result = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=2, heldout_pairs=2,
        ).run()
        self.assertTrue(all(
            episode.choice.chosen_action == "B" for episode in result.training
        ))
        self.assertTrue(all(
            episode.choice.chosen_action == "B" for episode in result.heldout
        ))
        self.assertEqual(result.heldout_primary.mean_chosen_action_probability, 0.5)

    def test_forged_update_readback_aborts_without_partial_success(self):
        fake = _FakeTokenPolicy(forged_update=True)
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=2, heldout_pairs=2,
        )
        with self.assertRaisesRegex(OpenWeightTrialAborted, "CHECK_READBACK") as caught:
            runner.run()
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(caught.exception.restoration_succeeded)
        self.assertTrue(fake.closed)

    def test_policy_infer_may_not_modify_weights_before_outcome(self):
        fake = _FakeTokenPolicy(mutate_during_infer="TRAIN")
        with self.assertRaisesRegex(
            OpenWeightTrialAborted, "TRAIN inference mutated"
        ) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertEqual(caught.exception.completed_episodes, 0)
        self.assertTrue(caught.exception.restoration_succeeded)

    def test_readonly_eval_mutation_is_rejected_and_restored(self):
        fake = _FakeTokenPolicy(mutate_during_infer="EVAL")
        with self.assertRaisesRegex(
            OpenWeightTrialAborted, "EVAL inference mutated"
        ) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertEqual(caught.exception.completed_episodes, 1)
        self.assertTrue(caught.exception.restoration_succeeded)

    def test_false_restore_receipt_prevents_success_claim(self):
        fake = _FakeTokenPolicy(dishonest_restore=True)
        with self.assertRaises(OpenWeightTrialAborted) as caught:
            OpenWeightTrialRunner(
                adapter=fake, encode_offer=_encode,
                training_pairs=1, heldout_pairs=1,
            ).run()
        self.assertFalse(caught.exception.restoration_succeeded)
        self.assertTrue(fake.closed)

    def test_missing_outcome_counts_failed_measurement_and_restores(self):
        fake = _FakeTokenPolicy()
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=2, heldout_pairs=2,
        )
        with mock.patch.object(OutcomeArena, "resolve", side_effect=TimeoutError("offline")):
            with self.assertRaises(OpenWeightTrialAborted) as caught:
                runner.run()
        self.assertEqual(caught.exception.phase, "RESOLVE")
        self.assertEqual(caught.exception.missing_outcomes, 1)
        self.assertTrue(caught.exception.restoration_succeeded)
        self.assertEqual(caught.exception.aborted_episodes, 5)
        self.assertTrue(fake.closed)

    def test_tampered_resolved_outcome_rejected_before_adapter_training(self):
        fake = _FakeTokenPolicy()
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=2, heldout_pairs=2,
        )
        real_resolve = OutcomeArena.resolve
        def altered(arena, ticket):
            receipt = real_resolve(arena, ticket)
            return replace(receipt, measured_quality=0.0)
        with mock.patch.object(OutcomeArena, "resolve", altered):
            with self.assertRaises(OpenWeightTrialAborted) as caught:
                runner.run()
        self.assertEqual(caught.exception.phase, "VALIDATE_OUTCOME")
        self.assertEqual(caught.exception.missing_outcomes, 1)
        self.assertTrue(fake.closed)

    def test_invalid_encoder_rejected_without_entering_policy_or_network(self):
        fake = _FakeTokenPolicy()
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=lambda _: (True, -1),
            training_pairs=1, heldout_pairs=1,
        )
        with self.assertRaises(OpenWeightTrialAborted) as caught:
            runner.run()
        self.assertEqual(caught.exception.phase, "ENCODE")
        self.assertFalse(any(entry[0] == "infer" for entry in fake.history))
        self.assertTrue(fake.closed)

    def test_trial_is_one_use(self):
        fake = _FakeTokenPolicy()
        runner = OpenWeightTrialRunner(
            adapter=fake, encode_offer=_encode,
            training_pairs=1, heldout_pairs=1,
        )
        runner.run()
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            runner.run()
