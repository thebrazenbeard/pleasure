"""Source-only bridge: executed numeric arena outcomes -> inert receipt chain.

This module reads *already completed* ClosedLoopReport snapshots. It does not
choose, resolve, train, restore, authorize, or authenticate anything. Numeric
parameter hashes establish source-local readback consistency only; an arena
evaluator is not an independent external or hosted-model evidence authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import fsum

from pleasure_experiment.closed_loop_experiment import (
    ARM_NO_RECURRENCE, ARM_ONLINE, ARM_ZERO, ARM_ZERO_NO_RECURRENCE,
    ArmReport, ClosedLoopReport, HeldoutEpisode, QualityMetrics, TrainingEpisode,
)
from pleasure_experiment.online_learning_receipts import (
    ActionCommitted, OnlineEvidenceJournal, OnlineEvidenceReceipt,
    OptimizerReported, OutcomeObserved, classify_digests, receipt_sha256,
)
from pleasure_experiment.outcome_arena import (
    EVIDENCE_CLASS as ARENA_EVIDENCE_CLASS,
    OutcomeArena,
)
from pleasure_experiment.online_weight_trial import _choice_digest
from pleasure_experiment.weight_delta_lab import (
    ModelBinding, digest_weights, predict_appraisal, single_outcome_step,
)
from pleasure_experiment.weight_first_trial import DecisionProbe, RecurrentState


ARM_MAP = {
    ARM_ONLINE: "correct_outcome",
    ARM_ZERO: "zero",
    ARM_NO_RECURRENCE: "correct_outcome",
    ARM_ZERO_NO_RECURRENCE: "zero",
}
BRIDGE_EVIDENCE_TIER = "SOURCE_ONLY_UNAUTHENTICATED"


@dataclass(frozen=True)
class ReadOnlyEvaluationEvidence:
    """Actual EVAL decision and outcome; explicitly no optimizer report."""

    task_id: str
    episode_id: str
    goal_instance_id: str
    action_commit_id: str
    action_choice: str
    decision_digest: str
    offer_nonce_sha256: str
    model_binding_sha256: str
    learned_weights_sha256: str
    evaluator_id: str
    rubric_sha256: str
    outcome_evidence_sha256: str
    predicted_quality: float
    measured_quality: float
    schema: str = "PLEASURE_READ_ONLY_EVAL_SOURCE_V1"
    update_status: str = "READ_ONLY_NO_UPDATE"
    capability_tier: str = BRIDGE_EVIDENCE_TIER
    phenomenology: str = "UNRESOLVED"


@dataclass(frozen=True)
class OnlineEvidenceBridgeResult:
    training_journal: OnlineEvidenceJournal
    heldout_evidence: tuple[ReadOnlyEvaluationEvidence, ...]
    capability_tier: str = BRIDGE_EVIDENCE_TIER
    phenomenology: str = "UNRESOLVED"


def _digest(value: object) -> str:
    """Canonical content fingerprint, not a signature or data-rights claim."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _arena_digest(*values: object) -> str:
    """Mirror the actual arena's documented deterministic source-local IDs."""
    return sha256(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _schedule_digest(offers: tuple) -> str:
    return sha256(json.dumps([
        (offer.task_id, offer.episode_id, offer.partition, offer.family_id,
         offer.pair_index, offer.side, offer.nonce,
         tuple(v.hex() for v in offer.features_a),
         tuple(v.hex() for v in offer.features_b))
        for offer in offers
    ], separators=(",", ":")).encode("ascii")).hexdigest()


def _quality(records: tuple[TrainingEpisode | HeldoutEpisode, ...]) -> QualityMetrics:
    if not records:
        raise ValueError("source quality metrics require nonempty episodes")
    predictions = tuple(
        record.choice.appraisal_a if record.choice.chosen_action == "A"
        else record.choice.appraisal_b for record in records
    )
    outcomes = tuple(record.outcome.measured_quality for record in records)
    selected_a = sum(record.choice.chosen_action == "A" for record in records)
    return QualityMetrics(
        episodes=len(records), chosen_a=selected_a,
        chosen_b=len(records) - selected_a,
        mean_realized_quality=fsum(outcomes) / len(records),
        mean_chosen_prediction_squared_error=fsum(
            (pred - observed) ** 2 for pred, observed in zip(predictions, outcomes)
        ) / len(records),
    )


def _next_recurrent_state(
    prior: RecurrentState, action: str, observed: float, arm: ArmReport,
) -> RecurrentState:
    """Replay chosen-only recurrence from the actual observed score."""
    a = arm.recurrent_policy.retention * prior.a
    b = arm.recurrent_policy.retention * prior.b
    impact = arm.recurrent_policy.outcome_gain * (2.0 * observed - 1.0)
    if action == "A":
        a += impact
    else:
        b += impact
    return RecurrentState(max(-1.0, min(1.0, a)), max(-1.0, min(1.0, b)))


def _validate_report(report: ClosedLoopReport, arm_id: str) -> ArmReport:
    if type(report) is not ClosedLoopReport:
        raise TypeError("report must be ClosedLoopReport")
    if type(arm_id) is not str or arm_id not in ARM_MAP:
        raise ValueError("unknown closed-loop arm")
    if type(report.binding) is not ModelBinding:
        raise ValueError("report model binding mismatch")
    matches = [arm for arm in report.arms if arm.arm_id == arm_id]
    if len(matches) != 1 or type(matches[0]) is not ArmReport:
        raise ValueError("report must contain exactly one requested arm")
    arm = matches[0]
    if report.hosted_model_modified or report.externally_authenticated or report.authorized_effects:
        raise ValueError("source-only bridge cannot ingest promoted host authority claims")
    if report.execution_target_kind != "SYNTHETIC_NUMERIC_SURROGATE" or report.phenomenology != "UNRESOLVED":
        raise ValueError("report execution target or phenomenology claim mismatch")
    if (len(arm.training_episodes) != arm.completed_training_episodes
            or arm.completed_training_episodes != arm.planned_training_episodes
            or len(arm.heldout_episodes) != arm.completed_heldout_episodes
            or arm.completed_heldout_episodes != arm.planned_heldout_episodes
            or arm.missing_outcome_count != 0):
        raise ValueError("incomplete or missing closed-loop episode record")
    if arm.planned_training_episodes <= 0 or arm.planned_heldout_episodes <= 0:
        raise ValueError("closed-loop evidence must include training and heldout trials")
    if arm.initial_weights_digest != report.base_weights_digest:
        raise ValueError("arm initial/base model digest mismatch")
    if (digest_weights(arm.initial_weights) != arm.initial_weights_digest
            or digest_weights(arm.learned_weights) != arm.learned_weights_digest):
        raise ValueError("arm numeric weight readback does not match exact digests")
    if arm_id in (ARM_ZERO, ARM_ZERO_NO_RECURRENCE) and arm.training_policy.learning_rate != 0.0:
        raise ValueError("zero-learning control has nonzero learning-rate policy")
    if arm_id in (ARM_NO_RECURRENCE, ARM_ZERO_NO_RECURRENCE) and (
        arm.recurrent_policy.retention != 0.0
        or arm.recurrent_policy.outcome_gain != 0.0
        or arm.recurrent_policy.decision_gain != 0.0
    ):
        raise ValueError("recurrence-disabled control has live recurrent policy")
    restore = arm.restore
    if (not restore.closed or restore.cancelled_pending_decision
            or restore.before_digest != arm.learned_weights_digest
            or restore.restored_digest != arm.initial_weights_digest
            or restore.update_count != len(arm.training_episodes)):
        raise ValueError("source-local restore/readback record mismatch")
    if arm.weight_update_count != sum(ep.update.update_detected for ep in arm.training_episodes):
        raise ValueError("arm weight update count mismatch")
    if _schedule_digest(tuple(ep.offer for ep in arm.training_episodes)) != report.train_schedule_digest:
        raise ValueError("training schedule digest mismatch")
    if _schedule_digest(tuple(ep.offer for ep in arm.heldout_episodes)) != report.eval_schedule_digest:
        raise ValueError("heldout schedule digest mismatch")
    independent = tuple(
        ep for ep in arm.heldout_episodes if ep.offer.side == ep.offer.pair_index % 2
    )
    if (arm.independent_training_pairs != len(arm.training_episodes)
            or arm.independent_heldout_pairs != len(independent)
            or len(independent) * 2 != len(arm.heldout_episodes)):
        raise ValueError("independent heldout/training pair count mismatch")
    if (arm.prequential != _quality(arm.training_episodes)
            or arm.heldout != _quality(arm.heldout_episodes)
            or arm.heldout_independent_primary != _quality(independent)):
        raise ValueError("aggregated prequential or heldout outcome metrics mismatch")
    reset_choice_count = sum(
        ep.choice.chosen_action != (
            "A" if ep.choice.appraisal_a > ep.choice.appraisal_b else "B"
        ) for ep in arm.heldout_episodes
    )
    primary_reset_count = sum(
        ep.choice.chosen_action != (
            "A" if ep.choice.appraisal_a > ep.choice.appraisal_b else "B"
        ) for ep in independent
    )
    if (arm.frozen_weight_state_reset_choice_differences != reset_choice_count
            or arm.frozen_weight_state_reset_primary_choice_differences != primary_reset_count):
        raise ValueError("fixed-weight state reset effect count mismatch")
    return arm


def _validate_episode(
    episode: TrainingEpisode | HeldoutEpisode, report: ClosedLoopReport,
    arena: OutcomeArena, arm: ArmReport, expected_weights: tuple[float, ...],
    expected_state: RecurrentState, session_nonce: str, *, train: bool, ordinal: int,
) -> None:
    if type(episode) is not (TrainingEpisode if train else HeldoutEpisode):
        raise ValueError("unexpected episode class or missing update")
    offer, choice, commitment, outcome = (
        episode.offer, episode.choice, episode.commitment, episode.outcome
    )
    # Reissue public deterministic offer IDs/features without resolving any
    # counterfactual outcome. This is source-level consistency, not custody.
    expected_offer = arena.issue(
        partition="TRAIN" if train else "EVAL",
        family_id=offer.family_id, pair_index=offer.pair_index, side=offer.side,
    )
    if expected_offer != offer:
        raise ValueError("arena offer identity/features/nonce mismatch")
    before_digest = digest_weights(expected_weights)
    if (choice.binding != report.binding
            or choice.probe_id != offer.task_id or choice.episode_id != offer.episode_id
            or choice.source_partition != offer.partition
            or choice.offer_nonce_digest != sha256(offer.nonce.encode("utf-8")).hexdigest()
            or choice.weights_digest != before_digest
            or choice.chosen_action not in ("A", "B")
            or choice.step_index != ordinal):
        raise ValueError("pre-outcome decision/model/weights readback mismatch")
    appraisal_a = predict_appraisal(expected_weights, offer.features_a)
    appraisal_b = predict_appraisal(expected_weights, offer.features_b)
    score_a = appraisal_a + arm.recurrent_policy.decision_gain * expected_state.a
    score_b = appraisal_b + arm.recurrent_policy.decision_gain * expected_state.b
    expected_action = "A" if score_a > score_b else "B"
    if (choice.appraisal_a != appraisal_a or choice.appraisal_b != appraisal_b
            or choice.score_a != score_a or choice.score_b != score_b
            or choice.chosen_action != expected_action):
        raise ValueError("pre-outcome appraisal/score/action numeric replay mismatch")
    probe = DecisionProbe(offer.task_id, offer.episode_id, report.binding,
                          offer.features_a, offer.features_b)
    expected_choice_digest = _choice_digest(
        report.binding, probe, ordinal, before_digest, expected_state,
        appraisal_a, appraisal_b, score_a, score_b, expected_action,
        offer.nonce, offer.partition, session_nonce,
    )
    if (choice.decision_digest != expected_choice_digest
            or choice.session_nonce_digest != sha256(session_nonce.encode("utf-8")).hexdigest()):
        raise ValueError("pre-outcome decision digest/nonce numeric replay mismatch")
    chosen_prediction = choice.appraisal_a if choice.chosen_action == "A" else choice.appraisal_b
    if not 0 <= chosen_prediction <= 1:
        raise ValueError("chosen prediction out of bounds")
    if (commitment.task_id != offer.task_id
            or commitment.episode_id != offer.episode_id
            or commitment.action != choice.chosen_action
            or commitment.model_id != report.binding.model_id
            or commitment.runtime_instance_id != report.binding.runtime_instance_id
            or commitment.decision_digest != choice.decision_digest):
        raise ValueError("arena commitment/action/model/decision binding mismatch")
    expected_commit = _arena_digest(
        "commit", offer.task_id, choice.chosen_action, report.binding.model_id,
        report.binding.runtime_instance_id, choice.decision_digest, offer.nonce,
    )
    if commitment.commit_id != expected_commit:
        raise ValueError("arena commit digest mismatch")
    if (outcome.task_id != offer.task_id
            or outcome.episode_id != offer.episode_id
            or outcome.partition != offer.partition
            or outcome.family_id != offer.family_id
            or outcome.action != choice.chosen_action
            or outcome.model_id != report.binding.model_id
            or outcome.runtime_instance_id != report.binding.runtime_instance_id
            or outcome.decision_digest != choice.decision_digest
            or outcome.evaluator_id != report.evaluator_id_declared
            or outcome.rubric_digest != report.rubric_digest_declared
            or outcome.evidence_class != ARENA_EVIDENCE_CLASS):
        raise ValueError("arena resolved outcome/action/evaluator/rubric mismatch")
    if type(outcome.measured_quality) not in (float, int) or not 0 <= outcome.measured_quality <= 1:
        raise ValueError("resolved outcome must be a finite [0,1] score")
    expected_outcome_id = _arena_digest(
        "outcome", commitment.commit_id, outcome.measured_quality, outcome.rubric_digest,
    )
    if outcome.evidence_id != expected_outcome_id:
        raise ValueError("arena outcome evidence digest mismatch")
    # Re-derive the CHOSEN action's actual deterministic arena resolution from
    # its public offer and committed choice. Comparing hashes against caller-
    # supplied quality alone would accept a self-consistent forged payoff.
    # This in-process oracle is still source-local, not an external custodian.
    replay_commitment = arena.commit(
        expected_offer, choice.chosen_action, choice.decision_digest,
        model_id=report.binding.model_id,
        runtime_instance_id=report.binding.runtime_instance_id,
    )
    replay_outcome = arena.resolve(replay_commitment)
    if replay_commitment != commitment or replay_outcome != outcome:
        raise ValueError("arena chosen-only outcome replay/readback mismatch")
    if train:
        update = episode.update
        if (episode.before_weights != expected_weights
                or digest_weights(episode.before_weights) != update.before_digest
                or digest_weights(episode.after_weights) != update.after_digest):
            raise ValueError("actual numeric before/after weight readback mismatch")
        if (update.decision_digest != choice.decision_digest
                or update.evidence_id != outcome.evidence_id
                or update.chosen_action != choice.chosen_action
                or update.prediction_before != chosen_prediction
                or update.measured_quality != outcome.measured_quality
                or update.before_digest != choice.weights_digest
                or update.step_index != ordinal + 1
                or update.evaluator_id_declared != outcome.evaluator_id
                or update.rubric_digest_declared != outcome.rubric_digest):
            raise ValueError("online optimizer update not bound to actual chosen outcome")
        chosen_features = offer.features_a if choice.chosen_action == "A" else offer.features_b
        feature_digest = _digest([v.hex() for v in chosen_features])
        if update.chosen_feature_digest != feature_digest:
            raise ValueError("online chosen-feature digest mismatch")
        expected_after = single_outcome_step(
            base_weights=arm.initial_weights,
            current_weights=expected_weights,
            chosen_features=chosen_features,
            observed_score=outcome.measured_quality,
            policy=arm.training_policy,
        )
        if episode.after_weights != expected_after:
            raise ValueError("actual numerical SGD after-weights replay mismatch")
        if (update.recurrent_state_before != expected_state
                or update.recurrent_state_after != _next_recurrent_state(
                    expected_state, choice.chosen_action, outcome.measured_quality, arm
                )):
            raise ValueError("actual chosen-outcome recurrent-state replay mismatch")
        changed = update.before_digest != update.after_digest
        if (update.update_detected is not changed or
                update.effect_classification != (
                    "WEIGHT_CHANGE_SOURCE_LOCAL" if changed else "NO_WEIGHT_CHANGE"
                )):
            raise ValueError("online update effect/readback classification mismatch")


def build_online_evidence_bundle(
    report: ClosedLoopReport,
    *,
    arm_id: str,
    session_id: str,
    session_epoch_id: str,
) -> OnlineEvidenceBridgeResult:
    """Encode completed TRAIN events, validate EVAL as separate read-only evidence.

    TRAIN: actual action -> actual chosen outcome -> actual OnlineUpdate readback.
    EVAL: actual action/outcome -> READ_ONLY_NO_UPDATE evidence. Heldout choices
    never become synthetic optimizer reports or authorization claims.
    """
    arm = _validate_report(report, arm_id)
    binding_digest = _digest(asdict(report.binding))
    journal = OnlineEvidenceJournal(
        session_id=session_id, session_epoch_id=session_epoch_id,
        model_binding_sha256=binding_digest,
    )
    arena = OutcomeArena(seed=report.seed, evaluator_id=report.evaluator_id_declared)
    if arena.rubric_digest != report.rubric_digest_declared:
        raise ValueError("arena rubric source binding mismatch")
    expected_weights = arm.initial_weights
    expected_state = RecurrentState()
    session_nonce = _digest(["CLOSED_LOOP_SESSION_V1", report.seed, arm_id])
    seen: set[tuple[str, str, str]] = set()
    for index, episode in enumerate(arm.training_episodes):
        train = True
        ordinal = index
        _validate_episode(
            episode, report, arena, arm, expected_weights,
            expected_state, session_nonce, train=train, ordinal=ordinal,
        )
        offer, choice, commitment, outcome = (
            episode.offer, episode.choice, episode.commitment, episode.outcome
        )
        key = (offer.task_id, offer.episode_id, offer.nonce)
        if key in seen:
            raise ValueError("duplicate arena task/episode/nonce in source report")
        seen.add(key)
        chosen_prediction = choice.appraisal_a if choice.chosen_action == "A" else choice.appraisal_b
        shared = dict(
            session_id=session_id,
            session_epoch_id=session_epoch_id,
            model_id=report.binding.model_id,
            model_revision=report.binding.model_revision,
            model_binding_sha256=binding_digest,
            runtime_instance_id=report.binding.runtime_instance_id,
            goal_instance_id=offer.family_id,
            task_id=offer.task_id,
            episode_id=offer.episode_id,
            decision_digest=choice.decision_digest,
            action_choice=choice.chosen_action,
            action_commit_id=commitment.commit_id,
            offer_nonce_sha256=choice.offer_nonce_digest,
        )
        seq = journal.next_sequence
        action = OnlineEvidenceReceipt(
            **shared,
            kind="ACTION_COMMITTED", event_id=commitment.commit_id,
            sequence=seq, previous_event_sha256=journal.head_sha256,
            recorded_at_ms=seq,
            payload=ActionCommitted(
                partition=offer.partition,
                goal_definition_sha256=_digest(
                    ["SOURCE_ONLY_FAMILY_GOAL_V1", offer.family_id,
                     report.binding.feature_schema_id]
                ),
                adapter_before_sha256=choice.weights_digest,
                predicted_quality=chosen_prediction,
                execution_receipt_claim_sha256=None,
                train_manifest_sha256=report.train_schedule_digest,
                holdout_manifest_sha256=report.eval_schedule_digest,
            ),
        )
        journal.append(action)
        outcome_receipt = OnlineEvidenceReceipt(
            **shared,
            kind="OUTCOME_OBSERVED", event_id=outcome.evidence_id,
            sequence=seq + 1, previous_event_sha256=receipt_sha256(action),
            recorded_at_ms=seq + 1,
            payload=OutcomeObserved(
                action_receipt_sha256=receipt_sha256(action),
                evidence_id=outcome.evidence_id,
                evaluator_id=outcome.evaluator_id,
                rubric_sha256=outcome.rubric_digest,
                outcome_evidence_sha256=_digest(asdict(outcome)),
                custody_claim_sha256=None,
                measured_quality=outcome.measured_quality,
            ),
        )
        journal.append(outcome_receipt)
        update = episode.update
        before, after = update.before_digest, update.after_digest
        arm_name = ARM_MAP[arm_id]
        claimed_steps = 1  # Exactly one source-local call to single_outcome_step.
        optimizer = OnlineEvidenceReceipt(
            **shared,
            kind="OPTIMIZER_REPORTED",
            event_id=_digest(["ONLINE_SOURCE_ONLY_STEP_V1", commitment.commit_id,
                              outcome.evidence_id, arm_name, claimed_steps]),
            sequence=seq + 2, previous_event_sha256=receipt_sha256(outcome_receipt),
            recorded_at_ms=seq + 2,
            payload=OptimizerReported(
                action_receipt_sha256=receipt_sha256(action),
                outcome_receipt_sha256=receipt_sha256(outcome_receipt),
                arm=arm_name,
                train_example_manifest_sha256=report.train_schedule_digest,
                holdout_exclusion_manifest_sha256=report.eval_schedule_digest,
                declared_optimizer_steps=claimed_steps,
                gradient_summary_claim_sha256=None,
                trainer_receipt_claim_sha256=None,
                adapter_before_sha256=before,
                adapter_after_sha256=after,
                checkpoint_claim_sha256=None,
                weight_change_evidence=classify_digests(before, after),
            ),
        )
        journal.append(optimizer)
        expected_weights = episode.after_weights
        expected_state = episode.update.recurrent_state_after
    if expected_weights != arm.learned_weights or digest_weights(expected_weights) != arm.learned_weights_digest:
        raise ValueError("final learned adapter digest disagrees with actual update readback")
    if expected_state != arm.final_recurrent_state:
        raise ValueError("final recurrent state disagrees with online update readback")
    heldout: list[ReadOnlyEvaluationEvidence] = []
    for episode in arm.heldout_episodes:
        _validate_episode(
            episode, report, arena, arm, arm.learned_weights,
            arm.final_recurrent_state, session_nonce, train=False,
            ordinal=len(arm.training_episodes),
        )
        offer, choice, commitment, outcome = (
            episode.offer, episode.choice, episode.commitment, episode.outcome
        )
        key = (offer.task_id, offer.episode_id, offer.nonce)
        if key in seen:
            raise ValueError("duplicate TRAIN/heldout task, episode, or nonce")
        seen.add(key)
        heldout.append(ReadOnlyEvaluationEvidence(
            task_id=offer.task_id,
            episode_id=offer.episode_id,
            goal_instance_id=offer.family_id,
            action_commit_id=commitment.commit_id,
            action_choice=choice.chosen_action,
            decision_digest=choice.decision_digest,
            offer_nonce_sha256=choice.offer_nonce_digest,
            model_binding_sha256=binding_digest,
            learned_weights_sha256=arm.learned_weights_digest,
            evaluator_id=outcome.evaluator_id,
            rubric_sha256=outcome.rubric_digest,
            outcome_evidence_sha256=_digest(asdict(outcome)),
            predicted_quality=(choice.appraisal_a if choice.chosen_action == "A"
                               else choice.appraisal_b),
            measured_quality=outcome.measured_quality,
        ))
    return OnlineEvidenceBridgeResult(journal, tuple(heldout))


def build_online_evidence_journal(
    report: ClosedLoopReport,
    *,
    arm_id: str,
    session_id: str,
    session_epoch_id: str,
) -> OnlineEvidenceJournal:
    """TRAIN-only causal journal; validates EVAL without pretending it trained."""
    return build_online_evidence_bundle(
        report, arm_id=arm_id, session_id=session_id,
        session_epoch_id=session_epoch_id,
    ).training_journal
