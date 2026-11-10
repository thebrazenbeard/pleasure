"""Matched, source-only action -> measured outcome -> online optimizer trial.

This is a deterministic *tiny numerical surrogate* experiment. Its arena and
learning policy run in one Python process, so neither hashes nor evaluator IDs
authenticate independent custody, host execution, persistent state, GPT model
weights, or subjective experience. Nothing here uses files or network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
import json

from pleasure_experiment.online_weight_trial import (
    ChosenOutcome,
    OnlineChoice,
    OnlineRecurrentPolicy,
    OnlineUpdate,
    OnlineWeightTrial,
    RestoreReceipt,
)
from pleasure_experiment.outcome_arena import (
    CommitTicket,
    EVIDENCE_CLASS as ARENA_EVIDENCE_CLASS,
    OutcomeArena,
    OutcomeReceipt,
    TaskOffer,
)
from pleasure_experiment.weight_delta_lab import (
    MAX_CASES,
    ModelBinding,
    PHENOMENOLOGY,
    TrainingPolicy,
    digest_weights,
)
from pleasure_experiment.weight_first_trial import DecisionProbe, RecurrentState


SCHEMA_VERSION = "PLEASURE_CLOSED_LOOP_SOURCE_V1"
EVIDENCE_CLASS = "SOURCE_ONLY_MATCHED_ACTION_OUTCOMES_UNVERIFIED"
ARM_ONLINE = "ONLINE_FEEDBACK"
ARM_ZERO = "ZERO_LEARNING_RATE"
ARM_NO_RECURRENCE = "RECURRENT_DISABLED"
ARM_ZERO_NO_RECURRENCE = "ZERO_LEARNING_RATE_RECURRENCE_DISABLED"
TRAIN_FAMILY = "online_feedback"
EVAL_FAMILY = "independent_transfer"
MAX_TRAIN_PAIRS = 64  # One realized TRAIN outcome per pair; sides alternate.
MAX_EVAL_PAIRS = MAX_CASES // 2  # Each pair reserves two holdout episodes.
TRAIN_PHASES = ("ISSUE", "CHOOSE", "COMMIT", "RESOLVE", "OBSERVE")
EVAL_PHASES = ("ISSUE", "EVALUATE", "COMMIT", "RESOLVE")


@dataclass(frozen=True)
class TrainingEpisode:
    offer: TaskOffer
    choice: OnlineChoice
    commitment: CommitTicket
    outcome: OutcomeReceipt
    update: OnlineUpdate
    before_weights: tuple[float, ...]
    after_weights: tuple[float, ...]
    phases: tuple[str, ...] = TRAIN_PHASES


@dataclass(frozen=True)
class HeldoutEpisode:
    offer: TaskOffer
    choice: OnlineChoice
    commitment: CommitTicket
    outcome: OutcomeReceipt
    phases: tuple[str, ...] = EVAL_PHASES


@dataclass(frozen=True)
class QualityMetrics:
    episodes: int
    chosen_a: int
    chosen_b: int
    mean_realized_quality: float
    mean_chosen_prediction_squared_error: float


@dataclass(frozen=True)
class ArmReport:
    arm_id: str
    initial_weights: tuple[float, ...]
    learned_weights: tuple[float, ...]
    initial_weights_digest: str
    learned_weights_digest: str
    training_policy: TrainingPolicy
    recurrent_policy: OnlineRecurrentPolicy
    final_recurrent_state: RecurrentState
    training_episodes: tuple[TrainingEpisode, ...]
    heldout_episodes: tuple[HeldoutEpisode, ...]
    prequential: QualityMetrics
    heldout_independent_primary: QualityMetrics
    # Diagnostic only: both mirrored sides per pair share the SAME option
    # payoff structure and are not independent statistical samples.
    heldout: QualityMetrics
    weight_update_count: int
    frozen_weight_state_reset_choice_differences: int
    frozen_weight_state_reset_primary_choice_differences: int
    independent_training_pairs: int
    independent_heldout_pairs: int
    planned_training_episodes: int
    completed_training_episodes: int
    planned_heldout_episodes: int
    completed_heldout_episodes: int
    missing_outcome_count: int
    restore: RestoreReceipt


@dataclass(frozen=True)
class ClosedLoopReport:
    seed: int
    binding: ModelBinding
    base_weights_digest: str
    evaluator_id_declared: str
    rubric_digest_declared: str
    train_schedule_digest: str
    eval_schedule_digest: str
    arms: tuple[ArmReport, ...]
    primary_heldout_selection_rule: str = "ONE_SIDE_PER_PAIR_INDEX_PARITY_V1"
    schema_version: str = SCHEMA_VERSION
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY
    execution_target_kind: str = "SYNTHETIC_NUMERIC_SURROGATE"
    binding_is_caller_declared: bool = True
    hosted_model_modified: bool = False
    externally_authenticated: bool = False
    # A full 2x2 arm grid is run, but each arm may train on DIFFERENT chosen
    # outcomes. This alone cannot identify a quality benefit from recurrence.
    full_factorial_controls_completed: bool = True
    recurrence_effect_causally_isolated: bool = False
    authorized_effects: tuple[str, ...] = ()


class ClosedLoopAborted(RuntimeError):
    """No partial success metric is issued after a failed measurement.

    A failed measurement counts as one missing/invalid outcome; subsequent
    scheduled episodes count as aborted, never as measured zero or silently
    dropped. These are source-local process counters, not a durable ledger.
    """

    def __init__(
        self, *, arm_id: str, planned: int, completed: int,
        missing: int, reason: str,
    ) -> None:
        super().__init__(
            f"{arm_id}: closed-loop aborted after {completed}/{planned} episodes; "
            f"missing/invalid outcomes={missing}; reason={reason}"
        )
        self.arm_id = arm_id
        self.planned_episodes = planned
        self.completed_episodes = completed
        self.missing_outcomes = missing
        self.aborted_episodes = planned - completed - missing


def _pair_count(value: int, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _probe(offer: TaskOffer, binding: ModelBinding) -> DecisionProbe:
    return DecisionProbe(
        probe_id=offer.task_id,
        episode_id=offer.episode_id,
        binding=binding,
        features_a=offer.features_a,
        features_b=offer.features_b,
    )


def _arena_local_digest(*fields: object) -> str:
    """Recompute the source-local arena ledger checksum, never a signature."""
    return sha256(
        json.dumps(fields, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def _validate_resolution(
    offer: TaskOffer, choice: OnlineChoice, commitment: CommitTicket,
    resolved: OutcomeReceipt, evaluator_id: str, rubric_digest: str,
) -> None:
    """Fail closed on missing, substituted, or wrongly attributed outcomes."""
    if not isinstance(resolved, OutcomeReceipt):
        raise ValueError("missing chosen-action outcome; training must stop")
    if (
        resolved.task_id != offer.task_id
        or resolved.episode_id != offer.episode_id
        or resolved.partition != offer.partition
        or resolved.family_id != offer.family_id
        or resolved.action != choice.chosen_action
        or resolved.decision_digest != choice.decision_digest
        or resolved.model_id != choice.binding.model_id
        or resolved.runtime_instance_id != choice.binding.runtime_instance_id
        or resolved.evaluator_id != evaluator_id
        or resolved.rubric_digest != rubric_digest
        or resolved.evidence_class != ARENA_EVIDENCE_CLASS
        or commitment.task_id != offer.task_id
        or commitment.episode_id != offer.episode_id
        or commitment.action != choice.chosen_action
        or commitment.decision_digest != choice.decision_digest
        or commitment.model_id != choice.binding.model_id
        or commitment.runtime_instance_id != choice.binding.runtime_instance_id
        or commitment.commit_id != _arena_local_digest(
            "commit", offer.task_id, choice.chosen_action,
            choice.binding.model_id, choice.binding.runtime_instance_id,
            choice.decision_digest, offer.nonce,
        )
    ):
        raise ValueError("chosen-action outcome/commitment binding mismatch")
    if isinstance(resolved.measured_quality, bool) or not isinstance(
        resolved.measured_quality, (int, float)
    ):
        raise ValueError("measured chosen-action quality must be finite in [0, 1]")
    try:
        quality = float(resolved.measured_quality)
    except (OverflowError, ValueError) as exc:
        raise ValueError("measured chosen-action quality must be finite in [0, 1]") from exc
    if not isfinite(quality) or not 0.0 <= quality <= 1.0:
        raise ValueError("measured chosen-action quality must be finite in [0, 1]")
    if not isinstance(resolved.evidence_id, str) or not resolved.evidence_id:
        raise ValueError("missing chosen-action evidence ID")
    if resolved.evidence_id != _arena_local_digest(
        "outcome", commitment.commit_id, resolved.measured_quality, rubric_digest,
    ):
        raise ValueError("source-local chosen-outcome evidence checksum mismatch")


def _quality(records: tuple[TrainingEpisode | HeldoutEpisode, ...]) -> QualityMetrics:
    # Each prediction was recorded before the particular realized quality was
    # resolved; the unchosen action's (counterfactual) score is never used.
    predicted = tuple(
        record.choice.appraisal_a if record.choice.chosen_action == "A"
        else record.choice.appraisal_b for record in records
    )
    observed = tuple(record.outcome.measured_quality for record in records)
    count = len(records)
    selected_a = sum(record.choice.chosen_action == "A" for record in records)
    return QualityMetrics(
        episodes=count, chosen_a=selected_a, chosen_b=count - selected_a,
        mean_realized_quality=fsum(observed) / count,
        mean_chosen_prediction_squared_error=fsum(
            (pred - value) ** 2 for pred, value in zip(predicted, observed)
        ) / count,
    )


def _schedule_digest(offers: tuple[TaskOffer, ...]) -> str:
    # Deterministic equality check for cross-arm public task allocation, not
    # secret randomness or independent-outcome authentication.
    fields = [
        (offer.task_id, offer.episode_id, offer.partition, offer.family_id,
         offer.pair_index, offer.side, offer.nonce,
         tuple(v.hex() for v in offer.features_a),
         tuple(v.hex() for v in offer.features_b))
        for offer in offers
    ]
    return sha256(json.dumps(fields, separators=(",", ":")).encode("ascii")).hexdigest()


def _run_arm(
    *, arm_id: str, seed: int, binding: ModelBinding,
    base_weights: tuple[float, float], evaluator_id: str,
    training_pairs: int, evaluation_pairs: int,
    policy: TrainingPolicy, recurrence: OnlineRecurrentPolicy,
) -> tuple[ArmReport, str, str, str]:
    arena = OutcomeArena(seed=seed, evaluator_id=evaluator_id)
    # Reserve unseen EVAL episode identities. Only their IDs, never features
    # or scores, are passed into the learner during training.
    eval_offers = tuple(
        arena.issue(partition="EVAL", family_id=EVAL_FAMILY, pair_index=index, side=side)
        for index in range(evaluation_pairs) for side in (0, 1)
    )
    trial = OnlineWeightTrial(
        base_weights=base_weights, binding=binding, training_policy=policy,
        expected_evaluator_id=evaluator_id,
        expected_rubric_digest=arena.rubric_digest,
        recurrent_policy=recurrence,
        reserved_holdout_episode_ids=tuple(offer.episode_id for offer in eval_offers),
        # Deliberately reproducible for paired experiments. Reusing this
        # caller-declared nonce across fresh runs does NOT prevent replay.
        trial_session_nonce=sha256(
            json.dumps(["CLOSED_LOOP_SESSION_V1", seed, arm_id],
                       separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    )
    train_records: list[TrainingEpisode] = []
    eval_records: list[HeldoutEpisode] = []
    train_offers: list[TaskOffer] = []
    planned = training_pairs + 2 * evaluation_pairs
    phase = "ISSUE"
    try:
        for index in range(training_pairs):
            # Mirrored sides of ONE pair expose the same two outcomes. Never
            # train on both as if they were independent observations; alternate
            # the displayed side across distinct pair indices instead.
            side = index % 2
            phase = "ISSUE"
            offer = arena.issue(partition="TRAIN", family_id=TRAIN_FAMILY,
                                pair_index=index, side=side)
            train_offers.append(offer)
            before_weights = trial.weights_readback()
            phase = "CHOOSE"
            choice = trial.choose(_probe(offer, binding), offer_nonce=offer.nonce,
                                  source_partition=offer.partition)
            phase = "COMMIT"
            commitment = arena.commit(
                offer, choice.chosen_action, choice.decision_digest,
                model_id=binding.model_id,
                runtime_instance_id=binding.runtime_instance_id,
            )
            phase = "RESOLVE"
            outcome = arena.resolve(commitment)
            phase = "VALIDATE_OUTCOME"
            _validate_resolution(offer, choice, commitment, outcome,
                                 evaluator_id, arena.rubric_digest)
            observed = ChosenOutcome(
                evidence_id=outcome.evidence_id,
                decision_digest=outcome.decision_digest,
                probe_id=offer.task_id, episode_id=offer.episode_id,
                binding=binding, chosen_action=outcome.action,
                evaluator_id=outcome.evaluator_id,
                rubric_digest=outcome.rubric_digest,
                measured_quality=outcome.measured_quality,
            )
            phase = "OBSERVE"
            update = trial.observe(observed)
            after_weights = trial.weights_readback()
            if (
                digest_weights(before_weights) != choice.weights_digest
                or update.before_digest != choice.weights_digest
                or update.after_digest != digest_weights(after_weights)
                or update.after_digest != trial.weights_digest
                or update.decision_digest != choice.decision_digest
                or update.evidence_id != outcome.evidence_id
            ):
                raise RuntimeError("online optimizer/readback binding mismatch")
            train_records.append(TrainingEpisode(
                offer, choice, commitment, outcome, update, before_weights, after_weights,
            ))

        learned_digest = trial.weights_digest
        learned_weights = trial.weights_readback()
        state = trial.state
        # Heldout scores are resolved AFTER all TRAIN updates; no outcomes are
        # admitted into the optimizer and evaluation never changes its state.
        for reserved_offer in eval_offers:
            phase = "ISSUE"
            # Explicitly reissue the same preregistered, publicly committed
            # heldout task immediately before its read-only decision.
            offer = arena.issue(
                partition="EVAL", family_id=EVAL_FAMILY,
                pair_index=reserved_offer.pair_index, side=reserved_offer.side,
            )
            if offer != reserved_offer:
                raise RuntimeError("heldout schedule diverged after preregistration")
            before_digest, before_step, before_state = (
                trial.weights_digest, trial.step_index, trial.state
            )
            phase = "EVALUATE"
            choice = trial.evaluate_choice(
                _probe(offer, binding), offer_nonce=offer.nonce,
                source_partition=offer.partition,
            )
            phase = "COMMIT"
            commitment = arena.commit(
                offer, choice.chosen_action, choice.decision_digest,
                model_id=binding.model_id,
                runtime_instance_id=binding.runtime_instance_id,
            )
            phase = "RESOLVE"
            outcome = arena.resolve(commitment)
            phase = "VALIDATE_OUTCOME"
            _validate_resolution(offer, choice, commitment, outcome,
                                 evaluator_id, arena.rubric_digest)
            phase = "CHECK_READBACK"
            if (
                trial.weights_digest != before_digest
                or trial.step_index != before_step
                or trial.state != before_state
            ):
                raise RuntimeError("heldout evaluation illegally changed training state")
            eval_records.append(HeldoutEpisode(offer, choice, commitment, outcome))
        initial_digest = digest_weights(base_weights)
    except Exception as exc:
        completed = len(train_records) + len(eval_records)
        # Count the failed outcome by typed execution phase, independent of
        # a sensor's arbitrary exception class or message. All remaining
        # unscheduled episodes count as aborted, never as success.
        invalid_measurement = phase in ("RESOLVE", "VALIDATE_OUTCOME")
        raise ClosedLoopAborted(
            arm_id=arm_id, planned=planned, completed=completed,
            missing=1 if invalid_measurement else 0,
            reason=f"{phase}: {exc}",
        ) from exc
    finally:
        # This state exists only in RAM; restoration is local, not persistent
        # checkpoint recovery or a provider-owned rollback guarantee.
        restoration = trial.restore()

    if restoration.restored_digest != initial_digest or not restoration.closed:
        raise RuntimeError("source-only baseline restore failed")
    training = tuple(train_records)
    evaluation = tuple(eval_records)
    # A single preregistered parity side PER independent heldout pair is the
    # primary quality sample. Both mirrored sides remain a diagnostic only;
    # averaging them can cancel opposite action preferences by symmetry.
    independent_evaluation = tuple(
        record for record in evaluation
        if record.offer.side == record.offer.pair_index % 2
    )
    if len(independent_evaluation) != evaluation_pairs:
        raise RuntimeError("heldout independent-pair selection diverged")
    # Direct, fixed-weight inference ablation: these read-only counterfactual
    # ACTIONS set recurrent state to zero while preserving each heldout
    # prediction/weight vector. No hypothetical outcome is generated or fed
    # into training, and changed choices are not a claim of improved quality.
    reset_choice_differences = sum(
        step.choice.chosen_action != (
            "A" if step.choice.appraisal_a > step.choice.appraisal_b else "B"
        )
        for step in evaluation
    )
    primary_reset_differences = sum(
        step.choice.chosen_action != (
            "A" if step.choice.appraisal_a > step.choice.appraisal_b else "B"
        )
        for step in independent_evaluation
    )
    return (
        ArmReport(
            arm_id=arm_id, initial_weights=base_weights,
            learned_weights=learned_weights,
            initial_weights_digest=initial_digest,
            learned_weights_digest=learned_digest, final_recurrent_state=state,
            training_policy=policy, recurrent_policy=recurrence,
            training_episodes=training, heldout_episodes=evaluation,
            prequential=_quality(training),
            heldout_independent_primary=_quality(independent_evaluation),
            heldout=_quality(evaluation),
            weight_update_count=sum(item.update.update_detected for item in training),
            frozen_weight_state_reset_choice_differences=reset_choice_differences,
            frozen_weight_state_reset_primary_choice_differences=primary_reset_differences,
            independent_training_pairs=training_pairs,
            independent_heldout_pairs=evaluation_pairs,
            planned_training_episodes=training_pairs,
            completed_training_episodes=len(training),
            planned_heldout_episodes=2 * evaluation_pairs,
            completed_heldout_episodes=len(evaluation),
            missing_outcome_count=0, restore=restoration,
        ),
        _schedule_digest(tuple(train_offers)), _schedule_digest(eval_offers),
        arena.rubric_digest,
    )


def run_closed_loop_experiment(
    *, binding: ModelBinding, seed: int = 12345,
    training_pairs: int = 24, evaluation_pairs: int = 24,
    base_weights: tuple[float, float] = (0.0, 0.0),
    learning_rate: float = 0.6, max_delta_l2: float = 0.8,
    evaluator_id: str = "source-only-outcome-oracle",
    recurrent_policy: OnlineRecurrentPolicy = OnlineRecurrentPolicy(),
) -> ClosedLoopReport:
    """Execute matched 2x2 arms without external effects or reward leakage.

    Evaluation quality can be worse than controls; a successful run reports
    measurements, never presupposes improvement or experiential valence. The
    fixed-weight state reset checks CHOICE changes only, not hypothetical
    selected-action quality or a statistically independent state effect.
    """
    train_count = _pair_count(training_pairs, name="training_pairs", maximum=MAX_TRAIN_PAIRS)
    eval_count = _pair_count(evaluation_pairs, name="evaluation_pairs", maximum=MAX_EVAL_PAIRS)
    if not isinstance(binding, ModelBinding):
        raise TypeError("binding must be ModelBinding")
    fixed_binding = ModelBinding(**binding.__dict__)
    if not isinstance(base_weights, tuple) or len(base_weights) != 2:
        raise ValueError("arena appraisal requires exactly two initial numeric weights")
    base_digest = digest_weights(base_weights)
    base = tuple(float(v) for v in base_weights)
    if not isinstance(recurrent_policy, OnlineRecurrentPolicy):
        raise TypeError("recurrent_policy must be OnlineRecurrentPolicy")
    recurrence = OnlineRecurrentPolicy(**recurrent_policy.__dict__)
    normal = TrainingPolicy(learning_rate, max_delta_l2, train_count, train_count)
    zero = TrainingPolicy(0.0, max_delta_l2, train_count, train_count)
    no_recurrence = OnlineRecurrentPolicy(retention=0.0, outcome_gain=0.0, decision_gain=0.0)
    configurations = (
        (ARM_ONLINE, normal, recurrence),
        (ARM_ZERO, zero, recurrence),
        (ARM_NO_RECURRENCE, normal, no_recurrence),
        (ARM_ZERO_NO_RECURRENCE, zero, no_recurrence),
    )
    arms = []
    schedule_digests: set[tuple[str, str, str]] = set()
    for arm_name, policy, recurrence_mode in configurations:
        arm, train_schedule, eval_schedule, rubric = _run_arm(
            arm_id=arm_name, seed=seed, binding=fixed_binding, base_weights=base,
            evaluator_id=evaluator_id, training_pairs=train_count,
            evaluation_pairs=eval_count, policy=policy, recurrence=recurrence_mode,
        )
        arms.append(arm)
        schedule_digests.add((train_schedule, eval_schedule, rubric))
    if len(schedule_digests) != 1:
        raise RuntimeError("matched-control schedule or rubric diverged across arms")
    train_schedule, eval_schedule, rubric = next(iter(schedule_digests))
    return ClosedLoopReport(
        seed=seed, binding=fixed_binding, base_weights_digest=base_digest,
        evaluator_id_declared=evaluator_id, rubric_digest_declared=rubric,
        train_schedule_digest=train_schedule, eval_schedule_digest=eval_schedule,
        arms=tuple(arms),
    )
