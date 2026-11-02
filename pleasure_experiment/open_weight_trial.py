"""Opt-in A/B outcome loop around a caller-supplied token-level model adapter.

An actual torch adapter can perform gradient updates; a test double can also
satisfy this protocol. This source-local loop never authenticates the model,
checkpoint, evaluator, provider, or a subjective experience. The deterministic
payoff arena is a simulated toy task, not a real-world task execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import fsum, isfinite
from threading import Lock
from typing import Callable, Protocol
import json

from pleasure_experiment.open_weight_adapter import (
    OpenWeightBinding, OpenWeightChoice, OpenWeightOutcome,
    OpenWeightRestore, OpenWeightUpdate, TorchChosenOutcomeAdapter,
    EVIDENCE_CLASS as ADAPTER_EVIDENCE_CLASS,
    PHENOMENOLOGY as ADAPTER_PHENOMENOLOGY,
)
from pleasure_experiment.outcome_arena import (
    CommitTicket, OutcomeArena, OutcomeReceipt, TaskOffer,
    EVIDENCE_CLASS as ARENA_EVIDENCE_CLASS, _hex as _arena_hex,
)


EVIDENCE_CLASS = "SOURCE_LOCAL_LM_CHOSEN_OUTCOME_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
MAX_TRAINING_PAIRS = 64
MAX_HELDOUT_PAIRS = 64
MAX_INPUT_TOKENS = 256
TRAIN_FAMILY = "open_weight_train"
EVAL_FAMILY = "open_weight_holdout"


class ActionModelAdapter(Protocol):
    @property
    def binding(self) -> object: ...

    @property
    def weights_digest(self) -> str: ...

    @property
    def frozen_backbone_digest(self) -> str: ...

    @property
    def step_index(self) -> int: ...

    @property
    def pending(self) -> bool: ...

    def infer(
        self, *, input_ids: tuple[int, ...], offer_nonce: str,
        episode_id: str, source_partition: str,
        score_bias: tuple[float, float] = (0.0, 0.0),
    ) -> object: ...

    def observe(self, outcome: OpenWeightOutcome) -> object: ...
    def restore(self) -> object: ...


@dataclass(frozen=True)
class OpenWeightTrainingEpisode:
    offer: TaskOffer
    input_ids_sha256: str
    choice: object
    commitment: CommitTicket
    outcome: OutcomeReceipt
    update: object


@dataclass(frozen=True)
class OpenWeightHeldoutEpisode:
    offer: TaskOffer
    input_ids_sha256: str
    choice: object
    commitment: CommitTicket
    outcome: OutcomeReceipt


@dataclass(frozen=True)
class OpenWeightPolicyMetrics:
    episodes: int
    chosen_a: int
    chosen_b: int
    mean_realized_quality: float
    mean_chosen_action_probability: float


@dataclass(frozen=True)
class OpenWeightTrialReport:
    binding: object
    seed: int
    initial_weights_digest: str
    learned_weights_digest: str
    frozen_backbone_digest: str
    evaluator_id_declared: str
    rubric_digest_declared: str
    training: tuple[OpenWeightTrainingEpisode, ...]
    heldout: tuple[OpenWeightHeldoutEpisode, ...]
    prequential: OpenWeightPolicyMetrics
    heldout_primary: OpenWeightPolicyMetrics
    heldout_mirrored_diagnostic: OpenWeightPolicyMetrics
    observed_weight_changes: int
    optimizer_steps_reported: int
    train_schedule_sha256: str
    heldout_schedule_sha256: str
    restored_weights_digest: str
    restored: bool
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY
    hosted_model_modified: bool = False
    externally_authenticated: bool = False
    reward_prediction_calibration_measured: bool = False
    backend_kind: str = "CALLER_SUPPLIED_ADAPTER_OR_TEST_DOUBLE"
    real_llm_weight_change_established: bool = False


class OpenWeightTrialAborted(RuntimeError):
    """No partial success receipt after any failed measurement or optimizer."""

    def __init__(
        self, *, phase: str, planned: int, completed: int,
        missing_outcomes: int, restoration_succeeded: bool, reason: str,
    ) -> None:
        super().__init__(
            f"open-weight trial aborted in {phase}: {completed}/{planned} completed; "
            f"missing outcomes={missing_outcomes}; restored={restoration_succeeded}; "
            f"reason={reason}"
        )
        self.phase = phase
        self.planned_episodes = planned
        self.completed_episodes = completed
        self.missing_outcomes = missing_outcomes
        self.aborted_episodes = planned - completed - missing_outcomes
        self.restoration_succeeded = restoration_succeeded


def _pair_count(value: int, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _token_ids(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not 1 <= len(values) <= MAX_INPUT_TOKENS:
        raise ValueError("encoder must return 1..256 token IDs as a tuple")
    if any(type(value) is not int or value < 0 or value >= 2**31 for value in values):
        raise ValueError("token IDs must be bounded nonnegative integers")
    return values


def _input_digest(ids: tuple[int, ...]) -> str:
    return sha256(json.dumps(list(ids), separators=(",", ":")).encode("ascii")).hexdigest()


def _schedule_digest(offers: tuple[TaskOffer, ...]) -> str:
    return sha256(json.dumps([
        [o.task_id, o.episode_id, o.partition, o.family_id,
         o.pair_index, o.side, o.nonce,
         [v.hex() for v in o.features_a], [v.hex() for v in o.features_b]]
        for o in offers
    ], separators=(",", ":")).encode("ascii")).hexdigest()


def _chosen_probability(choice: object) -> float:
    a, b = choice.prob_a, choice.prob_b
    if (
        type(a) not in (int, float) or type(b) not in (int, float)
        or not isfinite(a) or not isfinite(b) or not 0 <= a <= 1
        or not 0 <= b <= 1 or abs((a + b) - 1.0) > 1e-5
    ):
        raise ValueError("A/B policy probabilities must be finite and normalized")
    return float(a if choice.chosen_action == "A" else b)


def _metrics(
    records: tuple[OpenWeightTrainingEpisode | OpenWeightHeldoutEpisode, ...],
) -> OpenWeightPolicyMetrics:
    if not records:
        raise ValueError("policy metrics require at least one episode")
    chosen_a = sum(record.choice.chosen_action == "A" for record in records)
    return OpenWeightPolicyMetrics(
        episodes=len(records), chosen_a=chosen_a, chosen_b=len(records) - chosen_a,
        mean_realized_quality=fsum(r.outcome.measured_quality for r in records) / len(records),
        mean_chosen_action_probability=fsum(
            _chosen_probability(r.choice) for r in records
        ) / len(records),
    )


def _check_choice(
    choice: object, *, binding: object, offer: TaskOffer,
    ids: tuple[int, ...], weights_digest: str, step: int,
) -> None:
    if type(choice) is not OpenWeightChoice:
        raise TypeError("inference must return an OpenWeightChoice")
    if (
        choice.binding != binding or choice.episode_id != offer.episode_id
        or choice.source_partition != offer.partition
        or choice.chosen_action not in ("A", "B")
        or choice.offer_nonce_digest != sha256(offer.nonce.encode("utf-8")).hexdigest()
        or choice.input_ids_digest != _input_digest(ids)
        or choice.weights_digest != weights_digest or choice.step_index != step
    ):
        raise ValueError("model inference identity, partition or readback mismatch")
    _sha(choice.decision_digest, "model decision digest")
    _sha(choice.session_nonce_digest, "model session nonce digest")
    _sha(choice.frozen_backbone_digest, "frozen backbone digest")
    if (
        choice.evidence_class != ADAPTER_EVIDENCE_CLASS
        or choice.phenomenology != ADAPTER_PHENOMENOLOGY
    ):
        raise ValueError("inference attempted unsupported host or phenomenology claim")
    _chosen_probability(choice)
    # The runner passes zero score bias to every inference. Probabilities are
    # therefore the exact decision scores, and the adapter's tie rule is B.
    # A syntactically valid digest is not evidence that these fields agree.
    if (
        type(choice.score_a) not in (int, float)
        or type(choice.score_b) not in (int, float)
        or not isfinite(choice.score_a) or not isfinite(choice.score_b)
        or choice.score_a != choice.prob_a
        or choice.score_b != choice.prob_b
    ):
        raise ValueError("inference score/probability readback mismatch")
    expected_action = "A" if choice.score_a > choice.score_b else "B"
    if choice.chosen_action != expected_action:
        raise ValueError("chosen action does not match the zero-bias A/B policy")


def _check_outcome(
    offer: TaskOffer, choice: object, ticket: CommitTicket, outcome: OutcomeReceipt,
    evaluator_id: str, rubric: str,
) -> None:
    binding = choice.binding
    if (
        not isinstance(outcome, OutcomeReceipt)
        or outcome.task_id != offer.task_id or outcome.episode_id != offer.episode_id
        or outcome.partition != offer.partition or outcome.family_id != offer.family_id
        or outcome.action != choice.chosen_action
        or outcome.decision_digest != choice.decision_digest
        or outcome.model_id != binding.model_id
        or outcome.runtime_instance_id != binding.runtime_instance_id
        or outcome.evaluator_id != evaluator_id or outcome.rubric_digest != rubric
        or outcome.evidence_class != ARENA_EVIDENCE_CLASS
        or ticket.task_id != offer.task_id or ticket.episode_id != offer.episode_id
        or ticket.action != choice.chosen_action
        or ticket.model_id != binding.model_id
        or ticket.runtime_instance_id != binding.runtime_instance_id
        or ticket.decision_digest != choice.decision_digest
        or ticket.commit_id != _arena_hex(
            "commit", offer.task_id, choice.chosen_action, binding.model_id,
            binding.runtime_instance_id, choice.decision_digest, offer.nonce,
        )
    ):
        raise ValueError("committed chosen-action outcome identity mismatch")
    score = outcome.measured_quality
    if type(score) not in (int, float) or not isfinite(score) or not 0 <= score <= 1:
        raise ValueError("chosen-action quality missing or invalid")
    if outcome.evidence_id != _arena_hex(
        "outcome", ticket.commit_id, score, rubric,
    ):
        raise ValueError("chosen-action outcome checksum mismatch")


class OpenWeightTrialRunner:
    """One-use toy environment driver with opt-in adapter training and rollback."""

    def __init__(
        self, *, adapter: ActionModelAdapter,
        encode_offer: Callable[[TaskOffer], tuple[int, ...]],
        seed: int = 12345, training_pairs: int = 8, heldout_pairs: int = 8,
        evaluator_id: str = "source-local-open-weight-oracle",
        on_learned_weights: Callable[[str], None] | None = None,
    ) -> None:
        train_n = _pair_count(training_pairs, "training_pairs", MAX_TRAINING_PAIRS)
        eval_n = _pair_count(heldout_pairs, "heldout_pairs", MAX_HELDOUT_PAIRS)
        if not callable(encode_offer):
            raise TypeError("encode_offer must be callable")
        if on_learned_weights is not None and not callable(on_learned_weights):
            raise TypeError("on_learned_weights must be callable")
        binding = adapter.binding
        if type(binding) is not OpenWeightBinding:
            raise TypeError("adapter must bind an OpenWeightBinding")
        binding = OpenWeightBinding(**binding.__dict__)
        for name in ("model_id", "model_revision", "runtime_instance_id",
                     "checkpoint_sha256", "tokenizer_sha256"):
            value = getattr(binding, name, None)
            if name.endswith("_sha256"):
                _sha(value, name)
            elif type(value) is not str or not value or len(value) > 256:
                raise ValueError(f"{name} is required as a bounded model identity")
        initial = _sha(adapter.weights_digest, "initial adapter weights")
        frozen_backbone = _sha(
            adapter.frozen_backbone_digest, "frozen backbone weights"
        )
        if adapter.step_index != 0 or adapter.pending:
            raise ValueError("adapter must have unused baseline and no pending action")
        arena = OutcomeArena(seed=seed, evaluator_id=evaluator_id)
        eval_offers = tuple(
            arena.issue(partition="EVAL", family_id=EVAL_FAMILY,
                        pair_index=i, side=side)
            for i in range(eval_n) for side in (0, 1)
        )
        self._lock = Lock()
        self._adapter, self._encoder = adapter, encode_offer
        self._binding, self._arena = binding, arena
        self._initial, self._frozen_backbone = initial, frozen_backbone
        self._seed, self._evaluator = seed, evaluator_id
        self._train_n, self._eval_n = train_n, eval_n
        self._eval_offers = eval_offers
        self._on_learned_weights = on_learned_weights
        self._spent = False

    def run(self) -> OpenWeightTrialReport:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("trial already running")
        if self._spent:
            self._lock.release()
            raise RuntimeError("trial already consumed; use a fresh adapter")
        self._spent = True
        train: list[OpenWeightTrainingEpisode] = []
        evals: list[OpenWeightHeldoutEpisode] = []
        train_offers: list[TaskOffer] = []
        phase, error = "ISSUE", None
        restored, restored_digest, learned_digest = False, "", ""
        planned = self._train_n + 2 * self._eval_n
        try:
            for index in range(self._train_n):
                phase = "ISSUE"
                offer = self._arena.issue(
                    partition="TRAIN", family_id=TRAIN_FAMILY,
                    pair_index=index, side=index % 2,
                )
                train_offers.append(offer)
                phase = "ENCODE"
                ids = _token_ids(self._encoder(offer))
                before, step = self._adapter.weights_digest, self._adapter.step_index
                phase = "INFER"
                choice = self._adapter.infer(
                    input_ids=ids, offer_nonce=offer.nonce,
                    episode_id=offer.episode_id, source_partition="TRAIN",
                )
                phase = "VALIDATE_CHOICE"
                _check_choice(choice, binding=self._binding, offer=offer,
                              ids=ids, weights_digest=before, step=step)
                if (
                    self._adapter.weights_digest != before
                    or self._adapter.step_index != step
                    or not self._adapter.pending
                    or self._adapter.frozen_backbone_digest != self._frozen_backbone
                    or choice.frozen_backbone_digest != self._frozen_backbone
                ):
                    raise RuntimeError("TRAIN inference mutated parameter state before feedback")
                phase = "COMMIT"
                ticket = self._arena.commit(
                    offer, choice.chosen_action, choice.decision_digest,
                    model_id=self._binding.model_id,
                    runtime_instance_id=self._binding.runtime_instance_id,
                )
                phase = "RESOLVE"
                outcome = self._arena.resolve(ticket)
                phase = "VALIDATE_OUTCOME"
                _check_outcome(offer, choice, ticket, outcome,
                               self._evaluator, self._arena.rubric_digest)
                phase = "UPDATE"
                update = self._adapter.observe(OpenWeightOutcome(
                    evidence_id=outcome.evidence_id,
                    decision_digest=outcome.decision_digest,
                    episode_id=outcome.episode_id,
                    chosen_action=outcome.action,
                    evaluator_id=outcome.evaluator_id,
                    rubric_digest=outcome.rubric_digest,
                    measured_quality=outcome.measured_quality,
                ))
                phase = "CHECK_READBACK"
                if type(update) is not OpenWeightUpdate:
                    raise TypeError("adapter must return an OpenWeightUpdate")
                if (
                    update.evidence_class != ADAPTER_EVIDENCE_CLASS
                    or update.phenomenology != ADAPTER_PHENOMENOLOGY
                ):
                    raise ValueError("update attempted unsupported host or phenomenology claim")
                changed = update.before_digest != update.after_digest
                if (
                    update.before_digest != choice.weights_digest
                    or update.after_digest != self._adapter.weights_digest
                    or update.step_index != step + 1
                    or update.evidence_id != outcome.evidence_id
                    or update.decision_digest != choice.decision_digest
                    or update.update_detected is not changed
                    or not isinstance(update.optimizer_applied, bool)
                    or update.optimizer_applied is not changed
                    or not isinstance(update.optimizer_step_executed, bool)
                    or (changed and not update.optimizer_step_executed)
                    or update.frozen_backbone_digest != self._frozen_backbone
                    or self._adapter.frozen_backbone_digest != self._frozen_backbone
                    or self._adapter.pending
                ):
                    raise RuntimeError("chosen-outcome optimizer readback mismatch")
                train.append(OpenWeightTrainingEpisode(
                    offer, _input_digest(ids), choice, ticket, outcome, update,
                ))
            learned_digest = self._adapter.weights_digest
            for reserved in self._eval_offers:
                phase = "REISSUE_EVAL"
                offer = self._arena.issue(
                    partition="EVAL", family_id=EVAL_FAMILY,
                    pair_index=reserved.pair_index, side=reserved.side,
                )
                if offer != reserved:
                    raise ValueError("heldout preregistration diverged")
                phase = "ENCODE_EVAL"
                ids = _token_ids(self._encoder(offer))
                before, step = self._adapter.weights_digest, self._adapter.step_index
                phase = "EVAL_INFER"
                choice = self._adapter.infer(
                    input_ids=ids, offer_nonce=offer.nonce,
                    episode_id=offer.episode_id, source_partition="EVAL",
                )
                phase = "VALIDATE_EVAL_CHOICE"
                _check_choice(choice, binding=self._binding, offer=offer,
                              ids=ids, weights_digest=before, step=step)
                if (
                    self._adapter.weights_digest != before
                    or self._adapter.step_index != step
                    or self._adapter.pending
                    or self._adapter.frozen_backbone_digest != self._frozen_backbone
                    or choice.frozen_backbone_digest != self._frozen_backbone
                ):
                    raise RuntimeError("EVAL inference mutated model before commitment")
                phase = "EVAL_COMMIT"
                ticket = self._arena.commit(
                    offer, choice.chosen_action, choice.decision_digest,
                    model_id=self._binding.model_id,
                    runtime_instance_id=self._binding.runtime_instance_id,
                )
                phase = "EVAL_RESOLVE"
                outcome = self._arena.resolve(ticket)
                phase = "VALIDATE_OUTCOME"
                _check_outcome(offer, choice, ticket, outcome,
                               self._evaluator, self._arena.rubric_digest)
                phase = "CHECK_EVAL_READBACK"
                if (
                    self._adapter.weights_digest != before
                    or self._adapter.step_index != step
                    or self._adapter.frozen_backbone_digest != self._frozen_backbone
                    or self._adapter.pending
                ):
                    raise RuntimeError("heldout EVAL mutated model optimizer state")
                evals.append(OpenWeightHeldoutEpisode(
                    offer, _input_digest(ids), choice, ticket, outcome,
                ))
            learned_digest = self._adapter.weights_digest
            if self._on_learned_weights is not None:
                # Capture only *after* all heldout inference and *before*
                # mandatory exact restoration; never persist a restored base.
                phase = "CAPTURE_TRAINED_WEIGHTS"
                self._on_learned_weights(learned_digest)
                if (
                    self._adapter.weights_digest != learned_digest
                    or self._adapter.frozen_backbone_digest != self._frozen_backbone
                    or self._adapter.pending
                    or self._adapter.step_index != len(train)
                ):
                    raise RuntimeError("trained-weight capture mutated trial state")
        except Exception as exc:
            error = exc
        finally:
            if error is None:
                phase = "RESTORE"
            try:
                restore_receipt = self._adapter.restore()
                restored_digest = self._adapter.weights_digest
                restored = (
                    type(restore_receipt) is OpenWeightRestore
                    and restore_receipt.closed
                    and restore_receipt.optimizer_state_cleared
                    and restore_receipt.evidence_class == ADAPTER_EVIDENCE_CLASS
                    and restore_receipt.phenomenology == ADAPTER_PHENOMENOLOGY
                    and restore_receipt.restored_digest == self._initial
                    and restore_receipt.frozen_backbone_digest == self._frozen_backbone
                    and restored_digest == self._initial
                    and self._adapter.frozen_backbone_digest == self._frozen_backbone
                    and not self._adapter.pending
                )
            except Exception as exc:
                error = RuntimeError(
                    f"prior failure ({error}) and restoration failed ({exc})"
                ) if error is not None else exc
            self._lock.release()
        if error is not None or not restored:
            missing = int(phase in ("RESOLVE", "EVAL_RESOLVE", "VALIDATE_OUTCOME"))
            raise OpenWeightTrialAborted(
                phase=phase, planned=planned, completed=len(train) + len(evals),
                missing_outcomes=missing, restoration_succeeded=restored,
                reason=str(error) if error is not None else "restoration digest mismatch",
            ) from error
        train_records, eval_records = tuple(train), tuple(evals)
        primary = tuple(
            record for record in eval_records
            if record.offer.side == record.offer.pair_index % 2
        )
        if len(primary) != self._eval_n:
            raise RuntimeError("independent heldout parity schedule diverged")
        return OpenWeightTrialReport(
            binding=self._binding, seed=self._seed,
            initial_weights_digest=self._initial,
            learned_weights_digest=learned_digest,
            frozen_backbone_digest=self._frozen_backbone,
            evaluator_id_declared=self._evaluator,
            rubric_digest_declared=self._arena.rubric_digest,
            training=train_records, heldout=eval_records,
            prequential=_metrics(train_records),
            heldout_primary=_metrics(primary),
            heldout_mirrored_diagnostic=_metrics(eval_records),
            observed_weight_changes=sum(r.update.update_detected for r in train_records),
            optimizer_steps_reported=sum(r.update.optimizer_step_executed for r in train_records),
            train_schedule_sha256=_schedule_digest(tuple(train_offers)),
            heldout_schedule_sha256=_schedule_digest(self._eval_offers),
            restored_weights_digest=restored_digest, restored=True,
            backend_kind=(
                "SOURCE_LOCAL_TORCH_ADAPTER_UNAUTHENTICATED"
                if type(self._adapter) is TorchChosenOutcomeAdapter
                else "SOURCE_ONLY_TEST_DOUBLE_OR_UNKNOWN"
            ),
        )
