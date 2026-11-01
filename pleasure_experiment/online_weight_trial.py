"""Source-only online learning from the chosen action's observed outcome.

The tiny surrogate chooses an opaque A/B action, admits exactly one matching
caller-supplied outcome, then performs one bounded SGD step. No unchosen
outcome is inferred or trained on. Model-host access, evaluator authentication,
LLM training, and subjective experience are not established by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from secrets import token_hex
from threading import Lock
import json
import re

from pleasure_experiment.weight_delta_lab import (
    CAPABILITY_TIER,
    MAX_ABS_WEIGHT,
    MAX_CASES,
    ModelBinding,
    PHENOMENOLOGY,
    TrainingPolicy,
    _identifier,
    _real,
    _vector,
    digest_weights,
    predict_appraisal,
    single_outcome_step,
)
from pleasure_experiment.weight_first_trial import DecisionProbe, RecurrentState


EVIDENCE_CLASS = "SOURCE_ONLY_CHOSEN_OUTCOME_UNVERIFIED"


def _sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class OnlineRecurrentPolicy:
    """Outcome-only recurrent learning; both gains may be zero for ablations."""

    retention: float = 0.5
    outcome_gain: float = 0.4
    decision_gain: float = 0.2

    def __post_init__(self) -> None:
        for field in ("retention", "outcome_gain", "decision_gain"):
            object.__setattr__(
                self, field,
                _real(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )


@dataclass(frozen=True)
class ChosenOutcome:
    """Caller-declared measurement of a pending action, never a host attestation."""

    evidence_id: str
    decision_digest: str
    probe_id: str
    episode_id: str
    binding: ModelBinding
    chosen_action: str
    evaluator_id: str
    rubric_digest: str
    measured_quality: float

    def __post_init__(self) -> None:
        for field in ("evidence_id", "probe_id", "episode_id", "evaluator_id"):
            _identifier(getattr(self, field), field)
        _sha256(self.decision_digest, "decision_digest")
        _sha256(self.rubric_digest, "rubric_digest")
        if not isinstance(self.binding, ModelBinding):
            raise TypeError("outcome binding must be ModelBinding")
        object.__setattr__(self, "binding", ModelBinding(**self.binding.__dict__))
        if self.chosen_action not in ("A", "B") or type(self.chosen_action) is not str:
            raise ValueError("chosen_action must be A or B")
        object.__setattr__(
            self, "measured_quality",
            _real(self.measured_quality, "measured_quality", minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True)
class OnlineChoice:
    decision_digest: str
    binding: ModelBinding
    probe_id: str
    episode_id: str
    chosen_action: str
    appraisal_a: float
    appraisal_b: float
    score_a: float
    score_b: float
    source_partition: str
    offer_nonce_digest: str
    session_nonce_digest: str
    session_nonce_source: str
    weights_digest: str
    step_index: int
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


@dataclass(frozen=True)
class OnlineUpdate:
    decision_digest: str
    evidence_id: str
    chosen_action: str
    chosen_feature_digest: str
    prediction_before: float
    measured_quality: float
    before_digest: str
    after_digest: str
    step_index: int
    update_detected: bool
    effect_classification: str
    recurrent_state_before: RecurrentState
    recurrent_state_after: RecurrentState
    evaluator_id_declared: str
    rubric_digest_declared: str
    capability_tier: str = CAPABILITY_TIER
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY
    authorized_effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class RestoreReceipt:
    before_digest: str
    restored_digest: str
    update_count: int
    cancelled_pending_decision: bool
    state_after: RecurrentState
    closed: bool
    capability_tier: str = CAPABILITY_TIER
    phenomenology: str = PHENOMENOLOGY


def _choice_digest(
    binding: ModelBinding,
    probe: DecisionProbe,
    step_index: int,
    weights_digest: str,
    state: RecurrentState,
    appraisal_a: float,
    appraisal_b: float,
    score_a: float,
    score_b: float,
    action: str,
    offer_nonce: str,
    source_partition: str,
    trial_session_nonce: str,
) -> str:
    # Content binding only: this hash is not a signature or proof of execution.
    fields = [
        "ONLINE_CHOSEN_ACTION_V1",
        binding.model_id, binding.model_revision, binding.runtime_instance_id,
        binding.feature_schema_id, probe.probe_id, probe.episode_id, step_index,
        weights_digest, [v.hex() for v in probe.features_a],
        [v.hex() for v in probe.features_b], state.a.hex(), state.b.hex(),
        appraisal_a.hex(), appraisal_b.hex(), score_a.hex(), score_b.hex(), action,
        offer_nonce, source_partition, trial_session_nonce,
    ]
    return sha256(json.dumps(fields, separators=(",", ":")).encode("utf-8")).hexdigest()


def _feature_digest(features: tuple[float, ...]) -> str:
    return sha256(
        json.dumps([v.hex() for v in features], separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _recur_chosen(
    state: RecurrentState,
    action: str,
    outcome: float,
    policy: OnlineRecurrentPolicy,
) -> RecurrentState:
    # Every completed episode ages BOTH channels; measured impact is credited
    # only to the chosen action. No unchosen outcome is fabricated.
    a = policy.retention * state.a
    b = policy.retention * state.b
    impact = policy.outcome_gain * (2.0 * outcome - 1.0)
    if action == "A":
        a += impact
    else:
        b += impact
    return RecurrentState(max(-1.0, min(1.0, a)), max(-1.0, min(1.0, b)))


class OnlineWeightTrial:
    """Serialized READY -> PENDING -> READY chosen-outcome SGD trial.

    Attribute immutability protects against ordinary accidental in-process
    rebinding, not privileged Python memory tampering. `restore()` closes the
    instance; a fresh instance is required for another trial.
    """

    __slots__ = (
        "_lock", "_binding", "_training_policy", "_recurrent_policy",
        "_expected_evaluator_id", "_expected_rubric_digest", "_holdout_ids",
        "_trial_session_nonce", "_session_nonce_source",
        "_base", "_weights", "_state", "_pending", "_seen_episodes", "_evaluated_ids",
        "_seen_evidence", "_step_index", "_closed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("online weight trial bindings and state are guarded")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("online weight trial bindings and state are guarded")

    def __init__(
        self,
        *,
        base_weights: tuple[float, ...],
        binding: ModelBinding,
        training_policy: TrainingPolicy,
        expected_evaluator_id: str,
        expected_rubric_digest: str,
        recurrent_policy: OnlineRecurrentPolicy = OnlineRecurrentPolicy(),
        reserved_holdout_episode_ids: tuple[str, ...] = (),
        trial_session_nonce: str | None = None,
    ):
        base = _vector(base_weights, "base_weights", maximum=MAX_ABS_WEIGHT)
        if not isinstance(binding, ModelBinding):
            raise TypeError("binding must be ModelBinding")
        binding = ModelBinding(**binding.__dict__)
        if not isinstance(training_policy, TrainingPolicy):
            raise TypeError("training_policy must be TrainingPolicy")
        training_policy = TrainingPolicy(**training_policy.__dict__)
        if not isinstance(recurrent_policy, OnlineRecurrentPolicy):
            raise TypeError("recurrent_policy must be OnlineRecurrentPolicy")
        recurrent_policy = OnlineRecurrentPolicy(**recurrent_policy.__dict__)
        _identifier(expected_evaluator_id, "expected_evaluator_id")
        _sha256(expected_rubric_digest, "expected_rubric_digest")
        if not isinstance(reserved_holdout_episode_ids, tuple) or (
            len(reserved_holdout_episode_ids) > MAX_CASES
        ):
            raise ValueError("reserved holdout episode IDs must be a bounded tuple")
        holdout_ids = tuple(
            _identifier(value, "reserved holdout episode ID")
            for value in reserved_holdout_episode_ids
        )
        if len(set(holdout_ids)) != len(holdout_ids):
            raise ValueError("duplicate reserved holdout episode ID")
        if trial_session_nonce is None:
            session_nonce = token_hex(16)
            nonce_source = "GENERATED_SOURCE_LOCAL"
        else:
            session_nonce = _identifier(trial_session_nonce, "trial_session_nonce")
            nonce_source = "CALLER_DECLARED_REPLAY_EQUIVALENT_IF_REUSED"
        initial = dict(
            _lock=Lock(), _binding=binding, _training_policy=training_policy,
            _recurrent_policy=recurrent_policy,
            _expected_evaluator_id=expected_evaluator_id,
            _expected_rubric_digest=expected_rubric_digest,
            _holdout_ids=frozenset(holdout_ids), _base=base, _weights=base,
            _trial_session_nonce=session_nonce, _session_nonce_source=nonce_source,
            _state=RecurrentState(), _pending=None, _seen_episodes=set(),
            _evaluated_ids=set(),
            _seen_evidence=set(), _step_index=0, _closed=False,
        )
        for name, value in initial.items():
            object.__setattr__(self, name, value)

    @property
    def binding(self) -> ModelBinding:
        # Never return an alias of the binding stored inside the state machine.
        return ModelBinding(**self._binding.__dict__)

    @property
    def state(self) -> RecurrentState:
        with self._lock:
            return RecurrentState(self._state.a, self._state.b)

    @property
    def weights_digest(self) -> str:
        with self._lock:
            return digest_weights(self._weights)

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    @property
    def step_index(self) -> int:
        with self._lock:
            return self._step_index

    def weights_readback(self) -> tuple[float, ...]:
        with self._lock:
            return self._weights

    def _decision_locked(
        self, probe: DecisionProbe, *, offer_nonce: str, source_partition: str
    ) -> OnlineChoice:
        """Read-only evaluation of the EXACT policy used by both public paths."""
        pred_a = predict_appraisal(self._weights, probe.features_a)
        pred_b = predict_appraisal(self._weights, probe.features_b)
        score_a = pred_a + self._recurrent_policy.decision_gain * self._state.a
        score_b = pred_b + self._recurrent_policy.decision_gain * self._state.b
        action = "A" if score_a > score_b else "B"
        before_digest = digest_weights(self._weights)
        return OnlineChoice(
            decision_digest=_choice_digest(
                self._binding, probe, self._step_index, before_digest,
                self._state, pred_a, pred_b, score_a, score_b, action,
                offer_nonce, source_partition, self._trial_session_nonce,
            ),
            binding=self._binding,
            probe_id=probe.probe_id, episode_id=probe.episode_id,
            chosen_action=action, appraisal_a=pred_a, appraisal_b=pred_b,
            score_a=score_a, score_b=score_b,
            source_partition=source_partition,
            offer_nonce_digest=sha256(offer_nonce.encode("utf-8")).hexdigest(),
            session_nonce_digest=sha256(self._trial_session_nonce.encode("utf-8")).hexdigest(),
            session_nonce_source=self._session_nonce_source,
            weights_digest=before_digest, step_index=self._step_index,
        )

    def evaluate_choice(
        self, probe: DecisionProbe, *, offer_nonce: str, source_partition: str = "EVAL"
    ) -> OnlineChoice:
        """Preregistered heldout decision, no model/valence/SGD mutation.

        An internal evaluation-ID ledger prevents repeated EVAL and later
        TRAIN use of the same episode. Partition names are still declarations,
        not a trusted arena attestation.
        """
        if not isinstance(probe, DecisionProbe):
            raise TypeError("probe must be DecisionProbe")
        _identifier(offer_nonce, "offer_nonce")
        if source_partition != "EVAL" or type(source_partition) is not str:
            raise ValueError("read-only evaluation requires EVAL source partition")
        fixed = DecisionProbe(
            probe.probe_id, probe.episode_id, probe.binding, probe.features_a, probe.features_b
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("online trial is restored and closed")
            if self._pending is not None:
                raise RuntimeError("cannot evaluate while training outcome pending")
            if fixed.binding != self._binding:
                raise ValueError("probe model binding mismatch")
            if len(fixed.features_a) != len(self._weights):
                raise ValueError("probe feature dimensionality mismatch")
            if fixed.episode_id in self._seen_episodes:
                raise ValueError("training episode cannot also be evaluated")
            if fixed.episode_id not in self._holdout_ids:
                raise ValueError("EVAL episode must be preregistered as holdout")
            if fixed.episode_id in self._evaluated_ids:
                raise ValueError("duplicate evaluated episode ID")
            result = self._decision_locked(
                fixed, offer_nonce=offer_nonce, source_partition=source_partition
            )
            self._evaluated_ids.add(fixed.episode_id)
            return replace(result, binding=ModelBinding(**self._binding.__dict__))

    def choose(
        self, probe: DecisionProbe, *, offer_nonce: str, source_partition: str
    ) -> OnlineChoice:
        """Commit only an A/B decision, never an optimizer update."""
        if not isinstance(probe, DecisionProbe):
            raise TypeError("probe must be DecisionProbe")
        _identifier(offer_nonce, "offer_nonce")
        if source_partition != "TRAIN" or type(source_partition) is not str:
            raise ValueError("online SGD requires TRAIN source partition")
        fixed = DecisionProbe(
            probe.probe_id, probe.episode_id, probe.binding, probe.features_a, probe.features_b
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("online trial is restored and closed")
            if self._pending is not None:
                raise RuntimeError("pending decision requires a chosen outcome")
            if fixed.binding != self._binding:
                raise ValueError("probe model binding mismatch")
            if len(fixed.features_a) != len(self._weights):
                raise ValueError("probe feature dimensionality mismatch")
            if fixed.episode_id in self._evaluated_ids:
                raise ValueError("evaluated episode cannot be trained on")
            if fixed.episode_id in self._holdout_ids:
                raise ValueError("reserved holdout episode cannot be trained on")
            if fixed.episode_id in self._seen_episodes:
                raise ValueError("duplicate online episode ID")
            if len(self._seen_episodes) >= self._training_policy.maximum_examples or (
                self._step_index >= self._training_policy.maximum_steps
            ):
                raise RuntimeError("online training budget exhausted")
            receipt = self._decision_locked(
                fixed, offer_nonce=offer_nonce, source_partition=source_partition
            )
            object.__setattr__(self, "_pending", (receipt, fixed, offer_nonce))
            self._seen_episodes.add(fixed.episode_id)
            return replace(receipt, binding=ModelBinding(**self._binding.__dict__))

    def observe(self, outcome: ChosenOutcome) -> OnlineUpdate:
        """Admit ONLY the pending chosen action's declared numeric outcome."""
        if not isinstance(outcome, ChosenOutcome):
            raise TypeError("outcome must be ChosenOutcome")
        fixed = ChosenOutcome(
            outcome.evidence_id, outcome.decision_digest, outcome.probe_id,
            outcome.episode_id, outcome.binding, outcome.chosen_action,
            outcome.evaluator_id, outcome.rubric_digest, outcome.measured_quality,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("online trial is restored and closed")
            if self._pending is None:
                raise RuntimeError("no pending chosen-action decision")
            choice, probe, offer_nonce = self._pending
            if fixed.binding != self._binding:
                raise ValueError("outcome model binding mismatch")
            if (fixed.decision_digest, fixed.probe_id, fixed.episode_id, fixed.chosen_action) != (
                choice.decision_digest, choice.probe_id, choice.episode_id, choice.chosen_action
            ):
                raise ValueError("outcome does not match pending chosen action")
            if fixed.evaluator_id != self._expected_evaluator_id:
                raise ValueError("outcome evaluator ID mismatch")
            if fixed.rubric_digest != self._expected_rubric_digest:
                raise ValueError("outcome rubric digest mismatch")
            if fixed.evidence_id in self._seen_evidence:
                raise ValueError("duplicate online outcome evidence ID")
            before_digest = digest_weights(self._weights)
            if before_digest != choice.weights_digest:
                raise RuntimeError("pre-outcome weight readback mismatch")
            # Recover the committed choice from PRIVATE model/feature/state
            # snapshots. A caller can mutate its returned frozen DTO via
            # object.__setattr__, but that object is never the pending one.
            pre_a = predict_appraisal(self._weights, probe.features_a)
            pre_b = predict_appraisal(self._weights, probe.features_b)
            pre_score_a = pre_a + self._recurrent_policy.decision_gain * self._state.a
            pre_score_b = pre_b + self._recurrent_policy.decision_gain * self._state.b
            pre_action = "A" if pre_score_a > pre_score_b else "B"
            expected_digest = _choice_digest(
                self._binding, probe, self._step_index, before_digest, self._state,
                pre_a, pre_b, pre_score_a, pre_score_b, pre_action, offer_nonce,
                choice.source_partition, self._trial_session_nonce,
            )
            if expected_digest != choice.decision_digest or (
                pre_action != choice.chosen_action or pre_a != choice.appraisal_a or
                pre_b != choice.appraisal_b or pre_score_a != choice.score_a or
                pre_score_b != choice.score_b
            ):
                raise RuntimeError("pending decision readback mismatch")
            chosen_features = probe.features_a if choice.chosen_action == "A" else probe.features_b
            prediction = choice.appraisal_a if choice.chosen_action == "A" else choice.appraisal_b
            after_weights = single_outcome_step(
                base_weights=self._base, current_weights=self._weights,
                chosen_features=chosen_features, observed_score=fixed.measured_quality,
                policy=self._training_policy,
            )
            after_digest = digest_weights(after_weights)
            after_state = _recur_chosen(
                self._state, choice.chosen_action, fixed.measured_quality,
                self._recurrent_policy,
            )
            updated = after_digest != before_digest
            receipt = OnlineUpdate(
                decision_digest=choice.decision_digest, evidence_id=fixed.evidence_id,
                chosen_action=choice.chosen_action,
                chosen_feature_digest=_feature_digest(chosen_features),
                prediction_before=prediction, measured_quality=fixed.measured_quality,
                before_digest=before_digest, after_digest=after_digest,
                step_index=self._step_index + 1, update_detected=updated,
                effect_classification=("WEIGHT_CHANGE_SOURCE_LOCAL" if updated
                                       else "NO_WEIGHT_CHANGE"),
                recurrent_state_before=RecurrentState(self._state.a, self._state.b),
                recurrent_state_after=RecurrentState(after_state.a, after_state.b),
                evaluator_id_declared=fixed.evaluator_id,
                rubric_digest_declared=fixed.rubric_digest,
            )
            # Commit only after all validation, projection and receipt creation.
            object.__setattr__(self, "_weights", after_weights)
            object.__setattr__(self, "_state", after_state)
            object.__setattr__(self, "_step_index", self._step_index + 1)
            object.__setattr__(self, "_pending", None)
            self._seen_evidence.add(fixed.evidence_id)
            return receipt

    def restore(self) -> RestoreReceipt:
        """Close the trial and restore exact initial weights and recurrent state."""
        with self._lock:
            previous = digest_weights(self._weights)
            cancelled = self._pending is not None
            object.__setattr__(self, "_weights", self._base)
            object.__setattr__(self, "_state", RecurrentState())
            object.__setattr__(self, "_pending", None)
            object.__setattr__(self, "_closed", True)
            return RestoreReceipt(
                before_digest=previous, restored_digest=digest_weights(self._base),
                update_count=self._step_index, cancelled_pending_decision=cancelled,
                state_after=RecurrentState(self._state.a, self._state.b), closed=True,
            )

    def __enter__(self) -> OnlineWeightTrial:
        with self._lock:
            if self._closed:
                raise RuntimeError("online trial is restored and closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.restore()
        return False
