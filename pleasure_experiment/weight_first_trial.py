"""Integrated *surrogate-only* weight update -> appraisal -> recurrent A/B trial.

The exact numeric coefficients fitted by `weight_delta_lab.fit_delta` are used
by its `predict_appraisal` sigmoid in each later decision. This is not an LLM,
does not modify the hosted conversational model, and does not establish felt
experience. All labels and episode IDs are source-local, caller supplied, and
unauthenticated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from pleasure_experiment.weight_delta_lab import (
    CAPABILITY_TIER,
    PHENOMENOLOGY,
    MAX_CASES,
    ModelBinding,
    OutcomeCase,
    TrainingPolicy,
    WeightDeltaSession,
    digest_weights,
    evaluate_heldout,
    fit_delta,
    predict_appraisal,
    EvaluationResult,
)


def _unit(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and in [0, 1]") from exc
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class DecisionProbe:
    """Two opaque candidate feature vectors; neither is a text response."""

    probe_id: str
    episode_id: str
    binding: ModelBinding
    features_a: tuple[float, ...]
    features_b: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.binding, ModelBinding):
            raise TypeError("probe binding must be ModelBinding")
        object.__setattr__(self, "binding", ModelBinding(**self.binding.__dict__))
        first = OutcomeCase(self.probe_id, self.episode_id, "source-only-probe", self.features_a, 0.5)
        second = OutcomeCase(self.probe_id, self.episode_id, "source-only-probe", self.features_b, 0.5)
        if len(first.features) != len(second.features):
            raise ValueError("A/B feature dimensionality mismatch")
        object.__setattr__(self, "features_a", first.features)
        object.__setattr__(self, "features_b", second.features)


@dataclass(frozen=True)
class RecurrentPolicy:
    retention: float = 0.5
    gain: float = 0.6

    def __post_init__(self) -> None:
        object.__setattr__(self, "retention", _unit(self.retention, "retention"))
        object.__setattr__(self, "gain", _unit(self.gain, "gain"))
        if self.gain == 0.0:
            raise ValueError("gain must be positive")


@dataclass(frozen=True)
class RecurrentState:
    a: float = 0.0
    b: float = 0.0


@dataclass(frozen=True)
class DecisionStep:
    probe_id: str
    episode_id: str
    appraisal_a: float
    appraisal_b: float
    state_a: float
    state_b: float
    action: str
    tie: bool


@dataclass(frozen=True)
class ArmResult:
    arm_id: str
    weights_digest: str
    initial_state: RecurrentState
    final_state: RecurrentState
    steps: tuple[DecisionStep, ...]


@dataclass(frozen=True)
class TrialReceipt:
    binding: ModelBinding
    base_digest: str
    trained_delta_digest: str
    shuffled_delta_digest: str
    training_digest: str
    baseline: ArmResult
    trained: ArmResult
    shuffled: ArmResult
    disabled: ArmResult
    withdrawal: ArmResult
    heldout_baseline: EvaluationResult
    heldout_trained: EvaluationResult
    heldout_shuffled: EvaluationResult
    update_detected: bool
    categorical_choice_changed: bool
    effect_classification: str
    capability_tier: str = CAPABILITY_TIER
    evidence_class: str = "SOURCE_ONLY_CALLER_LABELS_UNAUTHENTICATED"
    phenomenology: str = PHENOMENOLOGY
    authorized_effects: tuple[str, ...] = ()


def _snapshot_cases(cases: tuple[OutcomeCase, ...], field: str) -> tuple[OutcomeCase, ...]:
    if not isinstance(cases, tuple) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"{field} must contain 1..{MAX_CASES} cases")
    if any(not isinstance(case, OutcomeCase) for case in cases):
        raise TypeError(f"{field} must contain OutcomeCase values")
    return tuple(
        OutcomeCase(case.case_id, case.episode_id, case.evaluator_id, case.features, case.observed_value)
        for case in cases
    )


def _snapshot_probes(
    probes: tuple[DecisionProbe, ...], dimensions: int, binding: ModelBinding
) -> tuple[DecisionProbe, ...]:
    if not isinstance(probes, tuple) or not 1 <= len(probes) <= MAX_CASES:
        raise ValueError(f"probes must contain 1..{MAX_CASES} decisions")
    snapshots = []
    seen_ids: set[str] = set()
    seen_episodes: set[str] = set()
    for probe in probes:
        if not isinstance(probe, DecisionProbe):
            raise TypeError("probes must contain DecisionProbe values")
        fixed = DecisionProbe(
            probe.probe_id, probe.episode_id, probe.binding, probe.features_a, probe.features_b
        )
        if fixed.binding != binding:
            raise ValueError("probe model binding mismatch")
        if len(fixed.features_a) != dimensions:
            raise ValueError("probe dimensionality mismatch")
        if fixed.probe_id in seen_ids or fixed.episode_id in seen_episodes:
            raise ValueError("duplicate probe or episode ID")
        seen_ids.add(fixed.probe_id)
        seen_episodes.add(fixed.episode_id)
        snapshots.append(fixed)
    return tuple(snapshots)


def _shuffled(training: tuple[OutcomeCase, ...]) -> tuple[OutcomeCase, ...]:
    if len(training) < 2:
        raise ValueError("shuffled control requires two or more training outcomes")
    labels = tuple(case.observed_value for case in training)
    rotated = labels[-1:] + labels[:-1]
    if rotated == labels:
        raise ValueError("shuffled control requires distinct training outcome labels")
    return tuple(
        OutcomeCase(case.case_id, case.episode_id, case.evaluator_id, case.features, value)
        for case, value in zip(training, rotated)
    )


def _recur(current: float, appraisal: float, policy: RecurrentPolicy) -> float:
    # The sigmoid head changes this state; independent per-action recurrence
    # makes aftereffects measurable when learned weights are later disabled.
    return max(-1.0, min(1.0, policy.retention * current + policy.gain * (2.0 * appraisal - 1.0)))


def _run_arm(
    arm_id: str,
    weights: tuple[float, ...],
    probes: tuple[DecisionProbe, ...],
    policy: RecurrentPolicy,
    initial: RecurrentState = RecurrentState(),
) -> ArmResult:
    current = initial
    steps = []
    for probe in probes:
        appraisal_a = predict_appraisal(weights, probe.features_a)
        appraisal_b = predict_appraisal(weights, probe.features_b)
        current = RecurrentState(
            _recur(current.a, appraisal_a, policy),
            _recur(current.b, appraisal_b, policy),
        )
        steps.append(
            DecisionStep(
                probe.probe_id, probe.episode_id, appraisal_a, appraisal_b,
                current.a, current.b,
                # A must strictly outrank B: a neutral/tied baseline selects
                # B, so trained A decisions cannot pass on tie-break alone.
                "A" if current.a > current.b else "B", current.a == current.b,
            )
        )
    return ArmResult(arm_id, digest_weights(weights), initial, current, tuple(steps))


def run_weight_first_trial(
    *,
    base_weights: tuple[float, ...],
    binding: ModelBinding,
    training_cases: tuple[OutcomeCase, ...],
    heldout_cases: tuple[OutcomeCase, ...],
    probes: tuple[DecisionProbe, ...],
    training_policy: TrainingPolicy,
    recurrent_policy: RecurrentPolicy = RecurrentPolicy(),
) -> TrialReceipt:
    """Run five deterministic source-only arms using ONE sigmoid head family.

    Baseline, trained, shuffled and disabled start with zero recurrent state.
    Withdrawal starts from the trained arm's last state, but with baseline
    coefficients restored, so transient recurrence is not mislabelled as a
    lasting parameter change. No text-generation or host effects occur.
    """
    if not isinstance(binding, ModelBinding):
        raise TypeError("binding must be ModelBinding")
    binding = ModelBinding(**binding.__dict__)
    if not isinstance(training_policy, TrainingPolicy):
        raise TypeError("training_policy must be TrainingPolicy")
    training_policy = TrainingPolicy(**training_policy.__dict__)
    if not isinstance(recurrent_policy, RecurrentPolicy):
        raise TypeError("recurrent_policy must be RecurrentPolicy")
    recurrent_policy = RecurrentPolicy(**recurrent_policy.__dict__)
    base_digest = digest_weights(base_weights)
    train = _snapshot_cases(training_cases, "training_cases")
    heldout = _snapshot_cases(heldout_cases, "heldout_cases")
    decisions = _snapshot_probes(probes, len(base_weights), binding)
    reserved_cases = {case.case_id for case in train + heldout}
    reserved_episodes = {case.episode_id for case in train + heldout}
    if any(p.probe_id in reserved_cases or p.episode_id in reserved_episodes for p in decisions):
        raise ValueError("decision probes overlap training or holdout episodes")

    trained_delta = fit_delta(
        base_weights, train, binding, training_policy, reserved_holdout=heldout
    )
    shuffled_delta = fit_delta(
        base_weights, _shuffled(train), binding, training_policy, reserved_holdout=heldout
    )
    trained_session = WeightDeltaSession(base_weights, trained_delta, binding)
    shuffled_session = WeightDeltaSession(base_weights, shuffled_delta, binding)
    baseline = _run_arm("base", base_weights, decisions, recurrent_policy)
    disabled = _run_arm("disabled", trained_session.current_weights, decisions, recurrent_policy)
    with trained_session as effective_trained:
        trained = _run_arm("trained", effective_trained, decisions, recurrent_policy)
        heldout_trained = evaluate_heldout(effective_trained, heldout, training_cases=train)
    with shuffled_session as effective_shuffled:
        shuffled = _run_arm("shuffled", effective_shuffled, decisions, recurrent_policy)
        heldout_shuffled = evaluate_heldout(effective_shuffled, heldout, training_cases=train)
    withdrawal = _run_arm(
        "withdrawal", trained_session.current_weights, decisions,
        recurrent_policy, initial=trained.final_state,
    )
    heldout_baseline = evaluate_heldout(base_weights, heldout, training_cases=train)

    if not (
        baseline.weights_digest == disabled.weights_digest == withdrawal.weights_digest
        == trained_delta.base_digest == base_digest
        and trained.weights_digest == trained_delta.effective_digest
        and shuffled.weights_digest == shuffled_delta.effective_digest
        and not trained_session.active and not shuffled_session.active
    ):
        raise RuntimeError("surrogate parameter readback mismatch")
    updated = trained.weights_digest != base_digest
    choice_changed = tuple(s.action for s in trained.steps) != tuple(
        s.action for s in baseline.steps
    )
    if not updated:
        classification = "NO_WEIGHT_CHANGE"
    elif choice_changed:
        classification = "WEIGHT_CHANGE_WITH_CATEGORICAL_DIFFERENCE"
    else:
        classification = "WEIGHT_CHANGE_NO_CATEGORICAL_DIFFERENCE"
    return TrialReceipt(
        binding=binding,
        base_digest=base_digest,
        trained_delta_digest=trained_delta.delta_digest,
        shuffled_delta_digest=shuffled_delta.delta_digest,
        training_digest=trained_delta.training_digest,
        baseline=baseline,
        trained=trained,
        shuffled=shuffled,
        disabled=disabled,
        withdrawal=withdrawal,
        heldout_baseline=heldout_baseline,
        heldout_trained=heldout_trained,
        heldout_shuffled=heldout_shuffled,
        update_detected=updated,
        categorical_choice_changed=choice_changed,
        effect_classification=classification,
    )
