"""Dependency-free laboratory for *synthetic* learned parameter changes.

This trains a tiny linear appraisal head on caller-supplied outcome labels. It
does not train, inspect, or alter an LLM, a hosted model, or this conversation's
inference runtime. The numeric update is real for this small surrogate, but a
source-local evaluator label is not authenticated independent evidence, and a
weight delta is not evidence of phenomenal pleasure.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp, hypot, isfinite
import json


CAPABILITY_TIER = "SURROGATE_WEIGHT_UPDATE_LAB"
PHENOMENOLOGY = "UNRESOLVED"
MAX_DIMENSIONS = 64
MAX_CASES = 256
MAX_STEPS = 128
MAX_ABS_WEIGHT = 10.0


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a nonempty string of at most 256 characters")
    return value


def _real(value: float, field: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and in [{minimum}, {maximum}]")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite and in [{minimum}, {maximum}]") from exc
    if not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be finite and in [{minimum}, {maximum}]")
    return 0.0 if value == 0.0 else value


def _vector(values: tuple[float, ...], field: str, *, maximum: float) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not 1 <= len(values) <= MAX_DIMENSIONS:
        raise ValueError(f"{field} must be a tuple of 1..{MAX_DIMENSIONS} values")
    return tuple(_real(value, field, minimum=-maximum, maximum=maximum) for value in values)


def digest_weights(weights: tuple[float, ...]) -> str:
    """Stable source-test checksum of the actual numeric tuple, not an attestation."""
    checked = _vector(weights, "weights", maximum=MAX_ABS_WEIGHT)
    raw = json.dumps([value.hex() for value in checked], separators=(",", ":")).encode("ascii")
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class ModelBinding:
    model_id: str
    model_revision: str
    runtime_instance_id: str
    feature_schema_id: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _identifier(getattr(self, field), field)


@dataclass(frozen=True)
class OutcomeCase:
    case_id: str
    episode_id: str
    evaluator_id: str
    features: tuple[float, ...]
    observed_value: float

    def __post_init__(self) -> None:
        for field in ("case_id", "episode_id", "evaluator_id"):
            _identifier(getattr(self, field), field)
        object.__setattr__(self, "features", _vector(self.features, "features", maximum=1.0))
        object.__setattr__(
            self,
            "observed_value",
            _real(self.observed_value, "observed_value", minimum=0.0, maximum=1.0),
        )


@dataclass(frozen=True)
class TrainingPolicy:
    learning_rate: float
    max_delta_l2: float
    maximum_examples: int
    maximum_steps: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "learning_rate", _real(self.learning_rate, "learning_rate", minimum=0.0, maximum=1.0)
        )
        object.__setattr__(
            self, "max_delta_l2", _real(self.max_delta_l2, "max_delta_l2", minimum=0.0, maximum=1.0)
        )
        if self.max_delta_l2 == 0.0:
            raise ValueError("max_delta_l2 must be positive")
        for field, ceiling in (("maximum_examples", MAX_CASES), ("maximum_steps", MAX_STEPS)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{field} must be an integer in [1, {ceiling}]")


@dataclass(frozen=True)
class WeightDelta:
    binding: ModelBinding
    base_digest: str
    delta: tuple[float, ...]
    delta_digest: str
    effective_digest: str
    training_digest: str
    policy_digest: str
    policy: TrainingPolicy
    capability_tier: str = CAPABILITY_TIER
    phenomenology: str = PHENOMENOLOGY


@dataclass(frozen=True)
class EvaluationResult:
    case_count: int
    mean_squared_error: float
    mean_absolute_error: float
    weights_digest: str
    evidence_class: str = "CALLER_LABELS_UNVERIFIED"
    split_status: str = "UNVERIFIED_WITHOUT_TRAINING_SET"


def _cases(values: tuple[OutcomeCase, ...], name: str) -> tuple[OutcomeCase, ...]:
    if not isinstance(values, tuple) or len(values) > MAX_CASES:
        raise ValueError(f"{name} must be a tuple containing at most {MAX_CASES} cases")
    snapshots = []
    seen_cases: set[str] = set()
    seen_episodes: set[str] = set()
    for case in values:
        if not isinstance(case, OutcomeCase):
            raise TypeError(f"{name} elements must be OutcomeCase")
        # Revalidate inputs even if a frozen dataclass was modified after init.
        fixed = OutcomeCase(
            case.case_id, case.episode_id, case.evaluator_id, case.features, case.observed_value
        )
        if fixed.case_id in seen_cases:
            raise ValueError(f"duplicate {name} case_id")
        if fixed.episode_id in seen_episodes:
            raise ValueError(f"duplicate {name} episode_id")
        seen_cases.add(fixed.case_id)
        seen_episodes.add(fixed.episode_id)
        snapshots.append(fixed)
    return tuple(snapshots)


def _disjoint(training: tuple[OutcomeCase, ...], holdout: tuple[OutcomeCase, ...]) -> None:
    ids = {case.case_id for case in training}
    episodes = {case.episode_id for case in training}
    if any(case.case_id in ids or case.episode_id in episodes for case in holdout):
        raise ValueError("training and holdout overlap")


def _case_digest(cases: tuple[OutcomeCase, ...]) -> str:
    rows = [
        [c.case_id, c.episode_id, c.evaluator_id, [f.hex() for f in c.features], c.observed_value.hex()]
        for c in cases
    ]
    return sha256(json.dumps(rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _policy_digest(policy: TrainingPolicy) -> str:
    values = [
        policy.learning_rate.hex(), policy.max_delta_l2.hex(),
        policy.maximum_examples, policy.maximum_steps,
    ]
    return sha256(json.dumps(values, separators=(",", ":")).encode("ascii")).hexdigest()


def _predict(weights: tuple[float, ...], features: tuple[float, ...]) -> float:
    logit = sum(weight * feature for weight, feature in zip(weights, features))
    if logit >= 0.0:
        return 1.0 / (1.0 + exp(-logit))
    shifted = exp(logit)
    return shifted / (1.0 + shifted)


def predict_appraisal(
    weights: tuple[float, ...], features: tuple[float, ...]
) -> float:
    """Evaluate the same sigmoid head whose coefficients `fit_delta` trains.

    The wrapper validates inputs independently at inference time. This is a
    numeric synthetic appraiser, not a hosted language-model inference API.
    """
    checked_weights = _vector(weights, "weights", maximum=MAX_ABS_WEIGHT)
    checked_features = _vector(features, "features", maximum=1.0)
    if len(checked_weights) != len(checked_features):
        raise ValueError("feature dimensionality mismatch")
    return _predict(checked_weights, checked_features)


def single_outcome_step(
    *,
    base_weights: tuple[float, ...],
    current_weights: tuple[float, ...],
    chosen_features: tuple[float, ...],
    observed_score: float,
    policy: TrainingPolicy,
) -> tuple[float, ...]:
    """One pure, bounded SGD step for ONE observed action's feature vector.

    The caller must have already matched the outcome to the actual pending
    choice. This function cannot authenticate measurement provenance or infer
    any unchosen/counterfactual outcome. The immutable baseline sets the
    cumulative L2 budget across sequential online updates.
    """
    base = _vector(base_weights, "base_weights", maximum=MAX_ABS_WEIGHT)
    current = _vector(current_weights, "current_weights", maximum=MAX_ABS_WEIGHT)
    features = _vector(chosen_features, "chosen_features", maximum=1.0)
    if not (len(base) == len(current) == len(features)):
        raise ValueError("single outcome feature dimensionality mismatch")
    if not isinstance(policy, TrainingPolicy):
        raise TypeError("policy must be TrainingPolicy")
    checked_policy = TrainingPolicy(**policy.__dict__)
    observed = _real(observed_score, "observed_score", minimum=0.0, maximum=1.0)
    if hypot(*(w - origin for origin, w in zip(base, current))) > (
        checked_policy.max_delta_l2 * (1.0 + 1e-12)
    ):
        raise ValueError("current weights exceed cumulative policy max_delta_l2")
    error = _predict(current, features) - observed
    candidate = tuple(
        max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, w - checked_policy.learning_rate * error * x))
        for w, x in zip(current, features)
    )
    displacement = tuple(w - origin for origin, w in zip(base, candidate))
    norm = hypot(*displacement)
    if norm > checked_policy.max_delta_l2:
        factor = checked_policy.max_delta_l2 / norm
        candidate = tuple(
            max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, origin + factor * d))
            for origin, d in zip(base, displacement)
        )
    digest_weights(candidate)  # Revalidate the admitted effective-weight domain.
    return candidate


def fit_delta(
    base_weights: tuple[float, ...],
    training_cases: tuple[OutcomeCase, ...],
    binding: ModelBinding,
    policy: TrainingPolicy,
    *,
    reserved_holdout: tuple[OutcomeCase, ...] = (),
) -> WeightDelta:
    """Fit bounded logistic-gradient weights, leaving the baseline immutable.

    `reserved_holdout` is checked for disjointness but is NEVER read for
    training or stopping decisions. These source-local labels aren't authenticated.
    """
    base = _vector(base_weights, "weights", maximum=MAX_ABS_WEIGHT)
    if not isinstance(binding, ModelBinding):
        raise TypeError("binding must be ModelBinding")
    binding = ModelBinding(**binding.__dict__)
    if not isinstance(policy, TrainingPolicy):
        raise TypeError("policy must be TrainingPolicy")
    policy = TrainingPolicy(**policy.__dict__)
    train = _cases(training_cases, "training")
    holdout = _cases(reserved_holdout, "holdout")
    if not train:
        raise ValueError("at least one training case required")
    if len(train) > policy.maximum_examples:
        raise ValueError("too many training cases")
    _disjoint(train, holdout)
    for case in train + holdout:
        if len(case.features) != len(base):
            raise ValueError("feature dimensionality mismatch")

    current = list(base)
    for _ in range(policy.maximum_steps):
        gradient = [0.0] * len(base)
        for case in train:
            error = _predict(tuple(current), case.features) - case.observed_value
            for index, feature in enumerate(case.features):
                gradient[index] += error * feature
        candidate = [
            weight - policy.learning_rate * grad / len(train)
            for weight, grad in zip(current, gradient)
        ]
        # The admitted coefficient domain is a second hard bound, distinct
        # from the cumulative delta L2 budget. Clip to this box first; the
        # subsequent ball projection is a convex combination with `base`,
        # which is itself inside the box.
        candidate = [max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, value)) for value in candidate]
        displacement = [value - initial for value, initial in zip(candidate, base)]
        norm = hypot(*displacement)
        if norm > policy.max_delta_l2:
            factor = policy.max_delta_l2 / norm
            candidate = [initial + factor * change for initial, change in zip(base, displacement)]
        current = [max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, value)) for value in candidate]

    delta_values = tuple(final - initial for final, initial in zip(current, base))
    effective = tuple(initial + change for initial, change in zip(base, delta_values))
    return WeightDelta(
        binding=binding,
        base_digest=digest_weights(base),
        delta=delta_values,
        delta_digest=digest_weights(delta_values),
        effective_digest=digest_weights(effective),
        training_digest=_case_digest(train),
        policy_digest=_policy_digest(policy),
        policy=policy,
    )


def apply_delta(
    base_weights: tuple[float, ...], delta: WeightDelta, binding: ModelBinding
) -> tuple[float, ...]:
    """Construct effective weights without mutating or replacing the base."""
    base = _vector(base_weights, "weights", maximum=MAX_ABS_WEIGHT)
    if not isinstance(delta, WeightDelta) or not isinstance(binding, ModelBinding):
        raise TypeError("delta and binding must have their declared types")
    binding = ModelBinding(**binding.__dict__)
    if delta.capability_tier != CAPABILITY_TIER or delta.phenomenology != PHENOMENOLOGY:
        raise ValueError("unsupported weight delta tier")
    if delta.binding != binding:
        raise ValueError("model binding mismatch")
    if delta.base_digest != digest_weights(base):
        raise ValueError("base weights digest mismatch")
    checked = _vector(delta.delta, "delta", maximum=MAX_ABS_WEIGHT)
    if delta.delta_digest != digest_weights(checked):
        raise ValueError("delta digest mismatch")
    if len(checked) != len(base):
        raise ValueError("delta dimensionality mismatch")
    if not isinstance(delta.policy, TrainingPolicy):
        raise TypeError("delta policy must be TrainingPolicy")
    policy = TrainingPolicy(**delta.policy.__dict__)
    if delta.policy_digest != _policy_digest(policy):
        raise ValueError("policy digest mismatch")
    if hypot(*checked) > policy.max_delta_l2 * (1.0 + 1e-12):
        raise ValueError("delta exceeds policy max_delta_l2")
    effective = tuple(initial + change for initial, change in zip(base, checked))
    if delta.effective_digest != digest_weights(effective):
        raise ValueError("effective weights digest mismatch")
    return effective


def evaluate_heldout(
    weights: tuple[float, ...],
    heldout_cases: tuple[OutcomeCase, ...],
    *,
    training_cases: tuple[OutcomeCase, ...] | None = None,
) -> EvaluationResult:
    """Score caller labels; split status is unverified without training IDs.

    Even DISJOINT_IDS_CHECKED certifies only an ID comparison of supplied
    inputs, not provenance, absence of semantic leakage, or independent review.
    """
    checked_weights = _vector(weights, "weights", maximum=MAX_ABS_WEIGHT)
    heldout = _cases(heldout_cases, "holdout")
    train = _cases(training_cases, "training") if training_cases is not None else ()
    if not heldout:
        raise ValueError("at least one holdout case required")
    _disjoint(train, heldout)
    for case in heldout:
        if len(case.features) != len(checked_weights):
            raise ValueError("feature dimensionality mismatch")
    errors = [_predict(checked_weights, case.features) - case.observed_value for case in heldout]
    return EvaluationResult(
        case_count=len(errors),
        mean_squared_error=sum(e * e for e in errors) / len(errors),
        mean_absolute_error=sum(abs(e) for e in errors) / len(errors),
        weights_digest=digest_weights(checked_weights),
        split_status=("DISJOINT_IDS_CHECKED" if train else "UNVERIFIED_WITHOUT_TRAINING_SET"),
    )


class WeightDeltaSession:
    """Scoped surrogate activation: disabling restores the original exact tuple.

    Ordinary attribute assignment/deletion cannot replace the validated
    snapshots. As for any in-process Python object, a privileged caller using
    `object.__setattr__` or memory introspection is outside this boundary.
    """

    __slots__ = ("_base", "_effective", "_active")

    def __init__(self, base_weights: tuple[float, ...], delta: WeightDelta, binding: ModelBinding):
        baseline = _vector(base_weights, "weights", maximum=MAX_ABS_WEIGHT)
        effective = apply_delta(baseline, delta, binding)
        object.__setattr__(self, "_base", baseline)
        object.__setattr__(self, "_effective", effective)
        object.__setattr__(self, "_active", False)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("validated weight session is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("validated weight session is immutable")

    @property
    def active(self) -> bool:
        return self._active

    @property
    def current_weights(self) -> tuple[float, ...]:
        return self._effective if self._active else self._base

    def enable(self) -> tuple[float, ...]:
        if self._active:
            raise RuntimeError("weight delta already enabled")
        object.__setattr__(self, "_active", True)
        return self._effective

    def disable(self) -> tuple[float, ...]:
        object.__setattr__(self, "_active", False)
        return self._base

    def __enter__(self) -> tuple[float, ...]:
        return self.enable()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.disable()
        return False
