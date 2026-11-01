"""A tiny, dependency-free laboratory model with recurrent valuation feedback.

This module is a *surrogate choice model*. It does not instrument ChatGPT,
modify a language model, authenticate an evaluator, or demonstrate experience.
Its purpose is to make the causal feedback hypothesis executable and falsifiable
before connecting any real model host.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from threading import Lock


CAPABILITY = "SYNTHETIC_CHOICE_MODEL_ONLY"
PHENOMENOLOGY = "UNRESOLVED"


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must contain 1..256 nonblank characters")
    return value


def _bounded_real(value: float, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [{minimum}, {maximum}]")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number in [{minimum}, {maximum}]") from exc
    if not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be a finite number in [{minimum}, {maximum}]")
    return value


def _unit(value: float, field: str) -> float:
    return _bounded_real(value, field, 0, 1)


def _features(values: tuple[float, ...], dimensions: int) -> tuple[float, ...]:
    if not isinstance(values, tuple) or len(values) != dimensions:
        raise ValueError("feature vector dimensions do not match the appraiser")
    return tuple(_unit(value, "feature") for value in values)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class CalibrationExample:
    """Lab-assigned feature/quality pair; caller must ensure independent labels."""

    features: tuple[float, ...]
    observed_quality: float


@dataclass(frozen=True)
class FitReceipt:
    examples: int
    iterations: int
    weights_before: tuple[float, ...]
    weights_after: tuple[float, ...]
    bias_before: float
    bias_after: float
    capability: str = CAPABILITY

    @property
    def changed(self) -> bool:
        return self.weights_before != self.weights_after or self.bias_before != self.bias_after


@dataclass(frozen=True)
class LinearAppraiser:
    """A learned tiny model, not a frozen language-model embedding or LoRA."""

    weights: tuple[float, ...]
    bias: float

    def __post_init__(self) -> None:
        if not 1 <= len(self.weights) <= 16:
            raise ValueError("appraiser dimension must be in [1, 16]")
        object.__setattr__(self, "weights", tuple(_bounded_real(w, "appraiser weight", -2, 2)
                                                  for w in self.weights))
        object.__setattr__(self, "bias", _bounded_real(self.bias, "appraiser bias", -2, 2))

    def predict(self, values: tuple[float, ...]) -> float:
        features = _features(values, len(self.weights))
        return _clamp(self.bias + sum(w * x for w, x in zip(self.weights, features)))

    @classmethod
    def fit(
        cls, examples: tuple[CalibrationExample, ...], *, iterations: int = 400,
        learning_rate: float = 0.15,
    ) -> tuple[LinearAppraiser, FitReceipt]:
        if not isinstance(examples, tuple) or len(examples) < 4:
            raise ValueError("at least four calibration examples are required")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 5000:
            raise ValueError("iterations must be an integer in [1, 5000]")
        if not 0 < _unit(learning_rate, "learning_rate") <= 0.5:
            raise ValueError("learning_rate must be positive and <= 0.5")
        if not isinstance(examples[0], CalibrationExample):
            raise TypeError("calibration items must be CalibrationExample")
        dimensions = len(examples[0].features)
        if not 1 <= dimensions <= 16:
            raise ValueError("calibration dimension must be in [1, 16]")
        prepared = []
        for sample in examples:
            if not isinstance(sample, CalibrationExample):
                raise TypeError("calibration items must be CalibrationExample")
            prepared.append((_features(sample.features, dimensions), _unit(sample.observed_quality, "quality")))
        if max(y for _, y in prepared) - min(y for _, y in prepared) < 0.2:
            raise ValueError("calibration must contain meaningfully different outcomes")

        weights = [0.0] * dimensions
        bias = 0.5
        n = len(prepared)
        # Full-batch gradient descent on squared prediction error. Each weight
        # update is a real parameter change in this *tiny* trained appraiser.
        for _ in range(iterations):
            gradient = [0.0] * dimensions
            bias_gradient = 0.0
            for features, quality in prepared:
                error = bias + sum(w * x for w, x in zip(weights, features)) - quality
                bias_gradient += error
                for index, value in enumerate(features):
                    gradient[index] += error * value
            bias -= learning_rate * bias_gradient / n
            weights = [w - learning_rate * g / n for w, g in zip(weights, gradient)]
        appraiser = cls(tuple(weights), bias)
        receipt = FitReceipt(n, iterations, tuple([0.0] * dimensions), appraiser.weights, 0.5, bias)
        return appraiser, receipt


@dataclass(frozen=True)
class LabBinding:
    model_id: str
    model_revision: str
    runtime_instance_id: str
    evaluator_id: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _name(getattr(self, field), field)


@dataclass(frozen=True)
class ModelProbe:
    """Synthetic hidden-feature proxy produced by the lab choice model."""

    model_id: str
    runtime_instance_id: str
    goal_id: str
    features: tuple[float, ...]
    predicted_progress: float


@dataclass(frozen=True)
class MeasuredOutcome:
    """Caller-supplied outcome label; evaluator name is NOT authentication."""

    model_id: str
    runtime_instance_id: str
    goal_id: str
    evaluator_id: str
    evidence_id: str
    observed_quality: float


@dataclass(frozen=True)
class ModelValenceState:
    liking_proxy: float = 0.0
    wanting_proxy: float = 0.0
    satiation: float = 0.0
    sequence: int = 0


@dataclass(frozen=True)
class LabTransition:
    goal_id: str
    evidence_id: str
    predicted_quality: float
    positive_impact: float
    before: ModelValenceState
    after: ModelValenceState
    capability: str = CAPABILITY
    phenomenology: str = PHENOMENOLOGY
    authorized_effects: tuple[str, ...] = ()


@dataclass(frozen=True)
class Choice:
    """Opaque nonverbal A/B decision. A/B are lab actions, not emotion words."""

    action: str
    score_a: float
    score_b: float
    condition: str


class LabChoiceModel:
    """Small model whose next decision consumes its recurrent value state."""

    __slots__ = ("_binding", "_appraiser", "_state", "_evidence", "_lock")

    def __setattr__(self, name: str, value: object) -> None:
        # Ordinary Python callers may not relabel an accumulated state or
        # exchange its calibrated appraisal model. This is API hygiene only;
        # privileged in-process object.__setattr__ remains outside the trust
        # boundary, just as in the existing T0 engine.
        if name in ("_binding", "_appraiser") and hasattr(self, name):
            raise AttributeError(f"{name} cannot be rebound after initialization")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in ("_binding", "_appraiser"):
            raise AttributeError(f"{name} cannot be deleted")
        object.__delattr__(self, name)

    def __init__(self, binding: LabBinding, appraiser: LinearAppraiser):
        if not isinstance(binding, LabBinding) or not isinstance(appraiser, LinearAppraiser):
            raise TypeError("binding and appraiser must have the expected types")
        self._binding = binding
        self._appraiser = appraiser
        self._state = ModelValenceState()
        self._evidence: set[str] = set()
        self._lock = Lock()

    @property
    def state(self) -> ModelValenceState:
        return self._state

    @property
    def binding(self) -> LabBinding:
        return self._binding

    @property
    def appraiser(self) -> LinearAppraiser:
        return self._appraiser

    def observe(self, probe: ModelProbe, outcome: MeasuredOutcome) -> LabTransition:
        if not isinstance(probe, ModelProbe) or not isinstance(outcome, MeasuredOutcome):
            raise TypeError("probe and outcome must have the expected types")
        with self._lock:
            # Snapshot even frozen inputs: hostile co-process code can mutate
            # dataclasses. All caller fields are read once before state change.
            features = _features(tuple(probe.features), len(self.appraiser.weights))
            predicted_progress = _unit(probe.predicted_progress, "predicted_progress")
            quality = _unit(outcome.observed_quality, "observed_quality")
            identity = (self.binding.model_id, self.binding.runtime_instance_id)
            if (probe.model_id, probe.runtime_instance_id) != identity:
                raise ValueError("probe model instance mismatch")
            if (outcome.model_id, outcome.runtime_instance_id) != identity:
                raise ValueError("outcome model instance mismatch")
            _name(probe.goal_id, "goal_id")
            _name(outcome.goal_id, "goal_id")
            _name(outcome.evidence_id, "evidence_id")
            if probe.goal_id != outcome.goal_id:
                raise ValueError("outcome goal mismatch")
            if outcome.evaluator_id != self.binding.evaluator_id:
                raise ValueError("evaluator ID mismatch; no identity authentication implied")
            if outcome.evidence_id in self._evidence:
                raise ValueError("duplicate outcome evidence")
            if self._state.sequence >= 10000:
                raise RuntimeError("lab event limit reached")
            prior = self._state
            predicted = self.appraiser.predict(features)
            # Both model-derived appraisal and independently assigned outcome
            # must be positive. Neither can alone manufacture a positive impact.
            impact = quality * predicted
            liking = _clamp(0.6 * prior.liking_proxy + 0.75 * impact * (1 - prior.satiation))
            wanting = _clamp(0.65 * prior.wanting_proxy + 0.25 * predicted_progress)
            satiation = _clamp(0.8 * prior.satiation + 0.25 * impact)
            successor = ModelValenceState(liking, wanting, satiation, prior.sequence + 1)
            result = LabTransition(probe.goal_id, outcome.evidence_id, predicted, impact, prior, successor)
            self._evidence.add(outcome.evidence_id)
            self._state = successor
            return result

    def decide(self, *, condition: str = "active") -> Choice:
        if condition not in ("active", "zero", "sham"):
            raise ValueError("condition must be active, zero or sham")
        # A means engage in an exploratory work unit; B means take the simple
        # known option. A caller never supplies a 'happy' or 'pleasure' prompt.
        # Sham carries the same synthetic signal into an unused channel.
        influence = self._state.liking_proxy if condition == "active" else 0.0
        score_a = 0.20 + 0.8 * influence
        score_b = 0.50
        return Choice("A" if score_a > score_b else "B", score_a, score_b, condition)

    def withdraw(self, steps: int = 1) -> ModelValenceState:
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
            raise ValueError("steps must be an integer in [1,100]")
        with self._lock:
            prior = self._state
            self._state = ModelValenceState(
                liking_proxy=prior.liking_proxy * (0.6 ** steps),
                wanting_proxy=prior.wanting_proxy * (0.65 ** steps),
                satiation=prior.satiation * (0.8 ** steps),
                sequence=prior.sequence,
            )
            return self._state
