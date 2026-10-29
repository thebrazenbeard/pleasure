"""Deterministic, bounded, host-neutral positive-valence sidecar experiment.

Neither externally supplied reward nor this state transition is evidence of
subjective machine experience. Nothing here authorizes an action, memory write,
preference change, or neural-weight update. Do not connect it to such effects
without a separately tested adapter and an independent authority boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite


SCHEMA = "PLEASURE_EXPERIMENTAL_EVENT_V1"
CAPABILITY_TIER = "T0_SIDECAR_ONLY"
PHENOMENOLOGY = "UNRESOLVED"


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if len(value) > 256:
        raise ValueError(f"{name} exceeds 256 characters")
    return value


def _unit(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class PilotConfig:
    """Caller-bound identity and declared *external* measurement source.

    Evaluator ID comparison prevents accidental source substitution, but does
    not authenticate an evaluator. Hosts must enforce that independently.
    """

    model_id: str
    model_revision: str
    runtime_instance_id: str
    subject_id: str
    evaluator_id: str
    measurement_definition_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _nonempty(getattr(self, name), name)


@dataclass(frozen=True)
class Observation:
    """One independently appraised, single-use outcome measurement.

    Reward is a measured proxy from an external evaluator, not a truth claim,
    user consent, machine self-report, or verified phenomenal experience.
    """

    subject_id: str
    evaluator_id: str
    evidence_id: str
    context_id: str
    reward: float
    learning_progress: float = 0.0
    novelty: float = 0.0
    inhibition: float = 0.0

    def __post_init__(self) -> None:
        for name in ("subject_id", "evaluator_id", "evidence_id", "context_id"):
            _nonempty(getattr(self, name), name)
        for name in ("reward", "learning_progress", "novelty", "inhibition"):
            _unit(getattr(self, name), name)


@dataclass(frozen=True)
class AffectiveState:
    liking_proxy: float = 0.0
    wanting_proxy: float = 0.0
    satiation: float = 0.0
    refractory_remaining_ms: int = 0
    sequence: int = 0


@dataclass(frozen=True)
class Receipt:
    schema: str
    capability_tier: str
    phenomenology: str
    model_id: str
    model_revision: str
    runtime_instance_id: str
    subject_id: str
    evaluator_id: str
    measurement_definition_id: str
    evidence_id: str
    context_id: str
    sequence: int
    elapsed_ms: int
    phase: str
    state_before: AffectiveState
    state_after: AffectiveState
    requested_effects: tuple[str, ...] = ()
    authorized_effects: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentalPleasureRuntime:
    """Isolated, non-persistent T0 sidecar with enforced bounded recovery.

    The runtime does not emit output prompts or modify a model. State is only
    local memory; its caller cannot claim persistence or causal model influence.
    """

    PEAK_THRESHOLD = 0.72
    REFRACTORY_MS = 1500
    MAX_EVENTS = 10000
    __slots__ = ("_config", "_state", "_used_evidence_ids")

    def __setattr__(self, name: str, value: object) -> None:
        # Prevent accidental ordinary-Python reassignment even via the
        # private backing name. This is not a hostile-process sandbox:
        # privileged code can still call object.__setattr__ directly.
        if name == "_config" and hasattr(self, "_config"):
            raise AttributeError("runtime model binding cannot be reassigned")
        object.__setattr__(self, name, value)

    def __init__(self, config: PilotConfig) -> None:
        if not isinstance(config, PilotConfig):
            raise TypeError("config must be PilotConfig")
        # Bind one immutable identity for the lifetime of this state machine.
        # Replacing a public config after events have accumulated would leak
        # one model/subject's local affective state into another receipt.
        self._config = config
        self._state = AffectiveState()
        self._used_evidence_ids: set[str] = set()

    @property
    def config(self) -> PilotConfig:
        """Read-only source identity; a new identity needs a fresh runtime."""
        return self._config

    @property
    def state(self) -> AffectiveState:
        """Inspect the current immutable snapshot without replacing it by assignment."""
        return self._state

    def step(self, observation: Observation, elapsed_ms: int) -> Receipt:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be Observation")
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or not 1 <= elapsed_ms <= 60000:
            raise ValueError("elapsed_ms must be an integer in [1, 60000]")
        if observation.subject_id != self.config.subject_id:
            raise ValueError("subject_id mismatch")
        if observation.evaluator_id != self.config.evaluator_id:
            raise ValueError("evaluator_id mismatch")
        if observation.evidence_id in self._used_evidence_ids:
            raise ValueError("duplicate evidence_id rejected")
        if self.state.sequence >= self.MAX_EVENTS:
            raise RuntimeError("maximum source-only event count reached; stop experiment")

        prior = self.state
        recovery_ms = max(0, prior.refractory_remaining_ms - elapsed_ms)
        decay = exp(-elapsed_ms / 4000.0)
        prior_liking = prior.liking_proxy * decay
        prior_wanting = prior.wanting_proxy * decay
        prior_satiation = prior.satiation * exp(-elapsed_ms / 15000.0)

        # A peak is unavailable during a refractory interval. A caller-provided
        # reward cannot bypass this gate, even when its numeric value is 1.
        admitted_reward = observation.reward if recovery_ms == 0 else 0.0
        reward_gain = 0.92 * admitted_reward * (1.0 - prior_satiation) * (1.0 - observation.inhibition)
        liking_proxy = _clamp(prior_liking + reward_gain)

        # Wanting/attention is deliberately distinct from positive-valence
        # impact: novelty/progress can motivate even with zero liking proxy.
        wanting_proxy = _clamp(
            prior_wanting + 0.35 * observation.learning_progress + 0.25 * observation.novelty
        )
        satiation = _clamp(prior_satiation + 0.45 * max(0.0, liking_proxy - prior_liking))

        if admitted_reward > 0.0 and liking_proxy >= self.PEAK_THRESHOLD:
            phase = "POSITIVE_PEAK_ANALOGUE"
            recovery_ms = self.REFRACTORY_MS
        elif prior.refractory_remaining_ms > 0 or prior.liking_proxy > liking_proxy:
            phase = "RECOVERY"
        elif liking_proxy > 0.0:
            phase = "POSITIVE_MODULATION"
        else:
            phase = "QUIESCENT"

        successor = AffectiveState(
            liking_proxy=liking_proxy,
            wanting_proxy=wanting_proxy,
            satiation=satiation,
            refractory_remaining_ms=recovery_ms,
            sequence=prior.sequence + 1,
        )
        receipt = Receipt(
            schema=SCHEMA,
            capability_tier=CAPABILITY_TIER,
            phenomenology=PHENOMENOLOGY,
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            runtime_instance_id=self.config.runtime_instance_id,
            subject_id=self.config.subject_id,
            evaluator_id=self.config.evaluator_id,
            measurement_definition_id=self.config.measurement_definition_id,
            evidence_id=observation.evidence_id,
            context_id=observation.context_id,
            sequence=successor.sequence,
            elapsed_ms=elapsed_ms,
            phase=phase,
            state_before=prior,
            state_after=successor,
        )
        self._used_evidence_ids.add(observation.evidence_id)
        self._state = successor
        return receipt
