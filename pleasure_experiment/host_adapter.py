"""Source-only boundary for a synthetic host-bound T0 Pleasure adapter.

This is NOT a trusted authorization service or an installed model adapter.
An actual host must own and authenticate the principal, supply trusted time,
and atomically verify-and-consume the consent/lease/scope/JTI on durable,
non-rollback storage shared by every reader. In-process Python cannot prove
those host properties or protect itself from a hostile co-resident caller.

This module never invokes model inference or performs a protected effect.
Its only admitted operation advances an isolated local T0 sidecar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from json import dumps
from math import isfinite
from typing import Callable

from .runtime import AffectiveState, ExperimentalPleasureRuntime, Observation, Receipt


class NotAuthorized(PermissionError):
    """A requested T0 transition failed the host-binding contract."""


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonblank string of at most 256 characters")


def _clock(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric host time")
    return float(value)


@dataclass(frozen=True)
class HostPrincipal:
    """Trusted-host assertion; no model-created object can authenticate itself."""

    model_id: str
    model_revision: str
    subject_id: str
    runtime_instance_id: str
    adapter_id: str
    adapter_revision: str
    principal_id: str
    host_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class HostPermission:
    """Scoped evidence to be rechecked and consumed by an external host."""

    principal: HostPrincipal
    capability_id: str
    operation: str
    context_id: str
    consent_ref: str
    lease_ref: str
    effect_scope: str
    jti: str
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.principal, HostPrincipal):
            raise ValueError("principal must be a host-bound identity declaration")
        for name in (
            "capability_id", "operation", "context_id", "consent_ref",
            "lease_ref", "effect_scope", "jti",
        ):
            _text(getattr(self, name), name)
        object.__setattr__(self, "expires_at", _clock(self.expires_at, "expires_at"))


# The callback must be provisioned by a host outside model influence. A
# process-local callback or dictionary is ONLY a unit-test double, never a
# durable consent service or cross-reader replay defense.
# The host must authenticate an independently admitted, exact request. Pass
# immutable canonical bytes instead of the mutable-by-ref Python observation:
# even a frozen dataclass can be changed using object.__setattr__.
HostAtomicConsume = Callable[[HostPermission, HostPrincipal, bytes, int, float], bool]


def _canonical_request(observation: Observation, elapsed_ms: int) -> bytes:
    return dumps(
        {"schema": "pleasure.t0.request.v1", "observation": asdict(observation), "elapsed_ms": elapsed_ms},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class HostBoundT0Adapter:
    """Only allow an external host-authorized, isolated T0 state transition."""

    CAPABILITY_ID = "pleasure.t0.step"
    OPERATION = "advance"
    EFFECT_SCOPE = "T0_SIDECAR_ONLY"
    __slots__ = ("_binding", "_engine", "_consume")

    def __init__(
        self,
        binding: HostPrincipal,
        engine: ExperimentalPleasureRuntime,
        consume: HostAtomicConsume,
    ) -> None:
        if not isinstance(binding, HostPrincipal):
            raise TypeError("binding must be HostPrincipal")
        if not isinstance(engine, ExperimentalPleasureRuntime):
            raise TypeError("engine must be ExperimentalPleasureRuntime")
        if not callable(consume):
            raise TypeError("trusted host consume callback is required")
        config = engine.config
        if (
            config.model_id,
            config.model_revision,
            config.subject_id,
            config.runtime_instance_id,
        ) != (
            binding.model_id,
            binding.model_revision,
            binding.subject_id,
            binding.runtime_instance_id,
        ):
            raise ValueError("engine and trusted host principal identity mismatch")
        self._binding = binding
        self._engine = engine
        self._consume = consume

    @property
    def state(self) -> AffectiveState:
        return self._engine.state

    def advance(
        self,
        permission: HostPermission,
        observation: Observation,
        *,
        elapsed_ms: int,
        now: float,
    ) -> Receipt:
        # The host is responsible for both caller authentication and the
        # timestamp; these Python arguments must not be model-controlled.
        now = _clock(now, "now")
        if not isinstance(permission, HostPermission) or type(observation) is not Observation:
            raise NotAuthorized("invalid admission evidence")
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, int) or not 1 <= elapsed_ms <= 60000:
            raise NotAuthorized("invalid T0 elapsed duration")
        # Copy once before calling external code. The callback receives only
        # immutable request bytes; the sidecar steps the very same snapshot.
        snapshot = replace(observation)
        if (
            permission.principal != self._binding
            or permission.capability_id != self.CAPABILITY_ID
            or permission.operation != self.OPERATION
            or permission.effect_scope != self.EFFECT_SCOPE
            or snapshot.context_id != permission.context_id
            or snapshot.subject_id != self._binding.subject_id
            or snapshot.evaluator_id != self._engine.config.evaluator_id
            or now >= permission.expires_at
        ):
            raise NotAuthorized("permission identity or scope mismatch")

        try:
            authorized = self._consume(permission, self._binding, _canonical_request(snapshot, elapsed_ms), elapsed_ms, now)
        except Exception:
            authorized = False
        if authorized is not True:
            raise NotAuthorized("host authorization not current or replayed")

        return self._engine.step(snapshot, elapsed_ms)
