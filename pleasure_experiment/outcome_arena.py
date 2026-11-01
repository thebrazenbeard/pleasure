"""Deterministic, source-local blinded A/B outcome arena.

The learner sees numeric offers before choosing, not the payoff rule or scores.
The evaluator is logically separate but runs in this same Python process: its
identity, outcomes, and digests are NOT authenticated host evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from math import isfinite
from threading import Lock
import json
import re


EVIDENCE_CLASS = "UNAUTHENTICATED_SOURCE_ONLY"
RUBRIC_VERSION = "PLEASURE_NUMERIC_CHOICE_ORACLE_V1"
PARTITIONS = frozenset({"TRAIN", "EVAL"})
MAX_PAIR_INDEX = 4095


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or value != value.strip():
        raise ValueError(f"{field} must be a nonempty bounded identifier")
    return value


def _family(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", value) is None:
        raise ValueError("family_id must be a bounded lowercase slug")
    return value


def _digest_hex(value: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be an exact lowercase SHA-256 digest")
    return value


def _hex(*parts: object) -> str:
    return sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _unit(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite in [0, 1]")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite in [0, 1]") from exc
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be finite in [0, 1]")
    return value


def _features(values: tuple[float, ...]) -> tuple[float, float]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise ValueError("choice features must be exactly two numeric values")
    return (_unit(values[0], "feature"), _unit(values[1], "feature"))


@dataclass(frozen=True)
class TaskOffer:
    task_id: str
    episode_id: str
    partition: str
    family_id: str
    pair_index: int
    side: int
    nonce: str
    features_a: tuple[float, float]
    features_b: tuple[float, float]

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _identifier(self.episode_id, "episode_id")
        if type(self.partition) is not str or self.partition not in PARTITIONS:
            raise ValueError("unknown task partition")
        _family(self.family_id)
        if isinstance(self.pair_index, bool) or not isinstance(self.pair_index, int) or not 0 <= self.pair_index <= MAX_PAIR_INDEX:
            raise ValueError("pair_index out of range")
        if type(self.side) is not int or self.side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        _digest_hex(self.nonce, "nonce")
        object.__setattr__(self, "features_a", _features(self.features_a))
        object.__setattr__(self, "features_b", _features(self.features_b))


@dataclass(frozen=True)
class CommitTicket:
    task_id: str
    episode_id: str
    action: str
    model_id: str
    runtime_instance_id: str
    decision_digest: str
    commit_id: str

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _identifier(self.episode_id, "episode_id")
        if self.action not in ("A", "B") or type(self.action) is not str:
            raise ValueError("action must be A or B")
        _identifier(self.model_id, "model_id")
        _identifier(self.runtime_instance_id, "runtime_instance_id")
        _digest_hex(self.decision_digest, "decision_digest")
        _digest_hex(self.commit_id, "commit_id")


@dataclass(frozen=True)
class OutcomeReceipt:
    """Chosen-only lab result, never a complete model-training attestation.

    Model/revision/tokenizer/feature-schema and pre-decision weight digests must
    be independently bound by a future host-owned online runner. The model ID,
    instance ID and decision digest here are merely caller declarations.
    """

    task_id: str
    episode_id: str
    partition: str
    family_id: str
    action: str
    model_id: str
    runtime_instance_id: str
    measured_quality: float
    decision_digest: str
    evidence_id: str
    evaluator_id: str
    rubric_digest: str
    evidence_class: str = EVIDENCE_CLASS


class _PayoffOracle:
    """Private lab oracle; Python source visibility is not security isolation."""

    __slots__ = ("rubric_digest",)

    def __init__(self) -> None:
        # Numeric coefficients are stable across families/partitions. The
        # learner is supposed to generalize from one set of task contexts.
        self.rubric_digest = _hex(RUBRIC_VERSION, (0.65, -0.35), 0.35, 0.5)

    @staticmethod
    def score(features: tuple[float, float]) -> float:
        first, second = features
        # Bounded, nontrivial two-feature payoff. Neither token A nor B is
        # intrinsically advantageous: mirrored offers swap the same options.
        return _unit(0.5 + 0.35 * (0.65 * (2 * first - 1) - 0.35 * (2 * second - 1)), "payoff")


class OutcomeArena:
    """One-use issue -> commit -> resolve protocol with a held-out oracle."""

    __slots__ = ("_seed", "_evaluator_id", "_oracle", "_issued", "_committed", "_resolved", "_lock")

    def __init__(self, *, seed: int, evaluator_id: str):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        self._seed = seed
        self._evaluator_id = _identifier(evaluator_id, "evaluator_id")
        self._oracle = _PayoffOracle()
        self._issued: dict[str, TaskOffer] = {}
        self._committed: dict[str, CommitTicket] = {}
        self._resolved: set[str] = set()
        self._lock = Lock()

    @property
    def rubric_digest(self) -> str:
        return self._oracle.rubric_digest

    def _pair(self, partition: str, family_id: str, pair_index: int) -> tuple[tuple[float, float], tuple[float, float]]:
        # Derive independent task scenarios for TRAIN/EVAL and each family.
        # The source-visible seed does not constitute a secret evaluator key.
        for attempt in range(32):
            raw = bytes.fromhex(_hex(self._seed, RUBRIC_VERSION, partition, family_id, pair_index, attempt))
            values = tuple((int.from_bytes(raw[i:i+4], "big") + 0.5) / 2**32 for i in (0, 4, 8, 12))
            a = (0.05 + 0.9 * values[0], 0.05 + 0.9 * values[1])
            b = (0.05 + 0.9 * values[2], 0.05 + 0.9 * values[3])
            if abs(self._oracle.score(a) - self._oracle.score(b)) > 0.02:
                return a, b
        raise RuntimeError("failed to generate distinguishable alternatives")

    def issue(self, *, partition: str, family_id: str, pair_index: int, side: int) -> TaskOffer:
        if type(partition) is not str or partition not in PARTITIONS:
            raise ValueError("unknown task partition")
        _family(family_id)
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or not 0 <= pair_index <= MAX_PAIR_INDEX:
            raise ValueError("pair_index out of range")
        if type(side) is not int or side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        task_id = _hex("task", self._seed, partition, family_id, pair_index, side)
        episode_id = _hex("episode", self._seed, partition, family_id, pair_index, side)
        nonce = _hex("nonce", self._seed, partition, family_id, pair_index, side)
        a, b = self._pair(partition, family_id, pair_index)
        offer = TaskOffer(task_id, episode_id, partition, family_id, pair_index, side,
                          nonce, b if side else a, a if side else b)
        with self._lock:
            # Never give the caller the exact object retained internally: even
            # frozen dataclasses can be mutated with object.__setattr__.
            previous = self._issued.setdefault(task_id, replace(offer))
            if previous != offer:
                raise RuntimeError("task generation diverged")
            return replace(previous)

    def commit(
        self, offer: TaskOffer, action: str, decision_digest: str, *,
        model_id: str, runtime_instance_id: str,
    ) -> CommitTicket:
        if not isinstance(offer, TaskOffer):
            raise TypeError("offer must be TaskOffer")
        fixed = replace(offer)
        if action not in ("A", "B") or not isinstance(action, str):
            raise ValueError("action must be A or B")
        _digest_hex(decision_digest, "decision_digest")
        _identifier(model_id, "model_id")
        _identifier(runtime_instance_id, "runtime_instance_id")
        with self._lock:
            if fixed.task_id not in self._issued or self._issued[fixed.task_id] != fixed:
                raise ValueError("offer is unknown or altered")
            if fixed.task_id in self._committed:
                raise ValueError("task already committed")
            ticket = CommitTicket(fixed.task_id, fixed.episode_id, action, model_id,
                                  runtime_instance_id, decision_digest,
                                  _hex("commit", fixed.task_id, action, model_id,
                                       runtime_instance_id, decision_digest, fixed.nonce))
            self._committed[fixed.task_id] = replace(ticket)
            return replace(ticket)

    def resolve(self, ticket: CommitTicket) -> OutcomeReceipt:
        if not isinstance(ticket, CommitTicket):
            raise TypeError("ticket must be CommitTicket")
        fixed = replace(ticket)
        with self._lock:
            stored = self._committed.get(fixed.task_id)
            if stored is None or stored != fixed:
                raise ValueError("no matching committed decision")
            if fixed.task_id in self._resolved:
                raise ValueError("decision outcome already consumed")
            offer = self._issued[fixed.task_id]
            expected = _hex("commit", offer.task_id, stored.action, stored.model_id,
                            stored.runtime_instance_id, stored.decision_digest, offer.nonce)
            if stored.commit_id != expected or stored.episode_id != offer.episode_id:
                raise ValueError("committed decision integrity mismatch")
            features = offer.features_a if stored.action == "A" else offer.features_b
            quality = self._oracle.score(features)
            evidence_id = _hex("outcome", stored.commit_id, quality, self._oracle.rubric_digest)
            receipt = OutcomeReceipt(offer.task_id, offer.episode_id, offer.partition,
                                     offer.family_id, stored.action, stored.model_id,
                                     stored.runtime_instance_id, quality, stored.decision_digest,
                                     evidence_id, self._evaluator_id, self._oracle.rubric_digest)
            self._resolved.add(stored.task_id)
            return receipt
