"""Candidate-conditioned, chosen-only v4 development head.

A zero-dependency numeric baseline that scores candidate features, not A/B
tokens. This is a TRAIN-only development mechanism, not a new pretrained-model
update, EVAL validation, externally authenticated receipt, or experience.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import exp, isfinite, log, sqrt
import re
from threading import RLock

FEATURE_SCHEMA = "numeric-candidate-xy-shared-sigmoid-v4"
EVIDENCE_CLASS = "SOURCE_LOCAL_CHOSEN_FEEDBACK_CANDIDATE_V4_DEVELOPMENT"
_EPISODE = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_EVIDENCE = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _finite(value: object, label: str, low: float, high: float) -> float:
    if type(value) not in (int, float) or not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be finite within [{low}, {high}]")
    return float(value)


def _candidate(value: object) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        raise TypeError("candidate must be exactly two floats")
    if any(type(v) is not float for v in value):
        raise TypeError("candidate x/y must be floats")
    return (_finite(value[0], "candidate x", 0.0, 1.0),
            _finite(value[1], "candidate y", 0.0, 1.0))


def _sigmoid(raw: float) -> float:
    if raw >= 0:
        return 1.0 / (1.0 + exp(-raw))
    x = exp(raw)
    return x / (1.0 + x)


def _basis(features: tuple[float, float]) -> tuple[float, float, float]:
    return (1.0, 2.0 * features[0] - 1.0, 2.0 * features[1] - 1.0)


@dataclass(frozen=True)
class CandidatePrediction:
    features_a: tuple[float, float]
    features_b: tuple[float, float]
    quality_a: float
    quality_b: float
    prob_a: float
    prob_b: float
    head_digest: str
    episode_id: str | None = None
    evidence_class: str = EVIDENCE_CLASS


@dataclass(frozen=True)
class CandidateCommit:
    decision: CandidatePrediction
    chosen_action: str
    chosen_propensity: float
    decision_digest: str
    evidence_class: str = EVIDENCE_CLASS


@dataclass(frozen=True)
class CandidateUpdate:
    episode_id: str
    evidence_id: str
    update_index: int
    before_digest: str
    after_digest: str
    effective_importance_weight: float
    gradient_l2: float
    evidence_class: str = EVIDENCE_CLASS


class CandidateValueHead:
    """Shared A/B candidate scorer with an exact chosen-feedback replay fence.

    Head features are (1,2x-1,2y-1). The same weights score A and B, so
    swapping feature tuples *exactly* swaps predicted qualities. TRAIN
    commits are distinct from EVAL inference. Local hashes are integrity
    checks, NOT third-party provenance or authority.
    """

    feature_schema = FEATURE_SCHEMA

    def __init__(self, *, allow_training: bool = False,
                 learning_rate: float = 0.5, max_updates: int = 128,
                 max_gradient_l2: float = 1.0) -> None:
        if type(allow_training) is not bool:
            raise TypeError("allow_training must be an explicit bool")
        self._rate = _finite(learning_rate, "learning_rate", 0.0, 4.0)
        self._grad_cap = _finite(max_gradient_l2, "gradient cap", 0.000001, 5.0)
        if type(max_updates) is not int or not 1 <= max_updates <= 4096:
            raise ValueError("bounded max_updates of 1..4096 required")
        self._allow_training = allow_training
        self._max_updates = max_updates
        self._weights: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._updates = 0
        self._seen_evidence: set[str] = set()
        self._seen_episodes: set[str] = set()
        self._pending: CandidatePrediction | CandidateCommit | None = None
        self._lock = RLock()

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    @property
    def head_digest(self) -> str:
        with self._lock:
            return _sha((FEATURE_SCHEMA, self._weights))

    def _quality(self, features: tuple[float, float]) -> float:
        basis = _basis(features)
        return _sigmoid(sum(w * x for w, x in zip(self._weights, basis)))

    def predict(self, features_a: tuple[float, float],
                features_b: tuple[float, float], *, gain: float = 8.0,
                base_prob_a: float = 0.5) -> CandidatePrediction:
        a, b = _candidate(features_a), _candidate(features_b)
        strength = _finite(gain, "gain", -16.0, 16.0)
        base = _finite(base_prob_a, "base probability", 0.000001, 0.999999)
        with self._lock:
            qa, qb = self._quality(a), self._quality(b)
            # The bounded probability intervention is symmetric for A/B
            # under a neutral reference policy. It is NOT used for TRAIN
            # action sampling, which must come from an external frozen policy.
            ratio = (base / (1.0-base))
            logit = log(ratio) + strength * (qa - qb)
            pa = _sigmoid(logit)
            return CandidatePrediction(
                features_a=a, features_b=b, quality_a=qa, quality_b=qb,
                prob_a=pa, prob_b=1.0-pa, head_digest=self.head_digest)

    def prepare_train(self, episode_id: str, features_a: tuple[float, float],
                      features_b: tuple[float, float]) -> CandidatePrediction:
        if type(episode_id) is not str or _EPISODE.fullmatch(episode_id) is None:
            raise ValueError("bounded TRAIN episode ID required")
        with self._lock:
            if not self._allow_training or self._rate == 0.0:
                raise PermissionError("explicit nonzero-rate TRAIN permission required")
            if episode_id in self._seen_episodes:
                raise ValueError("TRAIN episode replay detected")
            if self._pending is not None:
                raise ValueError("another TRAIN choice remains pending")
            if self._updates >= self._max_updates:
                raise ValueError("TRAIN update cap exceeded")
            decision = replace(self.predict(features_a, features_b),
                               episode_id=episode_id)
            self._pending = decision
            return decision

    def commit_train(self, decision: CandidatePrediction, chosen_action: str,
                     chosen_propensity: float) -> CandidateCommit:
        if type(chosen_action) is not str or chosen_action not in ("A", "B"):
            raise ValueError("chosen_action must be A or B")
        propensity = _finite(chosen_propensity, "chosen propensity", 0.000001, 1.0)
        with self._lock:
            if (type(decision) is not CandidatePrediction or
                self._pending is not decision):
                raise ValueError("no matching pending TRAIN decision")
            commit = CandidateCommit(
                decision=decision, chosen_action=chosen_action,
                chosen_propensity=propensity,
                decision_digest=_sha((
                    "candidate-v4-commit", decision.episode_id,
                    decision.head_digest, decision.features_a,
                    decision.features_b, chosen_action, propensity)),
            )
            self._pending = commit
            return commit

    def learn_chosen(self, commit: CandidateCommit, evidence_id: str,
                     measured_chosen_quality: float) -> CandidateUpdate:
        if type(commit) is not CandidateCommit:
            raise TypeError("precommitted chosen TRAIN decision required")
        if type(evidence_id) is not str or _EVIDENCE.fullmatch(evidence_id) is None:
            raise ValueError("source-local SHA256 evidence ID required")
        reward = _finite(measured_chosen_quality, "chosen quality", 0.0, 1.0)
        with self._lock:
            if self._pending is not commit:
                raise ValueError("no matching pending chosen TRAIN decision")
            if evidence_id in self._seen_evidence:
                raise ValueError("TRAIN evidence replay detected")
            if commit.decision.episode_id in self._seen_episodes:
                raise ValueError("TRAIN episode replay detected")
            if commit.decision.head_digest != self.head_digest:
                raise RuntimeError("head changed after TRAIN decision commitment")
            if self._updates >= self._max_updates:
                raise ValueError("TRAIN update cap exceeded")
            chosen = (commit.decision.features_a if commit.chosen_action == "A"
                      else commit.decision.features_b)
            before = self.head_digest
            prediction = self._quality(chosen)
            importance = min(5.0, 1.0 / commit.chosen_propensity)
            basis = _basis(chosen)
            common = (prediction - reward) * prediction * (1.0-prediction) * importance
            gradients = tuple(common*x for x in basis)
            gradient_l2 = sqrt(sum(v*v for v in gradients))
            scale = min(1.0, self._grad_cap / max(gradient_l2, 1e-12))
            updated = tuple(max(-8.0, min(8.0, w-self._rate*scale*g))
                            for w, g in zip(self._weights, gradients))
            if any(not isfinite(x) for x in updated):
                raise RuntimeError("nonfinite candidate update")
            self._weights = updated
            self._updates += 1
            self._seen_evidence.add(evidence_id)
            self._seen_episodes.add(commit.decision.episode_id)
            self._pending = None
            return CandidateUpdate(
                episode_id=commit.decision.episode_id, evidence_id=evidence_id,
                update_index=self._updates, before_digest=before,
                after_digest=self.head_digest,
                effective_importance_weight=importance,
                gradient_l2=gradient_l2,
            )

    def export_state(self) -> dict[str, object]:
        with self._lock:
            if self._pending is not None:
                raise ValueError("pending TRAIN choice cannot be exported")
            value: dict[str, object] = {
                "format": "CANDIDATE_VALUE_HEAD_V4_JSON_V1",
                "feature_schema": FEATURE_SCHEMA,
                "weights": list(self._weights),
                "updates": self._updates,
                "max_updates": self._max_updates,
                "learning_rate": self._rate,
                "max_gradient_l2": self._grad_cap,
                "seen_evidence_ids": sorted(self._seen_evidence),
                "seen_episode_ids": sorted(self._seen_episodes),
                "evidence_class": EVIDENCE_CLASS,
            }
            return {**value, "state_digest": _sha(value)}

    @classmethod
    def from_state(cls, state: dict[str, object], *, allow_training: bool = False,
                   min_updates: int = 0) -> "CandidateValueHead":
        if type(state) is not dict or type(min_updates) is not int or min_updates < 0:
            raise ValueError("bounded state and rollback floor required")
        expected = frozenset((
            "format", "feature_schema", "weights", "updates", "max_updates",
            "learning_rate", "max_gradient_l2", "seen_evidence_ids",
            "seen_episode_ids", "evidence_class", "state_digest"))
        if set(state) != expected:
            raise ValueError("exact candidate state schema required")
        payload = {k: v for k, v in state.items() if k != "state_digest"}
        if state["state_digest"] != _sha(payload):
            raise ValueError("candidate state digest mismatch")
        if (state["format"] != "CANDIDATE_VALUE_HEAD_V4_JSON_V1" or
            state["feature_schema"] != FEATURE_SCHEMA or
            state["evidence_class"] != EVIDENCE_CLASS):
            raise ValueError("unknown candidate state schema")
        weights = state["weights"]
        if (type(weights) is not list or len(weights) != 3 or
            any(type(w) is not float or not isfinite(w) or abs(w) > 8
                for w in weights)):
            raise ValueError("bounded float weights required")
        updates = state["updates"]
        if type(updates) is not int or updates < min_updates:
            raise ValueError("candidate state rollback detected")
        evidence, episodes = state["seen_evidence_ids"], state["seen_episode_ids"]
        if (type(evidence) is not list or type(episodes) is not list or
            len(evidence) != updates or len(episodes) != updates or
            len(set(evidence)) != len(evidence) or
            len(set(episodes)) != len(episodes) or
            any(type(x) is not str or _EVIDENCE.fullmatch(x) is None for x in evidence) or
            any(type(x) is not str or _EPISODE.fullmatch(x) is None for x in episodes)):
            raise ValueError("invalid consumed TRAIN ledger")
        head = cls(allow_training=allow_training,
                   learning_rate=state["learning_rate"],
                   max_gradient_l2=state["max_gradient_l2"],
                   max_updates=state["max_updates"])
        if updates > head._max_updates:
            raise ValueError("invalid update counter")
        head._weights = tuple(weights)
        head._updates = updates
        head._seen_evidence.update(evidence)
        head._seen_episodes.update(episodes)
        return head
