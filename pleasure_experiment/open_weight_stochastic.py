"""Reproducible, positivity-preserving A/B exploration and propensity logs.

This sampler is deliberately independent of the model and of outcome APIs.
The caller must obtain both action probabilities *before* seeing the chosen
outcome, make exactly one choice, and commit the returned action to its
chosen-only evaluator. A log is a locally reproducible trace, NOT an
authenticated experiment, unbiased reward estimator, or evidence of feeling.

The exploration mixture is epsilon-greedy in probability space:
    behavior(a) = (1 - epsilon) * model(a) + epsilon / 2.
Unlike argmax, both choices therefore retain known, strictly positive
propensities. Zero exploration is allowed *only* while both unblended model
probabilities are themselves positive. Epsilon is fixed when the sampler is
constructed; the caller cannot silently change it between decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
import random
import re
from threading import Lock


EVIDENCE_CLASS = "SOURCE_LOCAL_SEEDED_STOCHASTIC_ACTION_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}\Z")


def _float_probability(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite probability")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return result


def _decision_sha(payload: tuple[object, ...]) -> str:
    return sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
                  .encode("ascii")).hexdigest()


@dataclass(frozen=True)
class StochasticAction:
    """Immutable action commitment trace recorded before outcome resolution."""

    seed: int
    draw_index: int
    partition: str
    episode_id: str
    input_ids_sha256: str
    policy_id: str
    prob_a: float
    prob_b: float
    exploration_rate: float
    behavior_prob_a: float
    behavior_prob_b: float
    random_draw: float
    chosen_action: str
    chosen_propensity: float
    decision_digest: str
    evidence_class: str = EVIDENCE_CLASS
    phenomenology: str = PHENOMENOLOGY


class SeededActionSampler:
    """One private RNG per sampler; no mutation of global random/torch state.

    The seed only makes an experiment repeatable. It is NOT a cryptographic
    secret. Independent arms can use the same seed to couple their draws.
    """

    def __init__(self, *, seed: int, exploration_rate: float = 0.1) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        epsilon = _float_probability(exploration_rate, "exploration_rate")
        self.seed = seed
        self.exploration_rate = epsilon
        self._rng = random.Random(seed)
        self._index = 0
        self._lock = Lock()

    @property
    def draws_used(self) -> int:
        with self._lock:
            return self._index

    def sample(
        self,
        prob_a: float,
        prob_b: float,
        *,
        input_ids_sha256: str,
        partition: str,
        episode_id: str,
        policy_id: str,
    ) -> StochasticAction:
        """Draw one action, recording exact pre-outcome behavior propensity.

        A reward/outcome argument is intentionally absent. ``prob_a`` and
        ``prob_b`` are the model's normalized pre-exploration probabilities;
        the recorded behavior policy is the one used for sampling.
        """
        a = _float_probability(prob_a, "prob_a")
        b = _float_probability(prob_b, "prob_b")
        if abs((a + b) - 1.0) > 1e-6:
            raise ValueError("A/B model probabilities must sum to 1")
        if type(input_ids_sha256) is not str or _SHA256.fullmatch(input_ids_sha256) is None:
            raise ValueError("input_ids_sha256 must be a lowercase SHA-256 digest")
        if type(partition) is not str or partition not in ("TRAIN", "EVAL"):
            raise ValueError("partition must be TRAIN or EVAL")
        if type(episode_id) is not str or not 1 <= len(episode_id) <= 256 or episode_id != episode_id.strip():
            raise ValueError("episode_id must be a nonempty bounded identifier")
        if type(policy_id) is not str or _LABEL.fullmatch(policy_id) is None:
            raise ValueError("policy_id must be a bounded policy label")

        # A and B are normalized consistently, not independently rounded.
        normalized_a = a / (a + b)
        normalized_b = 1.0 - normalized_a
        behavior_a = (1.0 - self.exploration_rate) * normalized_a + self.exploration_rate / 2.0
        behavior_b = 1.0 - behavior_a
        if not 0.0 < behavior_a < 1.0 or not 0.0 < behavior_b < 1.0:
            raise ValueError("behavior policy must give both choices positive support")

        with self._lock:
            index = self._index
            draw = self._rng.random()
            self._index += 1
        chosen = "A" if draw < behavior_a else "B"
        propensity = behavior_a if chosen == "A" else behavior_b
        payload = (
            "PLEASURE_STOCHASTIC_CHOICE_V1", self.seed, index, partition,
            episode_id, input_ids_sha256, policy_id, normalized_a.hex(),
            normalized_b.hex(), self.exploration_rate.hex(), behavior_a.hex(),
            behavior_b.hex(), draw.hex(), chosen, propensity.hex(),
        )
        return StochasticAction(
            seed=self.seed, draw_index=index, partition=partition,
            episode_id=episode_id, input_ids_sha256=input_ids_sha256,
            policy_id=policy_id, prob_a=normalized_a, prob_b=normalized_b,
            exploration_rate=self.exploration_rate,
            behavior_prob_a=behavior_a, behavior_prob_b=behavior_b,
            random_draw=draw, chosen_action=chosen,
            chosen_propensity=propensity,
            decision_digest=_decision_sha(payload),
        )


def validate_stochastic_trace(rows: tuple[StochasticAction, ...]) -> None:
    """Replay and reject mutated, duplicated, reordered or forged local logs.

    This verifies deterministic consistency, not provenance or evaluator
    authenticity. A malicious producer who controls the seed can fabricate a
    whole *valid* trace; independent evidence remains necessary.
    """
    if type(rows) is not tuple or not rows:
        raise ValueError("stochastic trace must be a nonempty tuple")
    first = rows[0]
    if type(first) is not StochasticAction:
        raise ValueError("invalid stochastic trace record")
    sampler = SeededActionSampler(seed=first.seed, exploration_rate=first.exploration_rate)
    seen: set[tuple[str, str]] = set()
    for recorded in rows:
        if type(recorded) is not StochasticAction:
            raise ValueError("invalid stochastic trace record")
        episode_key = (recorded.partition, recorded.episode_id)
        if episode_key in seen:
            raise ValueError("same stochastic episode was sampled more than once")
        seen.add(episode_key)
        expected = sampler.sample(
            recorded.prob_a, recorded.prob_b,
            input_ids_sha256=recorded.input_ids_sha256,
            partition=recorded.partition, episode_id=recorded.episode_id,
            policy_id=recorded.policy_id,
        )
        if recorded != expected:
            raise ValueError("stochastic trace does not replay exactly")
