"""Source-local evaluation with *structurally* withheld payoff generators.

TRAIN exposes one smooth affine rule. EVAL offers instead come from three
different, never-trained rules: interactions, a discontinuous threshold and
a nonlinear preference basin. The learner API exposes numeric features and
*only the committed choice's* measured outcome; it does not reveal the rule,
counterfactual payoff or optimal action. After every EVAL commitment has been
resolved and the evaluation is sealed, a separate posthoc method exposes
regret and counterfactuals for scoring.

This is a source-local separation of capabilities, not a secret evaluator:
anyone with Python source/process access can inspect the rules or internals.
For authentic blinded deployment, the evaluator must run in a separately
controlled process/service with unseen generator definitions and sealed tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock

from pleasure_experiment.outcome_arena import (
    CommitTicket, OutcomeReceipt, TaskOffer, _digest_hex, _family,
    _hex, _identifier, _unit, MAX_PAIR_INDEX, PARTITIONS,
)


EVIDENCE_CLASS = "SOURCE_LOCAL_WITHHELD_GENERATOR_UNAUTHENTICATED"
TRAIN_FAMILY = "heldout_train_affine"
EVAL_FAMILIES = (
    "heldout_eval_interaction",
    "heldout_eval_threshold",
    "heldout_eval_basin",
)
ALL_FAMILIES = (TRAIN_FAMILY,) + EVAL_FAMILIES
GENERATOR_VERSION = "WITHHELD_SOURCE_LOCAL_PAYOFF_GENERATORS_V1"
_RULE_SPECS = {
    TRAIN_FAMILY: ("affine", 0.50, 0.32, 0.68, -0.32),
    EVAL_FAMILIES[0]: ("interaction", 0.50, 0.38),
    EVAL_FAMILIES[1]: ("threshold", 0.22, 0.56, 0.55, 0.12),
    EVAL_FAMILIES[2]: ("basin", 0.85, 0.75, 0.72, 0.24),
}


def _rule(family_id: str, features: tuple[float, float]) -> float:
    """Evaluator-private scoring; intentionally not part of public offers."""
    first, second = features
    if type(family_id) is not str or family_id not in _RULE_SPECS:
        raise ValueError("unknown withheld payoff generator")
    kind, *params = _RULE_SPECS[family_id]
    if kind == "affine":
        baseline, scale, weight_first, weight_second = params
        quality = baseline + scale * (weight_first * (2 * first - 1) + weight_second * (2 * second - 1))
    elif kind == "interaction":
        # Sign of the interaction changes as either coordinate crosses 0.5.
        baseline, scale = params
        quality = baseline + scale * ((2 * first - 1) * (2 * second - 1))
    elif kind == "threshold":
        # Discontinuous decision boundary, impossible for an affine rule.
        baseline, jump, boundary, slope = params
        quality = baseline + jump * (first > boundary) + slope * (second - 0.5)
    elif kind == "basin":
        # Nonmonotonic optimum in the interior of feature space.
        peak, curvature, center_first, center_second = params
        quality = peak - curvature * ((first - center_first) ** 2 + (second - center_second) ** 2)
    else:
        raise ValueError("unknown withheld payoff generator")
    return _unit(max(0.0, min(1.0, quality)), "quality")


def _generator_digest(family_id: str) -> str:
    # Pin the SOURCE specification identifier, not an external attestation.
    if family_id not in ALL_FAMILIES or type(family_id) is not str:
        raise ValueError("unknown withheld payoff generator")
    return _hex(GENERATOR_VERSION, family_id, _RULE_SPECS[family_id])


@dataclass(frozen=True)
class GeneratorManifest:
    train_family: str
    heldout_families: tuple[str, ...]
    train_generator_sha256: str
    heldout_generator_sha256: tuple[str, ...]
    suite_sha256: str
    evidence_class: str = EVIDENCE_CLASS
    externally_authenticated: bool = False


@dataclass(frozen=True)
class PosthocOutcome:
    """EVAL-only answer key, accessible only after evaluation has been sealed."""

    task_id: str
    episode_id: str
    family_id: str
    chosen_action: str
    chosen_quality: float
    unchosen_quality: float
    quality_a: float
    quality_b: float
    oracle_action: str
    oracle_quality: float
    regret: float
    generator_sha256: str
    evidence_class: str = EVIDENCE_CLASS


class WithheldPayoffArena:
    """Deterministic, matched-arm-compatible, one-use chosen feedback arena.

    The arena does not issue counterfactuals during either TRAIN or EVAL.
    Posthoc evaluation is available only once *all* issued EVAL decisions have
    resolved and `finalize_evaluation()` irrevocably freezes the arena.
    Use a fresh arena per treatment arm with the SAME seed and schedules.
    """

    __slots__ = (
        "_seed", "_evaluator_id", "_issued", "_committed", "_resolved",
        "_sealed", "_lock",
    )

    def __init__(self, *, seed: int, evaluator_id: str) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        self._seed = seed
        self._evaluator_id = _identifier(evaluator_id, "evaluator_id")
        self._issued: dict[str, TaskOffer] = {}
        self._committed: dict[str, CommitTicket] = {}
        self._resolved: set[str] = set()
        self._sealed = False
        self._lock = Lock()

    @property
    def manifest(self) -> GeneratorManifest:
        train_digest = _generator_digest(TRAIN_FAMILY)
        eval_digests = tuple(_generator_digest(family) for family in EVAL_FAMILIES)
        return GeneratorManifest(
            train_family=TRAIN_FAMILY, heldout_families=EVAL_FAMILIES,
            train_generator_sha256=train_digest,
            heldout_generator_sha256=eval_digests,
            suite_sha256=_hex(GENERATOR_VERSION, train_digest, eval_digests),
        )

    @property
    def rubric_digest(self) -> str:
        return self.manifest.suite_sha256

    def _pair(
        self, partition: str, family_id: str, pair_index: int,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        for attempt in range(128):
            digest = bytes.fromhex(_hex(
                GENERATOR_VERSION, self._seed, partition, family_id, pair_index, attempt,
            ))
            fractions = tuple(
                (int.from_bytes(digest[i:i + 4], "big") + 0.5) / 2**32
                for i in (0, 4, 8, 12)
            )
            a = (0.05 + 0.90 * fractions[0], 0.05 + 0.90 * fractions[1])
            b = (0.05 + 0.90 * fractions[2], 0.05 + 0.90 * fractions[3])
            if abs(_rule(family_id, a) - _rule(family_id, b)) > 0.02:
                return a, b
        raise RuntimeError("unable to issue distinguishable alternatives")

    def issue(
        self, *, partition: str, family_id: str, pair_index: int, side: int,
    ) -> TaskOffer:
        if type(partition) is not str or partition not in PARTITIONS:
            raise ValueError("unknown task partition")
        _family(family_id)
        if (
            (partition == "TRAIN" and family_id != TRAIN_FAMILY)
            or (partition == "EVAL" and family_id not in EVAL_FAMILIES)
        ):
            raise ValueError("generator family may not cross TRAIN/EVAL boundary")
        if type(pair_index) is not int or not 0 <= pair_index <= MAX_PAIR_INDEX:
            raise ValueError("pair_index out of range")
        if type(side) is not int or side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        a, b = self._pair(partition, family_id, pair_index)
        offer = TaskOffer(
            task_id=_hex("task", GENERATOR_VERSION, self._seed, partition, family_id, pair_index, side),
            episode_id=_hex("episode", GENERATOR_VERSION, self._seed, partition, family_id, pair_index, side),
            partition=partition, family_id=family_id, pair_index=pair_index, side=side,
            nonce=_hex("nonce", GENERATOR_VERSION, self._seed, partition, family_id, pair_index, side),
            features_a=b if side else a, features_b=a if side else b,
        )
        with self._lock:
            if self._sealed:
                raise ValueError("evaluation already sealed")
            previous = self._issued.setdefault(offer.task_id, replace(offer))
            if previous != offer:
                raise RuntimeError("source-local task generation diverged")
            return replace(previous)

    def commit(
        self, offer: TaskOffer, action: str, decision_digest: str,
        *, model_id: str, runtime_instance_id: str,
    ) -> CommitTicket:
        if not isinstance(offer, TaskOffer):
            raise TypeError("offer must be a TaskOffer")
        fixed = replace(offer)
        if type(action) is not str or action not in ("A", "B"):
            raise ValueError("action must be A or B")
        _digest_hex(decision_digest, "decision_digest")
        _identifier(model_id, "model_id")
        _identifier(runtime_instance_id, "runtime_instance_id")
        with self._lock:
            if self._sealed:
                raise ValueError("evaluation already sealed")
            if self._issued.get(fixed.task_id) != fixed:
                raise ValueError("offer is unknown or altered")
            if fixed.task_id in self._committed:
                raise ValueError("task already committed")
            ticket = CommitTicket(
                task_id=fixed.task_id, episode_id=fixed.episode_id, action=action,
                model_id=model_id, runtime_instance_id=runtime_instance_id,
                decision_digest=decision_digest,
                commit_id=_hex("commit", fixed.task_id, action, model_id,
                               runtime_instance_id, decision_digest, fixed.nonce),
            )
            self._committed[fixed.task_id] = replace(ticket)
            return replace(ticket)

    def resolve(self, ticket: CommitTicket) -> OutcomeReceipt:
        if not isinstance(ticket, CommitTicket):
            raise TypeError("ticket must be a CommitTicket")
        fixed = replace(ticket)
        with self._lock:
            if self._sealed:
                raise ValueError("evaluation already sealed")
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
            quality = _rule(offer.family_id, offer.features_a if stored.action == "A" else offer.features_b)
            rule_digest = _generator_digest(offer.family_id)
            receipt = OutcomeReceipt(
                task_id=offer.task_id, episode_id=offer.episode_id,
                partition=offer.partition, family_id=offer.family_id, action=stored.action,
                model_id=stored.model_id, runtime_instance_id=stored.runtime_instance_id,
                measured_quality=quality, decision_digest=stored.decision_digest,
                evidence_id=_hex("outcome", stored.commit_id, quality, rule_digest),
                evaluator_id=self._evaluator_id, rubric_digest=rule_digest,
                evidence_class=EVIDENCE_CLASS,
            )
            self._resolved.add(fixed.task_id)
            return receipt

    def finalize_evaluation(self) -> None:
        """Irrevocably forbid all further issue/commit/resolve operations."""
        with self._lock:
            if self._sealed:
                raise ValueError("evaluation already sealed")
            if not any(offer.partition == "EVAL" for offer in self._issued.values()):
                raise ValueError("evaluation requires at least one EVAL offer")
            if any(task_id not in self._resolved for task_id in self._issued):
                raise ValueError("unresolved tasks remain")
            self._sealed = True

    def posthoc_evaluate(self, offer: TaskOffer) -> PosthocOutcome:
        """Reveal oracle counterfactual ONLY after the full study is sealed."""
        if not isinstance(offer, TaskOffer):
            raise TypeError("offer must be a TaskOffer")
        fixed = replace(offer)
        with self._lock:
            if not self._sealed:
                raise PermissionError("posthoc evaluation requires sealed EVAL study")
            if fixed.partition != "EVAL":
                raise PermissionError("TRAIN counterfactual outcomes are never disclosed")
            if self._issued.get(fixed.task_id) != fixed or fixed.task_id not in self._resolved:
                raise ValueError("offer is unknown, altered or unresolved")
            selected = self._committed[fixed.task_id].action
            a, b = _rule(fixed.family_id, fixed.features_a), _rule(fixed.family_id, fixed.features_b)
            chosen, unchosen = (a, b) if selected == "A" else (b, a)
            oracle_action = "A" if a > b else "B"
            return PosthocOutcome(
                task_id=fixed.task_id, episode_id=fixed.episode_id,
                family_id=fixed.family_id, chosen_action=selected,
                chosen_quality=chosen, unchosen_quality=unchosen,
                quality_a=a, quality_b=b, oracle_action=oracle_action,
                oracle_quality=max(a, b), regret=max(a, b) - chosen,
                generator_sha256=_generator_digest(fixed.family_id),
            )
