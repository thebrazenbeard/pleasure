"""Inert, source-only online-learning evidence receipts and hash-chain journal.

ACTION_COMMITTED -> OUTCOME_OBSERVED -> OPTIMIZER_REPORTED or ABORTED is a
source-local *claim chain*. It neither runs an optimizer nor verifies external
evaluators, host execution, rights, signatures, or checkpoint custody.
In-memory replay resistance is reconstructed on import, but is not durable
across independent hosts or evidence stores. Phenomenology stays unresolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from threading import Lock
from typing import Any


RECEIPT_SCHEMA = "PLEASURE_ONLINE_LEARNING_EVIDENCE_V1"
JOURNAL_SCHEMA = "PLEASURE_ONLINE_LEARNING_JOURNAL_V1"
CAPABILITY_TIER = "SOURCE_ONLY_UNAUTHENTICATED"
PHENOMENOLOGY = "UNRESOLVED"
EVIDENCE_CLASS = "UNAUTHENTICATED_SOURCE_ONLY"
MAX_RECORD_BYTES = 16_384
MAX_JOURNAL_BYTES = 4_000_000
MAX_EVENTS = 10_000
ARMS = frozenset({"frozen", "zero", "correct_outcome", "shuffled_outcome", "corrupt_outcome"})
KINDS = frozenset({"ACTION_COMMITTED", "OUTCOME_OBSERVED", "OPTIMIZER_REPORTED", "ABORTED"})
ABORT_REASONS = frozenset({
    "ACTION_FAILED", "OUTCOME_MISSING", "OUTCOME_REJECTED", "POLICY_DENIED",
    "TRAINER_FAILED", "REPLAY_DETECTED", "WITHDRAWN",
})


def _identifier(value: Any, name: str) -> None:
    if (type(value) is not str or not value or len(value) > 256
            or value != value.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError(f"{name} must be a bounded identifier without control or boundary whitespace")


def _digest(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _int(value: Any, name: str, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _unit(value: Any, name: str) -> None:
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and within [0, 1]")


def _const(value: Any, name: str, expected: str) -> None:
    if type(value) is not str or value != expected:
        raise ValueError(f"{name} must equal {expected}")


def _fields(raw: Any, cls: type) -> dict:
    if type(raw) is not dict:
        raise ValueError(f"{cls.__name__} must be a JSON object")
    required = set(cls.__dataclass_fields__)
    if set(raw) != required:
        raise ValueError(f"{cls.__name__} fields mismatch")
    return raw


@dataclass(frozen=True, slots=True)
class ActionCommitted:
    partition: str
    goal_definition_sha256: str
    adapter_before_sha256: str
    predicted_quality: float
    execution_receipt_claim_sha256: str | None
    train_manifest_sha256: str
    holdout_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self.partition) is not str or self.partition not in ("TRAIN", "EVAL"):
            raise ValueError("action partition must be TRAIN or EVAL")
        for name in ("goal_definition_sha256", "adapter_before_sha256",
                     "train_manifest_sha256", "holdout_manifest_sha256"):
            _digest(getattr(self, name), name)
        _digest(self.execution_receipt_claim_sha256, "execution_receipt_claim_sha256", nullable=True)
        if self.execution_receipt_claim_sha256 is not None:
            raise ValueError("ACTION_COMMITTED cannot claim future host execution")
        _unit(self.predicted_quality, "predicted_quality")
        if self.train_manifest_sha256 == self.holdout_manifest_sha256:
            raise ValueError("training and holdout manifests must not be identical")

    @classmethod
    def from_dict(cls, raw: Any) -> ActionCommitted:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class OutcomeObserved:
    action_receipt_sha256: str
    evidence_id: str
    evaluator_id: str
    rubric_sha256: str
    outcome_evidence_sha256: str
    custody_claim_sha256: str | None
    measured_quality: float
    evidence_class: str = EVIDENCE_CLASS

    def __post_init__(self) -> None:
        for name in ("action_receipt_sha256", "evidence_id", "rubric_sha256",
                     "outcome_evidence_sha256"):
            _digest(getattr(self, name), name)
        _digest(self.custody_claim_sha256, "custody_claim_sha256", nullable=True)
        _identifier(self.evaluator_id, "evaluator_id")
        _unit(self.measured_quality, "measured_quality")
        _const(self.evidence_class, "evidence_class", EVIDENCE_CLASS)

    @classmethod
    def from_dict(cls, raw: Any) -> OutcomeObserved:
        return cls(**_fields(raw, cls))


def classify_digests(before: str, after: str) -> str:
    """A derived hash comparison, *never* authenticated gradient evidence."""
    _digest(before, "adapter_before_sha256")
    _digest(after, "adapter_after_sha256")
    return "DIGESTS_IDENTICAL" if before == after else "DIGESTS_DIFFER_UNVERIFIED"


@dataclass(frozen=True, slots=True)
class OptimizerReported:
    action_receipt_sha256: str
    outcome_receipt_sha256: str
    arm: str
    train_example_manifest_sha256: str
    holdout_exclusion_manifest_sha256: str
    declared_optimizer_steps: int
    gradient_summary_claim_sha256: str | None
    trainer_receipt_claim_sha256: str | None
    adapter_before_sha256: str
    adapter_after_sha256: str
    checkpoint_claim_sha256: str | None
    weight_change_evidence: str

    def __post_init__(self) -> None:
        if type(self.arm) is not str or self.arm not in ARMS:
            raise ValueError("unrecognized control arm")
        for name in ("action_receipt_sha256", "outcome_receipt_sha256",
                     "train_example_manifest_sha256", "holdout_exclusion_manifest_sha256",
                     "adapter_before_sha256", "adapter_after_sha256"):
            _digest(getattr(self, name), name)
        for name in ("gradient_summary_claim_sha256", "trainer_receipt_claim_sha256",
                     "checkpoint_claim_sha256"):
            _digest(getattr(self, name), name, nullable=True)
        _int(self.declared_optimizer_steps, "declared_optimizer_steps", minimum=0)
        if self.train_example_manifest_sha256 == self.holdout_exclusion_manifest_sha256:
            raise ValueError("train example and holdout exclusion manifest digests must differ")
        classification = classify_digests(self.adapter_before_sha256, self.adapter_after_sha256)
        if self.weight_change_evidence != classification:
            raise ValueError(f"weight_change_evidence must be derived as {classification}")
        if self.arm == "frozen" and self.declared_optimizer_steps != 0:
            raise ValueError("frozen control cannot claim optimizer steps")
        if classification == "DIGESTS_DIFFER_UNVERIFIED" and (
            self.arm in ("frozen", "zero") or self.declared_optimizer_steps == 0
        ):
            raise ValueError("control or no-step report cannot claim changed adapter digests")

    @classmethod
    def from_dict(cls, raw: Any) -> OptimizerReported:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class Aborted:
    action_receipt_sha256: str
    outcome_receipt_sha256: str | None
    reason: str
    restore_status: str = "NO_OPTIMIZER_STARTED"
    restore_receipt_claim_sha256: str | None = None

    def __post_init__(self) -> None:
        _digest(self.action_receipt_sha256, "action_receipt_sha256")
        _digest(self.outcome_receipt_sha256, "outcome_receipt_sha256", nullable=True)
        if type(self.reason) is not str or self.reason not in ABORT_REASONS:
            raise ValueError("unknown abort reason")
        if self.restore_status not in ("NO_OPTIMIZER_STARTED", "HOST_RESTORE_UNVERIFIED") or type(self.restore_status) is not str:
            raise ValueError("unknown restore_status")
        _digest(self.restore_receipt_claim_sha256, "restore_receipt_claim_sha256", nullable=True)
        if self.reason == "TRAINER_FAILED" and self.restore_status != "HOST_RESTORE_UNVERIFIED":
            raise ValueError("failed trainer requires unverified host restore and session halt")
        if self.restore_status == "NO_OPTIMIZER_STARTED" and self.restore_receipt_claim_sha256 is not None:
            raise ValueError("no-optimizer abort cannot claim a restore operation")

    @classmethod
    def from_dict(cls, raw: Any) -> Aborted:
        return cls(**_fields(raw, cls))


Payload = ActionCommitted | OutcomeObserved | OptimizerReported | Aborted
_PAYLOADS = {
    "ACTION_COMMITTED": ActionCommitted,
    "OUTCOME_OBSERVED": OutcomeObserved,
    "OPTIMIZER_REPORTED": OptimizerReported,
    "ABORTED": Aborted,
}


@dataclass(frozen=True, slots=True)
class OnlineEvidenceReceipt:
    kind: str
    event_id: str
    session_id: str
    session_epoch_id: str
    sequence: int
    previous_event_sha256: str | None
    recorded_at_ms: int
    model_id: str
    model_revision: str
    model_binding_sha256: str
    runtime_instance_id: str
    goal_instance_id: str
    task_id: str
    episode_id: str
    decision_digest: str
    action_choice: str
    action_commit_id: str
    offer_nonce_sha256: str
    payload: Payload
    schema: str = RECEIPT_SCHEMA
    capability_tier: str = CAPABILITY_TIER
    phenomenology: str = PHENOMENOLOGY

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError("unknown online evidence kind")
        for name in ("event_id", "model_binding_sha256", "decision_digest",
                     "action_commit_id", "offer_nonce_sha256"):
            _digest(getattr(self, name), name)
        _digest(self.previous_event_sha256, "previous_event_sha256", nullable=True)
        for name in ("session_id", "session_epoch_id", "model_id", "model_revision",
                     "runtime_instance_id", "goal_instance_id", "task_id", "episode_id"):
            _identifier(getattr(self, name), name)
        _int(self.sequence, "sequence", minimum=1)
        _int(self.recorded_at_ms, "recorded_at_ms", minimum=0)
        if type(self.action_choice) is not str or self.action_choice not in ("A", "B"):
            raise ValueError("action_choice must be A or B")
        if type(self.payload) is not _PAYLOADS[self.kind]:
            raise ValueError("payload type does not match event kind")
        _const(self.schema, "schema", RECEIPT_SCHEMA)
        _const(self.capability_tier, "capability_tier", CAPABILITY_TIER)
        _const(self.phenomenology, "phenomenology", PHENOMENOLOGY)

    @classmethod
    def from_dict(cls, raw: Any) -> OnlineEvidenceReceipt:
        attrs = dict(_fields(raw, cls))
        kind = attrs["kind"]
        if type(kind) is not str or kind not in _PAYLOADS:
            raise ValueError("unknown online evidence kind")
        attrs["payload"] = _PAYLOADS[kind].from_dict(attrs["payload"])
        return cls(**attrs)


def canonical_json(value: OnlineEvidenceReceipt) -> str:
    if type(value) is not OnlineEvidenceReceipt:
        raise TypeError("expected OnlineEvidenceReceipt")
    fixed = OnlineEvidenceReceipt.from_dict(asdict(value))
    encoded = json.dumps(asdict(fixed), separators=(",", ":"), sort_keys=True,
                         ensure_ascii=False, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("receipt exceeds source-only record size limit")
    return encoded


def receipt_sha256(value: OnlineEvidenceReceipt) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"invalid nonfinite JSON constant {value}")


def _identity(receipt: OnlineEvidenceReceipt) -> tuple[str, ...]:
    """Fields that must remain identical throughout a single causal cycle."""
    return (receipt.model_id, receipt.model_revision, receipt.model_binding_sha256,
            receipt.runtime_instance_id, receipt.goal_instance_id, receipt.task_id,
            receipt.episode_id, receipt.decision_digest, receipt.action_choice,
            receipt.action_commit_id, receipt.offer_nonce_sha256)


class OnlineEvidenceJournal:
    """One-session serial hash chain. No persistence, verification, or training."""

    __slots__ = ("session_id", "session_epoch_id", "model_binding_sha256",
                 "_receipts", "_hashes", "_used_event_ids", "_used_tasks",
                 "_used_episodes", "_used_nonces", "_used_outcomes",
                 "_action", "_outcome", "_last_adapter_sha256", "_blocked", "_lock")

    def __setattr__(self, name: str, value: Any) -> None:
        # Ordinary reassignment cannot re-label identity or bypass state.
        # Privileged Python can still mutate internals: this is not a sandbox.
        if hasattr(self, "_lock"):
            raise AttributeError("online evidence journal state is not publicly assignable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if hasattr(self, "_lock"):
            raise AttributeError("online evidence journal state is not publicly deletable")
        object.__delattr__(self, name)

    def __init__(self, *, session_id: str, session_epoch_id: str, model_binding_sha256: str) -> None:
        _identifier(session_id, "session_id")
        _identifier(session_epoch_id, "session_epoch_id")
        _digest(model_binding_sha256, "model_binding_sha256")
        self.session_id = session_id
        self.session_epoch_id = session_epoch_id
        self.model_binding_sha256 = model_binding_sha256
        self._receipts: list[OnlineEvidenceReceipt] = []
        self._hashes: list[str] = []
        self._used_event_ids: set[str] = set()
        self._used_tasks: set[str] = set()
        self._used_episodes: set[str] = set()
        self._used_nonces: set[str] = set()
        self._used_outcomes: set[str] = set()
        self._action: OnlineEvidenceReceipt | None = None
        self._outcome: OnlineEvidenceReceipt | None = None
        self._last_adapter_sha256: str | None = None
        self._blocked = False
        self._lock = Lock()

    @property
    def receipts(self) -> tuple[OnlineEvidenceReceipt, ...]:
        with self._lock:
            # Do not hand out aliases to retained journal snapshots. Frozen
            # dataclasses alone are not a hostile-process protection boundary.
            return tuple(OnlineEvidenceReceipt.from_dict(asdict(rec)) for rec in self._receipts)

    @property
    def head_sha256(self) -> str | None:
        with self._lock:
            return self._hashes[-1] if self._hashes else None

    @property
    def next_sequence(self) -> int:
        with self._lock:
            return len(self._receipts) + 1

    def append(self, receipt: OnlineEvidenceReceipt) -> str:
        """Atomically validate and consume one claimed event within this object."""
        if type(receipt) is not OnlineEvidenceReceipt:
            raise TypeError("receipt must be OnlineEvidenceReceipt")
        fixed = OnlineEvidenceReceipt.from_dict(asdict(receipt))
        fixed_hash = receipt_sha256(fixed)
        with self._lock:
            if len(self._receipts) >= MAX_EVENTS:
                raise RuntimeError("online evidence journal event limit reached")
            if self._blocked:
                raise RuntimeError("journal blocked by unverified restore; new session required")
            if (fixed.session_id, fixed.session_epoch_id, fixed.model_binding_sha256) != (
                self.session_id, self.session_epoch_id, self.model_binding_sha256
            ):
                raise ValueError("cross-session or cross-model evidence rejected")
            if fixed.sequence != len(self._receipts) + 1:
                raise ValueError("nonmonotonic journal sequence or replay")
            if fixed.previous_event_sha256 != (self._hashes[-1] if self._hashes else None):
                raise ValueError("broken journal hash chain")
            if fixed.event_id in self._used_event_ids:
                raise ValueError("duplicate event_id or replay")
            if self._receipts and fixed.recorded_at_ms <= self._receipts[-1].recorded_at_ms:
                raise ValueError("out-of-order, simultaneous, or future-labeled outcome")

            if fixed.kind == "ACTION_COMMITTED":
                if self._action is not None:
                    raise ValueError("previous action cycle not finalized")
                if (fixed.task_id in self._used_tasks or fixed.episode_id in self._used_episodes
                        or fixed.offer_nonce_sha256 in self._used_nonces
                        or fixed.action_commit_id in self._used_event_ids):
                    raise ValueError("duplicate action, episode, nonce, or commit ID")
                if fixed.event_id != fixed.action_commit_id:
                    raise ValueError("ACTION_COMMITTED event_id must bind exact action commit ID")
                action = fixed.payload
                if self._last_adapter_sha256 is not None and action.adapter_before_sha256 != self._last_adapter_sha256:
                    raise ValueError("model adapter digest continuity violation")
                object.__setattr__(self, "_action", fixed)
                object.__setattr__(self, "_outcome", None)
                self._used_tasks.add(fixed.task_id)
                self._used_episodes.add(fixed.episode_id)
                self._used_nonces.add(fixed.offer_nonce_sha256)
                object.__setattr__(self, "_last_adapter_sha256", action.adapter_before_sha256)
            else:
                if self._action is None or _identity(fixed) != _identity(self._action):
                    raise ValueError("missing action or cross-action/model/goal/episode/decision substitution")
                action_hash = receipt_sha256(self._action)
                if fixed.kind == "OUTCOME_OBSERVED":
                    if self._outcome is not None:
                        raise ValueError("duplicate outcome or replay")
                    outcome = fixed.payload
                    if outcome.action_receipt_sha256 != action_hash:
                        raise ValueError("outcome/action receipt hash mismatch")
                    if fixed.event_id != outcome.evidence_id or outcome.evidence_id in self._used_outcomes:
                        raise ValueError("duplicate or incorrectly bound outcome evidence ID")
                    object.__setattr__(self, "_outcome", fixed)
                    self._used_outcomes.add(outcome.evidence_id)
                elif fixed.kind == "OPTIMIZER_REPORTED":
                    if self._outcome is None:
                        raise ValueError("optimizer report requires prior outcome")
                    report = fixed.payload
                    if (report.action_receipt_sha256 != action_hash
                            or report.outcome_receipt_sha256 != receipt_sha256(self._outcome)):
                        raise ValueError("optimizer/action/outcome evidence digest mismatch")
                    if report.adapter_before_sha256 != self._action.payload.adapter_before_sha256:
                        raise ValueError("optimizer adapter-before digest mismatch")
                    if report.holdout_exclusion_manifest_sha256 != self._action.payload.holdout_manifest_sha256:
                        raise ValueError("optimizer locked holdout manifest mismatch")
                    if report.train_example_manifest_sha256 != self._action.payload.train_manifest_sha256:
                        raise ValueError("optimizer locked train manifest mismatch")
                    if self._action.payload.partition == "EVAL" and (
                        report.declared_optimizer_steps != 0 or report.weight_change_evidence != "DIGESTS_IDENTICAL"
                    ):
                        raise ValueError("protected EVAL action cannot be trained on")
                    object.__setattr__(self, "_last_adapter_sha256", report.adapter_after_sha256)
                    object.__setattr__(self, "_action", None)
                    object.__setattr__(self, "_outcome", None)
                elif fixed.kind == "ABORTED":
                    aborted = fixed.payload
                    if aborted.action_receipt_sha256 != action_hash:
                        raise ValueError("abort/action receipt digest mismatch")
                    expected_outcome = receipt_sha256(self._outcome) if self._outcome else None
                    if aborted.outcome_receipt_sha256 != expected_outcome:
                        raise ValueError("abort/outcome receipt digest mismatch")
                    if aborted.restore_status == "HOST_RESTORE_UNVERIFIED":
                        object.__setattr__(self, "_blocked", True)
                    object.__setattr__(self, "_action", None)
                    object.__setattr__(self, "_outcome", None)
                else:
                    raise AssertionError("unreachable receipt kind")

            self._used_event_ids.add(fixed.event_id)
            self._receipts.append(fixed)
            self._hashes.append(fixed_hash)
            return fixed_hash

    def export_json(self) -> str:
        with self._lock:
            if any(receipt_sha256(rec) != digest for rec, digest in zip(self._receipts, self._hashes)):
                raise RuntimeError("journal stored receipt digest changed unexpectedly")
            encoded = json.dumps({
                "schema": JOURNAL_SCHEMA,
                "session_id": self.session_id,
                "session_epoch_id": self.session_epoch_id,
                "model_binding_sha256": self.model_binding_sha256,
                "head_sha256": self._hashes[-1] if self._hashes else None,
                "events": [
                    {"receipt": asdict(rec), "sha256": digest}
                    for rec, digest in zip(self._receipts, self._hashes)
                ],
            }, separators=(",", ":"), sort_keys=True, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_JOURNAL_BYTES:
            raise ValueError("journal export exceeds size limit")
        return encoded

    @classmethod
    def import_json(cls, encoded: str) -> OnlineEvidenceJournal:
        if type(encoded) is not str or len(encoded.encode("utf-8")) > MAX_JOURNAL_BYTES:
            raise ValueError("journal import must be bounded JSON text")
        try:
            raw = json.loads(encoded, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid journal JSON: {exc}") from exc
        if type(raw) is not dict or set(raw) != {
            "schema", "session_id", "session_epoch_id", "model_binding_sha256", "head_sha256", "events"
        }:
            raise ValueError("journal envelope fields mismatch")
        _const(raw["schema"], "schema", JOURNAL_SCHEMA)
        if type(raw["events"]) is not list or len(raw["events"]) > MAX_EVENTS:
            raise ValueError("journal events must be a bounded list")
        journal = cls(session_id=raw["session_id"], session_epoch_id=raw["session_epoch_id"],
                      model_binding_sha256=raw["model_binding_sha256"])
        for item in raw["events"]:
            if type(item) is not dict or set(item) != {"receipt", "sha256"}:
                raise ValueError("journal event envelope fields mismatch")
            _digest(item["sha256"], "sha256")
            event = OnlineEvidenceReceipt.from_dict(item["receipt"])
            if receipt_sha256(event) != item["sha256"]:
                raise ValueError("journal event content digest mismatch")
            journal.append(event)
        _digest(raw["head_sha256"], "head_sha256", nullable=True)
        if journal.head_sha256 != raw["head_sha256"]:
            raise ValueError("journal head digest mismatch")
        return journal
