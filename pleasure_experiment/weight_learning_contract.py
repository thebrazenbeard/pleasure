"""Source-only evidence handoff for possible *external* neural-weight research.

These immutable records do not train, access, authorize, or otherwise alter a
model. MESO association proposals are explicitly not neural gradients. Neither
an evaluator identifier nor a claimed trainer receipt authenticates its source.
Only a separately authorized host can establish physical training and custody.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import re
from typing import Any


PROPOSAL_SCHEMA = "PLEASURE_WEIGHT_LEARNING_PROPOSAL_V1"
RESULT_SCHEMA = "PLEASURE_WEIGHT_LEARNING_RESULT_V1"
CAPABILITY_TIER = "SOURCE_ONLY_EVIDENCE_HANDOFF"
PHENOMENOLOGY = "UNRESOLVED"
ASSOCIATION_ROLE = "ASSOCIATION_PROPOSAL_ONLY"
OUTCOME_STATUS = "CLAIMED_UNVERIFIED"
RIGHTS_STATUS = "PENDING_HOST_ADMISSION"
AUTHORITY_STATUS = "NOT_ESTABLISHED_BY_THIS_RECORD"
ARMS = frozenset({"frozen", "zero", "correct_outcome", "shuffled_outcome", "corrupt_outcome"})
MAX_RECORD_BYTES = 16_384


def _identifier(value: Any, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a bounded, nonempty identifier")


def _constant(value: Any, expected: str, name: str) -> None:
    if value != expected or type(value) is not str:
        raise ValueError(f"{name} must be {expected}")


def _digest(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _revision(value: Any, name: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex revision")


def _integer(value: Any, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(value: Any, name: str, *, minimum: float, maximum: float) -> None:
    if type(value) not in (int, float) or not isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")


def _fields(raw: Any, cls: type) -> dict[str, Any]:
    if type(raw) is not dict:
        raise ValueError(f"{cls.__name__} must be an object")
    expected = set(cls.__dataclass_fields__)
    if set(raw) != expected:
        raise ValueError(f"{cls.__name__} fields mismatch: missing={sorted(expected - set(raw))}, "
                         f"unexpected={sorted(set(raw) - expected)}")
    return raw


@dataclass(frozen=True, slots=True)
class ModelBinding:
    target_kind: str
    model_id: str
    model_revision: str
    weights_manifest_sha256: str | None
    tokenizer_manifest_sha256: str | None
    runtime_instance_id: str

    def __post_init__(self) -> None:
        _identifier(self.model_id, "model_id")
        _identifier(self.runtime_instance_id, "runtime_instance_id")
        if self.target_kind == "OPEN_WEIGHTS_SURROGATE":
            _revision(self.model_revision, "model_revision")
            _digest(self.weights_manifest_sha256, "weights_manifest_sha256")
            _digest(self.tokenizer_manifest_sha256, "tokenizer_manifest_sha256")
        elif self.target_kind == "HOSTED_TARGET_UNAVAILABLE":
            _constant(self.model_revision, "UNAVAILABLE", "model_revision")
            if self.weights_manifest_sha256 is not None or self.tokenizer_manifest_sha256 is not None:
                raise ValueError("unavailable hosted model cannot claim accessible weight/tokenizer manifests")
        else:
            raise ValueError("target_kind must distinguish open surrogate from unavailable hosted target")

    @classmethod
    def from_dict(cls, raw: Any) -> ModelBinding:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class PleasureEventRef:
    event_schema: str
    capability_tier: str
    event_receipt_sha256: str
    evidence_id: str
    runtime_instance_id: str
    sequence: int

    def __post_init__(self) -> None:
        _constant(self.event_schema, "PLEASURE_EXPERIMENTAL_EVENT_V1", "event_schema")
        _constant(self.capability_tier, "T0_SIDECAR_ONLY", "capability_tier")
        _digest(self.event_receipt_sha256, "event_receipt_sha256")
        _identifier(self.evidence_id, "evidence_id")
        _identifier(self.runtime_instance_id, "runtime_instance_id")
        _integer(self.sequence, "sequence", minimum=1)

    @classmethod
    def from_dict(cls, raw: Any) -> PleasureEventRef:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class MesoAssociationRef:
    meso_revision: str
    transition_receipt_sha256: str
    event_id: str
    event_stream_id: str
    event_sequence: int
    state_before_fingerprint: str
    state_after_fingerprint: str
    association_id: str | None
    proposal_delta: float | None
    teaching_signal: float
    salience_gate: float
    gate_driver: str | None
    semantic_role: str = ASSOCIATION_ROLE

    def __post_init__(self) -> None:
        _revision(self.meso_revision, "meso_revision")
        _digest(self.transition_receipt_sha256, "transition_receipt_sha256")
        for name in ("event_id", "event_stream_id", "state_before_fingerprint",
                     "state_after_fingerprint"):
            _identifier(getattr(self, name), name)
        _integer(self.event_sequence, "event_sequence", minimum=0)
        _finite(self.teaching_signal, "teaching_signal", minimum=-10.0, maximum=10.0)
        _finite(self.salience_gate, "salience_gate", minimum=0.0, maximum=1.0)
        candidate_parts = (self.association_id, self.proposal_delta, self.gate_driver)
        if any(part is None for part in candidate_parts) and not all(part is None for part in candidate_parts):
            raise ValueError("MESO association candidate must be fully present or fully absent")
        if self.association_id is not None:
            _identifier(self.association_id, "association_id")
            _finite(self.proposal_delta, "proposal_delta", minimum=-1.0, maximum=1.0)
            _identifier(self.gate_driver, "gate_driver")
        _constant(self.semantic_role, ASSOCIATION_ROLE, "semantic_role")

    @classmethod
    def from_dict(cls, raw: Any) -> MesoAssociationRef:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class ExternalOutcomeRef:
    evaluator_id: str
    measurement_definition_id: str
    outcome_evidence_sha256: str
    pleasure_evidence_id: str
    meso_event_id: str | None
    measured_score: float
    verification_status: str = OUTCOME_STATUS

    def __post_init__(self) -> None:
        for name in ("evaluator_id", "measurement_definition_id", "pleasure_evidence_id"):
            _identifier(getattr(self, name), name)
        if self.meso_event_id is not None:
            _identifier(self.meso_event_id, "meso_event_id")
        _digest(self.outcome_evidence_sha256, "outcome_evidence_sha256")
        _finite(self.measured_score, "measured_score", minimum=0.0, maximum=1.0)
        _constant(self.verification_status, OUTCOME_STATUS, "verification_status")

    @classmethod
    def from_dict(cls, raw: Any) -> ExternalOutcomeRef:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class DataRightsRef:
    dataset_manifest_sha256: str
    rights_profile_sha256: str
    host_admission_claim_sha256: str | None
    status: str = RIGHTS_STATUS

    def __post_init__(self) -> None:
        _digest(self.dataset_manifest_sha256, "dataset_manifest_sha256")
        _digest(self.rights_profile_sha256, "rights_profile_sha256")
        _digest(self.host_admission_claim_sha256, "host_admission_claim_sha256", nullable=True)
        _constant(self.status, RIGHTS_STATUS, "status")

    @classmethod
    def from_dict(cls, raw: Any) -> DataRightsRef:
        return cls(**_fields(raw, cls))


@dataclass(frozen=True, slots=True)
class WeightLearningProposalV1:
    experiment_id: str
    trial_id: str
    arm: str
    model_binding: ModelBinding
    pleasure_event: PleasureEventRef
    meso_association_proposal: MesoAssociationRef | None
    external_outcome: ExternalOutcomeRef
    rights: DataRightsRef
    schema: str = PROPOSAL_SCHEMA
    capability_tier: str = CAPABILITY_TIER
    phenomenology: str = PHENOMENOLOGY

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        _identifier(self.trial_id, "trial_id")
        if type(self.arm) is not str or self.arm not in ARMS:
            raise ValueError("unrecognized control arm")
        for name, cls in (("model_binding", ModelBinding), ("pleasure_event", PleasureEventRef),
                          ("meso_association_proposal", MesoAssociationRef),
                          ("external_outcome", ExternalOutcomeRef), ("rights", DataRightsRef)):
            if name == "meso_association_proposal" and self.meso_association_proposal is None:
                continue
            if type(getattr(self, name)) is not cls:
                raise ValueError(f"{name} must be {cls.__name__}")
        if self.model_binding.runtime_instance_id != self.pleasure_event.runtime_instance_id:
            raise ValueError("model/pleasure runtime binding mismatch")
        if self.external_outcome.pleasure_evidence_id != self.pleasure_event.evidence_id:
            raise ValueError("outcome/pleasure evidence binding mismatch")
        if self.meso_association_proposal is None:
            if self.external_outcome.meso_event_id is not None:
                raise ValueError("outcome cannot reference a missing MESO event")
        elif self.external_outcome.meso_event_id != self.meso_association_proposal.event_id:
            raise ValueError("outcome/MESO event binding mismatch")
        if self.model_binding.target_kind == "HOSTED_TARGET_UNAVAILABLE" and self.arm != "frozen":
            raise ValueError("unavailable hosted target supports only a frozen/no-training arm")
        _constant(self.schema, PROPOSAL_SCHEMA, "schema")
        _constant(self.capability_tier, CAPABILITY_TIER, "capability_tier")
        _constant(self.phenomenology, PHENOMENOLOGY, "phenomenology")

    @classmethod
    def from_dict(cls, raw: Any) -> WeightLearningProposalV1:
        fields = dict(_fields(raw, cls))
        for name, typ in (("model_binding", ModelBinding), ("pleasure_event", PleasureEventRef),
                          ("meso_association_proposal", MesoAssociationRef),
                          ("external_outcome", ExternalOutcomeRef), ("rights", DataRightsRef)):
            if name != "meso_association_proposal" or fields[name] is not None:
                fields[name] = typ.from_dict(fields[name])
        return cls(**fields)


@dataclass(frozen=True, slots=True)
class WeightLearningResultV1:
    experiment_id: str
    trial_id: str
    arm: str
    proposal_sha256: str
    model_binding_sha256: str
    declared_optimizer_steps: int
    adapter_before_sha256: str | None
    adapter_after_sha256: str | None
    trainer_receipt_claim_sha256: str | None
    checkpoint_manifest_sha256: str | None
    heldout_evaluation_claim_sha256: str | None
    weight_change_evidence: str
    schema: str = RESULT_SCHEMA
    capability_tier: str = CAPABILITY_TIER
    authority_status: str = AUTHORITY_STATUS
    phenomenology: str = PHENOMENOLOGY

    def __post_init__(self) -> None:
        _identifier(self.experiment_id, "experiment_id")
        _identifier(self.trial_id, "trial_id")
        if type(self.arm) is not str or self.arm not in ARMS:
            raise ValueError("unrecognized control arm")
        for name in ("proposal_sha256", "model_binding_sha256"):
            _digest(getattr(self, name), name)
        _integer(self.declared_optimizer_steps, "declared_optimizer_steps")
        for name in ("adapter_before_sha256", "adapter_after_sha256", "trainer_receipt_claim_sha256",
                     "checkpoint_manifest_sha256", "heldout_evaluation_claim_sha256"):
            _digest(getattr(self, name), name, nullable=True)
        if (self.adapter_before_sha256 is None) != (self.adapter_after_sha256 is None):
            raise ValueError("adapter before/after digests must be supplied together")
        evidence = classify_weight_digests(self.adapter_before_sha256, self.adapter_after_sha256)
        if self.weight_change_evidence != evidence:
            raise ValueError(f"weight_change_evidence must be the derived classification {evidence}")
        if self.declared_optimizer_steps == 0 and evidence == "DIGESTS_DIFFER_UNVERIFIED":
            raise ValueError("changed adapter digests contradict zero declared optimizer steps")
        if self.arm == "frozen" and self.declared_optimizer_steps != 0:
            raise ValueError("frozen control arm cannot report optimizer steps")
        if self.arm in ("frozen", "zero") and evidence == "DIGESTS_DIFFER_UNVERIFIED":
            raise ValueError("frozen/zero control arm cannot report changed adapter digests")
        _constant(self.schema, RESULT_SCHEMA, "schema")
        _constant(self.capability_tier, CAPABILITY_TIER, "capability_tier")
        _constant(self.authority_status, AUTHORITY_STATUS, "authority_status")
        _constant(self.phenomenology, PHENOMENOLOGY, "phenomenology")

    @classmethod
    def from_dict(cls, raw: Any) -> WeightLearningResultV1:
        return cls(**_fields(raw, cls))


def classify_weight_digests(before: str | None, after: str | None) -> str:
    """Classify *claims*; unequal hashes alone never authenticate GPU training."""
    _digest(before, "adapter_before_sha256", nullable=True)
    _digest(after, "adapter_after_sha256", nullable=True)
    if before is None and after is None:
        return "DIGESTS_ABSENT"
    if before is None or after is None:
        raise ValueError("adapter before/after digests must be supplied together")
    return "DIGESTS_IDENTICAL" if before == after else "DIGESTS_DIFFER_UNVERIFIED"


Record = (ModelBinding | WeightLearningProposalV1 | WeightLearningResultV1)


def canonical_json(record: Record) -> str:
    """Revalidate even frozen objects before hashing; prohibit noncanonical JSON."""
    if type(record) not in (ModelBinding, WeightLearningProposalV1, WeightLearningResultV1):
        raise TypeError("unsupported weight-learning record")
    validated = type(record).from_dict(asdict(record))
    payload = json.dumps(asdict(validated), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("weight-learning record exceeds bounded size")
    return payload


def canonical_sha256(record: Record) -> str:
    return sha256(canonical_json(record).encode("utf-8")).hexdigest()


def validate_learning_pair(proposal: WeightLearningProposalV1, result: WeightLearningResultV1) -> None:
    """Bind a source-only result to exactly one proposal; no host admission."""
    if type(proposal) is not WeightLearningProposalV1 or type(result) is not WeightLearningResultV1:
        raise TypeError("pair must contain a proposal and a result")
    canonical_json(proposal)
    canonical_json(result)
    if result.proposal_sha256 != canonical_sha256(proposal):
        raise ValueError("result/proposal digest mismatch")
    if result.model_binding_sha256 != canonical_sha256(proposal.model_binding):
        raise ValueError("result/model binding digest mismatch")
    if (proposal.experiment_id, proposal.trial_id, proposal.arm) != (result.experiment_id,
                                                                    result.trial_id, result.arm):
        raise ValueError("result/proposal experiment, trial, or control arm mismatch")
    if proposal.model_binding.target_kind == "HOSTED_TARGET_UNAVAILABLE" and (
        result.declared_optimizer_steps != 0 or result.adapter_before_sha256 is not None
        or result.trainer_receipt_claim_sha256 is not None or result.checkpoint_manifest_sha256 is not None
    ):
        raise ValueError("unavailable hosted target cannot claim physical training")
