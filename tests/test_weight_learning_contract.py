"""Adversarial checks for an inert, versioned neural-learning evidence handoff."""

from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unittest

from pleasure_experiment.weight_learning_contract import (
    ASSOCIATION_ROLE,
    AUTHORITY_STATUS,
    CAPABILITY_TIER,
    DataRightsRef,
    ExternalOutcomeRef,
    MesoAssociationRef,
    ModelBinding,
    PleasureEventRef,
    WeightLearningProposalV1,
    WeightLearningResultV1,
    canonical_json,
    canonical_sha256,
    classify_weight_digests,
    validate_learning_pair,
)


HERE = Path(__file__).resolve().parents[1]
HEX_REVISION = "a" * 40


def digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def proposal(*, kind: str = "OPEN_WEIGHTS_SURROGATE", arm: str = "correct_outcome",
             with_meso: bool = True, with_candidate: bool = True) -> WeightLearningProposalV1:
    hosted = kind == "HOSTED_TARGET_UNAVAILABLE"
    meso = MesoAssociationRef(
        meso_revision="b" * 40,
        transition_receipt_sha256=digest("meso-receipt"),
        event_id="meso-event-001",
        event_stream_id="stream-1",
        event_sequence=8,
        state_before_fingerprint="before-state-1",
        state_after_fingerprint="after-state-1",
        association_id="association-001" if with_candidate else None,
        proposal_delta=0.04 if with_candidate else None,
        teaching_signal=0.8 if with_candidate else 0.0,
        salience_gate=0.5 if with_candidate else 0.0,
        gate_driver="semantic_salience" if with_candidate else None,
    ) if with_meso else None
    return WeightLearningProposalV1(
        experiment_id="positive-learning-study-1",
        trial_id="trial-001",
        arm="frozen" if hosted else arm,
        model_binding=ModelBinding(
            target_kind=kind,
            model_id="qwen3.5-4b-test-surrogate" if not hosted else "hosted-chat-target",
            model_revision="UNAVAILABLE" if hosted else HEX_REVISION,
            weights_manifest_sha256=None if hosted else digest("weights-manifest"),
            tokenizer_manifest_sha256=None if hosted else digest("tokenizer-manifest"),
            runtime_instance_id="study-instance-1",
        ),
        pleasure_event=PleasureEventRef(
            event_schema="PLEASURE_EXPERIMENTAL_EVENT_V1",
            capability_tier="T0_SIDECAR_ONLY",
            event_receipt_sha256=digest("pleasure-receipt"),
            evidence_id="outcome-001",
            runtime_instance_id="study-instance-1",
            sequence=1,
        ),
        meso_association_proposal=meso,
        external_outcome=ExternalOutcomeRef(
            evaluator_id="independent-evaluator-claimed",
            measurement_definition_id="heldout-goal-progress-v1",
            outcome_evidence_sha256=digest("external-evaluator-outcome"),
            pleasure_evidence_id="outcome-001",
            meso_event_id=meso.event_id if meso else None,
            measured_score=0.7,
        ),
        rights=DataRightsRef(
            dataset_manifest_sha256=digest("train-dev-protected-exclusions"),
            rights_profile_sha256=digest("rights-claim"),
            host_admission_claim_sha256=None,
        ),
    )


def result(item: WeightLearningProposalV1, *, steps: int = 0,
           before: str | None = None, after: str | None = None) -> WeightLearningResultV1:
    return WeightLearningResultV1(
        experiment_id=item.experiment_id,
        trial_id=item.trial_id,
        arm=item.arm,
        proposal_sha256=canonical_sha256(item),
        model_binding_sha256=canonical_sha256(item.model_binding),
        declared_optimizer_steps=steps,
        adapter_before_sha256=before,
        adapter_after_sha256=after,
        trainer_receipt_claim_sha256=None,
        checkpoint_manifest_sha256=None,
        heldout_evaluation_claim_sha256=None,
        weight_change_evidence=classify_weight_digests(before, after),
    )


class WeightLearningContractTests(unittest.TestCase):
    def test_schema_declares_closed_versioned_source_only_variants(self):
        payload = json.loads((HERE / "schemas" / "pleasure-weight-learning-v1.schema.json").read_text())
        self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(len(payload["oneOf"]), 2)
        for kind in ("proposal", "result", "modelBinding", "pleasureEvent",
                     "mesoAssociation", "externalOutcome", "dataRights"):
            self.assertIs(payload["$defs"][kind]["additionalProperties"], False)
        self.assertEqual(payload["$defs"]["proposal"]["properties"]["capability_tier"]["const"],
                         CAPABILITY_TIER)
        self.assertEqual(payload["$defs"]["result"]["properties"]["authority_status"]["const"],
                         AUTHORITY_STATUS)

    def test_schema_identifier_pattern_matches_python_identifier_validation(self):
        # jsonschema is intentionally not a dependency. Exercise the schema's
        # actual regex/length rules against the Python constructor's checks.
        schema = json.loads((HERE / "schemas" / "pleasure-weight-learning-v1.schema.json").read_text())
        rule = schema["$defs"]["identifier"]
        regex = re.compile(rule["pattern"])
        self.assertEqual(rule["minLength"], 1)
        self.assertEqual(rule["maxLength"], 256)

        cases = (
            "a", "a b", "a\tb", "emoji-🙂", "a" * 256, "a" * 257,
            "", " name", "name ", "\tname", "name\n", "name\x7f",
            "na\x00me", "na\x1bme", "na\x7fme", "na\x1cme",
            "\u0085name", "name\u00a0", "\u1680name", "name\u2000",
            "\u2028name", "name\u2029", "\u202fname", "name\u205f",
            "\u3000name", "a\u2000b", "a\u0085b", "\ufeffname",
        )
        for candidate in cases:
            with self.subTest(candidate=repr(candidate)):
                schema_accepts = (rule["minLength"] <= len(candidate) <= rule["maxLength"]
                                  and regex.fullmatch(candidate) is not None)
                try:
                    replace(proposal().model_binding, model_id=candidate)
                    python_accepts = True
                except ValueError:
                    python_accepts = False
                self.assertEqual(schema_accepts, python_accepts)

    def test_canonical_digest_stable_and_round_trip_strict(self):
        item = proposal()
        encoded = canonical_json(item)
        self.assertEqual(encoded, json.dumps(json.loads(encoded), separators=(",", ":"), sort_keys=True))
        self.assertEqual(WeightLearningProposalV1.from_dict(json.loads(encoded)), item)
        self.assertEqual(canonical_sha256(item), sha256(encoded.encode("utf-8")).hexdigest())
        self.assertEqual(canonical_sha256(WeightLearningProposalV1.from_dict(json.loads(encoded))),
                         canonical_sha256(item))
        self.assertNotIn("raw_prompt", encoded)
        self.assertNotIn("transcript", encoded)

        report = result(item, steps=1, before=digest("before"), after=digest("after"))
        self.assertEqual(WeightLearningResultV1.from_dict(json.loads(canonical_json(report))), report)
        validate_learning_pair(item, report)

    def test_arm_controls_and_hosted_unavailable_are_not_training_access(self):
        for arm in ("frozen", "zero", "correct_outcome", "shuffled_outcome", "corrupt_outcome"):
            with self.subTest(arm=arm):
                item = proposal(arm=arm)
                validate_learning_pair(item, result(item))
        hosted = proposal(kind="HOSTED_TARGET_UNAVAILABLE")
        self.assertEqual(hosted.model_binding.model_revision, "UNAVAILABLE")
        validate_learning_pair(hosted, result(hosted))
        with self.assertRaisesRegex(ValueError, "only a frozen"):
            replace(hosted, arm="correct_outcome")
        with self.assertRaisesRegex(ValueError, "cannot claim accessible"):
            replace(hosted.model_binding, weights_manifest_sha256=digest("invented-hosted-weights"))
        with self.assertRaisesRegex(ValueError, "frozen control arm cannot report optimizer steps"):
            result(hosted, steps=1)
        with self.assertRaisesRegex(ValueError, "cannot claim physical training"):
            validate_learning_pair(hosted, replace(result(hosted),
                                                   trainer_receipt_claim_sha256=digest("unverified-host-trainer")))

    def test_pleasure_only_trial_has_no_forced_meso_association(self):
        item = proposal(with_meso=False)
        self.assertIsNone(item.meso_association_proposal)
        self.assertIsNone(item.external_outcome.meso_event_id)
        self.assertEqual(WeightLearningProposalV1.from_dict(json.loads(canonical_json(item))), item)
        validate_learning_pair(item, result(item))
        with self.assertRaisesRegex(ValueError, "missing MESO event"):
            replace(item, external_outcome=replace(item.external_outcome,
                                                   meso_event_id="unbacked-meso-event"))

    def test_meso_zero_teaching_signal_without_candidate_or_delta(self):
        item = proposal(with_candidate=False)
        meso = item.meso_association_proposal
        self.assertIsNotNone(meso)
        self.assertEqual(meso.teaching_signal, 0.0)
        self.assertIsNone(meso.proposal_delta)
        self.assertIsNone(meso.association_id)
        self.assertEqual(WeightLearningProposalV1.from_dict(json.loads(canonical_json(item))), item)
        validate_learning_pair(item, result(item))
        with self.assertRaisesRegex(ValueError, "fully present or fully absent"):
            replace(meso, association_id="invented-association")

    def test_source_only_evidence_cannot_claim_authoritative_training_or_experience(self):
        item = proposal()
        report = result(item, steps=2, before=digest("before"), after=digest("after"))
        self.assertEqual(report.weight_change_evidence, "DIGESTS_DIFFER_UNVERIFIED")
        self.assertEqual(report.authority_status, "NOT_ESTABLISHED_BY_THIS_RECORD")
        self.assertEqual(report.phenomenology, "UNRESOLVED")
        self.assertEqual(item.phenomenology, "UNRESOLVED")
        self.assertEqual(item.meso_association_proposal.semantic_role, ASSOCIATION_ROLE)
        self.assertEqual(item.external_outcome.verification_status, "CLAIMED_UNVERIFIED")
        self.assertEqual(item.rights.status, "PENDING_HOST_ADMISSION")
        with self.assertRaisesRegex(ValueError, "authority_status"):
            replace(report, authority_status="EXTERNALLY_QUALIFIED")
        with self.assertRaisesRegex(ValueError, "verification_status"):
            replace(item.external_outcome, verification_status="VERIFIED")
        with self.assertRaisesRegex(ValueError, "semantic_role"):
            replace(item.meso_association_proposal, semantic_role="LORA_GRADIENT")

    def test_digest_classification_is_derived_and_not_a_training_proof(self):
        item = proposal()
        self.assertEqual(result(item).weight_change_evidence, "DIGESTS_ABSENT")
        self.assertEqual(result(item, steps=4, before=digest("same"), after=digest("same"))
                         .weight_change_evidence, "DIGESTS_IDENTICAL")
        with self.assertRaisesRegex(ValueError, "derived classification"):
            replace(result(item), weight_change_evidence="DIGESTS_DIFFER_UNVERIFIED")
        with self.assertRaisesRegex(ValueError, "supplied together"):
            result(item, steps=1, before=digest("only-before"))
        with self.assertRaisesRegex(ValueError, "contradict zero"):
            result(item, before=digest("before"), after=digest("after"))

    def test_frozen_and_zero_control_arms_reject_claimed_weight_changes(self):
        for arm in ("frozen", "zero"):
            with self.subTest(arm=arm):
                item = proposal(arm=arm)
                rejection = ("frozen control arm cannot report optimizer steps"
                             if arm == "frozen" else "frozen/zero control arm")
                with self.assertRaisesRegex(ValueError, rejection):
                    result(item, steps=3, before=digest("before"), after=digest("after"))
                unchanged = result(item, steps=0 if arm == "frozen" else 3,
                                   before=digest("same"), after=digest("same"))
                self.assertEqual(unchanged.weight_change_evidence, "DIGESTS_IDENTICAL")
                validate_learning_pair(item, unchanged)

    def test_strict_fields_no_raw_private_conversation_payload(self):
        item = proposal()
        raw = asdict(item)
        raw["private_transcript"] = "Do not copy a user's private text into this protocol"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            WeightLearningProposalV1.from_dict(raw)
        raw = asdict(item)
        raw["external_outcome"]["self_reward"] = 1.0
        with self.assertRaisesRegex(ValueError, "unexpected"):
            WeightLearningProposalV1.from_dict(raw)
        raw = asdict(item)
        del raw["model_binding"]["tokenizer_manifest_sha256"]
        with self.assertRaisesRegex(ValueError, "missing"):
            WeightLearningProposalV1.from_dict(raw)
        with self.assertRaisesRegex(ValueError, "must be an object"):
            WeightLearningProposalV1.from_dict([])

    def test_finite_values_and_bool_integer_bypass_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf, True, 10.1, -10.1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "teaching_signal must be finite"):
                    replace(proposal().meso_association_proposal, teaching_signal=invalid)
        for invalid in (math.nan, math.inf, True, 1.001, -0.1):
            with self.subTest(score=invalid):
                with self.assertRaisesRegex(ValueError, "measured_score must be finite"):
                    replace(proposal().external_outcome, measured_score=invalid)
        with self.assertRaisesRegex(ValueError, "sequence must be an integer"):
            replace(proposal().pleasure_event, sequence=True)
        with self.assertRaisesRegex(ValueError, "declared_optimizer_steps must be an integer"):
            replace(result(proposal()), declared_optimizer_steps=True)
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            replace(proposal().model_binding, weights_manifest_sha256="F" * 64)

    def test_cross_record_and_outcome_bindings_reject_substitution(self):
        item = proposal()
        report = result(item)
        with self.assertRaisesRegex(ValueError, "outcome/pleasure"):
            replace(item, external_outcome=replace(item.external_outcome, pleasure_evidence_id="other"))
        with self.assertRaisesRegex(ValueError, "outcome/MESO"):
            replace(item, external_outcome=replace(item.external_outcome, meso_event_id="other"))
        with self.assertRaisesRegex(ValueError, "runtime binding"):
            replace(item, pleasure_event=replace(item.pleasure_event, runtime_instance_id="other"))
        with self.assertRaisesRegex(ValueError, "result/proposal digest mismatch"):
            validate_learning_pair(item, replace(report, proposal_sha256=digest("wrong")))
        with self.assertRaisesRegex(ValueError, "model binding digest mismatch"):
            validate_learning_pair(item, replace(report, model_binding_sha256=digest("wrong")))
        with self.assertRaisesRegex(ValueError, "experiment, trial, or control arm mismatch"):
            validate_learning_pair(item, replace(report, trial_id="wrong-trial"))

    def test_revalidation_catches_privileged_in_process_mutation(self):
        item = proposal()
        object.__setattr__(item.external_outcome, "measured_score", math.nan)
        with self.assertRaisesRegex(ValueError, "measured_score must be finite"):
            canonical_json(item)


if __name__ == "__main__":
    unittest.main()
