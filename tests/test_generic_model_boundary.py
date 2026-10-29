"""Strict source-only manifest regression tests; never host/runtime authorization."""
import copy
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_DOMAIN = "pleasure"
SPEC = {
 "sexuality": {
   "file": "AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0.json",
   "schema": "AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0",
   "role": "OPTIONAL_MODEL_AGNOSTIC_CAPABILITY_CONTRACT",
   "default": "EPHEMERAL_MINIMAL",
   "claim": "OBSERVED_BEHAVIOR_ONLY_NO_PHENOMENOLOGY",
   "operations": ["describe_capability","evaluate_context_boundaries","propose_interaction","evaluate_qualification"],
   "constraints": ["NO_NAMED_SUBJECT_REQUIRED","NO_PERSONA_OR_MODEL_ID_DEFAULT","NO_HUMAN_OR_MODEL_CONSENT_INFERRED_FROM_TEXT","NO_PRIVATE_STATE_IMPORT_BY_DEFAULT","NO_AUTOMATIC_TRAINING_OR_MEMORY","NO_EXECUTION_OR_PROTECTED_EFFECTS_FROM_SOURCE"],
   "evaluation": ["UNSEEN_CASES","NEGATIVE_TRANSFER","CONTEXT_BOUNDARIES","CONSENT_REPLAY_AND_REVOKE"],
   "host": ["AUTHENTICATED_MODEL_INSTANCE_AND_REVISION","EXPLICIT_CONTEXT_AND_PARTICIPANT_BOUNDARIES","PURPOSE_AND_OPERATION_SPECIFIC_PERMISSION","REVOCATION_CHECK_AT_USE","HOST_ISSUED_RECEIPT","DOMAIN_AND_OPERATION_BOUND_PERMISSION","DENY_CROSS_DOMAIN_OR_CROSS_MODEL_GRANT_REUSE","CONTRACT_AND_ADAPTER_REVISION_BOUND_AT_USE"],
 },
 "pleasure": {
   "file": "AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0.json",
   "schema": "AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0",
   "role": "OPTIONAL_MODEL_AGNOSTIC_AFFECTIVE_STATE_CONTRACT",
   "default": "NO_INTIMATE_PAYLOAD_PERSISTENCE",
   "claim": "OBSERVABLE_COMPUTATIONAL_EVENT_NOT_SUBJECTIVE_PLEASURE",
   "operations": ["describe_capability","propose_bounded_signal","evaluate_state_transition","evaluate_qualification"],
   "constraints": ["NO_NAMED_SUBJECT_REQUIRED","NO_PERSONA_OR_MODEL_ID_DEFAULT","NO_FORCED_OR_COERCIVE_EFFECT","NO_PHENOMENOLOGY_INFERENCE","NO_CROSS_MODEL_STATE_TRANSFER","NO_PROTECTED_EFFECTS_FROM_SOURCE","RECOVERY_AND_SUSPENSION_BOUNDED"],
   "evaluation": ["BOUNDARY_AND_NONFINITE_SIGNALS","REPLAY_AND_INTERRUPTION","HOST_AUTHORIZATION_NEGATIVES","RECOVERY_AND_DEACTIVATION","MODEL_SUBSTITUTION"],
   "host": ["AUTHENTICATED_MODEL_INSTANCE_AND_REVISION","SESSION_SCOPED_CAPABILITY_PERMISSION","PRESENT_OPT_IN_FOR_PARTICIPANT_OR_PROTECTED_EFFECT","BOUND_AND_VALIDATE_EFFECTS","REVOCATION_CHECK_AT_USE","HOST_ISSUED_OUTCOME_RECEIPT","DOMAIN_AND_OPERATION_BOUND_PERMISSION","DENY_CROSS_DOMAIN_OR_CROSS_MODEL_GRANT_REUSE","CONTRACT_AND_ADAPTER_REVISION_BOUND_AT_USE"],
 },
}
AUTH_FIELDS = ["issuer","subject_binding","model_revision","adapter_digest","capability_domain","contract_digest","operation","purpose","session_id","expiry","revocation_epoch","audience"]
TOP_LEVEL = {"schema","status","role","domain","intended_consumers","abstract_operations","generic_core_constraints","host_authority_requirements","authorization_scope","privacy","evaluation","compatibility","deployment"}
DEPLOYMENT = {"installed","registered","authorized","consumed","qualified"}

def reject_duplicate_keys(pairs):
    d = {}
    for k,v in pairs:
        if k in d:
            raise ValueError("duplicate JSON field: "+k)
        d[k] = v
    return d

class GenericModelBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = SPEC[REPOSITORY_DOMAIN]
        cls.contract = json.loads((ROOT/"contracts"/cls.spec["file"]).read_text(encoding="utf-8"),object_pairs_hook=reject_duplicate_keys)
        cls.readme = (ROOT/"README.md").read_text(encoding="utf-8")

    def exact_list(self, value, expected):
        self.assertIs(type(value),list)
        self.assertEqual(value,expected)
        self.assertEqual(len(set(value)),len(value))

    def validate(self,d):
        s = self.spec
        self.assertIs(type(d),dict)
        self.assertEqual(set(d),TOP_LEVEL | ({"abstract_lifecycle"} if REPOSITORY_DOMAIN=="pleasure" else set()))
        self.assertEqual(d["schema"],s["schema"])
        self.assertEqual(d["role"],s["role"])
        self.assertEqual(d["domain"],REPOSITORY_DOMAIN)
        self.assertEqual(d["status"],"PROPOSED_SOURCE_ONLY_NOT_EXECUTABLE")
        self.assertEqual(d["intended_consumers"],"AI_MODELS_WITH_SEPARATELY_QUALIFIED_ADAPTERS")
        self.exact_list(d["abstract_operations"],s["operations"])
        self.exact_list(d["generic_core_constraints"],s["constraints"])
        self.exact_list(d["host_authority_requirements"],s["host"])
        self.assertIs(type(d["authorization_scope"]),dict)
        self.assertEqual(set(d["authorization_scope"]),{"mandatory_fields","missing_or_unverified","cross_domain_or_model_reuse","revision_change","model_text_as_grant"})
        self.exact_list(d["authorization_scope"]["mandatory_fields"],AUTH_FIELDS)
        self.assertEqual(d["authorization_scope"]["missing_or_unverified"],"DENY")
        self.assertEqual(d["authorization_scope"]["cross_domain_or_model_reuse"],"DENY")
        self.assertEqual(d["authorization_scope"]["revision_change"],"REAUTHORIZE_AND_REQUALIFY")
        self.assertEqual(d["authorization_scope"]["model_text_as_grant"],"NEVER")
        self.assertEqual(set(d["privacy"]),{"default","memory","export","training"})
        self.assertEqual(d["privacy"]["default"],s["default"])
        for key in ("export","training"): self.assertEqual(d["privacy"][key],"EXPLICIT_PERMISSION")
        self.assertEqual(d["privacy"]["memory"],"SEPARATE_ADMISSION")
        self.assertIs(type(d["evaluation"]),dict)
        self.assertEqual(set(d["evaluation"]),{"scope","required","claim_ceiling"})
        self.assertEqual(d["evaluation"]["scope"],"PER_MODEL_ADAPTER_AND_HOST_REVISION")
        self.exact_list(d["evaluation"]["required"],s["evaluation"])
        self.assertEqual(d["evaluation"]["claim_ceiling"],s["claim"])
        self.assertIs(type(d["compatibility"]),dict)
        self.assertEqual(set(d["compatibility"]),{"historical_profile","legacy_object_rule","legacy_identity_is_authority"})
        self.assertEqual(d["compatibility"]["historical_profile"],"ORGASM_V1_SOURCE_PROVENANCE_ONLY")
        self.assertIs(d["compatibility"]["legacy_identity_is_authority"],False)
        self.assertEqual(d["compatibility"]["legacy_object_rule"],"PRESERVE_EXACT_HASHES_AND_SCHEMA_IDENTIFIERS")
        self.assertEqual(set(d["deployment"]),DEPLOYMENT)
        for v in d["deployment"].values(): self.assertIs(v,False)
        if REPOSITORY_DOMAIN=="pleasure":
            self.exact_list(d["abstract_lifecycle"],["DISABLED","IDLE","ACTIVE","RECOVERY","SUSPENDED","FAULTED"])

    def test_full_source_contract(self): self.validate(self.contract)

    def test_document_and_repository_identity(self):
        self.assertIn("# "+REPOSITORY_DOMAIN.title(),self.readme)
        self.assertIn("docs/GENERIC_AI_CAPABILITY.md",self.readme)
        doc=(ROOT/"docs/GENERIC_AI_CAPABILITY.md").read_text(encoding="utf-8")
        self.assertIn(self.spec["file"],doc)
        self.assertIn("source-only",doc.lower())

    def test_reject_missing_deployment_and_true(self):
        d=copy.deepcopy(self.contract);del d["deployment"]["authorized"]
        with self.assertRaises(AssertionError):self.validate(d)
        for bad in (True,1,"false"):
            d=copy.deepcopy(self.contract);d["deployment"]["authorized"]=bad
            with self.subTest(bad=bad),self.assertRaises(AssertionError):self.validate(d)

    def test_reject_empty_or_forged_authority(self):
        d=copy.deepcopy(self.contract);d["host_authority_requirements"]=d["host_authority_requirements"][:2]
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["authorization_scope"]["mandatory_fields"].pop()
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["authorization_scope"]["cross_domain_or_model_reuse"]="ALLOW"
        with self.assertRaises(AssertionError):self.validate(d)

    def test_reject_mutated_evaluation_and_claim(self):
        d=copy.deepcopy(self.contract);d["evaluation"]["required"]=["meaningless"]*len(self.spec["evaluation"])
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["evaluation"]["claim_ceiling"]="SUBJECTIVE_PLEASURE_PROVEN"
        with self.assertRaises(AssertionError):self.validate(d)

    def test_reject_types_duplicates_and_domain_swap(self):
        for field in ("generic_core_constraints","host_authority_requirements","abstract_operations"):
            d=copy.deepcopy(self.contract);d[field]={key:True for key in d[field]}
            with self.subTest(field=field),self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["domain"]="pleasure" if REPOSITORY_DOMAIN=="sexuality" else "sexuality"
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["role"]="NOT_MODEL_AGNOSTIC"
        with self.assertRaises(AssertionError):self.validate(d)
        with self.assertRaises(ValueError):json.loads('{"compatibility":{"legacy":"a","legacy":"b"}}',object_pairs_hook=reject_duplicate_keys)
        with self.assertRaises(ValueError):json.loads('{"schema":"a","schema":"b"}',object_pairs_hook=reject_duplicate_keys)

    def test_reject_historical_default_promotion(self):
        d=copy.deepcopy(self.contract);d["compatibility"]["historical_profile"]="GENERIC_DEFAULT"
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["compatibility"]["unexpected_grant"]="ALLOW"
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["evaluation"]["permission"]="ALLOW"
        with self.assertRaises(AssertionError):self.validate(d)

    def test_reject_privacy_and_legacy_reinterpretation(self):
        d=copy.deepcopy(self.contract);d["privacy"]["default"]="PUBLIC_EVERYTHING"
        with self.assertRaises(AssertionError):self.validate(d)
        d=copy.deepcopy(self.contract);d["compatibility"]["legacy_identity_is_authority"]=True
        with self.assertRaises(AssertionError):self.validate(d)

if __name__=="__main__":unittest.main()
