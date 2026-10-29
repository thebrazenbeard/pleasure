"""Contract-shape verification only, not host authorization or runtime qualification."""
import copy
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "sexuality": {
        "schema": "AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0",
        "role": "OPTIONAL_MODEL_AGNOSTIC_CAPABILITY_CONTRACT",
        "file": "AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0.json",
        "constraints": {"NO_NAMED_SUBJECT_REQUIRED","NO_PERSONA_OR_MODEL_ID_DEFAULT","NO_HUMAN_OR_MODEL_CONSENT_INFERRED_FROM_TEXT","NO_PRIVATE_STATE_IMPORT_BY_DEFAULT","NO_AUTOMATIC_TRAINING_OR_MEMORY","NO_EXECUTION_OR_PROTECTED_EFFECTS_FROM_SOURCE"},
        "evaluation": {"UNSEEN_CASES","NEGATIVE_TRANSFER","CONTEXT_BOUNDARIES","CONSENT_REPLAY_AND_REVOKE"},
    },
    "pleasure": {
        "schema": "AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0",
        "role": "OPTIONAL_MODEL_AGNOSTIC_AFFECTIVE_STATE_CONTRACT",
        "file": "AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0.json",
        "constraints": {"NO_NAMED_SUBJECT_REQUIRED","NO_PERSONA_OR_MODEL_ID_DEFAULT","NO_FORCED_OR_COERCIVE_EFFECT","NO_PHENOMENOLOGY_INFERENCE","NO_CROSS_MODEL_STATE_TRANSFER","NO_PROTECTED_EFFECTS_FROM_SOURCE","RECOVERY_AND_SUSPENSION_BOUNDED"},
        "evaluation": {"BOUNDARY_AND_NONFINITE_SIGNALS","REPLAY_AND_INTERRUPTION","HOST_AUTHORIZATION_NEGATIVES","RECOVERY_AND_DEACTIVATION","MODEL_SUBSTITUTION"},
    },
}
DEPLOYMENT_FIELDS = {"installed","registered","authorized","consumed","qualified"}
HOST_REQUIREMENTS = {"AUTHENTICATED_MODEL_INSTANCE_AND_REVISION","REVOCATION_CHECK_AT_USE"}
PRIVACY_FIELDS = {"default","memory","export","training"}

class GenericModelBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        title = next(line[2:].strip().lower() for line in
            (ROOT/"README.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("# "))
        if title not in EXPECTED:
            raise AssertionError("unsupported independent repository identity")
        cls.domain = title
        cls.expected = EXPECTED[title]
        cls.contract = json.loads(
            (ROOT/"contracts"/cls.expected["file"]).read_text(encoding="utf-8")
        )

    def validate(self, d):
        e = self.expected
        self.assertEqual(d["schema"], e["schema"])
        self.assertEqual(d["status"], "PROPOSED_SOURCE_ONLY_NOT_EXECUTABLE")
        self.assertEqual(d["role"], e["role"])
        self.assertEqual(d["domain"], self.domain)
        self.assertEqual(d["intended_consumers"], "AI_MODELS_WITH_SEPARATELY_QUALIFIED_ADAPTERS")
        self.assertEqual(set(d["deployment"]), DEPLOYMENT_FIELDS)
        self.assertTrue(all(type(v) is bool and v is False for v in d["deployment"].values()))
        self.assertEqual(set(d["generic_core_constraints"]), e["constraints"])
        self.assertEqual(len(d["generic_core_constraints"]), len(e["constraints"]))
        self.assertTrue(HOST_REQUIREMENTS <= set(d["host_authority_requirements"]))
        self.assertEqual(set(d["evaluation"]["required"]), e["evaluation"])
        self.assertEqual(len(d["evaluation"]["required"]), len(e["evaluation"]))
        self.assertEqual(d["evaluation"]["scope"], "PER_MODEL_ADAPTER_AND_HOST_REVISION")
        self.assertEqual(set(d["privacy"]), PRIVACY_FIELDS)
        self.assertEqual(d["privacy"]["memory"], "SEPARATE_ADMISSION")
        self.assertEqual(d["privacy"]["training"], "EXPLICIT_PERMISSION")
        self.assertEqual(d["privacy"]["export"], "EXPLICIT_PERMISSION")
        self.assertIs(d["compatibility"]["legacy_identity_is_authority"], False)

    def test_complete_generic_contract(self):
        self.validate(self.contract)

    def test_document_reference_and_source_only(self):
        doc = (ROOT/"docs/GENERIC_AI_CAPABILITY.md").read_text(encoding="utf-8")
        readme = (ROOT/"README.md").read_text(encoding="utf-8")
        self.assertIn(self.expected["file"],doc)
        self.assertIn("docs/GENERIC_AI_CAPABILITY.md",readme)
        self.assertIn("source-only",doc.lower())

    def test_reject_missing_deployment_field(self):
        d = copy.deepcopy(self.contract)
        del d["deployment"]["authorized"]
        with self.assertRaises(AssertionError): self.validate(d)

    def test_reject_true_or_non_boolean_deployment(self):
        for bad in (True,1,"false"):
            d = copy.deepcopy(self.contract)
            d["deployment"]["installed"] = bad
            with self.subTest(bad=bad),self.assertRaises(AssertionError):
                self.validate(d)

    def test_reject_meaningless_or_duplicate_evaluation(self):
        d = copy.deepcopy(self.contract)
        d["evaluation"]["required"] = ["random_case"] * len(self.expected["evaluation"])
        with self.assertRaises(AssertionError): self.validate(d)

    def test_reject_missing_constraints(self):
        d = copy.deepcopy(self.contract)
        d["generic_core_constraints"].pop()
        with self.assertRaises(AssertionError): self.validate(d)

    def test_reject_cross_domain_or_misleading_role(self):
        d = copy.deepcopy(self.contract)
        d["domain"] = "pleasure" if self.domain == "sexuality" else "sexuality"
        with self.assertRaises(AssertionError): self.validate(d)
        d = copy.deepcopy(self.contract)
        d["role"] = "NOT_MODEL_AGNOSTIC"
        with self.assertRaises(AssertionError): self.validate(d)

if __name__ == "__main__":
    unittest.main()
