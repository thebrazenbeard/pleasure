"""Source-contract checks only; no runtime/experience qualification."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT.name
CONTRACTS = {
    "sexuality": ("AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0.json",
                   "AI_SEXUALITY_GENERIC_CAPABILITY_PROPOSAL_V0"),
    "pleasure": ("AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0.json",
                 "AI_PLEASURE_GENERIC_CAPABILITY_PROPOSAL_V0"),
}

class GenericModelBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_name, cls.schema = CONTRACTS[DOMAIN]
        cls.contract = json.loads((ROOT / "contracts" / file_name).read_text(encoding="utf-8"))

    def test_scope_is_generic_optional_and_nonexecuting(self):
        d = self.contract
        self.assertEqual(d["schema"], self.schema)
        self.assertEqual(d["status"], "PROPOSED_SOURCE_ONLY_NOT_EXECUTABLE")
        self.assertIn("MODEL_AGNOSTIC", d["role"])
        self.assertEqual(d["intended_consumers"], "AI_MODELS_WITH_SEPARATELY_QUALIFIED_ADAPTERS")
        self.assertEqual(d["domain"], DOMAIN)

    def test_source_never_claims_installation_or_authorization(self):
        self.assertTrue(self.contract["deployment"])
        self.assertTrue(all(value is False for value in self.contract["deployment"].values()))
        self.assertFalse(self.contract["compatibility"]["legacy_identity_is_authority"])
        self.assertIn("REVOCATION_CHECK_AT_USE", self.contract["host_authority_requirements"])

    def test_separate_privacy_and_qualification(self):
        self.assertEqual(self.contract["privacy"]["memory"], "SEPARATE_ADMISSION")
        self.assertEqual(self.contract["privacy"]["training"], "EXPLICIT_PERMISSION")
        self.assertEqual(self.contract["privacy"]["export"], "EXPLICIT_PERMISSION")
        self.assertEqual(self.contract["evaluation"]["scope"], "PER_MODEL_ADAPTER_AND_HOST_REVISION")
        self.assertGreaterEqual(len(self.contract["evaluation"]["required"]), 4)

    def test_generic_contract_and_readme_are_cross_referenced(self):
        expected_name, _ = CONTRACTS[DOMAIN]
        doc = (ROOT / "docs" / "GENERIC_AI_CAPABILITY.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(expected_name, doc)
        self.assertIn("docs/GENERIC_AI_CAPABILITY.md", readme)
        self.assertIn("model", doc.lower())
        self.assertIn("source-only", doc.lower())

    def test_no_default_named_persona_or_unbounded_effect(self):
        self.assertIn("NO_NAMED_SUBJECT_REQUIRED", self.contract["generic_core_constraints"])
        self.assertIn("NO_PERSONA_OR_MODEL_ID_DEFAULT", self.contract["generic_core_constraints"])
        self.assertNotIn("effect_grant", self.contract)
        self.assertNotIn("standing_consent", self.contract)

if __name__ == "__main__":
    unittest.main()
