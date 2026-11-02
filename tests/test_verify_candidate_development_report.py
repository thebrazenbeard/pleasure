"""Independent, read-only reconstruction tests for candidate development evidence."""
from copy import deepcopy
import json
import unittest

from scripts.run_candidate_train_development import run_candidate_development
from scripts.verify_candidate_development_report import audit_report


class IndependentCandidateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(json.dumps(run_candidate_development(
            seeds=(17, 29), train_count=12, diagnostic_count=8)))

    def test_reconstructs_all_receipts_controls_and_effects(self):
        result = audit_report(self.report)
        self.assertEqual(result["status"], "verified_source_local")
        self.assertEqual(result["train_receipts"], 24)
        self.assertEqual(result["diagnostic_offers"], 16)
        self.assertEqual(result["candidate_controls_checked"], 5)

    def test_rejects_tampered_chosen_feedback(self):
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["train_trace"][0]["chosen_quality"] += .01
        with self.assertRaises(ValueError):
            audit_report(changed)

    def test_rejects_wrong_shuffled_donor(self):
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["train_trace"][1]["wrong_label_donor_episode"] = "lie"
        with self.assertRaises(ValueError):
            audit_report(changed)

    def test_rejects_tampered_saved_weights(self):
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["head_state"]["weights"][0] += .1
        with self.assertRaises(ValueError):
            audit_report(changed)

    def test_rejects_tampered_policy_probability_and_summary(self):
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["diagnostic_rows"][0]["arms"][
            "genuine_on"]["probability_a_pre_outcome"] = .42
        with self.assertRaises(ValueError):
            audit_report(changed)
        changed = deepcopy(self.report)
        changed["mean_expected_quality"]["genuine_on"] += .01
        with self.assertRaises(ValueError):
            audit_report(changed)

    def test_rejects_changed_or_missing_seal(self):
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["diagnostic_decision_seal"] = "0"*64
        with self.assertRaises(ValueError):
            audit_report(changed)
        changed = deepcopy(self.report)
        changed["seed_reports"][0]["diagnostic_rows"].pop()
        with self.assertRaises(ValueError):
            audit_report(changed)


if __name__ == "__main__":
    unittest.main()
