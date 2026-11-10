"""Development-only matched-control smoke for numeric candidate learning."""
import unittest

from scripts.run_candidate_train_development import (
    TRAIN_ONLY_EVIDENCE, development_training_choices,
    run_candidate_development,
)


class CandidateDevelopmentTests(unittest.TestCase):
    def test_chosen_only_training_trace_is_precommitted_and_reproducible(self):
        first = development_training_choices(seed=17, count=24)
        second = development_training_choices(seed=17, count=24)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual(len({row.episode_id for row in first}), 24)
        for row in first:
            self.assertIn(row.chosen_action, ("A", "B"))
            self.assertEqual(row.propensity, .5)
            self.assertEqual(row.chosen_action,
                             "A" if row.random_draw < .5 else "B")
            self.assertTrue(0 <= row.chosen_quality <= 1)
            self.assertFalse(hasattr(row, "unchosen_quality"))
            self.assertEqual(len(row.evidence_id), 64)

    def test_same_train_data_genuine_and_shuffled_negative_control(self):
        report = run_candidate_development(
            seeds=(17, 29), train_count=64, diagnostic_count=32)
        self.assertEqual(report["evidence_class"], TRAIN_ONLY_EVIDENCE)
        self.assertEqual(report["seed_count"], 2)
        self.assertEqual(report["training_feedback_source"], "chosen_only")
        self.assertEqual(report["diagnostic_status"], "POSTHOC_DEVELOPMENT_NOT_CONFIRMATORY")
        self.assertEqual(set(report["arms"]), {
            "genuine_on", "genuine_off", "genuine_inverted",
            "shuffled_on", "frozen_on"})
        self.assertEqual(len(report["seed_reports"]), 2)
        for seed in report["seed_reports"]:
            self.assertEqual(seed["source_local_chosen_receipt_count"], 64)
            self.assertEqual(seed["genuine_updates"], 64)
            self.assertEqual(seed["shuffled_updates"], 64)
            self.assertEqual(seed["frozen_updates"], 0)
            self.assertTrue(seed["reloaded_prediction_exact"])
            self.assertTrue(seed["shuffled_labels_deranged"])
            self.assertEqual(seed["diagnostic_offer_count"], 32)
            self.assertEqual(seed["train_episodes_resolved_before_diagnostic"], True)

    def test_candidate_quality_benefit_on_synthetic_development_function(self):
        report = run_candidate_development(
            seeds=(17, 29, 43, 59), train_count=96, diagnostic_count=64)
        effects = report["mean_expected_quality"]
        self.assertGreater(effects["genuine_on"], effects["genuine_off"])
        self.assertGreater(effects["genuine_on"], effects["frozen_on"])
        self.assertGreater(effects["genuine_on"], effects["shuffled_on"])
        self.assertGreater(
            report["development_contrast_on_minus_frozen"], 0.001)

    def test_disallows_duplicate_or_invalid_schedule(self):
        for seeds, train, diag in (
            ((17, 17), 32, 16), ((-1,), 32, 16),
            ((17,), 0, 16), ((17,), 32, 0), ((17,), 99999, 16)):
            with self.subTest(seeds=seeds, train=train, diag=diag):
                with self.assertRaises(ValueError):
                    run_candidate_development(
                        seeds=seeds, train_count=train, diagnostic_count=diag)

    def test_separated_seeds_and_no_hidden_claim(self):
        report = run_candidate_development(
            seeds=(17,), train_count=32, diagnostic_count=16)
        self.assertEqual(report["seed_reports"][0]["seed"], 17)
        self.assertFalse(report["external_blind_custody"])
        self.assertFalse(report["pretrained_model_modified"])
        self.assertFalse(report["phenomenology_established"])


if __name__ == "__main__":
    unittest.main()
