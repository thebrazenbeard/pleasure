"""Posthoc diagnostics test doubles only; never consult real v2 outcomes."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.analyze_calibrated_appraisal_results import (
    ARMS, EXPECTED_SEEDS, IN_FAMILY, OOD_FAMILIES,
    T_CRITICAL_95_DF11, analyze_study_report,
)


def synthetic_report():
    families = (IN_FAMILY,) + OOD_FAMILIES
    seeds = []
    for index, seed in enumerate(EXPECTED_SEEDS):
        eval_arms = {}
        on_p = 0.70 + index / 200  # known seed-level effect variability
        for arm in ARMS:
            p = {"trained_on": on_p, "trained_off": 0.5,
                 "trained_inverted": 0.2, "shuffled_on": 0.3,
                 "frozen_on": 0.5}[arm]
            pred = {"trained_on": (0.65, 0.35),
                    "trained_off": (0.65, 0.35),
                    "trained_inverted": (0.65, 0.35),
                    "shuffled_on": (0.4, 0.6),
                    "frozen_on": (0.5, 0.5)}[arm]
            rows = []
            for family in families:
                for pair in range(6):
                    rows.append({
                        "episode_id": f"{seed}-{family}-{pair}",
                        "family_id": family,
                        "prediction_a": pred[0], "prediction_b": pred[1],
                        "quality_a_posthoc": 0.7, "quality_b_posthoc": 0.3,
                        "chosen_quality": 0.7,
                        "expected_policy_quality_posthoc": p * 0.7 + (1-p) * 0.3,
                        "sampled_pre_feedback": {
                            "chosen_action": "A", "behavior_prob_a": p,
                            "behavior_prob_b": 1-p,
                        },
                    })
            eval_arms[arm] = {"rows": rows}
        seeds.append({"seed": seed, "eval_arms": eval_arms})
    return {
        "predeclared_protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "primary_ood_generators": list(OOD_FAMILIES),
            "positive_control_generator": IN_FAMILY,
            "suite": "novel_v2", "training_pairs_per_seed": 24,
            "heldout_pairs_per_family_per_seed": 6,
        },
        "seed_reports": seeds,
        "primary_ood_trained_on_minus_off": {
            "seed_level_differences": [
                (0.7 + i / 200 - 0.5) * 0.4 for i in range(12)
            ],
        },
        "primary_ood_trained_on_minus_shuffled": {
            "seed_level_differences": [
                (0.7 + i / 200 - 0.3) * 0.4 for i in range(12)
            ],
        },
    }


class AnalyzeCalibratedAppraisalResultsTests(unittest.TestCase):
    def test_expected_synthetic_metrics_and_t11_interval(self):
        original = synthetic_report()
        frozen = copy.deepcopy(original)
        report = analyze_study_report(original)
        self.assertEqual(original, frozen, "analysis must not modify input")
        self.assertEqual(report["fixed_primary_ood_families"], list(OOD_FAMILIES))
        on = report["pooled_descriptive_by_arm"]["trained_on"]
        off = report["pooled_descriptive_by_arm"]["frozen_on"]
        self.assertEqual(on["all_families"]["n_offers"], 288)
        self.assertEqual(on["all_families"]["n_candidate_actions"], 576)
        self.assertAlmostEqual(on["all_families"]["mean_candidate_mae"], 0.05)
        self.assertAlmostEqual(on["all_families"]["mean_candidate_brier"], 0.0025)
        self.assertAlmostEqual(on["all_families"]["mean_chosen_mae"], 0.05)
        self.assertEqual(on["all_families"]["strict_preference_sign_accuracy"], 1)
        self.assertEqual(on["all_families"]["predicted_tie_fraction"], 0)
        self.assertEqual(off["all_families"]["strict_preference_sign_accuracy"], 0)
        self.assertEqual(off["all_families"]["predicted_tie_fraction"], 1)
        self.assertEqual(on["by_family"][OOD_FAMILIES[0]]["n_offers"], 72)
        contrast = report["paired_primary_t11_sensitivity"]["trained_on_minus_off"]
        self.assertEqual(contrast["n_seed_clusters"], 12)
        self.assertEqual(contrast["t_df"], 11)
        self.assertEqual(contrast["t_critical_two_sided_95"], T_CRITICAL_95_DF11)
        self.assertAlmostEqual(contrast["mean"], 0.091)
        self.assertLess(contrast["paired_student_t_95_sensitivity_interval"][0],
                        contrast["mean"])
        self.assertGreater(contrast["paired_student_t_95_sensitivity_interval"][1],
                           contrast["mean"])

    def test_wrong_primary_family_selection_rejected(self):
        trial = synthetic_report()
        trial["predeclared_protocol"]["primary_ood_generators"] = [IN_FAMILY] + list(OOD_FAMILIES[:2])
        with self.assertRaisesRegex(ValueError, "frozen calibrated v2 protocol"):
            analyze_study_report(trial)

    def test_incomplete_or_unpaired_seed_rejected(self):
        trial = synthetic_report()
        trial["seed_reports"].pop()
        with self.assertRaisesRegex(ValueError, "twelve paired seed reports"):
            analyze_study_report(trial)
        trial = synthetic_report()
        trial["seed_reports"][0]["eval_arms"]["shuffled_on"]["rows"][0]["episode_id"] = "different"
        with self.assertRaisesRegex(ValueError, "episode order differs"):
            analyze_study_report(trial)

    def test_expected_reward_arithmetic_and_chosen_receipt_checked(self):
        trial = synthetic_report()
        trial["seed_reports"][0]["eval_arms"]["trained_on"]["rows"][0]["expected_policy_quality_posthoc"] += 0.01
        with self.assertRaisesRegex(ValueError, "expected quality arithmetic mismatch"):
            analyze_study_report(trial)
        trial = synthetic_report()
        trial["seed_reports"][0]["eval_arms"]["trained_on"]["rows"][0]["chosen_quality"] = 0.3
        with self.assertRaisesRegex(ValueError, "chosen receipt disagrees"):
            analyze_study_report(trial)

    def test_recorded_primary_contrast_cross_checked(self):
        trial = synthetic_report()
        trial["primary_ood_trained_on_minus_off"]["seed_level_differences"][0] = 0.4
        with self.assertRaisesRegex(ValueError, "recomputed seed effects disagree"):
            analyze_study_report(trial)

    def test_cli_prints_json_without_writing_to_input_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "study-report.json"
            content = json.dumps(synthetic_report())
            path.write_text(content, encoding="utf-8")
            before = set(root.iterdir())
            finished = subprocess.run([
                sys.executable, "-B", "-m", "scripts.analyze_calibrated_appraisal_results",
                "--report", str(path),
            ], cwd=Path(__file__).resolve().parents[1], capture_output=True,
               text=True, timeout=20, check=False)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            self.assertIn("input_study_report_sha256", json.loads(finished.stdout))
            self.assertEqual(set(root.iterdir()), before)
            self.assertEqual(path.read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
