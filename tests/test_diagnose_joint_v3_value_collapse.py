"""v4 design-only diagnostics: no new EVAL or model mutation."""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

from scripts import diagnose_joint_v3_value_collapse as v4


def fixture():
    reports = []
    for seed in v4.SEEDS:
        arms = {}
        for arm in v4.ARMS:
            rows = []
            for family in v4.FAMILIES:
                for index in range(4):
                    offset = (seed - 163) * .0001
                    gap = offset + .000002 * index
                    if arm == "shuffled_joint_on":
                        gap = -gap
                    rows.append({
                        "episode_id": f"{seed}:{family}:{index}",
                        "family_id": family,
                        "prediction_a": .5 + gap / 2,
                        "prediction_b": .5 - gap / 2,
                        "quality_a_posthoc": .6 if index % 2 == 0 else .2,
                        "quality_b_posthoc": .2 if index % 2 == 0 else .6,
                    })
            arms[arm] = {"rows": rows}
        reports.append({"seed": seed, "eval_arms": arms})
    return {
        "evidence_class":
            "SOURCE_LOCAL_JOINT_LORA_SEALED_OUTCOME_V3_UNAUTHENTICATED",
        "seed_reports": reports,
    }


class V4ValueDiagnosticTests(TestCase):
    def test_collapsed_within_seed_gap_detected_without_false_utility_claim(self):
        result = v4.diagnose(fixture())
        self.assertEqual(result["seed_count"], 12)
        self.assertEqual(result["joint_on_n_offers"], 240)
        self.assertLess(result["within_over_between_predicted_gap_sd"], .01)
        self.assertIn("NOT_V4_CONFIRMATORY", result["evidence_class"])
        self.assertEqual(result["by_arm_and_family"]["joint_on"]
                         ["eval_latent_00_v3"]["n"], 48)

    def test_bad_class_and_incomplete_seed_rejected(self):
        report = fixture()
        report["evidence_class"] = "CLAIMED_EXTERNALLY_BLIND"
        with self.assertRaisesRegex(ValueError, "evidence class"):
            v4.diagnose(report)
        report = fixture()
        report["seed_reports"].pop()
        with self.assertRaisesRegex(ValueError, "seed schedule"):
            v4.diagnose(report)

    def test_mismatched_episode_or_control_rejected(self):
        report = fixture()
        report["seed_reports"][0]["eval_arms"]["head_only_on"]["rows"][0]["episode_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "unpaired"):
            v4.diagnose(report)
        report = fixture()
        report["seed_reports"][0]["eval_arms"]["shuffled_joint_on"]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "unmatched"):
            v4.diagnose(report)

    def test_corrupt_gap_rejected(self):
        report = fixture()
        report["seed_reports"][0]["eval_arms"]["joint_on"]["rows"][0]["prediction_a"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v4.diagnose(report)

    def test_pairwise_correlations_are_observational_only(self):
        report = fixture()
        result = v4.diagnose(report)
        self.assertTrue(isfinite(result["mean_within_seed_predicted_gap_sd"]))
        self.assertEqual(
            result["joint_on_vs_head_only_max_absolute_predicted_gap_difference"], 0)
        self.assertIn("observational", result["interpretation_guard"])


    def test_cli_refuses_unsealed_input_without_opening_report(self):
        import sys
        with TemporaryDirectory() as folder:
            with mock.patch.object(sys, "argv", ["v4-diagnostic", folder]):
                with mock.patch.object(v4, "verify_complete_study",
                                       side_effect=ValueError("unsealed")) as gate:
                    with mock.patch.object(Path, "read_bytes") as reader:
                        with self.assertRaisesRegex(ValueError, "unsealed"):
                            v4.main()
                        gate.assert_called_once()
                        reader.assert_not_called()
