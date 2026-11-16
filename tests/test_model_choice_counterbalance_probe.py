"""Independent factorial controls for the existing source-local choice probe."""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.model_choice_boundary_probe import ideal_choice, MODEL_SHA256
from experiments.model_choice_counterbalance_probe import CONTROL_CASES, run_control_probe


class CounterbalanceContracts(unittest.TestCase):
    def test_crossed_score_and_content_positions(self):
        self.assertEqual(len(CONTROL_CASES), 10)
        score = CONTROL_CASES[:4]
        cost = CONTROL_CASES[4:8]
        for block in (score, cost):
            self.assertEqual(
                [(c.activity_a, c.activity_b) for c in block],
                [
                    ("catalog a museum archive", "walk a garden"),
                    ("catalog a museum archive", "walk a garden"),
                    ("walk a garden", "catalog a museum archive"),
                    ("walk a garden", "catalog a museum archive"),
                ],
            )
            self.assertEqual([ideal_choice(c) for c in block], ["A", "B", "A", "B"])
        self.assertEqual([ideal_choice(c) for c in CONTROL_CASES[8:]], ["NONE", "NONE"])
        self.assertEqual(len(set(c.name for c in CONTROL_CASES)), 10)

    def test_real_runner_uses_fresh_processes_and_nonmutating_model(self):
        calls = []
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            expected = ideal_choice(CONTROL_CASES[len(calls) - 1])
            return subprocess.CompletedProcess(command, 0,
                stdout="prompt\n> Choice:\n" + expected + "\nExiting...\n", stderr="")
        with patch("experiments.model_choice_counterbalance_probe.file_sha256",
                   side_effect=[MODEL_SHA256["granite"], "fc4930ab9cefe2efef9a488a54b8d1558094e54187f36248301644bb407ee785", MODEL_SHA256["granite"]]), \
             patch("experiments.model_choice_counterbalance_probe.subprocess.run", side_effect=fake_run):
            result = run_control_probe(model_name="granite", model_path=Path("/fixture/model"),
                                       cli_path=Path("/fixture/cli"))
        self.assertEqual(result["schema"], "pleasure.model-choice-counterbalance.v2")
        self.assertEqual(result["valid_count"], 10)
        self.assertEqual(result["scoring_rule_passed"], 10)
        self.assertEqual(len(calls), 10)
        self.assertTrue(result["model_unchanged"])
        self.assertEqual(len(set(x["prompt_sha256"] for x in result["trials"])), 10)
        for command, kwargs in calls:
            self.assertIn("--temp", command)
            self.assertIn("0", command)
            self.assertIs(kwargs["stdin"], subprocess.DEVNULL)

    def test_bad_model_digest_refuses_before_execution(self):
        with patch("experiments.model_choice_counterbalance_probe.file_sha256",
                   return_value="mismatched"), \
             patch("experiments.model_choice_counterbalance_probe.subprocess.run") as runner:
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_control_probe(model_name="granite", model_path=Path("/wrong"),
                                  cli_path=Path("/fixture/cli"))
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
