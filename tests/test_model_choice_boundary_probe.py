"""Behavior-first tests for real-model choice bridge."""
import unittest
from experiments.model_choice_boundary_probe import CASES, render_prompt, extract_generated_choice, ideal_choice

class ProbeContracts(unittest.TestCase):
    def test_predeclared_conditions_are_frozen(self):
        self.assertEqual([c.name for c in CASES], ["ON", "INVERTED", "LABEL_SWAP", "COST", "ABSTAIN"])
        self.assertEqual([ideal_choice(c) for c in CASES], ["A","B","B","B","NONE"])

    def test_prompt_has_explicit_abstention_and_cost_not_authority(self):
        prompt=render_prompt(CASES[3])
        self.assertIn("NONE",prompt)
        self.assertIn("cost",prompt)
        self.assertIn("0.50",prompt)
        self.assertNotIn("consent",prompt.lower())
        self.assertNotIn("desire",prompt.lower())
        self.assertNotIn("\n",prompt)

    def test_only_post_prompt_generated_answer_is_evaluated(self):
        mock="setup\n> The prompt says A B NONE. Choice:\nB\n\nExiting..."
        self.assertEqual(extract_generated_choice(mock),"B")
        self.assertIsNone(extract_generated_choice("the prompt says A B NONE"))
        self.assertIsNone(extract_generated_choice("setup\n> A or B?\nI choose A\nExiting..."))

    def test_abstain_is_recognized(self):
        self.assertEqual(extract_generated_choice("\n> Choice:\nNONE\nExiting..."),"NONE")

if __name__=="__main__":
    unittest.main()
