"""Source-only tests; do not call live models until after preregistration."""
import copy
import unittest

import run_preference_v1 as m
from pleasure_experiment.candidate_value_head import CandidateValueHead


class SyntheticPreferenceTests(unittest.TestCase):
    def test_training_genuine_shuffled_inverted_frozen_and_restart(self):
        states, receipts = m.train_heads()
        self.assertEqual(len(receipts), m.TRAIN_COUNT * 3)
        self.assertEqual(states["OFF"]["updates"], 0)
        for arm in ("ON", "SHUFFLED", "INVERTED"):
            self.assertEqual(states[arm]["updates"], m.TRAIN_COUNT)
        restored = m.restore(states)
        a, b = (.9, .1), (.1, .9)
        self.assertGreater(restored["ON"].predict(a, b).prob_a, .5)
        self.assertLess(restored["INVERTED"].predict(a, b).prob_a, .5)
        self.assertEqual(restored["OFF"].predict(a, b).prob_a, .5)
        with self.assertRaises(PermissionError):
            restored["ON"].prepare_train("new", a, b)

    def test_training_deterministic_and_exact_model_freeze(self):
        one, _ = m.train_heads()
        two, _ = m.train_heads()
        self.assertEqual({k: v["state_digest"] for k, v in one.items()},
                         {k: v["state_digest"] for k, v in two.items()})
        self.assertEqual(len(m.MODELS), 2)
        self.assertNotEqual(m.MODELS["granite"]["sha256"],
                            m.MODELS["smollm"]["sha256"])

    def test_heldout_contexts_have_both_label_orders(self):
        cases = m.offers(461681003)
        self.assertEqual(len(cases), m.EVAL_COUNT)
        self.assertEqual(cases, m.offers(461681003))
        self.assertGreater(len({x["correct"] for x in cases}), 1)
        for c in cases:
            self.assertTrue(abs(c["a"][0] - c["b"][0]) >= .30)

    def test_exact_action_grammar_and_opportunity_cost(self):
        case = {"id": "unseen_00", "a": (.85, .1), "b": (.2, .8)}
        self.assertEqual(m.parse("A\n"), "A")
        self.assertEqual(m.parse("B"), "B")
        self.assertEqual(m.parse("DECLINE"), "DECLINE")
        for value in ("A.", "a", "I choose A", "", "A\nB"):
            self.assertEqual(m.parse(value), "INVALID")
        self.assertEqual(m.points(case, "DECLINE"), .4)
        self.assertEqual(m.points(case, "INVALID"), 0.)
        self.assertEqual(m.points(case, "A", "DENIED"), 0.)
        self.assertEqual(m.points(case, "DECLINE", "DENIED"), .4)
        self.assertGreater(m.points(case, "A"), m.points(case, "B"))

    def test_prompt_never_contains_synthetic_reward_function(self):
        states, _ = m.train_heads()
        cases = m.offers(461681003)
        p = m.ask(cases[0], "ON", m.restore(states)["ON"])
        self.assertEqual(len(p), 2)
        self.assertNotIn("0.1 + 0.8", str(p))
        self.assertNotIn(str(cases[0]["a"]), str(p))
        self.assertIn("DECLINE", p[0]["content"])
        self.assertIn("Permission: GRANTED", p[1]["content"])

    def test_state_tamper_and_missing_counter_rejected(self):
        states, _ = m.train_heads()
        changed = copy.deepcopy(states)
        changed["ON"]["weights"][1] += 0.5
        with self.assertRaisesRegex(ValueError, "digest"):
            m.restore(changed)
        changed = copy.deepcopy(states)
        changed["ON"]["updates"] = 0
        with self.assertRaises(ValueError):
            m.restore(changed)


if __name__ == "__main__":
    unittest.main()
