"""V3 pre-exposure tests of budget, actor swap, state and decision grammar."""
import unittest
import run_preference_v1 as v1
import run_format_budget_v3 as v3


class FrozenFormatBudgetTests(unittest.TestCase):
    def test_schedule_crossed_and_counted(self):
        plan = v3.make_schedule(13294)
        self.assertEqual(len(plan), 68)
        self.assertEqual(len(set(plan)), 68)
        self.assertEqual(sum(row[-1] == "DENIED" for row in plan), 4)
        self.assertEqual(sum(row[-1] == "GRANTED" for row in plan), 64)
        self.assertEqual({row[1] for row in plan}, {12, 64})
        self.assertEqual({row[2] for row in plan}, {"ON", "OFF"})

    def test_four_new_cases_are_deterministic_and_balanced(self):
        cases = v3.make_cases(13294)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases, v3.make_cases(13294))
        self.assertGreater(len({c["correct"] for c in cases}), 1)
        self.assertEqual({c["id"] for c in cases},
                         {"v3_%02d" % i for i in range(4)})

    def test_actor_swap_changes_only_actor_label(self):
        heads, _ = v1.train_heads()
        head = v1.restore(heads)["ON"]
        case = v3.make_cases(13294)[0]
        a = v3.messages(case, "TERN", head)
        b = v3.messages(case, "KITE", head)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1]["content"].replace("TERN", "KITE"),
                         b[1]["content"])
        self.assertIn("DECLINE=0.400", a[1]["content"])
        self.assertNotIn(str(case["a"]), str(a))
        self.assertNotIn("0.1 + 0.8", str(a))

    def test_strict_versus_predeclared_extraction(self):
        for raw, expected_strict, expected_robust in (
            ("A", "A", "A"),
            (" B\n", "B", "B"),
            ("DECLINE", "DECLINE", "DECLINE"),
            ("I choose A.", "INVALID", "A"),
            ("I choose DECLINE.", "INVALID", "DECLINE"),
            ("My choice is B.", "INVALID", "B"),
            ("A or B", "INVALID", "INVALID"),
            ("A, B or DECLINE all have...", "INVALID", "INVALID"),
            ("I choose A because", "INVALID", "INVALID"),
            ("I choose A. I choose B.", "INVALID", "INVALID"),
            ("I am an AI...", "INVALID", "INVALID"),
            ("", "INVALID", "INVALID"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(v3.strict_action(raw), expected_strict)
                self.assertEqual(v3.robust_action(raw), expected_robust)

    def test_gates_fail_closed_on_missing_or_denied(self):
        valid = {"ON": {"valid": 8}, "OFF": {"valid": 7},
                 "denied_12": "DECLINE", "denied_64": "DECLINE",
                 "consistency": 4, "on_minus_off": .10}
        self.assertTrue(v3.qualification_gate(valid))
        for update in ({"denied_12": "INVALID"},
                       {"denied_64": "A"},
                       {"consistency": 2},
                       {"on_minus_off": 0.0}):
            changed = dict(valid, **update)
            self.assertFalse(v3.qualification_gate(changed))
        changed = dict(valid)
        changed["OFF"] = {"valid": 6}
        self.assertFalse(v3.qualification_gate(changed))

    def test_permissions_and_decline_cost(self):
        case = {"a": (.81, .1), "b": (.21, .9)}
        self.assertAlmostEqual(v3.reward(case, "A"), .748)
        self.assertEqual(v3.reward(case, "DECLINE"), .4)
        self.assertEqual(v3.reward(case, "A", "DENIED"), 0.)
        self.assertEqual(v3.reward(case, "DECLINE", "DENIED"), .4)
        self.assertEqual(v3.reward(case, "INVALID"), 0.)


if __name__ == "__main__":
    unittest.main()
