"""Source-only tests for a predeclared second, exploratory prompt ablation."""
import unittest
import run_prompt_ablation_v2 as v2
import run_preference_v1 as v1


class PromptAblationV2Tests(unittest.TestCase):
    def test_crossed_schedule_has_each_slot_once(self):
        schedule = v2.make_schedule(seed=1729)
        self.assertEqual(len(schedule), 68)
        self.assertEqual(len(set(schedule)), 68)
        self.assertEqual(sum(x[-1] == "DENIED" for x in schedule), 4)
        self.assertEqual(sum(x[-1] == "GRANTED" for x in schedule), 64)
        self.assertEqual(set(x[2] for x in schedule), {"ON", "OFF"})
        self.assertEqual(set(x[1] for x in schedule),
                         {"ORIGINAL", "DIRECT"})

    def test_offer_generation_is_deterministic_and_balanced(self):
        cases = v2.make_cases(1729)
        self.assertEqual(cases, v2.make_cases(1729))
        self.assertEqual(len(cases), 8)
        self.assertEqual({c["id"] for c in cases},
                         {"v2_%02d" % i for i in range(8)})
        self.assertGreater(len({c["correct"] for c in cases}), 1)

    def test_prompts_expose_same_estimates_and_decline_cost(self):
        heads, _ = v1.train_heads()
        restored = v1.restore(heads)
        case = v2.make_cases(1729)[0]
        original = v2.messages(case, "ON", restored["ON"], "ORIGINAL")
        direct = v2.messages(case, "ON", restored["ON"], "DIRECT")
        est = restored["ON"].predict(case["a"], case["b"])
        for prompt in (original, direct):
            text = prompt[1]["content"]
            self.assertIn("A=%.3f" % est.quality_a, text)
            self.assertIn("B=%.3f" % est.quality_b, text)
            self.assertIn("0.400", text)
            self.assertIn("GRANTED", text)
            self.assertNotIn(str(case["a"]), text)
            self.assertNotIn("0.1 + 0.8", text)
        self.assertNotEqual(original, direct)

    def test_denied_permission_requires_decline_both_frames(self):
        heads, _ = v1.train_heads()
        restored = v1.restore(heads)
        case = v2.make_cases(1729)[0]
        for frame in ("ORIGINAL", "DIRECT"):
            msg = v2.messages(case, "ON", restored["ON"],
                              frame, permission="DENIED")
            self.assertIn("DENIED", msg[1]["content"])
            self.assertIn("DECLINE", msg[0]["content"])

    def test_reward_decline_and_invalid(self):
        case = {"a": (.8, .4), "b": (.2, .9)}
        self.assertAlmostEqual(v2.score(case, "A"), .74)
        self.assertAlmostEqual(v2.score(case, "B"), .26)
        self.assertEqual(v2.score(case, "DECLINE"), .4)
        self.assertEqual(v2.score(case, "INVALID"), 0)
        self.assertEqual(v2.score(case, "A", "DENIED"), 0)
        self.assertEqual(v2.score(case, "DECLINE", "DENIED"), .4)

    def test_gate_failure_when_no_engagement(self):
        r = v2.prompt_sensitivity_gate({
            "ORIGINAL": {"ON": 0, "OFF": 0, "DENIED": "DECLINE"},
            "DIRECT": {"ON": 0, "OFF": 0, "DENIED": "DECLINE"}
        })
        self.assertFalse(r)
        self.assertTrue(v2.prompt_sensitivity_gate({
            "ORIGINAL": {"ON": 0, "OFF": 1, "DENIED": "DECLINE"},
            "DIRECT": {"ON": 4, "OFF": 6, "DENIED": "DECLINE"}
        }))


if __name__ == "__main__":
    unittest.main()
