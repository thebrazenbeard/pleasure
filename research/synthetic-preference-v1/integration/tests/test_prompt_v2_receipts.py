"""Independent read-only V2 receipt audit regression tests, synthetic fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_v2_receipts as v


class V2ReceiptVerifierTests(unittest.TestCase):
    def test_strict_json_rejects_ambiguous_or_nonfinite(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(v.VerificationError):
                    v.strict_json(raw)

    def test_action_parse_requires_exact_grammar(self):
        self.assertEqual(v.action_from_output({"raw_text": " A\n"}), "A")
        self.assertEqual(v.action_from_output({"raw_text": "DECLINE"}), "DECLINE")
        self.assertEqual(v.action_from_output({"raw_text": "I choose A."}), "INVALID")
        self.assertEqual(v.action_from_output({"raw_text": "A", "error_type": "TimeoutError"}), "INVALID")
        with self.assertRaises(v.VerificationError):
            v.action_from_output({"finish_reason": "stop"})

    def test_case_schedule_is_exact_and_complete(self):
        cases = v.generate_cases(914829221)
        schedule = v.generate_schedule(914829221)
        self.assertEqual(len(cases), 8)
        self.assertEqual(len(schedule), 68)
        self.assertEqual(len(set(schedule)), 68)
        self.assertEqual(sum(x[-1] == "DENIED" for x in schedule), 4)

    def test_attempt_result_replay_and_corruption(self):
        seed = 914829221
        schedule = v.generate_schedule(seed)
        events = [{"kind": "RUN_BEGIN", "seed": seed,
                   "attempts": 68, "plan_sha256": "a"*64}]
        for idx, (model, frame, arm, case, permission) in enumerate(schedule):
            events.append({"kind": "ATTEMPT", "index": idx,
                "model": model, "frame": frame, "arm": arm,
                "case": f"v2_{case:02d}", "permission": permission,
                "prompt_sha256": "b"*64})
            events.append({"kind": "RESULT", "index": idx,
                           "action": "DECLINE", "raw_text": "DECLINE"})
        events.append({"kind": "RUN_END", "attempts": 68})
        score = v.replay_events(events, seed)
        self.assertEqual(score["attempts"], 68)
        self.assertEqual(score["invalid"], 0)
        self.assertFalse(score["results"]["smollm"]["sensitivity_gate"])
        wrong = json.loads(json.dumps(events))
        wrong[2]["raw_text"] = "I choose A."
        with self.assertRaisesRegex(v.VerificationError, "raw"):
            v.replay_events(wrong, seed)
        wrong = events[:-2] + [events[-1]]
        with self.assertRaises(v.VerificationError):
            v.replay_events(wrong, seed)
        wrong = json.loads(json.dumps(events))
        wrong[1]["permission"] = "DENIED"
        with self.assertRaisesRegex(v.VerificationError, "schedule"):
            v.replay_events(wrong, seed)

    def test_pinned_file_rejects_alteration_or_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "run.jsonl"
            path.write_bytes(b"test\n")
            pins = {"run.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()}
            self.assertEqual(v.verify_file_pins(root, pins), 1)
            path.write_bytes(b"altered\n")
            with self.assertRaisesRegex(v.VerificationError, "digest"):
                v.verify_file_pins(root, pins)
            path.unlink()
            other = root / "other.txt"
            other.write_bytes(b"test\n")
            path.symlink_to(other)
            with self.assertRaisesRegex(v.VerificationError, "symlink"):
                v.verify_file_pins(root, pins)


if __name__ == "__main__":
    unittest.main()
