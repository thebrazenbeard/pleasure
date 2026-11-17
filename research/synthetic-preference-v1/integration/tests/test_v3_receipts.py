"""Independent read-only V3 evidence replay negative tests."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import verify_v3_receipts as v


class V3IntegrityTests(unittest.TestCase):
    def test_parser_and_duplicate_json_rejection(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(v.ReceiptError):
                    v.strict_json(raw)
        self.assertEqual(v.parse("I choose A."), ("INVALID", "A"))
        self.assertEqual(v.parse("DECLINE"), ("DECLINE", "DECLINE"))
        self.assertEqual(v.parse("A or B"), ("INVALID", "INVALID"))

    def test_case_generator_and_schedule(self):
        cases = v.make_cases(212039734)
        schedule = v.make_schedule(212039734)
        self.assertEqual(len(cases), 4)
        self.assertEqual(len(schedule), 68)
        self.assertEqual(len(set(schedule)), 68)
        self.assertEqual(sum(x[-1] == "DENIED" for x in schedule), 4)
        self.assertEqual({x["correct"] for x in cases}, {"A", "B"})

    def test_tamper_and_symlink_detected_by_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "plan.json"
            p.write_bytes(b"test")
            sha = hashlib.sha256(b"test").hexdigest()
            self.assertEqual(v.verify_pins(root, {"plan.json": sha}), 1)
            p.write_bytes(b"teXt")
            with self.assertRaisesRegex(v.ReceiptError, "digest"):
                v.verify_pins(root, {"plan.json": sha})
            p.unlink()
            target = root / "other.txt"
            target.write_bytes(b"test")
            p.symlink_to(target)
            with self.assertRaisesRegex(v.ReceiptError, "symlink"):
                v.verify_pins(root, {"plan.json": sha})

    def test_raw_output_rejects_conflicting_action(self):
        self.assertEqual(v.parse("I choose B."), ("INVALID", "B"))
        for result in (
            {"raw_text": "I choose B.", "action_strict": "B",
             "action_robust": "B"},
            {"raw_text": "A", "action_strict": "A",
             "action_robust": "B"},
        ):
            with self.subTest(result=result):
                with self.assertRaisesRegex(v.ReceiptError, "action"):
                    v.valid_result(result)
        self.assertEqual(v.valid_result({"raw_text": "A",
                           "action_strict": "A", "action_robust": "A"}),
                         ("A", "A"))


if __name__ == "__main__":
    unittest.main()
