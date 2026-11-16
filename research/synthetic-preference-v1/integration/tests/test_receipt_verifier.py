"""Adversarial, source-only tests for the privately pinned V1 receipts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import verify_v1_receipts as v


class ReceiptVerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {"files": {}, "expected_attempts": 1}

    def save(self, key, value):
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        (self.root / key).write_bytes(raw)
        self.manifest["files"][key] = hashlib.sha256(raw).hexdigest()

    def test_genuine_file_hashes_recover_but_one_byte_mutation_fails(self):
        self.save("run.jsonl", b'{"kind":"RUN_BEGIN"}\n')
        self.save("train.json", {"synthetic": True})
        self.assertEqual(v.check_files(self.root, self.manifest), 2)
        with (self.root / "run.jsonl").open("ab") as out:
            out.write(b"x")
        with self.assertRaisesRegex(v.ReceiptVerificationError, "digest"):
            v.check_files(self.root, self.manifest)

    def test_missing_receipt_and_invalid_digest_rejected(self):
        self.save("train.json", {"a": 1})
        (self.root / "train.json").unlink()
        with self.assertRaisesRegex(v.ReceiptVerificationError, "missing"):
            v.check_files(self.root, self.manifest)
        (self.root / "train.json").write_text('{"a":1}')
        self.manifest["files"]["train.json"] = "not_a_sha256"
        with self.assertRaisesRegex(v.ReceiptVerificationError, "sha256"):
            v.check_files(self.root, self.manifest)

    def test_json_duplicate_keys_and_nonfinite_rejected(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(raw=raw):
                with self.assertRaises(v.ReceiptVerificationError):
                    v.strict_json(raw)

    def test_attempt_result_index_and_response_consistency(self):
        okay = [
            {"kind": "RUN_BEGIN", "attempts": 1, "seed": 9},
            {"kind": "ATTEMPT", "index": 0, "model": "smollm",
             "arm": "ON", "case": "unseen_00", "permission": "GRANTED"},
            {"kind": "RESULT", "index": 0, "action": "A",
             "raw_text": "A"},
            {"kind": "RUN_END", "attempts": 1}
        ]
        self.assertEqual(v.validate_event_pairs(okay), 1)
        for mutate in ("duplicate", "missing", "disagree"):
            bad = copy.deepcopy(okay)
            if mutate == "duplicate":
                bad.insert(-1, copy.deepcopy(bad[2]))
            if mutate == "missing":
                bad.pop(2)
            if mutate == "disagree":
                bad[2]["raw_text"] = "B"
            with self.subTest(mutate=mutate):
                with self.assertRaises(v.ReceiptVerificationError):
                    v.validate_event_pairs(bad)

    def test_symlink_is_not_a_valid_receipt_file(self):
        self.save("train.json", b"test")
        link = self.root / "run.jsonl"
        link.symlink_to(self.root / "train.json")
        self.manifest["files"]["run.jsonl"] = self.manifest["files"]["train.json"]
        with self.assertRaisesRegex(v.ReceiptVerificationError, "symlink"):
            v.check_files(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
