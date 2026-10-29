"""Keep current Pleasure navigation distinct from frozen Orgasm V1 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


class RepositoryRebrandTests(unittest.TestCase):
    def test_generic_contract_suite_is_present(self):
        path = ROOT / "tests/test_generic_model_boundary.py"
        self.assertTrue(path.is_file())
        import ast
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "GenericModelBoundaryTests"]
        self.assertEqual(len(classes), 1)
        methods = [node.name for node in classes[0].body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertIn("test_full_source_contract", methods)
        suite = unittest.defaultTestLoader.discover(
            start_dir=str(ROOT / "tests"), pattern="test_generic_model_boundary.py",
            top_level_dir=str(ROOT / "tests"))
        self.assertGreaterEqual(suite.countTestCases(), 7)
        result = unittest.TestResult()
        suite.run(result)
        self.assertGreaterEqual(result.testsRun, 7)
        self.assertEqual(result.skipped, [])
        self.assertTrue(result.wasSuccessful(), repr(result.errors + result.failures))

    def test_readme_repository_map_does_not_link_to_missing_local_files(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        referenced = re.findall(r"^- `((?:docs|snapshots)/[^`]+)`", readme, re.MULTILINE)
        self.assertTrue(referenced)
        for relative in referenced:
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file(), f"missing mapped file: {relative}")

    def test_current_repository_name_and_branch_are_pleasure(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
        integration = (ROOT / "docs/INTEGRATION.md").read_text(encoding="utf-8")
        self.assertIn("\n# Pleasure\n", readme)
        self.assertIn("thebrazenbeard/pleasure", readme)
        self.assertIn("pleasure/main", readme)
        self.assertTrue(status.startswith("# Pleasure status"))
        self.assertIn("pleasure/main", status)
        self.assertIn("thebrazenbeard/pleasure", integration)
        for text in (readme, status, integration):
            self.assertNotIn("`orgasm/main`", text)

    def test_legacy_contract_identifiers_still_point_to_frozen_objects(self) -> None:
        provenance = json.loads((ROOT / "PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["schema"], "VERA_ORGASM_REPOSITORY_PROVENANCE_V1")
        paths = [
            (provenance["contract_source"]["mirror_path"],
             provenance["contract_source"]["blob"]),
            (provenance["runtime_source"]["binding"]["mirror_path"],
             provenance["runtime_source"]["binding"]["blob"]),
            (provenance["runtime_source"]["authority_context_boundary"]["mirror_path"],
             provenance["runtime_source"]["authority_context_boundary"]["blob"]),
        ]
        for relative, expected in paths:
            with self.subTest(path=relative):
                self.assertEqual(git_blob_sha(ROOT / relative), expected)


if __name__ == "__main__":
    unittest.main()
