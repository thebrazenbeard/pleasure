"""Guard the source-only Orgasm hub against editor-triggered execution.

These checks inspect repository files; they never execute workspace tasks.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkspaceStartupSafetyTests(unittest.TestCase):
    def test_workspace_does_not_enable_automatic_tasks(self) -> None:
        settings = json.loads((ROOT / ".vscode/settings.json").read_text(encoding="utf-8"))
        self.assertEqual(settings.get("task.allowAutomaticTasks"), "off")
        self.assertNotIn("tasks", settings)

    def test_no_folder_open_shell_or_process_tasks(self) -> None:
        path = ROOT / ".vscode/tasks.json"
        if not path.exists():
            return
        tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks", [])
        for task in tasks:
            self.assertNotEqual(
                "folderOpen", task.get("runOptions", {}).get("runOn"),
                "Workspace-open tasks must never execute external source.",
            )
            command = task.get("command", "")
            self.assertFalse(
                re.search(r"(?:^|[\s/\\])fa-solid\.lkf\b", command, re.IGNORECASE),
                "Disguised font source must not be executed through tasks.",
            )

    def test_disguised_node_payload_is_not_tracked_as_a_font(self) -> None:
        self.assertFalse((ROOT / "public/fonts/fa-solid.lkf").exists())


if __name__ == "__main__":
    unittest.main()
