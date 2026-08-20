from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestRolloutDocumentedEntrypoints(unittest.TestCase):
    def test_public_release_scripts_are_explicit(self) -> None:
        self.assertEqual(
            sorted(path.name for path in (REPO_ROOT / "scripts").glob("*.py")),
            ["__init__.py", "download_release.py", "reproduce_results.py"],
        )

    def test_documented_trajectory_export_entrypoint_is_singular(self) -> None:
        self.assertTrue((REPO_ROOT / "workflows/trajectory/export_atif_trajectory.py").is_file())
        self.assertFalse((REPO_ROOT / "workflows/trajectory/export_atif_trajectories.py").exists())

    def test_documented_trajectory_evaluation_entrypoint_is_singular(self) -> None:
        self.assertTrue(
            (REPO_ROOT / "workflows/trajectory/trajectory_evaluation_programmatic.py").is_file()
        )

    def test_documented_extensionless_cleanup_entrypoint_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "workflows/rollout/cleanup_tb_question_resources").is_file())

    def test_rollout_entrypoints_do_not_delegate_back_to_scripts(self) -> None:
        for relpath in (
            "workflows/rollout/run_single_question.py",
            "workflows/rollout/evaluate_artifact.py",
        ):
            text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
            self.assertNotIn("from scripts", text)
            self.assertNotIn("scripts.", text)
