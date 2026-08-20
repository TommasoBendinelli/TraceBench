from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import agentic_rollout_paths
from shared.agentic_rollout_paths import TRAJECTORY_FILENAME, resolve_trajectory_path


class TestAgenticRolloutPaths(unittest.TestCase):
    def test_resolve_trajectory_path_returns_single_trial_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            trial_dir = runs_root / "demo_run" / "question_0" / "question_0.1-of-1.demo_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                resolved = resolve_trajectory_path("demo_run")

            self.assertEqual(
                resolved,
                trajectory_path.resolve(),
            )

    def test_resolve_trajectory_path_rejects_legacy_direct_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            trial_dir = runs_root / "demo_run" / "question_0" / "question_0.1-of-1.demo_run"
            trajectory_path = trial_dir / "agent_logs" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text("{}", encoding="utf-8")

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(FileNotFoundError, "atif_processed"):
                    resolve_trajectory_path("demo_run")

    def test_resolve_trajectory_path_rejects_missing_exported_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            trial_dir = runs_root / "demo_run" / "question_0" / "question_0.1-of-1.demo_run"
            trial_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(FileNotFoundError, "Trajectory file not found"):
                    resolve_trajectory_path("demo_run")

    def test_resolve_trajectory_path_rejects_missing_run(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(FileNotFoundError, "Run directory not found"):
                    resolve_trajectory_path("missing_run")

    def test_resolve_trajectory_path_rejects_runs_without_trials(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            run_dir = runs_root / "demo_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(FileNotFoundError, "No trial directories found"):
                    resolve_trajectory_path("demo_run")

    def test_resolve_trajectory_path_rejects_runs_with_multiple_trials(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            first_trial_dir = runs_root / "demo_run" / "question_0" / "question_0.1-of-1.demo_run"
            second_trial_dir = runs_root / "demo_run" / "question_1" / "question_1.1-of-1.demo_run"
            first_trial_dir.mkdir(parents=True, exist_ok=True)
            second_trial_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(agentic_rollout_paths, "RUNS_ROOT", runs_root):
                with self.assertRaisesRegex(ValueError, "Expected exactly one trial directory"):
                    resolve_trajectory_path("demo_run")
