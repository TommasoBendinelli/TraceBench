from __future__ import annotations

import concurrent.futures
import contextlib
import datetime as py_dt
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from click.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.rollout import question_run_orchestrator as orchestrator


class _FixedDateTime(py_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 4, 1, 10, 11, 12, tzinfo=tz)


class TestQuestionRunOrchestrator(unittest.TestCase):
    def _write_questions(
        self,
        tsenv_root: Path,
        model_name: str,
        questions: dict[str, dict[str, object]],
    ) -> Path:
        model_dir = tsenv_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "questions": questions,
        }
        (model_dir / "questions.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return model_dir

    def _make_repo_layout(self) -> tuple[TemporaryDirectory[str], Path, Path, Path, Path]:
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root_dir = Path(tmp_dir.name)
        tsenv_root = root_dir / "tsENV_questions"
        tsenv_root.mkdir(parents=True, exist_ok=True)
        driver_script = root_dir / "workflows" / "rollout" / "run_single_question.py"
        driver_script.parent.mkdir(parents=True, exist_ok=True)
        driver_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        adapter_script = root_dir / "terminal-bench" / "adapters" / "tsENV" / "run_adapter.py"
        adapter_script.parent.mkdir(parents=True, exist_ok=True)
        adapter_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        return tmp_dir, root_dir, tsenv_root, driver_script, adapter_script

    def _invoke_main(
        self,
        *,
        root_dir: Path,
        tsenv_root: Path,
        driver_script: Path,
        args: list[str],
        run_cmd_side_effect=None,
        noise_analysis_side_effect=None,
    ):
        runner = CliRunner()
        patchers = [
            mock.patch.object(orchestrator, "ROOT", root_dir),
            mock.patch.object(orchestrator, "TSENV_QUESTIONS_ROOT", tsenv_root),
            mock.patch.object(orchestrator, "DRIVER_SCRIPT", driver_script),
            mock.patch.object(
                orchestrator,
                "ADAPTER_SCRIPT",
                root_dir / "terminal-bench" / "adapters" / "tsENV" / "run_adapter.py",
            ),
            mock.patch.object(
                orchestrator,
                "DEFAULT_CONFIGURATION_DIR",
                root_dir / "run_configurations",
            ),
        ]
        if run_cmd_side_effect is not None:
            patchers.append(
                mock.patch.object(orchestrator, "_run_cmd", side_effect=run_cmd_side_effect)
            )
        if noise_analysis_side_effect is not None:
            patchers.append(
                mock.patch.object(
                    orchestrator,
                    "_write_noise_analysis_for_materialized_question",
                    side_effect=noise_analysis_side_effect,
                )
            )
        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            return runner.invoke(orchestrator.main, args)

    def _write_scores_json(self, run_dir: Path, *, agent_run_id: str) -> Path:
        scores_path = (
            run_dir
            / "question_0"
            / f"question_0.1-of-1.{agent_run_id}"
            / "scores.json"
        )
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(
            json.dumps(
                {
                    "agent_run_id": agent_run_id,
                    "is_correct_format": True,
                    "final_metric_test": {
                        "average_top1_accuracy": 1.0,
                        "average_shortlist_score": 1.0,
                        "average_num_answers": 1.0,
                    },
                    "sample_results": {
                        "sample_a": {
                            "predictions": ["label"],
                            "top1_correct": True,
                            "shortlist_score": 1.0,
                            "num_answers": 1.0,
                            "sample_type": "test",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return scores_path

    def test_driver_cmd_uses_documented_contract(self) -> None:
        cmd = orchestrator._driver_cmd(
            model_name="BallDrop",
            question_slug="q_demo",
            agent_id="gpt_5_5_codex_high",
            run_id="run_demo",
        )

        self.assertEqual(
            cmd,
            [
                sys.executable,
                str(orchestrator.DRIVER_SCRIPT),
                "--model",
                "BallDrop",
                "--question-slug",
                "q_demo",
                "--agent-id",
                "gpt_5_5_codex_high",
                "--agentic-run-id",
                "run_demo",
            ],
        )

    def test_main_accepts_explicit_documented_name(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(tsenv_root, "BallDrop", {"q_alpha": {"recipe_info": {}}})
        plan_path = root_dir / "run_configurations" / "basalt.json"

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--question-slug",
                    "q_alpha",
                    "--model",
                    "BallDrop",
                    "--name",
                    "basalt",
                    "--dry-run",
                ],
                run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["name"], "basalt")
        self.assertEqual(payload["timestamp"], "2026-04-01__10-11-12")

    def test_main_rejects_unknown_agent_id(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_alpha": {
                    "recipe_info": {"row_slug": "row_a", "shot_slug": "shot_a"},
                }
            },
        )

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=[
                "--tasks-dir",
                str(tsenv_root),
                "--agent-id",
                "does_not_exist",
                "--question-slug",
                "q_alpha",
                "--model",
                "BallDrop",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown --agent-id", result.output)

    def test_main_requires_exactly_one_selector(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_alpha": {
                    "recipe_info": {"row_slug": "row_a", "shot_slug": "shot_a"},
                }
            },
        )

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=[
                "--tasks-dir",
                str(tsenv_root),
                "--agent-id",
                "gpt_5_5_codex_high",
                "--question-slug",
                "q_alpha",
                "--row-slug",
                "row_a",
                "--model",
                "BallDrop",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("exactly one", result.output)

    def test_main_writes_documented_json_plan_and_runs_driver(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_alpha": {
                    "question_text": {
                        "first_sentence": "First",
                        "model_description": "Model",
                        "shared_description": (
                            "Shared {questions.<question_slug>.question_text.label_choices_json}"
                        ),
                        "possible_interventions_parameter": "\"alpha\" and \"beta\"",
                        "label_choices_json": ["alpha", "beta", "no parameter change"],
                        "task_instruction": "Question: determine which parameter changed.\n\nData\n\nFormat\n\nGeneric",
                        "ordered_field_agent_prompt": [
                            "first_sentence",
                            "model_description",
                            "shared_description",
                            "task_instruction",
                        ],
                    },
                    "recipe_info": {
                        "row_slug": "row_a",
                        "shot_slug": "shot_a",
                        "question_seed": 17,
                    },
                }
            },
        )
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"
        seen_cmds: list[list[str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            self.assertTrue(plan_path.exists())
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runs"][0]["question_slug"], "q_alpha")
            self.assertEqual(payload["runs"][0]["model"], "BallDrop")
            self.assertEqual(payload["runs"][0]["agent_id"], "gpt_5_5_codex_high")
            self.assertEqual(
                payload["runs"][0]["question_schema"]["recipe_info"]["row_slug"],
                "row_a",
            )
            self.assertEqual(
                payload["runs"][0]["agent_prompt"],
                'First\n\nModel\n\nShared ["alpha", "beta", "no parameter change"]\n\nQuestion: determine which parameter changed.\n\nData\n\nFormat\n\nGeneric',
            )
            self.assertEqual(payload["runs"][0]["tag"], "DEBUG")
            self.assertEqual(payload["runs"][0]["question.recipe_info.row_slug"], "row_a")
            self.assertEqual(payload["runs"][0]["question.recipe_info.shot_slug"], "shot_a")
            self.assertEqual(payload["runs"][0]["question.recipe_info.question_seed"], 17)
            self.assertEqual(payload["name"], "2026-04-01__10-11-12")
            self.assertEqual(payload["timestamp"], "2026-04-01__10-11-12")
            self.assertEqual(
                sorted(payload.keys()),
                [
                    "models",
                    "name",
                    "number_of_runs_in_parallel_per_agent",
                    "questions_root",
                    "runs",
                    "timestamp",
                    "total_runs",
                ],
            )
            seen_cmds.append(cmd)
            self.assertEqual(env["TSENV_AGENTIC_ROLLOUT_TAG"], "DEBUG")
            self.assertEqual(
                env["TSENV_AGENTIC_CONFIGURATION_FILE_NAME"],
                "2026-04-01__10-11-12.json",
            )
            self.assertEqual(env["TRACEBENCH_QUESTIONS_ROOT"], str(tsenv_root.resolve()))
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--question-slug",
                    "q_alpha",
                    "--model",
                    "BallDrop",
                ],
                run_cmd_side_effect=_fake_run_cmd,
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(len(seen_cmds), 1)
        self.assertEqual(
            seen_cmds[0],
            [
                sys.executable,
                str(driver_script),
                "--model",
                "BallDrop",
                "--question-slug",
                "q_alpha",
                "--agent-id",
                "gpt_5_5_codex_high",
                "--agentic-run-id",
                "2026-04-01__10-11-12__gpt_5_5_codex_high__balldrop__q_alpha",
            ],
        )
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_runs"], 1)
        self.assertEqual(payload["name"], "2026-04-01__10-11-12")
        self.assertEqual(payload["timestamp"], "2026-04-01__10-11-12")
        self.assertEqual(payload["questions_root"], str(tsenv_root.resolve()))
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertEqual(
            payload["runs"][0]["path_to_the_run"],
            str((root_dir / "terminal-bench" / "runs" / "2026-04-01__10-11-12__gpt_5_5_codex_high__balldrop__q_alpha").resolve()),
        )

    def test_main_row_slug_filters_questions(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "TransmissionLine",
            {
                "q_one": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_a"}},
                "q_two": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_b"}},
                "q_three": {"recipe_info": {"row_slug": "row_other", "shot_slug": "shot_b"}},
            },
        )
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--row-slug",
                    "row_shared",
                    "--model",
                    "TransmissionLine",
                    "--dry-run",
                ],
                run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [run["question_slug"] for run in payload["runs"]],
            ["q_one", "q_two"],
        )
        self.assertEqual(payload["total_runs"], 2)
        self.assertEqual(payload["name"], "2026-04-01__10-11-12")
        self.assertEqual(payload["timestamp"], "2026-04-01__10-11-12")
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_PENDING)
        self.assertEqual(payload["runs"][0]["path_to_the_run"], "")

    def test_main_shot_slug_filters_across_multiple_models_with_shared_question_slugs(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        shared_questions = {
            "q_shared": {"recipe_info": {"row_slug": "row_a", "shot_slug": "shot_shared"}},
            "q_other": {"recipe_info": {"row_slug": "row_b", "shot_slug": "shot_other"}},
        }
        self._write_questions(tsenv_root, "BallDrop", shared_questions)
        self._write_questions(tsenv_root, "BounceBall", shared_questions)
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--shot-slug",
                    "shot_shared",
                    "--model",
                    "BallDrop",
                    "--model",
                    "BounceBall",
                    "--dry-run",
                ],
                run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [(run["model"], run["question_slug"]) for run in payload["runs"]],
            [
                ("BallDrop", "q_shared"),
                ("BounceBall", "q_shared"),
            ],
        )
        self.assertEqual(payload["total_runs"], 2)

    def test_main_question_slug_runs_every_matching_model(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        shared_questions = {
            "q_shared": {"recipe_info": {"row_slug": "row_a", "shot_slug": "shot_shared"}},
            "q_other": {"recipe_info": {"row_slug": "row_b", "shot_slug": "shot_other"}},
        }
        self._write_questions(tsenv_root, "BallDrop", shared_questions)
        self._write_questions(tsenv_root, "BounceBall", shared_questions)
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--question-slug",
                    "q_shared",
                    "--model",
                    "BallDrop",
                    "--model",
                    "BounceBall",
                    "--dry-run",
                ],
                run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [(run["model"], run["question_slug"]) for run in payload["runs"]],
            [
                ("BallDrop", "q_shared"),
                ("BounceBall", "q_shared"),
            ],
        )
        self.assertEqual(payload["total_runs"], 2)

    def test_main_question_slug_still_errors_when_missing_from_all_requested_models(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_alpha": {
                    "recipe_info": {"row_slug": "row_a", "shot_slug": "shot_a"},
                }
            },
        )
        self._write_questions(
            tsenv_root,
            "BounceBall",
            {
                "q_beta": {
                    "recipe_info": {"row_slug": "row_b", "shot_slug": "shot_b"},
                }
            },
        )

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=[
                "--tasks-dir",
                str(tsenv_root),
                "--agent-id",
                "gpt_5_5_codex_high",
                "--question-slug",
                "q_missing",
                "--model",
                "BallDrop",
                "--model",
                "BounceBall",
                "--dry-run",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown --question-slug 'q_missing'.", result.output)

    def test_main_accepts_space_separated_multi_value_flags(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        shared_questions = {
            "q_shared": {"recipe_info": {"row_slug": "row_a", "shot_slug": "shot_shared"}},
            "q_other": {"recipe_info": {"row_slug": "row_b", "shot_slug": "shot_other"}},
        }
        self._write_questions(tsenv_root, "BallDrop", shared_questions)
        self._write_questions(tsenv_root, "BounceBall", shared_questions)
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "gpt_5_5_low_codex_low",
                    "--shot-slug",
                    "shot_shared",
                    "--model",
                    "BallDrop",
                    "BounceBall",
                    "--dry-run",
                ],
                run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [(run["agent_id"], run["model"], run["question_slug"]) for run in payload["runs"]],
            [
                ("gpt_5_5_codex_high", "BallDrop", "q_shared"),
                ("gpt_5_5_codex_high", "BounceBall", "q_shared"),
                ("gpt_5_5_low_codex_low", "BallDrop", "q_shared"),
                ("gpt_5_5_low_codex_low", "BounceBall", "q_shared"),
            ],
        )
        self.assertEqual(payload["total_runs"], 4)

    def test_main_call_adapter_materializes_each_selected_question_once(self) -> None:
        _, root_dir, tsenv_root, driver_script, adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_one": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_a"}},
                "q_two": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_b"}},
            },
        )
        plan_path = root_dir / "run_configurations" / "2026-04-01__10-11-12.json"
        seen_cmds: list[list[str]] = []
        seen_noise_analysis: list[tuple[str, str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            seen_cmds.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        def _fake_noise_analysis(*, model_name: str, question_slug: str, questions_root: Path):
            self.assertEqual(questions_root, tsenv_root.resolve())
            seen_noise_analysis.append((model_name, question_slug))
            return (
                root_dir
                / "terminal-bench"
                / "tasks_runtime"
                / "manual"
                / model_name
                / question_slug
                / "question_0"
                / "noise_analysis.json"
            )

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "gpt_5_5_low_codex_low",
                    "--row-slug",
                    "row_shared",
                    "--model",
                    "BallDrop",
                    "--call-adapter",
                    "--dry-run",
                ],
                run_cmd_side_effect=_fake_run_cmd,
                noise_analysis_side_effect=_fake_noise_analysis,
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(
            seen_cmds,
            [
                [
                    sys.executable,
                    str(adapter_script),
                    str((tsenv_root / "BallDrop").resolve()),
                    "--output-dir",
                    str(root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_one"),
                    "--overwrite",
                    "--only-question-id",
                    "q_one",
                ],
                [
                    sys.executable,
                    str(adapter_script),
                    str((tsenv_root / "BallDrop").resolve()),
                    "--output-dir",
                    str(root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_two"),
                    "--overwrite",
                    "--only-question-id",
                    "q_two",
                ],
            ],
        )
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertNotIn("call_adapter", payload)
        self.assertEqual(payload["total_runs"], 4)
        self.assertEqual(
            seen_noise_analysis,
            [("BallDrop", "q_one"), ("BallDrop", "q_two")],
        )

    def test_main_call_adapter_ignores_existing_documented_tmp_cache(self) -> None:
        _, root_dir, tsenv_root, driver_script, adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {
                "q_one": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_a"}},
                "q_two": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_b"}},
            },
        )
        (root_dir / "tmp" / "q_one").mkdir(parents=True)
        (
            root_dir
            / "terminal-bench"
            / "tasks_runtime"
            / "manual"
            / "BallDrop"
            / "q_one"
            / "question_0"
        ).mkdir(parents=True)
        seen_cmds: list[list[str]] = []
        seen_noise_analysis: list[tuple[str, str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            seen_cmds.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        def _fake_noise_analysis(*, model_name: str, question_slug: str, questions_root: Path):
            self.assertEqual(questions_root, tsenv_root.resolve())
            seen_noise_analysis.append((model_name, question_slug))
            return (
                root_dir
                / "terminal-bench"
                / "tasks_runtime"
                / "manual"
                / model_name
                / question_slug
                / "question_0"
                / "noise_analysis.json"
            )

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--row-slug",
                    "row_shared",
                    "--model",
                    "BallDrop",
                    "--call-adapter",
                    "--dry-run",
                ],
                run_cmd_side_effect=_fake_run_cmd,
                noise_analysis_side_effect=_fake_noise_analysis,
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(
            seen_cmds,
            [
                [
                    sys.executable,
                    str(adapter_script),
                    str((tsenv_root / "BallDrop").resolve()),
                    "--output-dir",
                    str(root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_one"),
                    "--overwrite",
                    "--only-question-id",
                    "q_one",
                ],
                [
                    sys.executable,
                    str(adapter_script),
                    str((tsenv_root / "BallDrop").resolve()),
                    "--output-dir",
                    str(root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_two"),
                    "--overwrite",
                    "--only-question-id",
                    "q_two",
                ],
            ],
        )
        self.assertEqual(
            seen_noise_analysis,
            [("BallDrop", "q_one"), ("BallDrop", "q_two")],
        )

    def test_main_call_adapter_materializes_despite_stale_documented_tmp_cache(self) -> None:
        _, root_dir, tsenv_root, driver_script, adapter_script = self._make_repo_layout()
        self._write_questions(
            tsenv_root,
            "BallDrop",
            {"q_one": {"recipe_info": {"row_slug": "row_shared", "shot_slug": "shot_a"}}},
        )
        (root_dir / "tmp" / "q_one").mkdir(parents=True)
        seen_cmds: list[list[str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            seen_cmds.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(orchestrator.dt, "datetime", _FixedDateTime):
            result = self._invoke_main(
                root_dir=root_dir,
                tsenv_root=tsenv_root,
                driver_script=driver_script,
                args=[
                    "--tasks-dir",
                    str(tsenv_root),
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--question-slug",
                    "q_one",
                    "--model",
                    "BallDrop",
                    "--call-adapter",
                    "--dry-run",
                ],
                run_cmd_side_effect=_fake_run_cmd,
                noise_analysis_side_effect=lambda *, model_name, question_slug, questions_root: (
                    root_dir
                    / "terminal-bench"
                    / "tasks_runtime"
                    / "manual"
                    / model_name
                    / question_slug
                    / "question_0"
                    / "noise_analysis.json"
                ),
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(
            seen_cmds,
            [
                [
                    sys.executable,
                    str(adapter_script),
                    str((tsenv_root / "BallDrop").resolve()),
                    "--output-dir",
                    str(root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_one"),
                    "--overwrite",
                    "--only-question-id",
                    "q_one",
                ],
            ],
        )

    def test_main_resumes_existing_plan_and_skips_done_runs(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        plan_path = root_dir / "run_configurations" / "plan.json"
        plan_payload = {
            "plan_version": 1,
            "created_at": "2026-04-01T10:11:12",
            "tasks_dir": str(tsenv_root),
            "models": ["BallDrop"],
            "selection": {"kind": "question-slug", "values": ["q_done", "q_pending"]},
            "number_of_runs_in_parallel_per_agent": 1,
            "tag": "DEBUG",
            "runs": [
                {
                    "question_slug": "q_done",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_done",
                    "status": orchestrator.PLAN_STATUS_DONE,
                    "path_to_the_run": str(root_dir / "terminal-bench" / "runs" / "run_done"),
                    "tag": "DEBUG",
                    "agent_prompt": "done prompt",
                    "error": None,
                },
                {
                    "question_slug": "q_pending",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_pending",
                    "status": orchestrator.PLAN_STATUS_PENDING,
                    "path_to_the_run": str(root_dir / "terminal-bench" / "runs" / "run_pending"),
                    "tag": "DEBUG",
                    "agent_prompt": "pending prompt",
                    "question_schema": {"question_id": "q_pending", "version": 7},
                    "error": None,
                },
            ],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
        seen_cmds: list[list[str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            seen_cmds.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=["--resume", "plan.json"],
            run_cmd_side_effect=_fake_run_cmd,
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(len(seen_cmds), 1)
        self.assertIn("run_pending", seen_cmds[0])
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["total_runs"], 2)
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertEqual(payload["runs"][1]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertEqual(
            payload["runs"][1]["question_schema"],
            {"question_id": "q_pending", "version": 7},
        )

    def test_resume_path_resolution_accepts_paths_and_preserves_bare_filename_behavior(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            with (
                mock.patch.object(orchestrator, "ROOT", root_dir),
                mock.patch.object(
                    orchestrator,
                    "DEFAULT_CONFIGURATION_DIR",
                    root_dir / "run_configurations",
                ),
            ):
                self.assertEqual(
                    orchestrator._resolve_resume_plan_path("plan").resolve(),
                    (root_dir / "run_configurations" / "plan.json").resolve(),
                )
                self.assertEqual(
                    orchestrator._resolve_resume_plan_path("custom/plan").resolve(),
                    (root_dir / "custom" / "plan.json").resolve(),
                )

    def test_main_resumes_existing_plan_from_explicit_path(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        plan_path = root_dir / "custom_configs" / "plan.json"
        plan_payload = {
            "plan_version": 1,
            "created_at": "2026-04-01T10:11:12",
            "tasks_dir": str(tsenv_root),
            "models": ["BallDrop"],
            "selection": {"kind": "question-slug", "values": ["q_pending"]},
            "number_of_runs_in_parallel_per_agent": 1,
            "tag": "DEBUG",
            "runs": [
                {
                    "question_slug": "q_pending",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_pending",
                    "status": orchestrator.PLAN_STATUS_PENDING,
                    "path_to_the_run": str(root_dir / "terminal-bench" / "runs" / "run_pending"),
                    "tag": "DEBUG",
                    "agent_prompt": "pending prompt",
                    "error": None,
                },
            ],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
        seen_cmds: list[list[str]] = []

        def _fake_run_cmd(cmd: list[str], *, cwd: Path | None = None, env=None):
            seen_cmds.append(cmd)
            return mock.Mock(returncode=0, stdout="", stderr="")

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=["--resume", str(plan_path)],
            run_cmd_side_effect=_fake_run_cmd,
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(len(seen_cmds), 1)
        self.assertIn("run_pending", seen_cmds[0])

    def test_main_resume_rejects_extra_options_and_preserves_plan(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        plan_path = root_dir / "run_configurations" / "plan.json"
        errored_run_dir = root_dir / "terminal-bench" / "runs" / "run_error"
        errored_run_dir.mkdir(parents=True)
        plan_payload = {
            "plan_version": 1,
            "created_at": "2026-04-01T10:11:12",
            "tasks_dir": str(tsenv_root),
            "models": ["BallDrop"],
            "selection": {"kind": "question-slug", "values": ["q_error"]},
            "number_of_runs_in_parallel_per_agent": 1,
            "tag": "DEBUG",
            "runs": [
                {
                    "question_slug": "q_error",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_error",
                    "status": orchestrator.PLAN_STATUS_ERROR,
                    "path_to_the_run": str(errored_run_dir),
                    "tag": "DEBUG",
                    "agent_prompt": "error prompt",
                    "error": "failed",
                },
            ],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=["--resume", "plan.json", "--model", "BallDrop", "--dry-run"],
            run_cmd_side_effect=AssertionError("dry-run should not launch subprocesses"),
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("do not pass any other option except --heal", result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_ERROR)
        self.assertEqual(payload["runs"][0]["error"], "failed")
        self.assertTrue(errored_run_dir.exists())

    def test_main_resume_reconciles_error_run_with_valid_scores(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        plan_path = root_dir / "run_configurations" / "plan.json"
        errored_run_dir = root_dir / "terminal-bench" / "runs" / "run_error"
        self._write_scores_json(errored_run_dir, agent_run_id="run_error")
        plan_payload = {
            "plan_version": 1,
            "created_at": "2026-04-01T10:11:12",
            "tasks_dir": str(tsenv_root),
            "models": ["BallDrop"],
            "selection": {"kind": "question-slug", "values": ["q_error"]},
            "number_of_runs_in_parallel_per_agent": 1,
            "tag": "DEBUG",
            "runs": [
                {
                    "question_slug": "q_error",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_error",
                    "status": orchestrator.PLAN_STATUS_ERROR,
                    "path_to_the_run": str(errored_run_dir),
                    "tag": "DEBUG",
                    "agent_prompt": "error prompt",
                    "error": "format-only harness failure",
                },
            ],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")
        seen_cmds: list[list[str]] = []

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=["--resume", "plan.json"],
            run_cmd_side_effect=lambda cmd, **kwargs: seen_cmds.append(cmd),
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Reconciled 1 ERROR run(s) with valid scores.json", result.output)
        self.assertEqual(seen_cmds, [])
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertNotIn("error", payload["runs"][0])

    def test_main_heal_resets_error_runs_and_deletes_recorded_run_dir(self) -> None:
        _, root_dir, tsenv_root, driver_script, _adapter_script = self._make_repo_layout()
        plan_path = root_dir / "run_configurations" / "heal.json"
        runs_root = root_dir / "terminal-bench" / "runs"
        done_run_dir = runs_root / "run_done"
        errored_run_dir = runs_root / "run_error"
        pending_run_dir = runs_root / "run_pending"
        done_run_dir.mkdir(parents=True)
        errored_run_dir.mkdir(parents=True)
        (done_run_dir / "marker.txt").write_text("done", encoding="utf-8")
        (errored_run_dir / "marker.txt").write_text("error", encoding="utf-8")
        plan_payload = {
            "plan_version": 1,
            "created_at": "2026-04-01T10:11:12",
            "tasks_dir": str(tsenv_root),
            "models": ["BallDrop"],
            "selection": {"kind": "question-slug", "values": ["q_done", "q_error", "q_pending"]},
            "number_of_runs_in_parallel_per_agent": 1,
            "tag": "DEBUG",
            "runs": [
                {
                    "question_slug": "q_done",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_done",
                    "status": orchestrator.PLAN_STATUS_DONE,
                    "path_to_the_run": str(done_run_dir),
                    "tag": "DEBUG",
                    "agent_prompt": "done prompt",
                    "error": None,
                },
                {
                    "question_slug": "q_error",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_error",
                    "status": orchestrator.PLAN_STATUS_ERROR,
                    "path_to_the_run": str(errored_run_dir),
                    "tag": "DEBUG",
                    "agent_prompt": "error prompt",
                    "error": "failed",
                },
                {
                    "question_slug": "q_pending",
                    "model": "BallDrop",
                    "agent_id": "gpt_5_5_codex_high",
                    "agentic_run_id": "run_pending",
                    "status": orchestrator.PLAN_STATUS_PENDING,
                    "path_to_the_run": str(pending_run_dir),
                    "tag": "DEBUG",
                    "agent_prompt": "pending prompt",
                    "error": "left alone",
                },
            ],
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan_payload), encoding="utf-8")

        result = self._invoke_main(
            root_dir=root_dir,
            tsenv_root=tsenv_root,
            driver_script=driver_script,
            args=["--resume", "heal.json", "--heal"],
            run_cmd_side_effect=lambda cmd, **kwargs: mock.Mock(returncode=0, stdout="", stderr=""),
        )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertEqual(payload["runs"][1]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertNotIn("error", payload["runs"][1])
        self.assertEqual(payload["runs"][2]["status"], orchestrator.PLAN_STATUS_DONE)
        self.assertNotIn("error", payload["runs"][2])
        self.assertTrue(done_run_dir.exists())
        self.assertFalse(errored_run_dir.exists())

    def test_run_plan_uses_parallelism_per_agent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            plan_path = Path(tmp_dir) / "plan.json"
            plan = {
                "plan_version": 1,
                "created_at": "2026-04-01T10:11:12",
                "tasks_dir": str(Path(tmp_dir)),
                "models": ["BallDrop"],
                "selection": {"kind": "question-slug", "values": ["q_a", "q_b"]},
                "number_of_runs_in_parallel_per_agent": 2,
                "tag": "DEBUG",
                "runs": [
                    {
                        "question_slug": "q_a",
                        "model": "BallDrop",
                        "agent_id": "gpt_5_5_codex_high",
                        "agentic_run_id": "run_a",
                        "status": orchestrator.PLAN_STATUS_PENDING,
                        "path_to_the_run": str(Path(tmp_dir) / "terminal-bench" / "runs" / "run_a"),
                        "tag": "DEBUG",
                        "agent_prompt": "prompt a",
                        "error": None,
                    },
                    {
                        "question_slug": "q_b",
                        "model": "BallDrop",
                        "agent_id": "gpt_5_5_low_codex_low",
                        "agentic_run_id": "run_b",
                        "status": orchestrator.PLAN_STATUS_PENDING,
                        "path_to_the_run": str(Path(tmp_dir) / "terminal-bench" / "runs" / "run_b"),
                        "tag": "DEBUG",
                        "agent_prompt": "prompt b",
                        "error": None,
                    },
                ],
            }
            orchestrator._write_json_file(plan_path, plan)
            seen_max_workers: list[int] = []

            class _FakeExecutor:
                def __init__(self, max_workers: int):
                    seen_max_workers.append(max_workers)

                def submit(self, fn, *args, **kwargs):
                    future: concurrent.futures.Future[str | None] = concurrent.futures.Future()
                    try:
                        future.set_result(fn(*args, **kwargs))
                    except Exception as exc:  # pragma: no cover
                        future.set_exception(exc)
                    return future

                def shutdown(self, wait: bool = True) -> None:
                    return None

            with mock.patch.object(
                orchestrator.concurrent.futures,
                "ThreadPoolExecutor",
                _FakeExecutor,
            ):
                with mock.patch.object(
                    orchestrator,
                    "_run_cmd",
                    return_value=mock.Mock(returncode=0, stdout="", stderr=""),
                ):
                    orchestrator._run_plan(
                        plan,
                        plan_path=plan_path,
                        number_runs_in_parallel_per_agent=2,
                    )

        self.assertEqual(seen_max_workers, [2, 2])

    def test_run_plan_treats_failed_command_with_scores_as_done(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            run_dir = root_dir / "terminal-bench" / "runs" / "run_scored"
            self._write_scores_json(run_dir, agent_run_id="run_scored")
            plan_path = root_dir / "plan.json"
            plan = {
                "plan_version": 1,
                "created_at": "2026-04-01T10:11:12",
                "tasks_dir": str(root_dir),
                "models": ["BallDrop"],
                "selection": {"kind": "question-slug", "values": ["q_scored"]},
                "number_of_runs_in_parallel_per_agent": 1,
                "tag": "DEBUG",
                "runs": [
                    {
                        "question_slug": "q_scored",
                        "model": "BallDrop",
                        "agent_id": "gpt_5_5_codex_high",
                        "agentic_run_id": "run_scored",
                        "status": orchestrator.PLAN_STATUS_PENDING,
                        "path_to_the_run": str(run_dir),
                        "tag": "DEBUG",
                        "agent_prompt": "prompt",
                    },
                ],
            }
            orchestrator._write_json_file(plan_path, plan)

            with (
                mock.patch.object(orchestrator, "ROOT", root_dir),
                mock.patch.object(
                    orchestrator,
                    "_run_cmd",
                    side_effect=RuntimeError("driver failed after writing scores"),
                ),
            ):
                orchestrator._run_plan(
                    plan,
                    plan_path=plan_path,
                    number_runs_in_parallel_per_agent=1,
                )

            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["runs"][0]["status"], orchestrator.PLAN_STATUS_DONE)
            self.assertNotIn("error", payload["runs"][0])


if __name__ == "__main__":
    unittest.main()
