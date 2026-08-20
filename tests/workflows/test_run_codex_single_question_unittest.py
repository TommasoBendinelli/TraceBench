from __future__ import annotations

import contextlib
import json
import io
import os
import subprocess
import sys
import unittest
import datetime as py_dt
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.agentic_rollout_paths import LIGHT_TRAJECTORY_FILENAME, TRAJECTORY_FILENAME
from workflows.rollout import run_single_question as runner

NPM_PACKAGE = "@openai/codex@0.118"
INSTALL_COMMAND = f"npm install -g {NPM_PACKAGE}"


def _tsenv_payload(interventions: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "ground_truth_information": {
            "interventions": interventions,
        }
    }


class _FixedDateTime(py_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return cls(2026, 3, 28, 9, 10, 11, tzinfo=tz)


class TestRunCodexSingleQuestion(unittest.TestCase):
    def test_python_executable_preserves_virtual_environment_symlink(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base_python = root / "base" / "python3"
            base_python.parent.mkdir()
            base_python.write_bytes(b"")
            venv_python = root / ".venv" / "bin" / "python3"
            venv_python.parent.mkdir(parents=True)
            try:
                venv_python.symlink_to(base_python)
            except OSError as exc:
                self.skipTest(f"Symbolic links are unavailable: {exc}")

            with (
                mock.patch.dict(runner.os.environ, {}, clear=False),
                mock.patch.object(runner.sys, "executable", str(venv_python)),
            ):
                runner.os.environ.pop("TRACEBENCH_PYTHON", None)
                selected = runner._python_executable()

            self.assertEqual(selected, venv_python.absolute())
            self.assertNotEqual(selected, base_python.absolute())

    def test_python_executable_supports_relative_override_with_spaces(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            relative_python = Path("custom environment") / "python"
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                expected = Path.cwd() / relative_python
                with mock.patch.dict(
                    runner.os.environ,
                    {"TRACEBENCH_PYTHON": str(relative_python)},
                    clear=False,
                ):
                    selected = runner._python_executable()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(selected, expected)

    def test_python_executable_runs_inside_uv_environment_with_pandas(self) -> None:
        selected = runner._python_executable()

        proc = subprocess.run(
            [
                str(selected),
                "-c",
                (
                    "import pandas, sys; "
                    "assert sys.prefix != sys.base_prefix; "
                    "print(pandas.__version__)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.strip())

    def test_first_diff_for_sample_accepts_documented_list_and_legacy_scalar(self) -> None:
        payload = _tsenv_payload(
            {
                "list_sample": {"first_diff": [None, 3.0, 2.0]},
                "legacy_sample": {"first_diff": 4.0},
                "null_sample": {"first_diff": [None]},
            }
        )

        self.assertEqual(runner._first_diff_for_sample(payload, "list_sample"), 2.0)
        self.assertEqual(runner._first_diff_for_sample(payload, "legacy_sample"), 4.0)
        self.assertIsNone(runner._first_diff_for_sample(payload, "null_sample"))

    def test_normalized_noise_analysis_payload_rejects_malformed_snr_values(self) -> None:
        self.assertEqual(
            runner._normalized_noise_analysis_payload(
                {"global": [1.0, "-inf", None], "local": []}
            ),
            {"global": [1.0, "-inf", None], "local": []},
        )
        self.assertIsNone(
            runner._normalized_noise_analysis_payload({"global": ["not-snr"], "local": []})
        )

    def _prepare_single_question_run(
        self,
        *,
        eval_mode: str,
        postprocessor_returncodes: tuple[int, int, int] = (0, 0, 0),
        n_resolved: int = 1,
        n_unresolved: int = 0,
        configuration_file_name: str | None = None,
    ) -> tuple[int, list[list[str]], Path, str]:
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        root_dir = Path(tmp_dir.name)
        terminal_bench_dir = root_dir / "terminal-bench"
        dataset_dir = terminal_bench_dir / "tasks_runtime" / "manual" / "TransmissionLine" / "q_demo"
        question_root = dataset_dir / "question_0"
        question_root.mkdir(parents=True, exist_ok=True)
        (question_root / "train_samples").mkdir(parents=True, exist_ok=True)
        (question_root / "test_samples").mkdir(parents=True, exist_ok=True)
        (question_root / "train_samples" / "11111111111111111111111111111111.parquet").write_bytes(b"train")
        (question_root / "test_samples" / "22222222222222222222222222222222.parquet").write_bytes(b"test")
        payload_root = question_root / "agent_payload"
        (payload_root / "train_samples").mkdir(parents=True, exist_ok=True)
        (payload_root / "test_samples").mkdir(parents=True, exist_ok=True)
        (payload_root / "train_samples" / "11111111111111111111111111111111.parquet").write_bytes(b"train")
        (payload_root / "test_samples" / "22222222222222222222222222222222.parquet").write_bytes(b"test")
        (payload_root / "train_labels.json").write_text(
            json.dumps({"11111111111111111111111111111111.parquet": "alpha"}),
            encoding="utf-8",
        )
        (question_root / "scenario_info.json").write_text(
            json.dumps(
                {
                    "eval_mode": eval_mode,
                    "train_samples": ["train_samples/11111111111111111111111111111111.parquet"],
                    "test_samples": ["test_samples/22222222222222222222222222222222.parquet"],
                }
            ),
            encoding="utf-8",
        )

        run_dir = terminal_bench_dir / "runs" / "run_demo"
        trial_dir = run_dir / "question_0" / "question_0.1-of-1.run_demo"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "run_id": "run_demo",
                    "agent_name": "codex",
                    "model_name": "gpt-5.4",
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "n_resolved": n_resolved,
                    "n_unresolved": n_unresolved,
                    "accuracy": 1.0 if n_resolved else 0.0,
                    "results": [
                        {
                            "task_id": "question_0",
                            "trial_name": "question_0.1-of-1.run_demo",
                            "task_hash": "q_demo",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "results.json").write_text(
            json.dumps(
                {
                    "is_resolved": True,
                    "failure_mode": None,
                    "parser_results": {},
                }
            ),
            encoding="utf-8",
        )
        (trial_dir / "scenario_info.json").write_text(
            json.dumps({"question_id": "q_demo", "eval_mode": eval_mode}),
            encoding="utf-8",
        )
        (trial_dir / "agentic-final-response.json").write_text(
            json.dumps({"final_answer": "alpha"}),
            encoding="utf-8",
        )

        recorded_postprocessors: list[list[str]] = []

        def _fake_postprocessor(cmd: list[str], *, cwd: Path, label: str) -> subprocess.CompletedProcess:
            recorded_postprocessors.append(cmd)
            if str(root_dir / "workflows" / "rollout" / "evaluate_artifact.py") in cmd:
                (run_dir / "accuracy_summary.json").write_text(
                    json.dumps(
                        {
                            "evaluated_trials": 1,
                            "errored_trials": 0,
                            "total_evaluable_answers": 1,
                            "total_correct_answers": 1,
                            "batch_accuracy": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
                (trial_dir / "scores.json").write_text(
                    json.dumps(
                        {
                            "metrics": {"accuracy": 1.0},
                            "agent_answer": {"final_answer": "alpha"},
                            "ground_truth": {"answer": "alpha"},
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, postprocessor_returncodes[0], "", "")
            if str(root_dir / "workflows" / "rollout" / "export_atif_trajectory.py") in cmd:
                agent_logs_dir = trial_dir / "agent_logs"
                atif_processed_dir = agent_logs_dir / "atif_processed"
                atif_processed_dir.mkdir(parents=True, exist_ok=True)
                agent_logs_dir.mkdir(parents=True, exist_ok=True)
                (atif_processed_dir / TRAJECTORY_FILENAME).write_text(
                    json.dumps({"agent_id": "gpt_5_5_codex_high", "session_id": "run_demo", "steps": []}),
                    encoding="utf-8",
                )
                (atif_processed_dir / LIGHT_TRAJECTORY_FILENAME).write_text(
                    json.dumps({"agent_id": "gpt_5_5_codex_high", "session_id": "run_demo", "steps": []}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, postprocessor_returncodes[1], "", "")
            if str(root_dir / "workflows" / "trajectory" / "trajectory_evaluation_programmatic.py") in cmd:
                (run_dir / "trajectory_evaluation.json").write_text(
                    json.dumps({"agentic_run_id": "run_demo"}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, postprocessor_returncodes[2], "", "")
            raise AssertionError(f"Unexpected postprocessor command: {cmd}")

        stdout = io.StringIO()
        with mock.patch.object(runner, "ROOT_DIR", root_dir):
            with mock.patch.object(runner, "TERMINAL_BENCH_DIR", terminal_bench_dir):
                with mock.patch.object(runner, "PYTHON_EXE", Path(sys.executable)):
                    with mock.patch.object(runner, "_resolve_question_source", return_value=("TransmissionLine", root_dir / "tsENV_questions" / "TransmissionLine")):
                        with mock.patch.object(
                            runner,
                            "_load_question_payload",
                            return_value=(
                                _tsenv_payload(
                                    {
                                        "11111111111111111111111111111111": {
                                            "changed_parameter": "alpha"
                                        }
                                    }
                                ),
                                {"train_samples": ["dataframes/11111111111111111111111111111111.parquet"]},
                            ),
                        ):
                            with mock.patch.object(runner, "_materialize_dataset"):
                                with mock.patch.object(runner, "_validate_agent"):
                                    with mock.patch.object(runner, "_populate_api_key"):
                                        with mock.patch.object(runner, "dotenv_values"):
                                            with mock.patch.object(runner, "_tb_run_cmd", return_value=["tb"]):
                                                with mock.patch.object(runner, "_run_tb_with_timeout", return_value=0):
                                                    with mock.patch.object(
                                                        runner,
                                                        "_run_postprocessor",
                                                        side_effect=_fake_postprocessor,
                                                    ):
                                                        env_patch = {}
                                                        if configuration_file_name is not None:
                                                            env_patch[
                                                                "TSENV_AGENTIC_CONFIGURATION_FILE_NAME"
                                                            ] = configuration_file_name
                                                        with mock.patch.dict(
                                                            runner.os.environ,
                                                            env_patch,
                                                            clear=False,
                                                        ):
                                                            with contextlib.redirect_stdout(stdout):
                                                                exit_code = runner.main(
                                                                    [
                                                                        "--model",
                                                                        "TransmissionLine",
                                                                        "--question-slug",
                                                                        "q_demo",
                                                                        "--agent-id",
                                                                        "gpt_5_5_codex_high",
                                                                        "--agentic-run-id",
                                                                        "run_demo",
                                                                    ]
                                                                )

        return exit_code, recorded_postprocessors, run_dir, stdout.getvalue()

    def test_parser_rejects_legacy_positional_contract(self) -> None:
        parser = runner._build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["q_demo", "gpt_5_5_codex_high", "run_demo"])

    def test_parser_rejects_batch_size_flag(self) -> None:
        parser = runner._build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--model",
                    "TransmissionLine",
                    "--question-slug",
                    "q_demo",
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--agentic-run-id",
                    "run_demo",
                    "--batch-size",
                    "2",
                ]
            )

    def test_unknown_agent_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "unknown AGENT_ID"):
            runner.main(
                [
                    "--model",
                    "TransmissionLine",
                    "--question-slug",
                    "q_demo",
                    "--agent-id",
                    "codex",
                    "--agentic-run-id",
                    "run_demo",
                ]
            )

    def test_removed_keep_container_alias_is_rejected(self) -> None:
        parser = runner._build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--keep_contaienr",
                    "--model",
                    "TransmissionLine",
                    "--question-slug",
                    "q_demo",
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--agentic-run-id",
                    "run_demo",
                ]
            )

    def test_compute_run_id_uses_profile_id_and_dataset_model(self) -> None:
        with mock.patch.object(runner.dt, "datetime", _FixedDateTime):
            run_id = runner._compute_run_id(
                "gpt_5_5_codex_high",
                "BallDrop",
                "q_demo",
            )

        self.assertEqual(
            run_id,
            "2026-03-28__09-10-11__gpt_5_5_codex_high__balldrop__q_demo",
        )

    def test_export_materialized_samples_copies_files_and_writes_train_labels(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            question_root = root_dir / "materialized" / "question_0"
            payload_root = question_root / "agent_payload"
            train_a = payload_root / "train_samples" / "11111111111111111111111111111111.parquet"
            train_b = payload_root / "train_samples" / "22222222222222222222222222222222.parquet"
            test_a = payload_root / "test_samples" / "33333333333333333333333333333333.parquet"
            train_a.parent.mkdir(parents=True, exist_ok=True)
            test_a.parent.mkdir(parents=True, exist_ok=True)
            train_a.write_bytes(b"train-a")
            train_b.write_bytes(b"train-b")
            test_a.write_bytes(b"test-a")
            (payload_root / "train_labels.json").write_text(
                json.dumps(
                    {
                        "11111111111111111111111111111111.parquet": "alpha",
                        "22222222222222222222222222222222.parquet": "beta",
                    }
                ),
                encoding="utf-8",
            )
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "train_samples": [
                            "train_samples/11111111111111111111111111111111.parquet",
                            "train_samples/22222222222222222222222222222222.parquet",
                        ],
                        "test_samples": [
                            "test_samples/33333333333333333333333333333333.parquet"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            question = {
                "train_samples": [
                    "dataframes/11111111111111111111111111111111.parquet",
                    "dataframes/22222222222222222222222222222222.parquet",
                ]
            }
            payload = {
                "ground_truth_information": {
                    "interventions": {
                        "11111111111111111111111111111111": {"changed_parameter": "alpha"},
                        "22222222222222222222222222222222": {"changed_parameter": "beta"},
                    }
                }
            }

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                export_dir = runner._export_materialized_samples(
                    question_root=question_root,
                    question=question,
                    payload=payload,
                    question_slug="q_demo",
                    agentic_run_id="agentic_run_demo",
                )

            self.assertEqual(
                export_dir,
                root_dir / "tmp" / "agentic_run_demo" / "agent_payload",
            )
            self.assertEqual(
                (export_dir / "train_samples" / train_a.name).read_bytes(),
                b"train-a",
            )
            self.assertEqual(
                (export_dir / "train_samples" / train_b.name).read_bytes(),
                b"train-b",
            )
            self.assertEqual(
                (export_dir / "test_samples" / test_a.name).read_bytes(),
                b"test-a",
            )
            self.assertFalse((export_dir / "scenario_info.json").exists())
            self.assertEqual(
                json.loads((export_dir / "train_labels.json").read_text(encoding="utf-8")),
                {
                    "11111111111111111111111111111111.parquet": "alpha",
                    "22222222222222222222222222222222.parquet": "beta",
                },
            )

    def test_export_materialized_samples_skips_empty_train_labels_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            question_root = root_dir / "materialized" / "question_0"
            payload_root = question_root / "agent_payload"
            test_a = payload_root / "test_samples" / "33333333333333333333333333333333.parquet"
            test_a.parent.mkdir(parents=True, exist_ok=True)
            test_a.write_bytes(b"test-a")
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "train_samples": [],
                        "test_samples": [
                            "test_samples/33333333333333333333333333333333.parquet"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            question = {"train_samples": []}
            payload = {"ground_truth_information": {"interventions": {}}}

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                export_dir = runner._export_materialized_samples(
                    question_root=question_root,
                    question=question,
                    payload=payload,
                    question_slug="q_demo",
                    agentic_run_id="agentic_run_demo",
                )

            self.assertEqual(
                export_dir,
                root_dir / "tmp" / "agentic_run_demo" / "agent_payload",
            )
            self.assertEqual(
                (export_dir / "test_samples" / test_a.name).read_bytes(),
                b"test-a",
            )
            self.assertFalse((export_dir / "train_labels.json").exists())

    def test_export_materialized_samples_replaces_existing_run_tmp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            export_dir = root_dir / "tmp" / "agentic_run_demo" / "agent_payload"
            export_dir.mkdir(parents=True)
            (export_dir / "sentinel.txt").write_text("keep", encoding="utf-8")
            question_root = root_dir / "materialized" / "question_0"
            payload_root = question_root / "agent_payload"
            payload_root.mkdir(parents=True)
            (payload_root / "replacement.txt").write_text("new", encoding="utf-8")
            (question_root / "scenario_info.json").write_text(
                json.dumps({"train_samples": [], "test_samples": []}),
                encoding="utf-8",
            )

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                returned = runner._export_materialized_samples(
                    question_root=question_root,
                    question={"train_samples": []},
                    payload={"ground_truth_information": {"interventions": {}}},
                    question_slug="q_demo",
                    agentic_run_id="agentic_run_demo",
                )

            self.assertEqual(returned, export_dir)
            self.assertFalse((export_dir / "sentinel.txt").exists())
            self.assertEqual((export_dir / "replacement.txt").read_text(encoding="utf-8"), "new")

    def test_write_noise_analysis_uses_documented_task_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            question_root = root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_demo" / "question_0"
            question_root.mkdir(parents=True)
            model_dir = root_dir / "models" / "simulink" / "BallDrop"
            model_dir.mkdir(parents=True)
            (model_dir / "experiment_config.json").write_text(
                json.dumps(
                    {
                        "observable_signals": {
                            "signal_type": {
                                "Position": {"type": "continuous", "envelope_size": 9},
                                "Velocity": {"type": "continuous", "envelope_size": 9},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            scenario_info = {
                "train_samples": ["train_samples/11111111111111111111111111111111.parquet"],
                "test_samples": ["test_samples/22222222222222222222222222222222.parquet"],
            }
            payload = _tsenv_payload(
                {
                    "11111111111111111111111111111111": {"first_diff": 1.5},
                    "22222222222222222222222222222222": {"first_diff": 2.5},
                }
            )

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                with mock.patch.object(
                    runner,
                    "materialize",
                    side_effect=lambda sample_uuid, noise_level, seed, first_diff, tsenv_model_root: (
                        {"sample_uuid": sample_uuid, "noise_level": noise_level, "seed": seed},
                        {"global": [float(seed), float(first_diff)], "local": [float(seed + 1)]},
                    ),
                ):
                    with mock.patch.object(
                        runner,
                        "quantify_analysis",
                        side_effect=lambda clean_df, noisy_df, signal_type, first_diff, local_radius_rows: {
                            "global": [f"quant-global-{noisy_df['noise_level']}", first_diff],
                            "local": [f"quant-local-{noisy_df['noise_level']}", local_radius_rows],
                        },
                    ):
                        artifact_path = runner._write_noise_analysis_artifact(
                            question_root=question_root,
                            source_dir=root_dir / "tsENV_questions" / "BallDrop",
                            model_name="BallDrop",
                            question_slug="q_demo",
                            question={"recipe_info": {"noise_level": "high", "question_seed": 7}},
                            payload=payload,
                            scenario_info=scenario_info,
                        )

            self.assertEqual(artifact_path, question_root / "noise_analysis.json")
            self.assertEqual(
                json.loads(artifact_path.read_text(encoding="utf-8")),
                {
                    "test_samples/22222222222222222222222222222222.parquet": {
                        "global": [7.0, 2.5],
                        "local": [8.0],
                    },
                    "train_samples/11111111111111111111111111111111.parquet": {
                        "global": [7.0, 1.5],
                        "local": [8.0],
                    },
                },
            )

    def test_write_noise_analysis_for_materialized_question_backfills_adapter_output(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            question_root = root_dir / "terminal-bench" / "tasks_runtime" / "manual" / "BallDrop" / "q_demo" / "question_0"
            source_dir = root_dir / "tsENV_questions" / "BallDrop"
            question_root.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            (source_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "questions": {
                            "q_demo": {
                                "recipe_info": {
                                    "noise_level": "high",
                                    "question_seed": 7,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "train_samples": [],
                        "test_samples": ["test_samples/22222222222222222222222222222222.parquet"],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                runner,
                "materialize",
                side_effect=lambda sample_uuid, noise_level, seed, first_diff, tsenv_model_root: (
                    {
                        "sample_uuid": sample_uuid,
                        "noise_level": noise_level,
                        "seed": seed,
                        "root": str(tsenv_model_root),
                    },
                    {"global": [float(seed), 1.0], "local": [float(first_diff)]},
                ),
            ):
                with mock.patch.object(
                    runner,
                    "quantify_analysis",
                    side_effect=lambda clean_df, noisy_df, signal_type, first_diff, local_radius_rows: {
                        "global": [float(noisy_df["seed"]), noisy_df["noise_level"]],
                    },
                ):
                    artifact_path = runner.write_noise_analysis_for_materialized_question(
                        model_name="BallDrop",
                        question_id="q_demo",
                        repo_root=root_dir,
                    )

            self.assertEqual(artifact_path, question_root / "noise_analysis.json")
            self.assertEqual(
                json.loads(artifact_path.read_text(encoding="utf-8")),
                {
                    "test_samples/22222222222222222222222222222222.parquet": {
                        "global": [7.0, 1.0],
                        "local": [-1.0],
                    }
                },
            )

    def test_run_single_question_skips_materialization_when_task_folder_exists(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            terminal_bench_dir = root_dir / "terminal-bench"
            question_root = terminal_bench_dir / "tasks_runtime" / "manual" / "TransmissionLine" / "q_demo" / "question_0"
            question_root.mkdir(parents=True)
            source_question = {"train_samples": []}
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "eval_mode": "direct",
                        "train_samples": [],
                        "test_samples": [],
                        "question_schema": source_question,
                    }
                ),
                encoding="utf-8",
            )
            run_dir = terminal_bench_dir / "runs" / "run_demo"
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.run_demo"
            trial_dir.mkdir(parents=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"run_id": "run_demo", "agent_name": "codex", "model_name": "gpt-5.4"}),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "n_resolved": 1,
                        "n_unresolved": 0,
                        "accuracy": 1.0,
                        "results": [
                            {
                                "task_id": "question_0",
                                "trial_name": "question_0.1-of-1.run_demo",
                                "task_hash": "q_demo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "results.json").write_text(
                json.dumps({"is_resolved": True, "failure_mode": None, "parser_results": {}}),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps({"question_id": "q_demo", "eval_mode": "direct"}),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"final_answer": "alpha"}),
                encoding="utf-8",
            )

            materialize_dataset = mock.Mock()
            noise_analysis = mock.Mock()
            export_dir = root_dir / "tmp" / "run_demo" / "agent_payload"
            export_materialized = mock.Mock(return_value=export_dir)
            captured_env: dict[str, str] = {}
            stdout = io.StringIO()
            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                with mock.patch.object(runner, "TERMINAL_BENCH_DIR", terminal_bench_dir):
                    with mock.patch.object(runner, "PYTHON_EXE", Path(sys.executable)):
                        with mock.patch.object(
                            runner,
                            "_resolve_question_source",
                            return_value=("TransmissionLine", root_dir / "tsENV_questions" / "TransmissionLine"),
                        ):
                            with mock.patch.object(
                                runner,
                                "_load_question_payload",
                                return_value=({"questions": {}}, source_question),
                            ):
                                with mock.patch.object(runner, "_materialize_dataset", materialize_dataset):
                                    with mock.patch.object(runner, "_write_noise_analysis_artifact", noise_analysis):
                                        with mock.patch.object(runner, "_export_materialized_samples", export_materialized):
                                            with mock.patch.object(runner, "_validate_agent"):
                                                with mock.patch.object(runner, "_populate_api_key"):
                                                    with mock.patch.object(runner, "dotenv_values"):
                                                        with mock.patch.object(runner, "_tb_run_cmd", return_value=["tb"]):
                                                            with mock.patch.object(
                                                                runner,
                                                                "_run_tb_with_timeout",
                                                                side_effect=lambda cmd, *, cwd, env, timeout_sec: (
                                                                    captured_env.update(env) or 0
                                                                ),
                                                            ):
                                                                with mock.patch.object(
                                                                    runner,
                                                                    "_run_postprocessor",
                                                                    return_value=subprocess.CompletedProcess(["post"], 0, "", ""),
                                                                ):
                                                                    with contextlib.redirect_stdout(stdout):
                                                                        exit_code = runner.run_single_question(
                                                                            question_id="q_demo",
                                                                            agent_profile_id="gpt_5_5_codex_high",
                                                                            run_id="run_demo",
                                                                            model_name="TransmissionLine",
                                                                        )

            self.assertEqual(exit_code, 0)
            self.assertIn("Reusing materialized single-question dataset", stdout.getvalue())
            materialize_dataset.assert_not_called()
            noise_analysis.assert_called_once()
            export_materialized.assert_called_once()
            self.assertEqual(captured_env.get("T_BENCH_TASK_PAYLOAD_PATH"), str(export_dir))

    def test_run_single_question_rematerializes_when_question_schema_mismatches(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            terminal_bench_dir = root_dir / "terminal-bench"
            question_root = terminal_bench_dir / "tasks_runtime" / "manual" / "TransmissionLine" / "q_demo" / "question_0"
            question_root.mkdir(parents=True)
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "eval_mode": "direct",
                        "train_samples": [],
                        "test_samples": [],
                        "question_schema": {"version": 1},
                    }
                ),
                encoding="utf-8",
            )
            captured_env: dict[str, str] = {}
            source_question = {"version": 2, "train_samples": []}

            def _materialize_dataset(_source_dir: Path, _dataset_dir: Path, _question_id: str) -> None:
                (question_root / "scenario_info.json").write_text(
                    json.dumps(
                        {
                            "eval_mode": "direct",
                            "train_samples": [],
                            "test_samples": [],
                            "question_schema": source_question,
                        }
                    ),
                    encoding="utf-8",
                )

            run_dir = terminal_bench_dir / "runs" / "run_demo"
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.run_demo"
            trial_dir.mkdir(parents=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"run_id": "run_demo", "agent_name": "codex", "model_name": "gpt-5.4"}),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "n_resolved": 1,
                        "n_unresolved": 0,
                        "accuracy": 1.0,
                        "results": [
                            {
                                "task_id": "question_0",
                                "trial_name": "question_0.1-of-1.run_demo",
                                "task_hash": "q_demo",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "results.json").write_text(
                json.dumps({"is_resolved": True, "failure_mode": None, "parser_results": {}}),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps({"question_id": "q_demo", "eval_mode": "direct"}),
                encoding="utf-8",
            )

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                with mock.patch.object(runner, "TERMINAL_BENCH_DIR", terminal_bench_dir):
                    with mock.patch.object(runner, "PYTHON_EXE", Path(sys.executable)):
                        with mock.patch.object(
                            runner,
                            "_resolve_question_source",
                            return_value=("TransmissionLine", root_dir / "tsENV_questions" / "TransmissionLine"),
                        ):
                            with mock.patch.object(
                                runner,
                                "_load_question_payload",
                                return_value=({"questions": {}}, source_question),
                            ):
                                with mock.patch.object(runner, "_materialize_dataset", side_effect=_materialize_dataset) as materialize_dataset:
                                    with mock.patch.object(runner, "_write_noise_analysis_artifact"):
                                        export_dir = root_dir / "tmp" / "run_demo" / "agent_payload"
                                        with mock.patch.object(runner, "_export_materialized_samples", return_value=export_dir):
                                            with mock.patch.object(runner, "_validate_agent"):
                                                with mock.patch.object(runner, "_populate_api_key"):
                                                    with mock.patch.object(runner, "dotenv_values"):
                                                        with mock.patch.object(runner, "_tb_run_cmd", return_value=["tb"]):
                                                            with mock.patch.object(
                                                                runner,
                                                                "_run_tb_with_timeout",
                                                                side_effect=lambda cmd, *, cwd, env, timeout_sec: (
                                                                    captured_env.update(env) or 0
                                                                ),
                                                            ):
                                                                with mock.patch.object(
                                                                    runner,
                                                                    "_run_postprocessor",
                                                                    return_value=subprocess.CompletedProcess(["post"], 0, "", ""),
                                                                ):
                                                                    exit_code = runner.run_single_question(
                                                                        question_id="q_demo",
                                                                        agent_profile_id="gpt_5_5_codex_high",
                                                                        run_id="run_demo",
                                                                        model_name="TransmissionLine",
                                                                    )

            self.assertEqual(exit_code, 0)
            materialize_dataset.assert_called_once()
            self.assertEqual(captured_env.get("T_BENCH_TASK_PAYLOAD_PATH"), str(export_dir))

    def test_run_single_question_ignores_stale_tmp_cache_without_task_folder(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            terminal_bench_dir = root_dir / "terminal-bench"
            (root_dir / "tmp" / "q_demo").mkdir(parents=True)
            question_root = terminal_bench_dir / "tasks_runtime" / "manual" / "TransmissionLine" / "q_demo" / "question_0"
            captured_env: dict[str, str] = {}

            def _materialize_dataset(_source_dir: Path, _dataset_dir: Path, _question_id: str) -> None:
                question_root.mkdir(parents=True)
                (question_root / "scenario_info.json").write_text(
                    json.dumps({"eval_mode": "direct", "train_samples": [], "test_samples": []}),
                    encoding="utf-8",
                )

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                with mock.patch.object(runner, "TERMINAL_BENCH_DIR", terminal_bench_dir):
                    with mock.patch.object(runner, "PYTHON_EXE", Path(sys.executable)):
                        with mock.patch.object(
                            runner,
                            "_resolve_question_source",
                            return_value=("TransmissionLine", root_dir / "tsENV_questions" / "TransmissionLine"),
                        ):
                            with mock.patch.object(
                                runner,
                                "_load_question_payload",
                                return_value=({"questions": {}}, {"train_samples": []}),
                            ):
                                with mock.patch.object(runner, "_materialize_dataset", side_effect=_materialize_dataset) as materialize_dataset:
                                    with mock.patch.object(runner, "_write_noise_analysis_artifact"):
                                        export_dir = root_dir / "tmp" / "run_demo" / "agent_payload"
                                        with mock.patch.object(runner, "_export_materialized_samples", return_value=export_dir) as export_materialized:
                                            with mock.patch.object(runner, "_validate_agent"):
                                                with mock.patch.object(runner, "_populate_api_key"):
                                                    with mock.patch.object(runner, "dotenv_values"):
                                                        with mock.patch.object(runner, "_tb_run_cmd", return_value=["tb"]):
                                                            with mock.patch.object(
                                                                runner,
                                                                "_run_tb_with_timeout",
                                                                side_effect=lambda cmd, *, cwd, env, timeout_sec: (
                                                                    captured_env.update(env) or 124
                                                                ),
                                                            ):
                                                                with mock.patch.object(runner, "_cleanup_run_resources"):
                                                                    exit_code = runner.run_single_question(
                                                                        question_id="q_demo",
                                                                        agent_profile_id="gpt_5_5_codex_high",
                                                                        run_id="run_demo",
                                                                        model_name="TransmissionLine",
                                                                    )

            self.assertEqual(exit_code, 124)
            materialize_dataset.assert_called_once()
            export_materialized.assert_called_once()
            self.assertEqual(captured_env.get("T_BENCH_TASK_PAYLOAD_PATH"), str(export_dir))

    def test_validate_explicit_run_id_rejects_path_traversal(self) -> None:
        with self.assertRaises(SystemExit):
            runner._validate_explicit_run_id("../bad")

    def test_ground_truth_context_uses_documented_recipe_field(self) -> None:
        self.assertTrue(
            runner._uses_ground_truth_context(
                {"recipe_info": {"textual_context": "ground_truth"}}
            )
        )
        self.assertFalse(
            runner._uses_ground_truth_context(
                {"recipe_info": {"textual_context": "high"}}
            )
        )

    def test_api_key_uses_process_environment_before_dotenv(self) -> None:
        env = {"OPENAI_API_KEY": "from-process"}

        with mock.patch.object(runner, "dotenv_values") as read_dotenv:
            runner._populate_api_key(env, "OPENAI_API_KEY")

        self.assertEqual(env["OPENAI_API_KEY"], "from-process")
        read_dotenv.assert_not_called()

    def test_api_key_loads_selected_key_from_repo_dotenv(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            (root_dir / ".env").write_text(
                "GEMINI_API_KEY=from-dotenv\n",
                encoding="utf-8",
            )
            env: dict[str, str] = {}

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                runner._populate_api_key(env, "GEMINI_API_KEY")

        self.assertEqual(env["GEMINI_API_KEY"], "from-dotenv")

    def test_blank_process_api_key_falls_back_to_dotenv(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root_dir = Path(tmp_dir)
            (root_dir / ".env").write_text(
                "ANTHROPIC_API_KEY=from-dotenv\n",
                encoding="utf-8",
            )
            env = {"ANTHROPIC_API_KEY": "  "}

            with mock.patch.object(runner, "ROOT_DIR", root_dir):
                runner._populate_api_key(env, "ANTHROPIC_API_KEY")

        self.assertEqual(env["ANTHROPIC_API_KEY"], "from-dotenv")

    def test_missing_api_key_names_selected_variable_without_leaking_values(self) -> None:
        legacy_credentials = {
            "CODEX_AUTH_JSON_BASE64": "legacy-codex-secret",
            "GEMINI_CLI_AUTH": "legacy-gemini-secret",
            "CLAUDE_CODE_AUTH_DIR": "legacy-claude-secret",
            "OPENROUTER_API": "legacy-openrouter-secret",
        }
        selected_keys = (
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
        )

        with TemporaryDirectory() as tmp_dir:
            with mock.patch.object(runner, "ROOT_DIR", Path(tmp_dir)):
                for selected_key in selected_keys:
                    with self.subTest(selected_key=selected_key):
                        with self.assertRaises(SystemExit) as raised:
                            runner._populate_api_key(
                                dict(legacy_credentials),
                                selected_key,
                            )
                        message = str(raised.exception)
                        self.assertIn(selected_key, message)
                        for secret in legacy_credentials.values():
                            self.assertNotIn(secret, message)

    def test_api_key_variable_name_must_be_configured(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no API key environment variable"):
            runner._populate_api_key({}, "  ")

    def test_tb_run_cmd_appends_reasoning_agent_kwarg(self) -> None:
        cmd = runner._tb_run_cmd(
            run_id="run_demo",
            model="BallDrop",
            question_id="q_demo",
            agent="codex",
            agent_model="gpt-5.4",
            installation_command=INSTALL_COMMAND,
            reasoning="high",
            timeout_sec=120,
            keep_container=False,
        )

        self.assertEqual(cmd[:4], ["uv", "run", "tb", "run"])
        self.assertIn("--agent-kwarg", cmd)
        self.assertIn(f"installation_command={INSTALL_COMMAND}", cmd)
        self.assertIn("reasoning=high", cmd)
        self.assertNotIn("--cleanup-images", cmd)

    def test_tb_run_cmd_only_requests_image_cleanup_for_single_run(self) -> None:
        cmd = runner._tb_run_cmd(
            run_id="run_demo",
            model="BallDrop",
            question_id="q_demo",
            agent="codex",
            agent_model="gpt-5.4",
            installation_command=None,
            reasoning=None,
            timeout_sec=120,
            keep_container=False,
            cleanup_images=True,
        )

        self.assertIn("--cleanup", cmd)
        self.assertIn("--cleanup-images", cmd)

    def test_tb_run_cmd_does_not_cleanup_images_when_container_is_kept(self) -> None:
        cmd = runner._tb_run_cmd(
            run_id="run_demo",
            model="BallDrop",
            question_id="q_demo",
            agent="codex",
            agent_model="gpt-5.4",
            installation_command=None,
            reasoning=None,
            timeout_sec=120,
            keep_container=True,
            cleanup_images=True,
        )

        self.assertIn("--no-cleanup", cmd)
        self.assertNotIn("--cleanup-images", cmd)

    def test_main_always_runs_evaluate_accuracy_and_analyze_trajectory_for_code(self) -> None:
        exit_code, postprocessors, run_dir, stdout = self._prepare_single_question_run(eval_mode="code")

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            postprocessors,
            [
                [
                    str(Path(sys.executable)),
                    str(run_dir.parents[2] / "workflows" / "rollout" / "evaluate_artifact.py"),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "rollout"
                        / "export_atif_trajectory.py"
                    ),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "trajectory"
                        / "trajectory_evaluation_programmatic.py"
                    ),
                    "run_demo",
                ],
            ],
        )
        run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(run_metadata["agent_id"], "gpt_5_5_codex_high")
        self.assertEqual(run_metadata["reasoning"], "high")
        scenario = json.loads(
            (
                run_dir
                / "question_0"
                / "question_0.1-of-1.run_demo"
                / "scenario_info.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(scenario["agent_id"], "gpt_5_5_codex_high")
        self.assertEqual(scenario["tag"], "DEBUG")
        self.assertTrue((run_dir / "accuracy_summary.json").exists())
        self.assertTrue((run_dir / "question_0" / "agentic-final-response.json").exists())
        self.assertTrue((run_dir / "question_0" / "agent_logs" / "rollout").exists())
        self.assertTrue(
            (
                run_dir
                / "question_0"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            ).exists()
        )
        self.assertIn("atif_trajectory.json:", stdout)
        self.assertIn("atif_trajectory_light.json:", stdout)
        self.assertIn(str(run_dir / "question_0" / "question_0.1-of-1.run_demo" / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME), stdout)
        self.assertIn(str(run_dir / "question_0" / "question_0.1-of-1.run_demo" / "agent_logs" / "atif_processed" / LIGHT_TRAJECTORY_FILENAME), stdout)
        self.assertTrue((run_dir / "trajectory_evaluation.json").exists())

    def test_main_records_configuration_file_name_from_orchestrator_env(self) -> None:
        exit_code, _postprocessors, run_dir, _stdout = self._prepare_single_question_run(
            eval_mode="code",
            configuration_file_name="plan.json",
        )

        self.assertEqual(exit_code, 0)
        trial_scenario = json.loads(
            (
                run_dir
                / "question_0"
                / "question_0.1-of-1.run_demo"
                / "scenario_info.json"
            ).read_text(encoding="utf-8")
        )
        copied_scenario = json.loads(
            (run_dir / "question_0" / "scenario_info.json").read_text(encoding="utf-8")
        )
        self.assertEqual(trial_scenario["configuration_file_name"], "plan.json")
        self.assertEqual(copied_scenario["configuration_file_name"], "plan.json")

    def test_main_always_runs_evaluate_accuracy_and_analyze_trajectory_for_non_code(self) -> None:
        exit_code, postprocessors, run_dir, _stdout = self._prepare_single_question_run(
            eval_mode="direct",
            postprocessor_returncodes=(1, 1, 1),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            postprocessors,
            [
                [
                    str(Path(sys.executable)),
                    str(run_dir.parents[2] / "workflows" / "rollout" / "evaluate_artifact.py"),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "rollout"
                        / "export_atif_trajectory.py"
                    ),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "trajectory"
                        / "trajectory_evaluation_programmatic.py"
                    ),
                    "run_demo",
                ],
            ],
        )
        self.assertTrue((run_dir / "accuracy_summary.json").exists())
        self.assertTrue((run_dir / "trajectory_evaluation.json").exists())

    def test_main_runs_open_ended_accuracy_without_judge_helper(self) -> None:
        exit_code, postprocessors, run_dir, _stdout = self._prepare_single_question_run(
            eval_mode="open-ended",
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            postprocessors,
            [
                [
                    str(Path(sys.executable)),
                    str(run_dir.parents[2] / "workflows" / "rollout" / "evaluate_artifact.py"),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "rollout"
                        / "export_atif_trajectory.py"
                    ),
                    "run_demo",
                ],
                [
                    str(Path(sys.executable)),
                    str(
                        run_dir.parents[2]
                        / "workflows"
                        / "trajectory"
                        / "trajectory_evaluation_programmatic.py"
                    ),
                    "run_demo",
                ],
            ],
        )
        run_metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(run_metadata["agent_id"], "gpt_5_5_codex_high")
        self.assertEqual(run_metadata["reasoning"], "high")

    def test_main_does_not_run_postprocessors_for_unresolved_run(self) -> None:
        exit_code, postprocessors, run_dir, _stdout = self._prepare_single_question_run(
            eval_mode="code",
            n_resolved=0,
            n_unresolved=1,
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(postprocessors, [])
        self.assertFalse((run_dir / "accuracy_summary.json").exists())

    def test_materialize_batches_splits_question_directory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "dataset"
            question_root = dataset_dir / "question_0"
            test_dir = question_root / "test_samples"
            train_dir = question_root / "train_samples"
            payload_dir = question_root / "agent_payload"
            test_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)
            payload_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "a.parquet").write_bytes(b"a")
            (test_dir / "b.parquet").write_bytes(b"b")
            (train_dir / "train.parquet").write_bytes(b"train")
            (question_root / "train_labels.json").write_text('{"train":"alpha"}', encoding="utf-8")
            (question_root / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "test_samples": [
                            "test_samples/a.parquet",
                            "test_samples/b.parquet",
                        ],
                        "test_samples_source_paths": [
                            "source/a.parquet",
                            "source/b.parquet",
                        ],
                        "test_sample_labels": {
                            "source/a.parquet": "alpha",
                            "source/b.parquet": "beta",
                        },
                    }
                ),
                encoding="utf-8",
            )

            count, returned_root = runner._materialize_batches(dataset_dir, batch_size=1)

            self.assertEqual(count, 2)
            self.assertEqual(returned_root, question_root)
            second_root = dataset_dir / "question_1"
            second_scenario = json.loads((second_root / "scenario_info.json").read_text(encoding="utf-8"))
            self.assertEqual(second_scenario["test_samples"], ["test_samples/b.parquet"])
            self.assertTrue((second_root / "agent_payload" / "test_samples" / "b.parquet").exists())
            self.assertFalse((second_root / "agent_payload" / "test_samples" / "a.parquet").exists())


if __name__ == "__main__":
    unittest.main()
