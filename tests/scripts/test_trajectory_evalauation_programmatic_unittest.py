from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import click
from click.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.agentic_rollout_paths import TRAJECTORY_FILENAME
from workflows.trajectory import trajectory_evaluation_programmatic as trajectory_evalauation_programmatic

SHELL_SUBCOMMANDS = ["python", "cat"]


class TestTrajectoryEvalauationProgrammatic(unittest.TestCase):
    def _empty_cat_subcommand_analysis(self) -> dict[str, object]:
        return {
            "total_invocations": 0,
            "pattern": {
                "heredoc_write": {
                    "create_files": {
                        "python_files": {},
                        "other": {},
                    },
                },
                "file_concat_to_stdout": 0,
                "file_concat_to_file": 0,
                "pipe_into_cat": 0,
                "other": 0,
            },
            "followup_command_counts": {},
            "execution_after_creation": {
                "executed_in_same_shell_snippet": 0,
                "not_executed_in_same_shell_snippet": 0,
            },
            "heredoc_delimiters": {},
        }

    def _empty_python_subcommand_analysis(self) -> dict[str, object]:
        return {
            "total_invocations": 0,
            "failed_invocations": 0,
            "total_stdout_lines": 0,
            "total_stderr_lines": 0,
            "total_output_bytes": 0,
            "invocation_modes": {
                "dash_c": 0,
                "heredoc": 0,
                "script_path": 0,
                "other": 0,
            },
            "lines_of_code": {
                "script_total": 0,
            },
            "library_calls": {},
            "detailed": {},
        }

    def _mock_agent_profile_lookup(self, agent_id: str) -> dict[str, object]:
        if agent_id == "gemini_3_1_pro_high":
            return {
                "agent_id": "gemini_3_1_pro_high",
                "cost": {
                    "input_token": 2.5,
                    "cached_token": 0.625,
                    "completion_token": 8.5,
                },
                "available_tools": {
                    "exec_command": {"parse_method": False, "subcommands": []},
                    "read_file": {"parse_method": False, "subcommands": []},
                    "run_shell_command": {
                        "parse_method": "tree_sitter_bash",
                        "subcommands": SHELL_SUBCOMMANDS,
                    },
                    "write_file": {"parse_method": False, "subcommands": []},
                },
            }
        if agent_id == "claude_4_opus_high":
            return {
                "agent_id": "claude_4_opus_high",
                "cost": {
                    "input_token": 5.0,
                    "cached_token": 0.5,
                    "completion_token": 25.0,
                },
                "available_tools": {
                    "Bash": {
                        "parse_method": "tree_sitter_bash",
                        "subcommands": SHELL_SUBCOMMANDS,
                    },
                    "Read": {"parse_method": "file_path", "subcommands": []},
                    "Write": {"parse_method": False, "subcommands": []},
                    "Edit": {"parse_method": "file_path", "subcommands": []},
                    "TodoWrite": {"parse_method": False, "subcommands": []},
                    "ToolSearch": {"parse_method": False, "subcommands": []},
                    "Agent": {"parse_method": False, "subcommands": []},
                },
            }
        if agent_id == "minimax-m2.7":
            return {
                "agent_id": "minimax-m2.7",
                "cost": {
                    "input_token": 0.3,
                    "cached_token": 0.03,
                    "completion_token": 1.2,
                },
                "available_tools": {},
            }
        raise click.ClickException(f"Unknown agent_id: {agent_id}")

    def _write_run_metadata(
        self,
        run_dir: Path,
        *,
        agent_id: str | None = "gemini_3_1_pro_high",
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        (run_dir / "run_metadata.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_analyze_artifacts_counts_only_agent_visible_plots(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            trial_dir = Path(tmp_dir) / "run" / "question_0" / "question_0.1-of-1.run"
            trajectory_path = (
                trial_dir
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            artifacts_dir = trial_dir / "artifacts"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "used.png").write_bytes(b"png")
            (artifacts_dir / "unused.png").write_bytes(b"png")

            analysis = trajectory_evalauation_programmatic._analyze_artifacts(  # noqa: SLF001
                trajectory_path,
                {
                    "steps": [
                        {
                            "source": "agent",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call_1",
                                    "function_name": "view_image",
                                    "arguments": {"path": "artifacts/used.png"},
                                }
                            ],
                            "observation": [
                                {"source_call_id": "call_1", "content": "opened used.png"}
                            ],
                        }
                    ]
                },
            )

            self.assertEqual(analysis["plots"], ["used.png"])

    def test_analyze_artifacts_ignores_failed_plot_views(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            trial_dir = Path(tmp_dir) / "run" / "question_0" / "question_0.1-of-1.run"
            trajectory_path = (
                trial_dir
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            artifacts_dir = trial_dir / "artifacts"
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "missing.png").write_bytes(b"png")

            analysis = trajectory_evalauation_programmatic._analyze_artifacts(  # noqa: SLF001
                trajectory_path,
                {
                    "steps": [
                        {
                            "source": "agent",
                            "tool_calls": [
                                {
                                    "tool_call_id": "call_1",
                                    "function_name": "view_image",
                                    "arguments": {"path": "artifacts/missing.png"},
                                }
                            ],
                            "observation": [
                                {
                                    "source_call_id": "call_1",
                                    "content": "No such file or directory",
                                }
                            ],
                        }
                    ]
                },
            )

            self.assertEqual(analysis["plots"], [])

    def test_main_returns_requested_metrics(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_cost_usd": 1.25,
                            "total_prompt_tokens": 0,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_steps": 7,
                        },
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {"tool_call_id": "call_1", "function_name": "exec_command"},
                                    {"tool_call_id": "call_2", "function_name": "read_file"},
                                    {
                                        "tool_call_id": "call_3",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print(1)\""},
                                    },
                                    {"tool_call_id": "call_4", "function_name": "write_file"},
                                ],
                                "observation": [
                                    {"source_call_id": "call_1", "content": "ok"},
                                    {"source_call_id": "call_2", "content": "ok"},
                                    {"source_call_id": "call_3", "content": "1\n"},
                                    {"source_call_id": "call_4", "content": "ok"},
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            payload = json.loads(result.output)
            self.assertEqual(payload["agentic_run_id"], "traj_run")
            self.assertEqual(payload["agent_id"], "gemini_3_1_pro_high")
            self.assertEqual(payload["cost_usd"], 0.0)
            self.assertEqual(
                json.loads((run_dir / "trajectory_evaluation.json").read_text(encoding="utf-8")),
                payload,
            )
            self.assertEqual(
                payload["number_of_interactions"],
                {
                    "user": 1,
                    "system": 0,
                    "agent": {
                        "tool_call_steps": 4,
                        "tool_observations": 4,
                        "steps": 1,
                    },
                },
            )
            self.assertEqual(
                payload["tool_calls"],
                {
                    "exec_command": {"count": 1},
                    "read_file": {"count": 1},
                    "run_shell_command": {
                        "count": 1,
                        "commands": {
                            "python": {
                                "count": 1,
                                "call_unique": 1,
                            },
                        },
                    },
                    "write_file": {"count": 1},
                },
            )
            self.assertEqual(
                payload["artifact_analysis"],
                {
                    "python_scripts": [],
                    "json_files": [],
                    "plots": [],
                },
            )
            self.assertEqual(
                payload["tokens"],
                {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "avg_prompt_tokens_per_iteration": 0.0,
                    "avg_completion_tokens_per_iteration": 0.0,
                },
            )
            self.assertEqual(
                payload["subcommand_analysis"],
                {
                    "python": {
                        "total_invocations": 1,
                        "failed_invocations": 0,
                        "total_stdout_lines": 1,
                        "total_stderr_lines": 0,
                        "total_output_bytes": 2,
                        "invocation_modes": {
                            "dash_c": 1,
                            "heredoc": 0,
                            "script_path": 0,
                            "other": 0,
                        },
                        "lines_of_code": {
                            "script_total": 0,
                        },
                        "library_calls": {},
                        "detailed": {
                            "dash_c_1": {
                                "executions": 1,
                                "lines_of_code": 1,
                                "library_calls": {},
                                "total_stdout_lines": 1,
                                "total_stderr_lines": 0,
                                "total_output_bytes": 2,
                            }
                        },
                    },
                },
            )
            self.assertNotIn(
                "inline_total",
                payload["subcommand_analysis"]["python"]["lines_of_code"],
            )
            self.assertNotIn(
                "inline_by_invocation",
                payload["subcommand_analysis"]["python"]["lines_of_code"],
            )
            self.assertNotIn(
                "script_by_path",
                payload["subcommand_analysis"]["python"]["lines_of_code"],
            )
            self.assertNotIn("modules_called", payload["subcommand_analysis"]["python"])
            self.assertEqual(payload["atif_trajectory_path"], str(trajectory_path.resolve()))
            self.assertNotIn(
                "description",
                payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"],
            )

    def test_main_accepts_run_path(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "external_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "agent_id": "gemini_3_1_pro_high",
                        "run_id": "metadata_run_id",
                    }
                ),
                encoding="utf-8",
            )
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.metadata_run_id"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_prompt_tokens": 0,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_steps": 1,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                trajectory_evalauation_programmatic,
                "_agentic_profile_by_id",
                side_effect=self._mock_agent_profile_lookup,
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["--path", str(run_dir)],
                )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["agentic_run_id"], "metadata_run_id")
        self.assertEqual(payload["atif_trajectory_path"], str(trajectory_path.resolve()))

    def test_main_rejects_path_and_run_id_misuse(self) -> None:
        runner = CliRunner()

        result = runner.invoke(trajectory_evalauation_programmatic.main, [])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Pass exactly one of <agentic_run_id> or --path.", result.output)

        result = runner.invoke(
            trajectory_evalauation_programmatic.main,
            ["traj_run", "--path", "/tmp"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Pass exactly one of <agentic_run_id> or --path.", result.output)

    def test_main_requires_tree_sitter_for_shell_parsed_tools(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_cost_usd": 0.5,
                            "total_prompt_tokens": 0,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_steps": 1,
                        },
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "echo ok"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "ok\n"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_load_tree_sitter_bash_parser",
                    return_value=None,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Shell parsing requires tree_sitter and tree_sitter_bash", result.output)

    def test_main_requires_total_steps_for_token_averages(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_cost_usd": 0.5,
                            "total_prompt_tokens": 12,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 6,
                        },
                        "steps": [
                            {"source": "user"},
                            {"source": "agent"},
                            {"source": "agent"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("final_metrics.total_steps must be a positive integer", result.output)

    def test_main_ignores_missing_or_invalid_tool_calls(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": {"function_name": "exec_command"},
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {"function_name": ""},
                                    {},
                                    "not-an-object",
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": ""},
                                    },
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {},
                                    },
                                ],
                                "observation": [
                                    {"source_call_id": "call_1", "content": ""},
                                    {"source_call_id": "call_2", "content": ""},
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            payload = json.loads(result.output)
            self.assertEqual(
                payload["number_of_interactions"],
                {
                    "user": 1,
                    "system": 0,
                    "agent": {
                        "tool_call_steps": 2,
                        "tool_observations": 2,
                        "steps": 2,
                    },
                },
            )
            self.assertEqual(
                payload["tool_calls"],
                {
                    "exec_command": {"count": 0},
                    "read_file": {"count": 0},
                    "run_shell_command": {"count": 0, "commands": {}},
                    "write_file": {"count": 0},
                },
            )
            self.assertEqual(payload["subcommand_analysis"], {})

    def test_main_parses_shell_commands_by_depth(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {
                                            "command": "echo hi | grep h && wc -c"
                                        },
                                    },
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {
                                            "command": "echo $(python -c 'print(1)')"
                                        },
                                    },
                                ],
                                "observation": [
                                    {"source_call_id": "call_1", "content": "1\n"},
                                    {"source_call_id": "call_2", "content": "1\n"},
                                ],
                            },
                            {"source": "user"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            payload = json.loads(result.output)
            self.assertEqual(
                payload["tool_calls"],
                {
                    "exec_command": {"count": 0},
                    "read_file": {"count": 0},
                    "run_shell_command": {
                        "count": 5,
                        "commands": {
                            "echo": {
                                "count": 2,
                                "call_unique": 2,
                            },
                            "grep": {
                                "count": 1,
                                "call_unique": 1,
                            },
                            "python": {
                                "count": 1,
                                "call_unique": 1,
                            },
                            "wc": {
                                "count": 1,
                                "call_unique": 1,
                            },
                        },
                    },
                    "write_file": {"count": 0},
                },
            )
            self.assertNotIn("cat", payload["subcommand_analysis"])
            self.assertEqual(
                payload["subcommand_analysis"]["python"],
                {
                    "total_invocations": 1,
                    "failed_invocations": 0,
                    "total_stdout_lines": 1,
                    "total_stderr_lines": 0,
                    "total_output_bytes": 2,
                    "invocation_modes": {
                        "dash_c": 1,
                        "heredoc": 0,
                        "script_path": 0,
                        "other": 0,
                    },
                    "lines_of_code": {
                        "script_total": 0,
                    },
                    "library_calls": {},
                    "detailed": {
                        "dash_c_1": {
                            "executions": 1,
                            "lines_of_code": 1,
                            "library_calls": {},
                            "total_stdout_lines": 1,
                            "total_stderr_lines": 0,
                            "total_output_bytes": 2,
                        }
                    },
                },
            )
            self.assertEqual(
                payload["number_of_interactions"],
                {
                    "user": 1,
                    "system": 0,
                    "agent": {
                        "tool_call_steps": 2,
                        "tool_observations": 2,
                        "steps": 1,
                    },
                },
            )

    def test_main_counts_failed_python_invocations_from_observation_text(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"raise RuntimeError('x')\""},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": (
                                                "Traceback (most recent call last):\n"
                                                "  File \"<string>\", line 1, in <module>\n"
                                                "RuntimeError: x\n"
                                            ),
                                        }
                                    ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_invocations"], 1)
        self.assertEqual(payload["subcommand_analysis"]["python"]["failed_invocations"], 1)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stdout_lines"], 1)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stderr_lines"], 2)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_output_bytes"], 90)

    def test_main_counts_one_failed_python_invocation_when_only_one_fails_in_shell_snippet(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {
                                            "command": (
                                                "python -c \"print(1)\"\n"
                                                "python -c \"raise RuntimeError('x')\""
                                            )
                                        },
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": (
                                                "1\n"
                                                "Traceback (most recent call last):\n"
                                                "  File \"<string>\", line 1, in <module>\n"
                                                "RuntimeError: x\n"
                                            ),
                                        }
                                    ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            payload["tool_calls"],
            {
                "exec_command": {"count": 0},
                "read_file": {"count": 0},
                "run_shell_command": {
                    "count": 2,
                    "commands": {
                        "python": {
                            "count": 2,
                            "call_unique": 1,
                        },
                    },
                },
                "write_file": {"count": 0},
            },
        )
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_invocations"], 2)
        self.assertEqual(payload["subcommand_analysis"]["python"]["failed_invocations"], 1)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stdout_lines"], 2)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stderr_lines"], 2)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_output_bytes"], 92)
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"],
            {
                "dash_c_1": {
                    "executions": 1,
                    "lines_of_code": 1,
                    "library_calls": {},
                    "total_stdout_lines": 0,
                    "total_stderr_lines": 0,
                    "total_output_bytes": 0,
                },
                "dash_c_2": {
                    "executions": 1,
                    "lines_of_code": 1,
                    "library_calls": {},
                    "total_stdout_lines": 0,
                    "total_stderr_lines": 0,
                    "total_output_bytes": 0,
                },
            },
        )

    def test_main_counts_python_output_volume_from_successful_observation_text(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            observation_text = "hello\nworld\n"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('hello'); print('world')\""},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": observation_text,
                                        }
                                    ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["subcommand_analysis"]["python"]["failed_invocations"], 0)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stdout_lines"], 2)
        self.assertEqual(payload["subcommand_analysis"]["python"]["total_stderr_lines"], 0)
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["total_output_bytes"],
            len(observation_text.encode("utf-8")),
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"],
            {
                "dash_c_1": {
                    "executions": 1,
                    "lines_of_code": 1,
                    "library_calls": {},
                    "total_stdout_lines": 2,
                    "total_stderr_lines": 0,
                    "total_output_bytes": len(observation_text.encode("utf-8")),
                }
            },
        )
        self.assertNotIn("modules_called", payload["subcommand_analysis"]["python"])
        self.assertNotIn("by_script", payload["subcommand_analysis"]["python"])

    def test_main_includes_heredoc_python_runs_in_detailed(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            observation_text = "x\n"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {
                                            "command": (
                                                "python <<'PY'\n"
                                                "import json\n"
                                                "json.dumps({'x': 1})\n"
                                                "print('x')\n"
                                                "PY\n"
                                            )
                                        },
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": observation_text,
                                        }
                                    ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"],
            {
                "heredoc_1": {
                    "executions": 1,
                    "lines_of_code": 3,
                    "library_calls": {
                        "json": 1,
                    },
                    "total_stdout_lines": 1,
                    "total_stderr_lines": 0,
                    "total_output_bytes": len(observation_text.encode("utf-8")),
                }
            },
        )

    def test_main_aggregates_python_stats_by_script_target(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text(
                "import json\nfrom pathlib import Path\nprint(Path('x'))\n",
                encoding="utf-8",
            )
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": "ok\n",
                                        }
                                    ],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_2",
                                            "content": (
                                                "Traceback (most recent call last):\n"
                                                "RuntimeError: boom\n"
                                            ),
                                        }
                                    ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"],
            {
                "analyze.py": {
                    "executions": 2,
                    "lines_of_code": 3,
                    "library_calls": {
                        "pathlib": 2,
                    },
                    "total_stdout_lines": 1,
                    "total_stderr_lines": 2,
                    "total_output_bytes": 57,
                }
            },
        )

    def test_main_keeps_unresolved_python_script_target_in_detailed(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python missing.py"},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": "missing\n",
                                        }
                                    ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["missing.py"],
            {
                "executions": 1,
                "lines_of_code": None,
                "library_calls": {},
                "total_stdout_lines": 1,
                "total_stderr_lines": 0,
                "total_output_bytes": len("missing\n".encode("utf-8")),
            },
        )
        self.assertNotIn(
            "inline_total",
            payload["subcommand_analysis"]["python"]["lines_of_code"],
        )
        self.assertNotIn(
            "inline_by_invocation",
            payload["subcommand_analysis"]["python"]["lines_of_code"],
        )
        self.assertNotIn(
            "script_by_path",
            payload["subcommand_analysis"]["python"]["lines_of_code"],
        )

    def test_main_preserves_python_by_script_first_seen_order(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "test_rule.py").write_text("print('first')\n", encoding="utf-8")
            (artifacts_dir / "analyze.py").write_text("print('second')\n", encoding="utf-8")
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python test_rule.py"},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_1",
                                            "content": "first\n",
                                        }
                                    ],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_2",
                                            "content": "inline\n",
                                        }
                                    ],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_3",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [
                                        {
                                            "source_call_id": "call_3",
                                            "content": "second\n",
                                        }
                                    ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            list(payload["subcommand_analysis"]["python"]["detailed"].keys()),
            ["test_rule.py", "dash_c_1", "analyze.py"],
        )

    def test_main_rejects_undocumented_verbose_option(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text(
                "import numpy as np\nimport pandas as pd\nnp.abs([-1])\nnp.log([1])\npd.read_parquet('x.parquet')\n",
                encoding="utf-8",
            )
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "ok\n"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run", "--verbose"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such option", result.output)
        self.assertIn("--verbose", result.output)

    def test_main_llm_adds_descriptions_to_detailed_entries_in_order(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text("print('script')\n", encoding="utf-8")
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "script\n"}],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_2", "content": "inline\n"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            codex_calls: list[str] = []
            tqdm_calls: list[dict[str, object]] = []

            def _fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                self.assertEqual(
                    cmd,
                    ["codex", "exec", "--json", "--model", "gpt-5.3-codex", "--", "-"],
                )
                codex_calls.append(str(kwargs.get("input") or ""))
                outputs = [
                    json.dumps(
                        {
                            "message": json.dumps(
                                {
                                    "description": "Plots script sample output and saves figures",
                                    "number_of_semantically_relevant_numbers": 3,
                                    "intent_class": "exploring",
                                    "specific_subtype": 5,
                                }
                            )
                        }
                    ),
                    json.dumps(
                        {
                            "message": json.dumps(
                                {
                                    "description": "Prints inline sample output",
                                    "number_of_semantically_relevant_numbers": 1,
                                    "intent_class": "exploiting",
                                    "specific_subtype": -1,
                                }
                            )
                        }
                    ),
                ]
                return subprocess.CompletedProcess(cmd, 0, outputs[len(codex_calls) - 1], "")

            def _fake_tqdm(iterable: object, **kwargs: object) -> object:
                tqdm_calls.append(dict(kwargs))
                return iterable

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_fake_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=_fake_tqdm),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

            raw_root = trajectory_path.parent / "codex_description_raw"
            self.assertTrue(raw_root.is_dir())
            first_dir = raw_root / "001_analyze.py"
            second_dir = raw_root / "002_dash_c_1"
            self.assertTrue(first_dir.is_dir())
            self.assertTrue(second_dir.is_dir())
            self.assertIn("Entry key: analyze.py", (first_dir / "prompt.txt").read_text(encoding="utf-8"))
            self.assertEqual(
                (first_dir / "stdout.txt").read_text(encoding="utf-8"),
                json.dumps(
                    {
                        "message": json.dumps(
                            {
                                "description": "Plots script sample output and saves figures",
                                "number_of_semantically_relevant_numbers": 3,
                                "intent_class": "exploring",
                                "specific_subtype": 5,
                            }
                        )
                    }
                ),
            )
            self.assertEqual((first_dir / "stderr.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(
                json.loads((first_dir / "metadata.json").read_text(encoding="utf-8")),
                {
                    "entry_key": "analyze.py",
                    "description": "Plots script sample output and saves figures",
                    "number_of_semantically_relevant_numbers": 3,
                    "intent_class": "exploring",
                    "specific_subtype": 5,
                    "returncode": 0,
                    "codex_command": ["codex", "exec", "--json", "--model", "gpt-5.3-codex", "--", "-"],
                    "executions": 1,
                },
            )
            first_prompt = (first_dir / "prompt.txt").read_text(encoding="utf-8")
            self.assertIn("Return only a JSON object with exactly these keys:", first_prompt)
            self.assertIn(
                "description, number_of_semantically_relevant_numbers, intent_class, specific_subtype",
                first_prompt,
            )
            self.assertIn("Executions: 1", first_prompt)
            self.assertIn("Calls:\n- python analyze.py", first_prompt)
            self.assertIn("Observed output:\n- script", first_prompt)
            self.assertNotIn("Code:\n", first_prompt)
            self.assertIn("Entry key: dash_c_1", (second_dir / "prompt.txt").read_text(encoding="utf-8"))
            second_prompt = (second_dir / "prompt.txt").read_text(encoding="utf-8")
            self.assertIn("Executions: 1", second_prompt)
            self.assertIn("Calls:\n- python -c \"print('inline')\"", second_prompt)
            self.assertIn("Observed output:\n- inline", second_prompt)
            self.assertNotIn("Code:\n", second_prompt)

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        detailed = payload["subcommand_analysis"]["python"]["detailed"]
        self.assertEqual(
            detailed["analyze.py"]["description"],
            "Plots script sample output and saves figures",
        )
        self.assertEqual(detailed["analyze.py"]["number_of_semantically_relevant_numbers"], 3)
        self.assertEqual(detailed["analyze.py"]["intent_class"], "exploring")
        self.assertEqual(detailed["analyze.py"]["specific_subtype"], 5)
        self.assertEqual(detailed["dash_c_1"]["description"], "Prints inline sample output")
        self.assertEqual(detailed["dash_c_1"]["number_of_semantically_relevant_numbers"], 1)
        self.assertEqual(detailed["dash_c_1"]["intent_class"], "exploiting")
        self.assertEqual(detailed["dash_c_1"]["specific_subtype"], -1)
        self.assertEqual(len(codex_calls), 2)
        self.assertIn("Entry key: analyze.py", codex_calls[0])
        self.assertIn("Entry key: dash_c_1", codex_calls[1])
        self.assertIn("Return only a JSON object with exactly these keys:", codex_calls[0])
        self.assertIn(
            "description, number_of_semantically_relevant_numbers, intent_class, specific_subtype",
            codex_calls[0],
        )
        self.assertIn("Executions: 1", codex_calls[0])
        self.assertIn("Calls:\n- python analyze.py", codex_calls[0])
        self.assertIn("Observed output:\n- script", codex_calls[0])
        self.assertIn("Executions: 1", codex_calls[1])
        self.assertIn("Calls:\n- python -c \"print('inline')\"", codex_calls[1])
        self.assertIn("Observed output:\n- inline", codex_calls[1])
        self.assertNotIn("Code:\n", codex_calls[0])
        self.assertNotIn("Code:\n", codex_calls[1])
        self.assertEqual(len(tqdm_calls), 1)
        self.assertEqual(tqdm_calls[0]["desc"], "Codex descriptions")
        self.assertEqual(tqdm_calls[0]["unit"], "entry")

    def test_main_llm_calls_codex_per_invocation_for_repeated_entry_key(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text("print('script')\n", encoding="utf-8")
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "first output\n"}],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_2", "content": "second output\n"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            codex_calls: list[str] = []

            def _fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                codex_calls.append(str(kwargs.get("input") or ""))
                payloads = [
                    {
                        "description": "First analyze.py execution",
                        "number_of_semantically_relevant_numbers": 1,
                        "intent_class": "exploring",
                        "specific_subtype": 2,
                    },
                    {
                        "description": "Second analyze.py execution",
                        "number_of_semantically_relevant_numbers": 2,
                        "intent_class": "exploiting",
                        "specific_subtype": -1,
                    },
                ]
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    json.dumps({"message": json.dumps(payloads[len(codex_calls) - 1])}),
                    "",
                )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_fake_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

            raw_root = trajectory_path.parent / "codex_description_raw"
            self.assertTrue((raw_root / "001_analyze.py").is_dir())
            self.assertTrue((raw_root / "002_analyze.py").is_dir())

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(len(codex_calls), 2)
        self.assertIn("Calls:\n- python analyze.py", codex_calls[0])
        self.assertIn("Observed output:\n- first output", codex_calls[0])
        self.assertNotIn("second output", codex_calls[0])
        self.assertIn("Calls:\n- python analyze.py", codex_calls[1])
        self.assertIn("Observed output:\n- second output", codex_calls[1])
        self.assertNotIn("first output", codex_calls[1])
        payload = json.loads(result.output)
        detailed = payload["subcommand_analysis"]["python"]["detailed"]["analyze.py"]
        self.assertEqual(detailed["description"], "Second analyze.py execution")
        self.assertEqual(detailed["number_of_semantically_relevant_numbers"], 2)
        self.assertEqual(detailed["intent_class"], "exploiting")
        self.assertEqual(detailed["specific_subtype"], -1)

    def test_main_llm_prompt_keeps_full_observed_output(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text("print('script')\n", encoding="utf-8")
            long_output = "Output: " + ("segment-" * 80) + "TAIL_MARKER_123\n"
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": long_output}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            codex_calls: list[str] = []

            def _fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                codex_calls.append(str(kwargs.get("input") or ""))
                payload = {
                    "description": "Analyzes long output without truncating the observed results",
                    "number_of_semantically_relevant_numbers": 1,
                    "intent_class": "exploring",
                    "specific_subtype": 3,
                }
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"message": json.dumps(payload)}), "")

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_fake_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

            prompt_text = (trajectory_path.parent / "codex_description_raw" / "001_analyze.py" / "prompt.txt").read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("TAIL_MARKER_123", prompt_text)
        self.assertIn("TAIL_MARKER_123", codex_calls[0])
        self.assertEqual(codex_calls[0].count("TAIL_MARKER_123"), 1)

    def test_main_llm_parallel_preserves_output_order_and_raw_folder_numbering(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.traj_run"
            trajectory_path = trial_dir / "agent_logs" / "atif_processed" / TRAJECTORY_FILENAME
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts_dir = trial_dir / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            (artifacts_dir / "analyze.py").write_text("print('script')\n", encoding="utf-8")
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 2},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python analyze.py"},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "script\n"}],
                            },
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_2",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_2", "content": "inline\n"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completion_order: list[str] = []
            progress_updates: list[int] = []
            tqdm_calls: list[dict[str, object]] = []

            def _fake_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                prompt = str(kwargs.get("input") or "")
                if "Entry key: analyze.py" in prompt:
                    time.sleep(0.05)
                    completion_order.append("analyze.py")
                    payload = {
                        "description": "Plots script sample output and saves figures",
                        "number_of_semantically_relevant_numbers": 3,
                        "intent_class": "exploring",
                        "specific_subtype": 5,
                    }
                else:
                    time.sleep(0.01)
                    completion_order.append("dash_c_1")
                    payload = {
                        "description": "Prints inline sample output",
                        "number_of_semantically_relevant_numbers": 1,
                        "intent_class": "exploiting",
                        "specific_subtype": -1,
                    }
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    json.dumps({"message": json.dumps(payload)}),
                    "",
                )

            class _FakeProgress:
                def __init__(self, *args: object, **kwargs: object) -> None:
                    tqdm_calls.append(dict(kwargs))

                def update(self, value: int) -> None:
                    progress_updates.append(int(value))

                def close(self) -> None:
                    return None

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_fake_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda *args, **kwargs: _FakeProgress(*args, **kwargs)),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm", "--parallel", "2"],
                )

            self.assertEqual(completion_order, ["dash_c_1", "analyze.py"])
            self.assertEqual(sum(progress_updates), 2)
            self.assertEqual(len(tqdm_calls), 1)
            self.assertEqual(tqdm_calls[0]["total"], 2)
            raw_root = trajectory_path.parent / "codex_description_raw"
            self.assertTrue((raw_root / "001_analyze.py").is_dir())
            self.assertTrue((raw_root / "002_dash_c_1").is_dir())

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            list(payload["subcommand_analysis"]["python"]["detailed"].keys()),
            ["analyze.py", "dash_c_1"],
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["analyze.py"]["description"],
            "Plots script sample output and saves figures",
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["analyze.py"]["intent_class"],
            "exploring",
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["analyze.py"]["specific_subtype"],
            5,
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["description"],
            "Prints inline sample output",
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["intent_class"],
            "exploiting",
        )
        self.assertEqual(
            payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["specific_subtype"],
            -1,
        )

    def test_main_parallel_requires_llm_and_rejects_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run", "--parallel", "2"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--parallel requires --llm.", result.output)

        result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run", "--llm", "--parallel", "0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("0 is not in the range x>=1", result.output)

    def test_main_without_llm_does_not_use_tqdm(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch(
                    "workflows.trajectory.trajectory_evaluation_programmatic.tqdm",
                    side_effect=AssertionError("tqdm should not be used without --llm"),
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_main_llm_sets_description_to_null_when_codex_call_fails(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "inline\n"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def _failing_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                return subprocess.CompletedProcess(cmd, 1, "", "boom")

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_failing_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

            raw_root = trajectory_path.parent / "codex_description_raw"
            self.assertTrue(raw_root.is_dir())
            entry_dir = raw_root / "001_dash_c_1"
            self.assertTrue(entry_dir.is_dir())
            self.assertIn("Entry key: dash_c_1", (entry_dir / "prompt.txt").read_text(encoding="utf-8"))
            self.assertEqual((entry_dir / "stdout.txt").read_text(encoding="utf-8"), "")
            self.assertEqual((entry_dir / "stderr.txt").read_text(encoding="utf-8"), "boom")
            self.assertEqual(
                json.loads((entry_dir / "metadata.json").read_text(encoding="utf-8")),
                {
                    "entry_key": "dash_c_1",
                    "description": None,
                    "number_of_semantically_relevant_numbers": None,
                    "intent_class": None,
                    "specific_subtype": None,
                    "returncode": 1,
                    "codex_command": ["codex", "exec", "--json", "--model", "gpt-5.3-codex", "--", "-"],
                    "executions": 1,
                },
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["description"])
        self.assertIsNone(
            payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"][
                "number_of_semantically_relevant_numbers"
            ]
        )
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["intent_class"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["specific_subtype"])

    def test_main_llm_sets_enriched_fields_to_null_when_intent_class_is_invalid(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "inline\n"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def _invalid_subprocess_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                payload = {
                    "description": "Prints inline sample output",
                    "number_of_semantically_relevant_numbers": 1,
                    "intent_class": "unknown",
                    "specific_subtype": -1,
                }
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"message": json.dumps(payload)}), "")

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value="codex"),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.subprocess.run", side_effect=_invalid_subprocess_run),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["description"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["number_of_semantically_relevant_numbers"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["intent_class"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["specific_subtype"])

    def test_main_llm_sets_enriched_fields_to_null_when_codex_binary_is_missing(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print('inline')\""},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": "inline\n"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.shutil.which", return_value=None),
                mock.patch("workflows.trajectory.trajectory_evaluation_programmatic.tqdm", side_effect=lambda iterable, **kwargs: iterable),
            ):
                result = runner.invoke(
                    trajectory_evalauation_programmatic.main,
                    ["traj_run", "--llm"],
                )

            raw_root = trajectory_path.parent / "codex_description_raw"
            self.assertTrue(raw_root.is_dir())
            entry_dir = raw_root / "001_dash_c_1"
            self.assertTrue(entry_dir.is_dir())
            self.assertEqual((entry_dir / "stdout.txt").read_text(encoding="utf-8"), "")
            self.assertEqual((entry_dir / "stderr.txt").read_text(encoding="utf-8"), "")
            self.assertEqual(
                json.loads((entry_dir / "metadata.json").read_text(encoding="utf-8")),
                {
                    "entry_key": "dash_c_1",
                    "description": None,
                    "number_of_semantically_relevant_numbers": None,
                    "intent_class": None,
                    "specific_subtype": None,
                    "returncode": -1,
                    "codex_command": [],
                    "executions": 1,
                    "error": "Could not find 'codex' in PATH.",
                },
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["description"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["number_of_semantically_relevant_numbers"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["intent_class"])
        self.assertIsNone(payload["subcommand_analysis"]["python"]["detailed"]["dash_c_1"]["specific_subtype"])

    def test_main_summarizes_cat_patterns_and_created_files(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {
                                            "command": (
                                                "cat << 'EOF' > analyze.py\n"
                                                "print('x')\n"
                                                "EOF\n"
                                                "python analyze.py\n"
                                                "cat a.txt b.txt > out.txt\n"
                                                "cat file.txt\n"
                                                "echo hi | cat > merged.txt\n"
                                            )
                                        },
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": ""}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            payload = json.loads(result.output)
            self.assertEqual(
                payload["tool_calls"],
                {
                    "exec_command": {"count": 0},
                    "read_file": {"count": 0},
                    "run_shell_command": {
                        "count": 6,
                        "commands": {
                            "cat": {
                                "count": 4,
                                "call_unique": 1,
                            },
                            "echo": {
                                "count": 1,
                                "call_unique": 1,
                            },
                            "python": {
                                "count": 1,
                                "call_unique": 1,
                            },
                        },
                    },
                    "write_file": {"count": 0},
                },
            )
            self.assertEqual(
                payload["subcommand_analysis"],
                {
                    "cat": {
                        "total_invocations": 4,
                        "pattern": {
                            "heredoc_write": {
                                "create_files": {
                                    "python_files": {
                                        "analyze.py": 1,
                                    },
                                    "other": {},
                                },
                            },
                            "file_concat_to_stdout": 1,
                            "file_concat_to_file": 1,
                            "pipe_into_cat": 1,
                            "other": 0,
                        },
                        "followup_command_counts": {
                            "python": 1,
                        },
                        "execution_after_creation": {
                            "executed_in_same_shell_snippet": 1,
                            "not_executed_in_same_shell_snippet": 2,
                        },
                        "heredoc_delimiters": {
                            "EOF": 1,
                        },
                    },
                    "python": {
                        "total_invocations": 1,
                        "failed_invocations": 0,
                        "total_stdout_lines": 0,
                        "total_stderr_lines": 0,
                        "total_output_bytes": 0,
                        "invocation_modes": {
                            "dash_c": 0,
                            "heredoc": 0,
                            "script_path": 1,
                            "other": 0,
                        },
                        "lines_of_code": {
                            "script_total": 0,
                        },
                        "library_calls": {},
                        "detailed": {
                            "analyze.py": {
                                "executions": 1,
                                "lines_of_code": None,
                                "library_calls": {},
                                "total_stdout_lines": 0,
                                "total_stderr_lines": 0,
                                "total_output_bytes": 0,
                            }
                        },
                    },
                },
            )
            self.assertEqual(
                payload["number_of_interactions"],
	                {
	                    "user": 1,
	                    "system": 0,
	                    "agent": {
	                        "tool_call_steps": 1,
	                        "tool_observations": 1,
	                        "steps": 1,
	                    },
	                },
	            )

    def test_main_keeps_bash_tool_name_for_claude_parsed_output(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="claude_4_opus_high")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "function_name": "Bash",
                                        "arguments": {
                                            "command": "cat << 'EOF' > analyze.py\nprint('x')\nEOF\npython analyze.py\n"
                                        },
                                    },
                                    {
                                        "function_name": "Read",
                                        "arguments": {"file_path": "/app/analyze.py"},
                                    },
                                    {
                                        "function_name": "Read",
                                        "arguments": {"file_path": "/app/overview_1.png"},
                                    },
                                    {
                                        "function_name": "Read",
                                        "arguments": {"file_path": "/app/analyze.py"},
                                    },
                                    {
                                        "function_name": "Read",
                                        "arguments": {},
                                    },
                                ],
                                "observation": [
                                    {"source_call_id": "call_1", "content": ""},
                                    {"source_call_id": "call_2", "content": ""},
                                    {"source_call_id": "call_3", "content": ""},
                                    {"source_call_id": "call_4", "content": ""},
                                    {"source_call_id": "call_5", "content": ""},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(
            payload["tool_calls"],
            {
                "Bash": {
                    "count": 2,
                    "commands": {
                        "cat": {
                            "count": 1,
                            "call_unique": 1,
                        },
                        "python": {
                            "count": 1,
                            "call_unique": 1,
                        },
                    },
                },
                "Read": {
                    "count": 3,
                    "read": {
                        "/app/analyze.py": 2,
                        "/app/overview_1.png": 1,
                    },
                },
                "Write": {"count": 0},
                "Edit": {"count": 0, "read": {}},
                "TodoWrite": {"count": 0},
                "ToolSearch": {"count": 0},
                "Agent": {"count": 0},
            },
        )
        self.assertEqual(
            payload["subcommand_analysis"],
            {
                "cat": {
                    "total_invocations": 1,
                    "pattern": {
                        "heredoc_write": {
                            "create_files": {
                                "python_files": {
                                    "analyze.py": 1,
                                },
                                "other": {},
                            },
                        },
                        "file_concat_to_stdout": 0,
                        "file_concat_to_file": 0,
                        "pipe_into_cat": 0,
                        "other": 0,
                    },
                    "followup_command_counts": {
                        "python": 1,
                    },
                    "execution_after_creation": {
                        "executed_in_same_shell_snippet": 1,
                        "not_executed_in_same_shell_snippet": 0,
                    },
                    "heredoc_delimiters": {
                        "EOF": 1,
                    },
                },
                "python": {
                    "total_invocations": 1,
                    "failed_invocations": 0,
                    "total_stdout_lines": 0,
                    "total_stderr_lines": 0,
                    "total_output_bytes": 0,
                    "invocation_modes": {
                        "dash_c": 0,
                        "heredoc": 0,
                        "script_path": 1,
                        "other": 0,
                    },
                    "lines_of_code": {
                        "script_total": 0,
                    },
                    "library_calls": {},
                    "detailed": {
                        "analyze.py": {
                            "executions": 1,
                            "lines_of_code": None,
                            "library_calls": {},
                            "total_stdout_lines": 0,
                            "total_stderr_lines": 0,
                            "total_output_bytes": 0,
                        }
                    },
                },
            },
        )
        self.assertNotIn("by_script", payload["subcommand_analysis"]["python"])
        self.assertNotIn("script_targets", payload["subcommand_analysis"]["python"])
        self.assertNotIn("artifact_script_targets", payload["subcommand_analysis"]["python"])
        self.assertNotIn("unresolved_script_targets", payload["subcommand_analysis"]["python"])

    def test_main_errors_when_shell_command_cannot_be_parsed(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {
                                        "tool_call_id": "call_1",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "echo $("},
                                    }
                                ],
                                "observation": [{"source_call_id": "call_1", "content": ""}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Failed to parse run_shell_command", result.output)

    def test_main_errors_when_run_directory_is_missing(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "runs"
            with mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["missing_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Run directory not found", result.output)

    def test_main_errors_when_trajectory_file_is_missing(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata(run_dir)

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Trajectory file not found", result.output)

    def test_main_rejects_nonstandard_user_and_system_interaction_counts(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 99},
                        "steps": [
                            {"source": "system"},
                            {"source": "agent"},
                            {"source": "unknown"},
                            {"message": "missing source"},
                            "not-an-object",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("number_of_interactions.user must be 1", result.output)

    def test_main_rejects_tool_call_observation_mismatch(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_cost_usd": 0.5,
                            "total_prompt_tokens": 0,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 0,
                            "total_steps": 1,
                        },
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {"tool_call_id": "call_1", "function_name": "read_file"},
                                ],
                                "observation": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("tool_call_steps and tool_observations must match", result.output)

    def test_main_errors_when_tool_calls_are_not_allowed_for_agent_profile(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="gemini_3_1_pro_high")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_cost_usd": 0.5, "total_prompt_tokens": 0, "total_cached_tokens": 0, "total_completion_tokens": 0, "total_steps": 1},
                        "steps": [
                            {"source": "user"},
                            {
                                "source": "agent",
                                "tool_calls": [
                                    {"tool_call_id": "call_1", "function_name": "read_file"},
                                    {"tool_call_id": "call_2", "function_name": "write_file"},
                                    {
                                        "tool_call_id": "call_3",
                                        "function_name": "run_shell_command",
                                        "arguments": {"command": "python -c \"print(1)\""},
                                    },
                                ],
                                "observation": [
                                    {"source_call_id": "call_1", "content": ""},
                                    {"source_call_id": "call_2", "content": ""},
                                    {"source_call_id": "call_3", "content": ""},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def _restricted_agent_profile_lookup(agent_id: str) -> dict[str, object]:
                profile = self._mock_agent_profile_lookup(agent_id)
                return {
                    **profile,
                    "available_tools": {"read_file": {"parse_method": False, "subcommands": []}},
                }

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=_restricted_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid tools: run_shell_command, write_file", result.output)
        self.assertIn("Allowed tools: read_file", result.output)

    def test_main_computes_cost_from_run_metadata_agent_profile(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="gemini_3_1_pro_high")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_prompt_tokens": 100,
                            "total_cached_tokens": 20,
                            "total_completion_tokens": 10,
                            "total_steps": 2,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["agent_id"], "gemini_3_1_pro_high")
        self.assertAlmostEqual(payload["cost_usd"], 0.0002975, places=10)

    def test_main_uses_minimax_low_alias_for_profile_but_keeps_output_agent_id(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="minimax-m2.7_low")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "agent_id": "minimax-m2.7_low",
                        "final_metrics": {
                            "total_prompt_tokens": 100,
                            "total_cached_tokens": 10,
                            "total_completion_tokens": 5,
                            "total_steps": 1,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["agent_id"], "minimax-m2.7_low")
        self.assertAlmostEqual(payload["cost_usd"], 0.00003359, places=10)

    def test_main_normalizes_non_cached_prompt_tokens_before_cost(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="claude_4_opus_high")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {
                            "total_prompt_tokens": 47,
                            "total_cached_tokens": 2_153_784,
                            "total_completion_tokens": 115_409,
                            "total_steps": 2,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["agent_id"], "claude_4_opus_high")
        self.assertEqual(payload["tokens"]["total_prompt_tokens"], 2_153_831)
        self.assertAlmostEqual(payload["cost_usd"], 3.962352, places=10)
        self.assertGreater(payload["cost_usd"], 0)

    def test_main_computes_cost_from_trajectory_agent_id_when_run_metadata_agent_id_missing(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id=None)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "agent_id": "gemini_3_1_pro_high",
                        "final_metrics": {
                            "total_prompt_tokens": 10,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 2,
                            "total_steps": 2,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["agent_id"], "gemini_3_1_pro_high")
        self.assertAlmostEqual(payload["cost_usd"], 0.000042, places=10)

    def test_main_errors_when_agent_profile_cannot_be_resolved(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="unknown_agent")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "agent_id": "also_unknown",
                        "final_metrics": {
                            "total_prompt_tokens": 10,
                            "total_cached_tokens": 0,
                            "total_completion_tokens": 2,
                        },
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Could not resolve a known agentic profile", result.output)

    def test_main_errors_when_cost_tokens_are_missing(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir, agent_id="gemini_3_1_pro_high")
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(
                json.dumps(
                    {
                        "final_metrics": {"total_steps": 2},
                        "steps": [{"source": "user"}],
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root),
                mock.patch.object(
                    trajectory_evalauation_programmatic,
                    "_agentic_profile_by_id",
                    side_effect=self._mock_agent_profile_lookup,
                ),
            ):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("missing required final_metrics fields to compute cost", result.output)
        self.assertIn("total_prompt_tokens", result.output)
        self.assertIn("total_cached_tokens", result.output)
        self.assertIn("total_completion_tokens", result.output)

    def test_main_errors_when_payload_is_not_an_object(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "traj_run"
            self._write_run_metadata(run_dir)
            trajectory_path = (
                run_dir
                / "question_0"
                / "question_0.1-of-1.traj_run"
                / "agent_logs"
                / "atif_processed"
                / TRAJECTORY_FILENAME
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_path.write_text(json.dumps([]), encoding="utf-8")

            with mock.patch.object(trajectory_evalauation_programmatic, "RUNS_ROOT", runs_root):
                result = runner.invoke(trajectory_evalauation_programmatic.main, ["traj_run"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("must contain a JSON object", result.output)


if __name__ == "__main__":
    unittest.main()
