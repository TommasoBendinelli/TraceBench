from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.agentic_rollout_paths import LIGHT_TRAJECTORY_FILENAME, TRAJECTORY_FILENAME
from workflows.trajectory import export_atif_trajectory


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rfc_cost(*, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    return (
        (prompt_tokens - cached_tokens) * 5.0
        + cached_tokens * 0.5
        + completion_tokens * 30.0
    ) / 1_000_000


def _write_minimal_codex_run(run_dir: Path) -> Path:
    trial_dir = run_dir / "question_0" / f"question_0.1-of-1.{run_dir.name}"
    agent_logs_dir = trial_dir / "agent_logs"
    agent_logs_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "scenario_info.json").write_text(
        json.dumps(
            {
                "agent_id": "gpt_5_5_codex_high",
                "question_id": "question_0",
                "tag": "DEBUG",
            }
        ),
        encoding="utf-8",
    )

    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "agent_id": "gpt_5_5_codex_high",
                "agent_name": "codex",
                "model_name": "gpt-5.4",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "question_0",
                        "trial_name": f"question_0.1-of-1.{run_dir.name}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        agent_logs_dir / "rollout-test.jsonl",
        [
            {
                "timestamp": "2026-03-30T10:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": "sess_1", "cli_version": "0.120.0"},
            },
            {
                "timestamp": "2026-03-30T10:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "task"}],
                },
            },
            {
                "timestamp": "2026-03-30T10:00:02.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "pwd"}),
                    "call_id": "call_exec",
                },
            },
            {
                "timestamp": "2026-03-30T10:00:03.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call_exec",
                    "output": "Command: pwd\nOutput:\n/app\n",
                },
            },
            {
                "timestamp": "2026-03-30T10:00:04.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 12,
                            "cached_input_tokens": 2,
                            "output_tokens": 3,
                            "reasoning_output_tokens": 1,
                            "total_tokens": 15,
                        },
                        "total_token_usage": {
                            "input_tokens": 12,
                            "cached_input_tokens": 2,
                            "output_tokens": 3,
                            "reasoning_output_tokens": 1,
                            "total_tokens": 15,
                        },
                    },
                },
            },
        ],
    )
    return agent_logs_dir


class TestExportAtifTrajectory(unittest.TestCase):
    def test_build_trajectory_from_gemini_normalizes_list_content(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            session_path = Path(tmp_dir) / "session.json"
            session_path.write_text(
                json.dumps(
                    {
                        "sessionId": "gemini_session_1",
                        "messages": [
                            {
                                "timestamp": "2026-03-30T10:00:00.000Z",
                                "type": "user",
                                "content": [{"text": "first line"}, {"text": "second line"}],
                            },
                            {
                                "timestamp": "2026-03-30T10:00:01.000Z",
                                "type": "gemini",
                                "content": [{"text": "analysis block"}, {"kind": "note"}],
                                "thoughts": [],
                                "toolCalls": [
                                    {
                                        "id": "call_read",
                                        "name": "read_file",
                                        "timestamp": "2026-03-30T10:00:02.000Z",
                                        "args": {"path": "README.md"},
                                        "result": [
                                            {
                                                "functionResponse": {
                                                    "response": {"output": "contents"}
                                                }
                                            }
                                        ],
                                    }
                                ],
                                "tokens": {
                                    "input": 12,
                                    "output": 3,
                                    "cached": 2,
                                    "thoughts": 1,
                                    "tool": 0,
                                },
                                "model": "gemini-3.1-pro-preview",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            trajectory = export_atif_trajectory.build_trajectory_from_gemini(
                session_path=session_path,
                session_id="trial_1",
                agent_profile_id="gemini_3_1_pro_high",
                model_name="gemini-3.1-pro-preview",
            )

            self.assertEqual(trajectory.session_id, "trial_1")
            self.assertEqual(trajectory.agent.name, "gemini_3_1_pro_high")
            self.assertEqual(trajectory.agent.version, "1.00")
            self.assertEqual(trajectory.steps[0].source, "USER")
            self.assertEqual(trajectory.steps[0].message, "first line\nsecond line")
            self.assertEqual(
                trajectory.steps[1].message,
                'analysis block\n{"kind": "note"}',
            )
            self.assertEqual(trajectory.steps[1].metrics.prompt_tokens, 12)
            self.assertEqual(trajectory.steps[1].metrics.cached_tokens, 2)
            self.assertEqual(trajectory.steps[1].metrics.completion_tokens, 4)
            self.assertIsNone(trajectory.steps[0].duration)
            self.assertEqual(trajectory.steps[1].duration, 1.0)
            self.assertEqual(trajectory.steps[1].tool_calls[0].function_name, "read_file")
            self.assertEqual(
                set(trajectory.steps[1].tool_calls[0].model_dump()),
                {"tool_call_id", "function_name", "arguments"},
            )
            self.assertEqual(
                trajectory.steps[1].observation[0].timestamp,
                "2026-03-30T10:00:02.000Z",
            )
            self.assertEqual(trajectory.steps[1].observation[0].duration, 1.0)
            self.assertEqual(trajectory.final_metrics.total_duration, 1.0)
            self.assertEqual(trajectory.final_metrics.total_tool_duration, 1.0)
            self.assertAlmostEqual(
                trajectory.final_metrics.total_cost_usd,
                ((10 * 2.0) + (2 * 0.2) + (4 * 12.0)) / 1_000_000,
            )

    def test_build_codex_trajectory_parses_rollout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "session_meta",
                        "payload": {
                            "id": "sess_1",
                            "cli_version": "0.120.0",
                            "model_provider": "openai",
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "solve task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:03.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "inspecting files"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:04.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "ls", "workdir": "/app"}),
                            "call_id": "call_exec",
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:05.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_exec",
                            "output": "Command: ls\nOutput:\ntrain_samples\n",
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:06.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call",
                            "status": "completed",
                            "call_id": "call_patch",
                            "name": "apply_patch",
                            "input": "*** Begin Patch\n*** End Patch\n",
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:07.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "custom_tool_call_output",
                            "call_id": "call_patch",
                            "output": json.dumps(
                                {
                                    "output": "Success. Updated the following files:\nM /app/rule.py\n",
                                    "metadata": {"exit_code": 0},
                                }
                            ),
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:08.000Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "last_token_usage": {
                                    "input_tokens": 25,
                                    "cached_input_tokens": 5,
                                    "output_tokens": 4,
                                    "reasoning_output_tokens": 3,
                                    "total_tokens": 29,
                                },
                                "total_token_usage": {
                                    "input_tokens": 25,
                                    "cached_input_tokens": 5,
                                    "output_tokens": 4,
                                    "reasoning_output_tokens": 3,
                                    "total_tokens": 29,
                                }
                            },
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertEqual(trajectory.agent.name, "gpt_5_5_codex_high")
            self.assertEqual(trajectory.agent.version, "1.00")
            self.assertEqual(
                [step.source for step in trajectory.steps],
                ["USER", "AGENT", "AGENT", "AGENT"],
            )
            self.assertEqual(trajectory.steps[2].tool_calls[0].function_name, "exec_command")
            self.assertEqual(
                trajectory.steps[2].observation[0].content,
                "Command: ls\nOutput:\ntrain_samples\n",
            )
            self.assertEqual(
                set(trajectory.steps[2].tool_calls[0].model_dump()),
                {"tool_call_id", "function_name", "arguments"},
            )
            self.assertEqual(
                trajectory.steps[2].observation[0].timestamp,
                "2026-03-30T10:00:05.000Z",
            )
            self.assertIsNone(trajectory.steps[0].duration)
            self.assertEqual(trajectory.steps[1].duration, 1.0)
            self.assertEqual(trajectory.steps[2].duration, 1.0)
            self.assertEqual(trajectory.steps[2].observation[0].duration, 1.0)
            self.assertEqual(trajectory.steps[3].duration, 2.0)
            self.assertEqual(trajectory.steps[3].observation[0].duration, 1.0)
            self.assertIsNone(trajectory.steps[0].metrics)
            self.assertEqual(trajectory.steps[1].metrics.prompt_tokens, 25)
            self.assertEqual(trajectory.steps[1].metrics.cached_tokens, 5)
            self.assertEqual(trajectory.steps[1].metrics.completion_tokens, 7)
            self.assertEqual(trajectory.steps[3].tool_calls[0].function_name, "apply_patch")
            self.assertIn("Updated the following files", trajectory.steps[3].observation[0].content)
            self.assertIsNone(trajectory.steps[2].metrics)
            self.assertIsNone(trajectory.steps[3].metrics)
            self.assertEqual(trajectory.final_metrics.total_prompt_tokens, 25)
            self.assertEqual(trajectory.final_metrics.total_cached_tokens, 5)
            self.assertEqual(trajectory.final_metrics.total_completion_tokens, 7)
            self.assertEqual(trajectory.final_metrics.total_duration, 4.0)
            self.assertEqual(trajectory.final_metrics.total_tool_duration, 2.0)
            self.assertEqual(
                sum((step.metrics.prompt_tokens if step.metrics else 0) for step in trajectory.steps),
                trajectory.final_metrics.total_prompt_tokens,
            )
            self.assertEqual(
                sum(
                    (step.metrics.completion_tokens if step.metrics else 0)
                    for step in trajectory.steps
                ),
                trajectory.final_metrics.total_completion_tokens,
            )
            self.assertEqual(
                sum((step.metrics.cached_tokens if step.metrics else 0) for step in trajectory.steps),
                trajectory.final_metrics.total_cached_tokens,
            )
            self.assertAlmostEqual(
                trajectory.final_metrics.total_cost_usd,
                _rfc_cost(prompt_tokens=25, cached_tokens=5, completion_tokens=7),
            )

    def test_build_codex_trajectory_discards_leading_system_steps(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "dev instructions"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "solve task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertEqual([step.step_id for step in trajectory.steps], [1, 2])
            self.assertEqual([step.source for step in trajectory.steps], ["USER", "AGENT"])
            self.assertEqual(trajectory.steps[0].message, "solve task")
            self.assertEqual(trajectory.steps[1].message, "done")
            self.assertEqual(trajectory.steps[1].duration, 1.0)
            self.assertEqual(trajectory.final_metrics.total_steps, 2)
            self.assertEqual(trajectory.final_metrics.total_duration, 1.0)

    def test_build_codex_trajectory_keeps_system_step_after_user(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "solve task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "cron update"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "done"}],
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertEqual(
                [step.source for step in trajectory.steps],
                ["USER", "SYSTEM", "AGENT"],
            )
            self.assertEqual(trajectory.steps[1].message, "cron update")
            self.assertIsNone(trajectory.steps[1].duration)
            self.assertEqual(trajectory.steps[2].duration, 1.0)

    def test_main_exports_codex_run_from_documented_path_option(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "runs" / "codex_run"
            agent_logs_dir = _write_minimal_codex_run(run_dir)

            result = runner.invoke(export_atif_trajectory.main, ["--path", str(run_dir)])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            atif_processed_dir = agent_logs_dir / "atif_processed"
            trajectory_path = atif_processed_dir / TRAJECTORY_FILENAME
            light_trajectory_path = atif_processed_dir / LIGHT_TRAJECTORY_FILENAME
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(light_trajectory_path.exists())
            self.assertFalse((agent_logs_dir / TRAJECTORY_FILENAME).exists())
            self.assertFalse((agent_logs_dir / LIGHT_TRAJECTORY_FILENAME).exists())
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            light_payload = json.loads(light_trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["session_id"], "codex_run")
            self.assertEqual(payload["agent_id"], "gpt_5_5_codex_high")
            self.assertEqual(payload["question_id"], "question_0")
            self.assertEqual(payload["tag"], "DEBUG")
            self.assertEqual(
                set(payload),
                {
                    "tag",
                    "question_id",
                    "agent_id",
                    "session_id",
                    "final_metrics",
                    "is_context_compacted",
                    "steps",
                },
            )
            self.assertFalse(payload["is_context_compacted"])
            self.assertEqual(payload["final_metrics"]["total_completion_tokens"], 4)
            self.assertEqual(payload["final_metrics"]["total_duration"], 1.0)
            self.assertEqual(payload["final_metrics"]["total_tool_duration"], 1.0)
            self.assertEqual(payload["steps"][0]["source"], "USER")
            self.assertNotIn("duration", payload["steps"][0])
            self.assertEqual(payload["steps"][1]["tool_calls"][0]["function_name"], "exec_command")
            self.assertEqual(payload["steps"][1]["duration"], 1.0)
            self.assertNotIn("timestamp", payload["steps"][1]["tool_calls"][0])
            self.assertEqual(payload["steps"][1]["metrics"]["prompt_tokens"], 12)
            self.assertEqual(payload["steps"][1]["metrics"]["cached_tokens"], 2)
            self.assertEqual(payload["steps"][1]["metrics"]["completion_tokens"], 4)
            self.assertEqual(payload["steps"][1]["observation"][0]["duration"], 1.0)
            self.assertAlmostEqual(
                payload["final_metrics"]["total_cost_usd"],
                _rfc_cost(prompt_tokens=12, cached_tokens=2, completion_tokens=4),
            )
            self.assertEqual(
                payload["steps"][1]["observation"][0]["timestamp"],
                "2026-03-30T10:00:03.000Z",
            )
            self.assertEqual(set(light_payload), {"agent_id", "session_id", "steps"})
            self.assertEqual(light_payload["agent_id"], payload["agent_id"])
            self.assertEqual(light_payload["session_id"], payload["session_id"])
            self.assertEqual(
                set(light_payload["steps"][1]),
                {"step_id", "source", "message", "tool_calls", "observation"},
            )
            self.assertNotIn("duration", light_payload["steps"][1])
            self.assertEqual(
                light_payload["steps"][1]["tool_calls"],
                payload["steps"][1]["tool_calls"],
            )
            self.assertNotIn("timestamp", light_payload["steps"][1]["tool_calls"][0])
            self.assertEqual(
                light_payload["steps"][1]["observation"],
                payload["steps"][1]["observation"],
            )

    def test_target_context_compaction_detects_summarization_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            agent_logs_dir = Path(tmp_dir) / "agent_logs"
            agent_logs_dir.mkdir()
            (agent_logs_dir / "trajectory.summarization-1-summary.json").write_text(
                "{}",
                encoding="utf-8",
            )
            target = export_atif_trajectory.TrajectoryTarget(
                source_type="terminus",
                rollout_path=None,
                gemini_session_path=None,
                episode_root=agent_logs_dir,
                opencode_storage=None,
                opencode_session_path=None,
                output_path=agent_logs_dir / TRAJECTORY_FILENAME,
                session_id="run_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name=None,
                exported_agent_id="gpt_5_5_codex_high",
                question_id="question_0",
                tag="DEBUG",
            )

            self.assertTrue(export_atif_trajectory.target_has_context_compaction(target))

    def test_target_context_compaction_detects_claude_compact_boundary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "agent_logs" / ".claude" / "projects" / "session.jsonl"
            rollout_path.parent.mkdir(parents=True)
            _write_jsonl(
                rollout_path,
                [
                    {
                        "type": "system",
                        "subtype": "compact_boundary",
                        "content": "Conversation compacted",
                    },
                ],
            )
            target = export_atif_trajectory.TrajectoryTarget(
                source_type="claude",
                rollout_path=rollout_path,
                gemini_session_path=None,
                episode_root=rollout_path.parents[3],
                opencode_storage=None,
                opencode_session_path=None,
                output_path=rollout_path.parents[3] / "atif_processed" / TRAJECTORY_FILENAME,
                session_id="run_1",
                agent_profile_id="claude_4_opus_high",
                model_name=None,
                exported_agent_id="claude_4_opus_high",
                question_id="question_0",
                tag="DEBUG",
            )

            self.assertTrue(export_atif_trajectory.target_has_context_compaction(target))

    def test_target_context_compaction_detects_claude_continuation_message(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "agent_logs" / ".claude" / "projects" / "session.jsonl"
            rollout_path.parent.mkdir(parents=True)
            _write_jsonl(
                rollout_path,
                [
                    {
                        "type": "user",
                        "message": {
                            "role": "user",
                            "content": "This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion.",
                        },
                    },
                ],
            )
            target = export_atif_trajectory.TrajectoryTarget(
                source_type="claude",
                rollout_path=rollout_path,
                gemini_session_path=None,
                episode_root=rollout_path.parents[3],
                opencode_storage=None,
                opencode_session_path=None,
                output_path=rollout_path.parents[3] / "atif_processed" / TRAJECTORY_FILENAME,
                session_id="run_1",
                agent_profile_id="claude_4_opus_high",
                model_name=None,
                exported_agent_id="claude_4_opus_high",
                question_id="question_0",
                tag="DEBUG",
            )

            self.assertTrue(export_atif_trajectory.target_has_context_compaction(target))

    def test_target_context_compaction_ignores_claude_last_prompt_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "agent_logs" / ".claude" / "projects" / "session.jsonl"
            rollout_path.parent.mkdir(parents=True)
            _write_jsonl(
                rollout_path,
                [
                    {
                        "type": "last-prompt",
                        "lastPrompt": "original task prompt",
                    },
                ],
            )
            target = export_atif_trajectory.TrajectoryTarget(
                source_type="claude",
                rollout_path=rollout_path,
                gemini_session_path=None,
                episode_root=rollout_path.parents[3],
                opencode_storage=None,
                opencode_session_path=None,
                output_path=rollout_path.parents[3] / "atif_processed" / TRAJECTORY_FILENAME,
                session_id="run_1",
                agent_profile_id="claude_4_opus_high",
                model_name=None,
                exported_agent_id="claude_4_opus_high",
                question_id="question_0",
                tag="DEBUG",
            )

            self.assertFalse(export_atif_trajectory.target_has_context_compaction(target))

    def test_resolve_agent_fields_rejects_legacy_spark_codex_alias(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown agent_id 'gpt_5_3_high_spark_codex'"):
            export_atif_trajectory.resolve_agent_fields(
                {
                    "run_id": "legacy_run",
                    "agent_id": "gpt_5_3_high_spark_codex",
                    "agent_name": "codex",
                    "model_name": "gpt-5.3-codex",
                    "reasoning": "high",
                },
                agent_name=None,
                model_name=None,
            )

    def test_resolve_agent_fields_rejects_unknown_non_aliased_agent_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown agent_id 'missing_agent'"):
            export_atif_trajectory.resolve_agent_fields(
                {
                    "run_id": "unknown_agent_run",
                    "agent_id": "missing_agent",
                    "agent_name": "codex",
                    "model_name": "gpt-5.3-codex",
                },
                agent_name=None,
                model_name=None,
            )

    def test_scenario_agent_profile_alias_resolves_minimax_low_for_profile_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            scenario_info_path = Path(tmp_dir) / "scenario_info.json"
            scenario_info_path.write_text("{}", encoding="utf-8")

            self.assertEqual(
                export_atif_trajectory._profile_id_from_scenario_agent_id(
                    "minimax-m2.7_low",
                    scenario_info_path,
                ),
                "minimax-m2.7",
            )

    def test_main_exports_documented_run_id(self) -> None:
        runner = CliRunner()
        original_runs_root = export_atif_trajectory.RUNS_ROOT
        with TemporaryDirectory() as tmp_dir:
            runs_root = Path(tmp_dir) / "terminal-bench" / "runs"
            run_dir = runs_root / "codex_run"
            agent_logs_dir = _write_minimal_codex_run(run_dir)
            export_atif_trajectory.RUNS_ROOT = runs_root
            try:
                result = runner.invoke(export_atif_trajectory.main, ["codex_run"])
            finally:
                export_atif_trajectory.RUNS_ROOT = original_runs_root

            self.assertEqual(result.exit_code, 0, msg=result.output)
            atif_processed_dir = agent_logs_dir / "atif_processed"
            self.assertTrue((atif_processed_dir / TRAJECTORY_FILENAME).exists())
            self.assertTrue((atif_processed_dir / LIGHT_TRAJECTORY_FILENAME).exists())
            self.assertFalse((agent_logs_dir / TRAJECTORY_FILENAME).exists())
            self.assertFalse((agent_logs_dir / LIGHT_TRAJECTORY_FILENAME).exists())

    def test_main_reports_invalid_flat_opencode_session_json(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "opencode_run"
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.opencode_run"
            agent_logs_dir = trial_dir / "agent_logs"
            agent_logs_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "agent_id": "minimax-m2.7_low",
                        "question_id": "question_0",
                        "tag": "DEBUG",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "opencode_run",
                        "agent_id": "minimax-m2.7",
                        "agent_name": "opencode",
                        "model_name": "minimax/minimax-m2.7",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "question_0",
                                "trial_name": "question_0.1-of-1.opencode_run",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            session_path = agent_logs_dir / "opencode-session-ses_invalid.json"
            session_path.write_text('{"info": {"id": "ses_invalid"}, "messages": [', encoding="utf-8")

            result = runner.invoke(export_atif_trajectory.main, ["--path", str(run_dir)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid JSON", result.output)
        self.assertIn(str(session_path), result.output)

    def test_main_exports_flat_opencode_session_json(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "opencode_run"
            trial_dir = run_dir / "question_0" / "question_0.1-of-1.opencode_run"
            agent_logs_dir = trial_dir / "agent_logs"
            agent_logs_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "agent_id": "minimax-m2.7_low",
                        "question_id": "question_0",
                        "tag": "DEBUG",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "opencode_run",
                        "agent_id": "minimax-m2.7",
                        "agent_name": "opencode",
                        "model_name": "minimax/minimax-m2.7",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "results": [
                            {
                                "task_id": "question_0",
                                "trial_name": "question_0.1-of-1.opencode_run",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (agent_logs_dir / "opencode-session-ses_valid.json").write_text(
                json.dumps(
                    {
                        "info": {"id": "ses_valid"},
                        "messages": [
                            {
                                "info": {
                                    "role": "user",
                                    "time": {"created": 1777760188598},
                                    "id": "msg_user",
                                },
                                "parts": [{"type": "text", "text": "task"}],
                            },
                            {
                                "info": {
                                    "role": "assistant",
                                    "time": {"created": 1777760198598},
                                    "tokens": {
                                        "input": 10,
                                        "output": 3,
                                        "cache": {"read": 2},
                                    },
                                    "cost": 0.01,
                                    "modelID": "minimax/minimax-m2.7",
                                    "providerID": "openrouter",
                                    "id": "msg_assistant",
                                },
                                "parts": [{"type": "text", "text": "answer"}],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = runner.invoke(export_atif_trajectory.main, ["--path", str(run_dir)])

            trajectory_path = agent_logs_dir / "atif_processed" / TRAJECTORY_FILENAME
            light_path = agent_logs_dir / "atif_processed" / LIGHT_TRAJECTORY_FILENAME

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(trajectory_path.exists())
            self.assertTrue(light_path.exists())
            self.assertFalse((agent_logs_dir / TRAJECTORY_FILENAME).exists())
            payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["agent_id"], "minimax-m2.7_low")
            self.assertEqual([step["message"] for step in payload["steps"]], ["task", "answer"])
            self.assertEqual(payload["final_metrics"]["total_prompt_tokens"], 10)
            self.assertEqual(payload["final_metrics"]["total_cached_tokens"], 2)

    def test_main_rejects_missing_scenario_export_metadata(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "codex_run"
            agent_logs_dir = _write_minimal_codex_run(run_dir)
            scenario_info_path = agent_logs_dir.parent / "scenario_info.json"
            scenario_info_path.write_text(
                json.dumps({"agent_id": "gpt_5_5_codex_high", "question_id": "question_0"}),
                encoding="utf-8",
            )

            result = runner.invoke(export_atif_trajectory.main, ["--path", str(run_dir)])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("tag missing or empty", result.output)

    def test_main_rejects_unknown_path_or_run_id(self) -> None:
        runner = CliRunner()
        original_runs_root = export_atif_trajectory.RUNS_ROOT
        with TemporaryDirectory() as tmp_dir:
            export_atif_trajectory.RUNS_ROOT = Path(tmp_dir) / "terminal-bench" / "runs"
            try:
                result = runner.invoke(export_atif_trajectory.main, ["missing_run"])
            finally:
                export_atif_trajectory.RUNS_ROOT = original_runs_root

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("neither an existing path nor a run id", result.output)

    def test_main_rejects_missing_input_selector(self) -> None:
        runner = CliRunner()
        result = runner.invoke(export_atif_trajectory.main, [])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Provide exactly one of <agentic_run_id> or --path PATH", result.output)

    def test_main_rejects_both_run_id_and_path(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "codex_run"
            _write_minimal_codex_run(run_dir)

            result = runner.invoke(
                export_atif_trajectory.main,
                ["codex_run", "--path", str(run_dir)],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Provide exactly one of <agentic_run_id> or --path PATH", result.output)

    def test_build_codex_trajectory_ignores_token_count_without_last_usage(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": None,
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertIsNone(trajectory.steps[0].metrics)
            self.assertIsNone(trajectory.steps[1].metrics)
            self.assertEqual(trajectory.final_metrics.total_steps, 2)
            self.assertEqual(trajectory.final_metrics.total_prompt_tokens, 0)
            self.assertEqual(trajectory.final_metrics.total_cached_tokens, 0)
            self.assertEqual(trajectory.final_metrics.total_completion_tokens, 0)
            self.assertEqual(trajectory.final_metrics.total_cost_usd, 0.0)

    def test_build_codex_trajectory_normalizes_non_cached_input_totals(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "last_token_usage": {
                                    "input_tokens": 5,
                                    "cached_input_tokens": 9,
                                    "output_tokens": 2,
                                    "reasoning_output_tokens": 1,
                                    "total_tokens": 8,
                                },
                                "total_token_usage": {
                                    "input_tokens": 5,
                                    "cached_input_tokens": 9,
                                    "output_tokens": 2,
                                    "reasoning_output_tokens": 1,
                                    "total_tokens": 8,
                                },
                            },
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertEqual(trajectory.final_metrics.total_prompt_tokens, 14)
            self.assertEqual(trajectory.final_metrics.total_cached_tokens, 9)
            self.assertEqual(
                trajectory.final_metrics.total_cost_usd,
                _rfc_cost(prompt_tokens=14, cached_tokens=9, completion_tokens=3),
            )

    def test_build_codex_trajectory_rejects_missing_user_step(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "first step must have source USER"):
                export_atif_trajectory.build_codex_trajectory(
                    rollout_path=rollout_path,
                    session_id="trial_1",
                    agent_profile_id="gpt_5_5_codex_high",
                    model_name="gpt-5.4",
                )

    def test_build_codex_trajectory_keeps_only_last_user_step(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "old answer"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:04.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "follow up"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:06.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "call_id": "call_exec",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "pwd"}),
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:07.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_exec",
                            "output": "Command: pwd\nOutput:\n/app\n",
                        },
                    },
                ],
            )

            trajectory = export_atif_trajectory.build_codex_trajectory(
                rollout_path=rollout_path,
                session_id="trial_1",
                agent_profile_id="gpt_5_5_codex_high",
                model_name="gpt-5.4",
            )

            self.assertEqual([step.step_id for step in trajectory.steps], [1, 2])
            self.assertEqual([step.source for step in trajectory.steps], ["USER", "AGENT"])
            self.assertEqual(trajectory.steps[0].message, "follow up")
            self.assertEqual(trajectory.steps[1].tool_calls[0].tool_call_id, "call_exec")
            self.assertEqual(trajectory.steps[1].duration, 2.0)
            self.assertEqual(trajectory.steps[1].observation[0].duration, 1.0)
            self.assertEqual(trajectory.final_metrics.total_steps, 2)
            self.assertEqual(trajectory.final_metrics.total_duration, 2.0)
            self.assertEqual(trajectory.final_metrics.total_tool_duration, 1.0)

    def test_build_codex_trajectory_rejects_negative_step_duration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "step 2 duration cannot be negative"):
                export_atif_trajectory.build_codex_trajectory(
                    rollout_path=rollout_path,
                    session_id="trial_1",
                    agent_profile_id="gpt_5_5_codex_high",
                    model_name="gpt-5.4",
                )

    def test_build_codex_trajectory_rejects_negative_observation_duration(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            rollout_path = Path(tmp_dir) / "rollout-test.jsonl"
            _write_jsonl(
                rollout_path,
                [
                    {
                        "timestamp": "2026-03-30T10:00:00.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "task"}],
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:02.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "pwd"}),
                            "call_id": "call_exec",
                        },
                    },
                    {
                        "timestamp": "2026-03-30T10:00:01.000Z",
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_exec",
                            "output": "Command: pwd\nOutput:\n/app\n",
                        },
                    },
                ],
            )

            with self.assertRaisesRegex(ValueError, "step 2 observation duration cannot be negative"):
                export_atif_trajectory.build_codex_trajectory(
                    rollout_path=rollout_path,
                    session_id="trial_1",
                    agent_profile_id="gpt_5_5_codex_high",
                    model_name="gpt-5.4",
                )

    def test_documented_total_duration_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_duration cannot be negative"):
            export_atif_trajectory._documented_total_duration(
                {
                    "steps": [
                        {"source": "USER"},
                        {"source": "AGENT", "duration": -1.0},
                    ]
                }
            )

    def test_documented_duration_totals_sum_values_and_ignore_missing(self) -> None:
        payload = {
            "steps": [
                {"source": "USER"},
                {
                    "source": "AGENT",
                    "duration": 1.5,
                    "observation": [{"duration": 0.25}, {"content": "missing duration"}],
                },
                {
                    "source": "AGENT",
                    "duration": 2,
                    "observation": [{"duration": 0.75}],
                },
            ]
        }

        self.assertEqual(export_atif_trajectory._documented_total_duration(payload), 3.5)
        self.assertEqual(export_atif_trajectory._documented_total_tool_duration(payload), 1.0)

    def test_documented_total_tool_duration_rejects_negative_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "total_tool_duration cannot be negative"):
            export_atif_trajectory._documented_total_tool_duration(
                {
                    "steps": [
                        {
                            "source": "AGENT",
                            "observation": [{"duration": -0.1}],
                        }
                    ]
                }
            )

    def test_missing_cost_profile_yields_nan_total_cost(self) -> None:
        profile = dict(export_atif_trajectory.AGENTIC_PROFILES_BY_ID["gpt_5_5_codex_high"])
        profile.pop("cost", None)
        export_atif_trajectory.AGENTIC_PROFILES_BY_ID["missing_cost_agent"] = profile
        try:
            final_metrics = export_atif_trajectory._build_final_metrics_from_values(
                step_count=1,
                agent_profile_id="missing_cost_agent",
                prompt_tokens=10,
                cached_tokens=2,
                completion_tokens=3,
            )
        finally:
            export_atif_trajectory.AGENTIC_PROFILES_BY_ID.pop("missing_cost_agent", None)

        self.assertTrue(math.isnan(final_metrics.total_cost_usd))

    def test_final_metrics_normalize_non_cached_input_totals(self) -> None:
        final_metrics = export_atif_trajectory._build_final_metrics_from_values(
            step_count=3,
            agent_profile_id="gpt_5_5_codex_high",
            prompt_tokens=47,
            cached_tokens=2_153_784,
            completion_tokens=115_409,
        )

        self.assertEqual(final_metrics.total_prompt_tokens, 2_153_831)
        self.assertEqual(final_metrics.total_cached_tokens, 2_153_784)
        self.assertEqual(
            final_metrics.total_cost_usd,
            _rfc_cost(
                prompt_tokens=2_153_831,
                cached_tokens=2_153_784,
                completion_tokens=115_409,
            ),
        )


if __name__ == "__main__":
    unittest.main()
