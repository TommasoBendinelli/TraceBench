from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.rollout import run_single_question as wrapper


class TestRunSingleQuestionWrapper(unittest.TestCase):
    def test_main_forwards_documented_cli_to_internal_runner(self) -> None:
        with mock.patch.object(wrapper.internal_runner, "run_single_question", return_value=0) as mocked:
            exit_code = wrapper.main(
                [
                    "--model",
                    "BallDrop",
                    "--question-slug",
                    "q_demo",
                    "--agent-id",
                    "gpt_5_5_codex_high",
                    "--agentic-run-id",
                    "run_demo",
                    "--keep-container",
                ]
            )

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(
            question_id="q_demo",
            agent_profile_id="gpt_5_5_codex_high",
            run_id="run_demo",
            model_name="BallDrop",
            keep_container=True,
            single_run=False,
            tag="DEBUG",
            configuration_file_name=None,
            questions_root=None,
        )

    def test_main_rejects_legacy_positional_invocation(self) -> None:
        with self.assertRaises(SystemExit):
            wrapper.main(["q_demo", "gpt_5_5_codex_high", "run_demo"])

    def test_main_rejects_batch_size_flag(self) -> None:
        with self.assertRaises(SystemExit):
            wrapper.main(
                [
                    "--model",
                    "BallDrop",
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

    def test_main_reads_rollout_tag_from_environment(self) -> None:
        with mock.patch.dict(
            wrapper.os.environ,
            {"TSENV_AGENTIC_ROLLOUT_TAG": "EXPERIMENT"},
        ):
            with mock.patch.object(wrapper.internal_runner, "run_single_question", return_value=0) as mocked:
                exit_code = wrapper.main(
                    [
                        "--model",
                        "BallDrop",
                        "--question-slug",
                        "q_demo",
                        "--agent-id",
                        "gpt_5_5_codex_high",
                        "--agentic-run-id",
                        "run_demo",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(mocked.call_args.kwargs["tag"], "EXPERIMENT")


if __name__ == "__main__":
    unittest.main()
