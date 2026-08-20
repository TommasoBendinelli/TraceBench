from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd
from click.testing import CliRunner

from shared.scores_schema import validate_scores_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflows.rollout import evaluate_accuracy


def _questions_payload(
    *,
    question_id: str,
    test_samples: list[str],
    interventions: dict[str, dict[str, object]],
    description: str = "description",
) -> dict[str, object]:
    return {
        "ground_truth_information": {
            "model_description": description,
            "interventions": interventions,
        },
        "questions": {
            question_id: {
                "question_text": {"model_description": description},
                "test_samples": test_samples,
            }
        },
    }


class TestEvaluateAccuracy(unittest.TestCase):
    def test_direct_multi_answer_scoring_uses_documented_formula(self) -> None:
        choices = ["A", "B", "no parameter changed"]

        predicted = evaluate_accuracy._resolve_classification_choices(["B", "A"], choices)

        self.assertEqual(predicted, ["B", "A"])
        self.assertEqual(evaluate_accuracy._score_direct_prediction(predicted, "A", choices), 0.5)
        self.assertEqual(
            evaluate_accuracy._score_direct_prediction(["A"], "A", choices),
            1.0,
        )
        self.assertEqual(
            evaluate_accuracy._score_direct_prediction(["B"], "A", choices),
            0.0,
        )

    def test_scores_payload_uses_null_num_answers_for_sample_errors(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "scores.json"
            payload = evaluate_accuracy._build_scores_payload(
                agent_run_id="demo_run",
                sample_results={
                    "ok": {
                        "predictions": ["A"],
                        "top1_correct": True,
                        "shortlist_score": 1.0,
                        "num_answers": 2.0,
                        "sample_type": "test",
                    },
                    "bad": {
                        "predictions": [],
                        "top1_correct": False,
                        "shortlist_score": 0.0,
                        "num_answers": None,
                        "sample_type": "test",
                        "error": "ValueError: demo",
                    },
                },
            )

            evaluate_accuracy._write_json(output_path, payload)
            written = output_path.read_text(encoding="utf-8")

            self.assertEqual(payload["final_metric_test"]["average_num_answers"], 2.0)
            self.assertIsNone(payload["sample_results"]["bad"]["num_answers"])
            self.assertIn('"num_answers": null', written)
            self.assertNotIn("NaN", written)

    def test_score_code_rule_artifact_records_sample_errors(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tsenv_root = root / "tsENV_questions"
            model_root = tsenv_root / "DemoModel"
            trial_dir = root / "runs" / "demo_run" / "question_code" / "trial"
            model_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (model_root / "model_record.json").write_text("{}", encoding="utf-8")
            (model_root / "sample_manifest.json").write_text("{}", encoding="utf-8")
            (trial_dir / "rule.py").write_text("def predict(df):\n    return ['A']\n", encoding="utf-8")

            manifest_item = mock.Mock(
                test_samples=["test_a", "test_b"],
                other_samples=["other_a"],
            )

            def fake_result(*, run_id: str, **_: object) -> dict[str, object]:
                if run_id == "test_b":
                    return {
                        "predicted_label": "",
                        "top1_correct": False,
                        "shortlist_score": 0.0,
                        "answer_count": 0,
                        "error": "ValueError: bad prediction",
                    }
                return {
                    "predicted_label": ["A", "B"] if run_id == "test_a" else "B",
                    "top1_correct": run_id == "test_a",
                    "shortlist_score": 0.5 if run_id == "test_a" else 0.0,
                    "answer_count": 2 if run_id == "test_a" else 1,
                    "error": None,
                }

            with mock.patch.object(evaluate_accuracy, "TSENV_ROOT", tsenv_root), mock.patch.object(
                evaluate_accuracy,
                "_load_tsenv_question_lookup",
                return_value=("DemoModel", {"payload": True}, {"question_id": "q_code"}),
            ), mock.patch.object(
                evaluate_accuracy,
                "_question_manifest_item",
                return_value=manifest_item,
            ), mock.patch.object(
                evaluate_accuracy,
                "label_for_question_sample",
                return_value="A",
            ), mock.patch.object(
                evaluate_accuracy,
                "materialize",
                return_value=object(),
            ), mock.patch.object(
                evaluate_accuracy,
                "evaluate_rule_on_dataframe",
                side_effect=fake_result,
            ):
                sample_results = evaluate_accuracy._score_code_rule_artifact(
                    trial_dir=trial_dir,
                    question_row={
                        "question_id": "q_code",
                        "multiple_choices": ["A", "B"],
                    },
                    run_metadata={"dataset_path": str(root / "tasks" / "DemoModel" / "dataset")},
                    resolved_noise_level="low",
                    resolved_seed=7,
                )

            self.assertEqual(list(sample_results), ["test_a", "test_b", "other_a"])
            self.assertEqual(sample_results["test_a"]["predictions"], ["A", "B"])
            self.assertEqual(sample_results["test_a"]["num_answers"], 2.0)
            self.assertEqual(sample_results["test_b"]["top1_correct"], False)
            self.assertEqual(sample_results["test_b"]["shortlist_score"], 0.0)
            self.assertIsNone(sample_results["test_b"]["num_answers"])
            self.assertEqual(sample_results["test_b"]["error"], "ValueError: bad prediction")
            self.assertEqual(sample_results["other_a"]["sample_type"], "other")

    def test_load_question_index_creates_aliases_from_run_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "runs" / "demo_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            trial_dir = run_dir / "question_alias" / "question_alias.1-of-1.demo_run"
            trial_dir.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "demo_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_alias",
                                "trial_name": "question_alias.1-of-1.demo_run",
                                "task_hash": "q_alias_hash",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_alias",
                        "eval_mode": "direct",
                        "context": "high",
                        "shot": "zero_shot",
                        "test_samples": ["test_samples/a.parquet"],
                        "test_sample_labels": {"dataframes/a.parquet": "X"},
                        "multiple_choices": ["X", "Y"],
                    }
                ),
                encoding="utf-8",
            )

            index = evaluate_accuracy.load_question_index(run_dir)

            self.assertIn("q_alias", index)
            self.assertIn("question_alias", index)
            self.assertIn("q_alias_hash", index)
            self.assertEqual(index["q_alias"]["benchmark"], "tsenv_cls")
            self.assertEqual(index["q_alias"]["variant"], "high_zero_shot")

    def test_load_tsenv_question_lookup_accepts_dataset_slug_when_scenario_uses_hash(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metadata_dir = root / "tsENV_questions" / "BallDrop"
            dataset_root = root / "tasks_runtime" / "manual" / "BallDrop" / "frost_01234-willow_0"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "questions": {
                            "frost_01234-willow_0": {
                                "question_hash": "hash_from_scenario",
                                "question_text": {},
                                "recipe_info": {},
                            }
                        },
                        "ground_truth_information": {"interventions": {}},
                        "label_int_mapping": {},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "TSENV_ROOT", root / "tsENV_questions"):
                model_name, _, question = evaluate_accuracy._load_tsenv_question_lookup(
                    {"dataset_path": str(dataset_root)},
                    "hash_from_scenario",
                )

            self.assertEqual(model_name, "BallDrop")
            self.assertEqual(question["question_id"], "frost_01234-willow_0")
            self.assertEqual(question["question_hash"], "hash_from_scenario")

    def test_main_scores_code_and_direct_and_writes_summary(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "demo_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            run_dir.mkdir(parents=True, exist_ok=True)
            dataset_root.mkdir(parents=True, exist_ok=True)

            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "demo_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_code",
                                "trial_name": "question_code.1-of-1.demo_run",
                                "task_hash": "q_code",
                            },
                            {
                                "task_id": "question_direct",
                                "trial_name": "question_direct.1-of-1.demo_run",
                                "task_hash": "q_direct",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            code_trial_dir = run_dir / "question_code" / "question_code.1-of-1.demo_run"
            code_trial_dir.mkdir(parents=True, exist_ok=True)
            (code_trial_dir / "results.json").write_text(
                json.dumps(
                    {
                        "task_hash": "q_code",
                        "task_id": "question_code",
                        "trial_name": "question_code.1-of-1.demo_run",
                    }
                ),
                encoding="utf-8",
            )
            (code_trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_code",
                        "question_hash": "q_code",
                        "eval_mode": "code",
                        "context": "high",
                        "shot": "few_shot",
                        "recipe": "recipe_code",
                        "recipe_info": {
                            "noise_profile": "high",
                            "noise_seed": 13,
                        },
                        "multiple_choices": ["A", "B"],
                        "test_samples": [
                            "test_samples/code_0.parquet",
                            "test_samples/code_1.parquet",
                        ],
                        "test_samples_source_paths": [
                            "dataframes/source_code_0.parquet",
                            "dataframes/source_code_1.parquet",
                        ],
                        "test_sample_labels": {
                            "dataframes/source_code_0.parquet": "A",
                            "dataframes/source_code_1.parquet": "B",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (code_trial_dir / "rule.py").write_text(
                "def predict(df):\n"
                "    return 'A'\n",
                encoding="utf-8",
            )
            code_data_dir = dataset_root / "question_code" / "test_samples"
            code_data_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"value": [0.0]}).to_parquet(code_data_dir / "code_0.parquet")
            pd.DataFrame({"value": [1.0]}).to_parquet(code_data_dir / "code_1.parquet")

            direct_trial_dir = run_dir / "question_direct" / "question_direct.1-of-1.demo_run"
            direct_trial_dir.mkdir(parents=True, exist_ok=True)
            (direct_trial_dir / "results.json").write_text(
                json.dumps(
                    {
                        "direct_0.parquet": "X",
                        "direct_1.parquet": "Y",
                    }
                ),
                encoding="utf-8",
            )
            (direct_trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_direct",
                        "question_hash": "q_direct",
                        "eval_mode": "direct",
                        "context": "low",
                        "shot": "one_shot",
                        "recipe": "recipe_direct",
                        "multiple_choices": ["X", "Y"],
                        "test_samples": [
                            "test_samples/direct_0.parquet",
                            "test_samples/direct_1.parquet",
                        ],
                        "test_samples_source_paths": [
                            "dataframes/source_direct_0.parquet",
                            "dataframes/source_direct_1.parquet",
                        ],
                        "test_sample_labels": {
                            "dataframes/source_direct_0.parquet": "X",
                            "dataframes/source_direct_1.parquet": "Y",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root), mock.patch.object(
                evaluate_accuracy,
                "_score_code_rule_artifact",
                return_value={
                    "code_0": {
                        "predictions": ["A"],
                        "top1_correct": True,
                        "shortlist_score": 0.52,
                        "num_answers": 1.0,
                        "sample_type": "test",
                    },
                    "code_1": {
                        "predictions": ["A"],
                        "top1_correct": False,
                        "shortlist_score": 0.28,
                        "num_answers": 1.4,
                        "sample_type": "test",
                    },
                },
            ) as mock_score_code_rule_artifact:
                result = runner.invoke(evaluate_accuracy.main, ["demo_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn(
                f"Saved scores.json to {(code_trial_dir / 'scores.json').resolve()}",
                result.output,
            )
            self.assertIn(
                f"Saved scores.json to {(direct_trial_dir / 'scores.json').resolve()}",
                result.output,
            )
            self.assertNotIn("Saved accuracy summary", result.output)
            summary = json.loads((run_dir / "accuracy_summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["batch_accuracy"], 0.75)
            self.assertEqual(summary["total_evaluable_answers"], 4)
            self.assertEqual(summary["mode_counts"], {"code": 1, "direct": 1})
            code_scores = json.loads((code_trial_dir / "scores.json").read_text(encoding="utf-8"))
            direct_scores = json.loads((direct_trial_dir / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(code_scores["agent_run_id"], "demo_run")
            self.assertEqual(code_scores["final_metric_test"]["average_top1_accuracy"], 0.5)
            self.assertEqual(code_scores["final_metric_test"]["average_shortlist_score"], 0.4)
            self.assertEqual(code_scores["final_metric_test"]["average_num_answers"], 1.2)
            self.assertEqual(
                code_scores["final_metric_other"],
                {
                    "average_top1_accuracy": 0.0,
                    "average_shortlist_score": 0.0,
                    "average_num_answers": 0.0,
                },
            )
            self.assertEqual(code_scores["sample_results"]["code_0"]["sample_type"], "test")
            self.assertEqual(code_scores["sample_results"]["code_0"]["top1_correct"], True)
            self.assertEqual(code_scores["sample_results"]["code_1"]["top1_correct"], False)
            self.assertEqual(direct_scores["agent_run_id"], "demo_run")
            self.assertEqual(direct_scores["is_correct_format"], True)
            self.assertEqual(direct_scores["final_metric_test"]["average_top1_accuracy"], 1.0)
            self.assertEqual(direct_scores["final_metric_test"]["average_shortlist_score"], 1.0)
            self.assertEqual(
                direct_scores["final_metric_test"]["average_num_answers"],
                1.0,
            )
            self.assertEqual(direct_scores["sample_results"]["direct_0"]["predictions"], ["X"])
            self.assertEqual(direct_scores["sample_results"]["direct_0"]["top1_correct"], True)
            self.assertEqual(direct_scores["sample_results"]["direct_1"]["top1_correct"], True)
            validate_scores_payload(code_scores, path=str(code_trial_dir / "scores.json"))
            validate_scores_payload(direct_scores, path=str(direct_trial_dir / "scores.json"))
            mock_score_code_rule_artifact.assert_called_once_with(
                trial_dir=code_trial_dir.resolve(),
                question_row=mock.ANY,
                run_metadata=mock.ANY,
                resolved_noise_level="high",
                resolved_seed=13,
            )

    def test_main_accepts_run_dir_via_path(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "open_run"
            dataset_root = (
                root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            )
            trial_dir = run_dir / "question_open" / "question_open.1-of-1.open_run"
            dataset_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "open_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_open",
                                "trial_name": "question_open.1-of-1.open_run",
                                "task_hash": "q_open",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_open",
                        "eval_mode": "open-ended",
                        "context": "high",
                        "shot": "zero_shot",
                        "multiple_choices": ["X", "Y"],
                        "test_samples": ["test_samples/a.parquet"],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"test_samples/a.parquet": "X"}),
                encoding="utf-8",
            )
            metadata_dir = root / "tsENV_questions" / "TransmissionLine"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    _questions_payload(
                        question_id="q_open",
                        test_samples=["dataframes/a.parquet"],
                        interventions={"a": {"changed_parameter": "X"}},
                    )
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root), mock.patch.object(
                evaluate_accuracy, "TSENV_ROOT", root / "tsENV_questions"
            ):
                result = runner.invoke(
                    evaluate_accuracy.main,
                    ["--path", str(run_dir)],
                )

            self.assertEqual(result.exit_code, 0, msg=result.output)
            summary = json.loads((run_dir / "accuracy_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["agentic_run_id"], "open_run")

    def test_main_rejects_both_run_id_and_path(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "demo_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = runner.invoke(
                evaluate_accuracy.main,
                ["demo_run", "--path", str(run_dir)],
            )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Pass exactly one", result.output)

    def test_main_requires_run_id_or_path(self) -> None:
        result = CliRunner().invoke(evaluate_accuracy.main, [])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Pass exactly one", result.output)

    def test_direct_multi_answer_predictions_are_stored_as_arrays(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "direct_run"
            dataset_root = (
                root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            )
            trial_dir = run_dir / "question_direct" / "question_direct.1-of-1.direct_run"
            dataset_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "direct_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_direct",
                                "trial_name": "question_direct.1-of-1.direct_run",
                                "task_hash": "q_direct",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_direct",
                        "question_hash": "q_direct",
                        "eval_mode": "direct",
                        "context": "low",
                        "shot": "one_shot",
                        "multiple_choices": ["X", "Y", "No parameter changed"],
                        "test_samples": ["test_samples/direct_0.parquet"],
                        "test_samples_source_paths": ["dataframes/source_direct_0.parquet"],
                        "test_sample_labels": {
                            "dataframes/source_direct_0.parquet": "X",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "results.json").write_text(
                json.dumps({"direct_0.parquet": ["X", "Y"]}),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root):
                result = runner.invoke(evaluate_accuracy.main, ["direct_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            scores = json.loads((trial_dir / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(scores["is_correct_format"], True)
            self.assertEqual(scores["final_metric_test"]["average_top1_accuracy"], 1.0)
            self.assertEqual(scores["final_metric_test"]["average_shortlist_score"], 0.5)
            self.assertEqual(scores["final_metric_test"]["average_num_answers"], 2.0)
            self.assertEqual(scores["sample_results"]["direct_0"]["predictions"], ["X", "Y"])
            self.assertEqual(scores["sample_results"]["direct_0"]["top1_correct"], True)
            validate_scores_payload(scores, path=str(trial_dir / "scores.json"))

    def test_direct_prefixed_results_fallback_marks_incorrect_format(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "direct_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            trial_dir = run_dir / "question_direct" / "question_direct.1-of-1.direct_run"
            dataset_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "direct_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_direct",
                                "trial_name": "question_direct.1-of-1.direct_run",
                                "task_hash": "q_direct",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_direct",
                        "question_hash": "q_direct",
                        "eval_mode": "direct",
                        "context": "low",
                        "shot": "one_shot",
                        "multiple_choices": ["X", "Y"],
                        "test_samples": ["test_samples/direct_0.parquet"],
                        "test_samples_source_paths": ["dataframes/source_direct_0.parquet"],
                        "test_sample_labels": {"dataframes/source_direct_0.parquet": "X"},
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "results.json").write_text(
                json.dumps({"test_samples/direct_0.parquet": ["Y", "X"]}),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root):
                result = runner.invoke(evaluate_accuracy.main, ["direct_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            scores = json.loads((trial_dir / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(scores["is_correct_format"], False)
            self.assertEqual(scores["final_metric_test"]["average_top1_accuracy"], 0.0)
            self.assertEqual(scores["final_metric_test"]["average_shortlist_score"], 0.5)
            self.assertEqual(scores["sample_results"]["direct_0"]["predictions"], ["Y", "X"])
            validate_scores_payload(scores, path=str(trial_dir / "scores.json"))

    def test_missing_artifact_skips_trial_without_scores(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "direct_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            trial_dir = run_dir / "question_direct" / "question_direct.1-of-1.direct_run"
            dataset_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "direct_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_direct",
                                "trial_name": "question_direct.1-of-1.direct_run",
                                "task_hash": "q_direct",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_direct",
                        "question_hash": "q_direct",
                        "eval_mode": "direct",
                        "context": "low",
                        "shot": "one_shot",
                        "multiple_choices": ["X", "Y"],
                        "test_samples": ["test_samples/direct_0.parquet"],
                        "test_samples_source_paths": ["dataframes/source_direct_0.parquet"],
                        "test_sample_labels": {"dataframes/source_direct_0.parquet": "X"},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root):
                result = runner.invoke(evaluate_accuracy.main, ["direct_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertFalse((trial_dir / "scores.json").exists())
            summary = json.loads((run_dir / "accuracy_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["evaluated_trials"], 0)
            self.assertEqual(summary["skipped_trials"], 1)
            self.assertEqual(summary["total_evaluable_answers"], 0)

    def test_main_scores_open_trials_without_judge_helper(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            runs_root = root / "runs"
            run_dir = runs_root / "open_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "demo_dataset"
            trial_dir = run_dir / "question_open" / "question_open.1-of-1.open_run"
            dataset_root.mkdir(parents=True, exist_ok=True)
            trial_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "run_id": "open_run",
                        "dataset_path": str(dataset_root),
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
                                "task_id": "question_open",
                                "trial_name": "question_open.1-of-1.open_run",
                                "task_hash": "q_open",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_open",
                        "eval_mode": "open-ended",
                        "context": "high",
                        "shot": "zero_shot",
                        "multiple_choices": ["X", "Y"],
                        "test_samples": ["test_samples/a.parquet"],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"test_samples/a.parquet": "X"}),
                encoding="utf-8",
            )
            metadata_dir = root / "tsENV_questions" / "TransmissionLine"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    _questions_payload(
                        question_id="q_open",
                        test_samples=["dataframes/a.parquet"],
                        interventions={"a": {"changed_parameter": "X"}},
                    )
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "RUNS_ROOT", runs_root), mock.patch.object(
                evaluate_accuracy, "TSENV_ROOT", root / "tsENV_questions"
            ):
                result = runner.invoke(evaluate_accuracy.main, ["open_run"])

            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn(
                f"Saved scores.json to {(trial_dir / 'scores.json').resolve()}",
                result.output,
            )
            summary = json.loads((run_dir / "accuracy_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["evaluated_trials"], 1)
            self.assertEqual(summary["errored_trials"], 0)
            self.assertTrue((trial_dir / "scores.json").exists())

    def test_score_open_trial_writes_per_sample_scores(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tsenv_root = root / "tsENV_questions"
            run_dir = root / "runs" / "open_run"
            dataset_root = root / "tasks_runtime" / "manual" / "TransmissionLine" / "open_dataset"
            trial_dir = run_dir / "question_open" / "question_open.1-of-1.open_run"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_open",
                        "eval_mode": "open-ended",
                        "context": "high",
                        "shot": "zero_shot",
                        "instruction_agent_format": "Explain the intervention.",
                        "test_samples": [
                            "test_samples/child_1.parquet",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"test_samples/child_1.parquet": "Z"}),
                encoding="utf-8",
            )
            metadata_dir = tsenv_root / "TransmissionLine"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "ground_truth_information": {
                            "model_description": "Physics explanation.",
                            "shared_description": (
                                "The simulation lasts until `end_time_input_s` and is sampled at `sampling_rate_hz`.\n\n"
                                "The intervention trajectory is created by changing one and one parameter only at the `intervention_time`."
                            ),
                            "interventions": {
                                "child_1": {
                                    "initial_parameters": {"gravity": 9.81},
                                    "intervention_time": 8.5,
                                    "changed_parameter": "Z",
                                    "new_value": 30,
                                }
                            },
                        },
                        "questions": {
                            "q_open": {
                                "question_text": {
                                    "model_description": "Open prompt description."
                                },
                                "test_samples": [
                                    "dataframes/child_1.parquet",
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "TSENV_ROOT", tsenv_root):
                payload = evaluate_accuracy._score_open_trial(
                    entry={
                        "task_hash": "q_open",
                        "task_id": "question_open",
                        "trial_name": "question_open.1-of-1.open_run",
                    },
                    trial_dir=trial_dir,
                    question_row={
                        "question_id": "q_open",
                        "multiple_choices": ["Z", "Y"],
                        "test_samples": ["test_samples/child_1.parquet"],
                    },
                    run_metadata={
                        "run_id": "open_run",
                        "dataset_path": str(dataset_root),
                        "agent_name": "codex",
                        "model_name": "gpt-5.4",
                    },
                )

            self.assertEqual(payload["final_metric_test"]["average_top1_accuracy"], 1.0)
            written = json.loads((trial_dir / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(written["agent_run_id"], "open_run")
            self.assertEqual(written["sample_results"]["child_1"]["predictions"], ["Z"])
            self.assertEqual(written["sample_results"]["child_1"]["top1_correct"], True)
            self.assertEqual(written["sample_results"]["child_1"]["shortlist_score"], 1.0)
            self.assertEqual(written["sample_results"]["child_1"]["sample_type"], "test")

    def test_score_open_trial_derives_labels_from_interventions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tsenv_root = root / "tsENV_questions"
            run_dir = root / "runs" / "open_run"
            dataset_root = root / "tasks_runtime" / "manual" / "BallDrop" / "open_dataset"
            trial_dir = run_dir / "question_open" / "question_open.1-of-1.open_run"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_open",
                        "eval_mode": "open-ended",
                        "context": "high",
                        "shot": "zero_shot",
                        "instruction_agent_format": "Classify all test samples.",
                        "test_samples": [
                            "test_samples/a.parquet",
                            "test_samples/b.parquet",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps(
                    {
                        "test_samples/a.parquet": "drag_coeff",
                        "test_samples/b.parquet": "gravity",
                    }
                ),
                encoding="utf-8",
            )
            metadata_dir = tsenv_root / "BallDrop"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    {
                        "ground_truth_information": {
                            "model_description": "Physics explanation.",
                            "interventions": {
                                "a": {"changed_parameter": "drag_coeff"},
                                "b": {"changed_parameter": "gravity"},
                            },
                        },
                        "questions": {
                            "q_open": {
                                "question_text": {
                                    "model_description": "Open prompt description."
                                },
                                "test_samples": [
                                    "dataframes/a.parquet",
                                    "dataframes/b.parquet",
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "TSENV_ROOT", tsenv_root):
                payload = evaluate_accuracy._score_open_trial(
                    entry={
                        "task_hash": "q_open",
                        "task_id": "question_open",
                        "trial_name": "question_open.1-of-1.open_run",
                    },
                    trial_dir=trial_dir,
                    question_row={
                        "question_id": "q_open",
                        "multiple_choices": ["drag_coeff", "restitution", "mass", "gravity"],
                        "test_samples": [
                            "test_samples/a.parquet",
                            "test_samples/b.parquet",
                        ],
                    },
                    run_metadata={
                        "run_id": "open_run",
                        "dataset_path": str(dataset_root),
                        "agent_name": "codex",
                        "model_name": "gpt-5.4",
                    },
                )

            self.assertEqual(payload["final_metric_test"]["average_top1_accuracy"], 1.0)
            self.assertEqual(payload["sample_results"]["a"]["predictions"], ["drag_coeff"])
            self.assertEqual(payload["sample_results"]["b"]["predictions"], ["gravity"])

    def test_score_open_trial_marks_unparseable_prediction_as_maybe(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tsenv_root = root / "tsENV_questions"
            run_dir = root / "runs" / "open_run"
            dataset_root = root / "tasks_runtime" / "manual" / "BallDrop" / "open_dataset"
            trial_dir = run_dir / "question_open" / "question_open.1-of-1.open_run"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "scenario_info.json").write_text(
                json.dumps(
                    {
                        "question_id": "q_open",
                        "eval_mode": "open-ended",
                        "context": "high",
                        "shot": "zero_shot",
                        "instruction_agent_format": "Classify all test samples.",
                        "test_samples": ["test_samples/a.parquet"],
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"test_samples/a.parquet": "I am not sure"}),
                encoding="utf-8",
            )
            metadata_dir = tsenv_root / "BallDrop"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            (metadata_dir / "questions.json").write_text(
                json.dumps(
                    _questions_payload(
                        question_id="q_open",
                        test_samples=["dataframes/a.parquet"],
                        interventions={"a": {"changed_parameter": "drag_coeff"}},
                        description="Open prompt description.",
                    )
                ),
                encoding="utf-8",
            )

            with mock.patch.object(evaluate_accuracy, "TSENV_ROOT", tsenv_root):
                payload = evaluate_accuracy._score_open_trial(
                    entry={
                        "task_hash": "q_open",
                        "task_id": "question_open",
                        "trial_name": "question_open.1-of-1.open_run",
                    },
                    trial_dir=trial_dir,
                    question_row={
                        "question_id": "q_open",
                        "multiple_choices": ["drag_coeff", "restitution", "mass", "gravity"],
                        "test_samples": ["test_samples/a.parquet"],
                    },
                    run_metadata={
                        "run_id": "open_run",
                        "dataset_path": str(dataset_root),
                        "agent_name": "codex",
                        "model_name": "gpt-5.4",
                    },
                )

            self.assertEqual(payload["final_metric_test"]["average_top1_accuracy"], 0.0)
            self.assertEqual(payload["sample_results"]["a"]["predictions"], ["I am not sure"])
            self.assertEqual(payload["sample_results"]["a"]["top1_correct"], False)
            self.assertEqual(payload["sample_results"]["a"]["shortlist_score"], 0.0)

    def test_resolve_open_answer_raises_when_interventions_missing_sample(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing ground-truth label"):
            evaluate_accuracy._resolve_open_answer(
                question_row={"question_id": "q_open"},
                question={
                    "question_text": {"model_description": "description"},
                    "test_samples": ["dataframes/a.parquet", "dataframes/b.parquet"],
                },
                metadata_payload={
                    "ground_truth_information": {
                        "interventions": {
                            "a": {"changed_parameter": "drag_coeff"},
                        }
                    }
                },
                agent_payload={"test_samples/a.parquet": "drag_coeff"},
            )

    def test_score_non_open_trial_rejects_legacy_scalar_benchmark(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            trial_dir = root / "runs" / "legacy_run" / "question_ucr" / "question_ucr.1-of-1.legacy_run"
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "results.json").write_text(
                json.dumps(
                    {
                        "task_hash": "q_ucr",
                        "task_id": "question_ucr",
                        "trial_name": "question_ucr.1-of-1.legacy_run",
                    }
                ),
                encoding="utf-8",
            )
            (trial_dir / "agentic-final-response.json").write_text(
                json.dumps({"final_answer": "alpha"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "legacy scalar-answer scoring"):
                evaluate_accuracy._score_non_open_trial(
                    entry={
                        "task_hash": "q_ucr",
                        "task_id": "question_ucr",
                        "trial_name": "question_ucr.1-of-1.legacy_run",
                    },
                    trial_dir=trial_dir,
                    question_row={
                        "question_id": "q_ucr",
                        "question_root": str(root / "dataset" / "question_ucr"),
                        "benchmark": "ucr",
                        "variant": "ucr_none_one_shot",
                        "eval_mode": "direct",
                    },
                    run_metadata={
                        "agent_name": "codex",
                        "model_name": "gpt-5.4",
                    },
                )


if __name__ == "__main__":
    unittest.main()
