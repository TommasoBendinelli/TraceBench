from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ADAPTER_DIR = Path(__file__).resolve().parents[2] / "terminal-bench" / "adapters" / "tsENV"
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))

from adapter import IndustrialTimeSeriesAdapter, _join_tsenv_question_text, _normalize_tsenv_question  # noqa: E402
from shared.prompts import render_tsenv_agent_prompt  # noqa: E402


def _write_bundle(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataframes").mkdir()
    pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]}).to_parquet(
        path / "dataframes" / "sample.parquet"
    )
    pd.DataFrame({"signal": [3.0, 4.0], "time": [0.0, 1.0]}).to_parquet(
        path / "dataframes" / "train.parquet"
    )
    (path / "noise_adder.py").write_text(
        "\n".join(
            [
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "",
                "def quantify_noise(clean, noisy, reference):",
                "    return {'global': [0.0], 'local': [None]}",
                "",
                "def add_noise(src, seed=0, noise_level='low', ref=None):",
                "    return src, quantify_noise(src, src, ref)",
            ]
        ),
        encoding="utf-8",
    )
    (path / "questions.json").write_text(
        json.dumps(
            {
                "questions": {
                    "wanted": {
                        "train_samples": ["dataframes/train.parquet"],
                        "test_samples": ["dataframes/sample.parquet"],
                        "question_text": {
                            "first_sentence": "First",
                            "model_description": "Model",
                            "shared_description": "Shared {questions.<question_slug>.question_text.label_choices_json}",
                            "possible_interventions_parameter": "\"changed\"",
                            "label_choices_json": ["changed", "no parameter change"],
                            "task_instruction": "Question: determine which parameter changed.\n\nData\n\nFormat\n\nGeneric",
                            "ordered_field_agent_prompt": [
                                "first_sentence",
                                "model_description",
                                "shared_description",
                                "task_instruction",
                            ],
                        },
                        "recipe_info": {
                            "type_of_request": "direct",
                            "desc_level": "high",
                            "noise_level": "none",
                            "number_train_samples_per_class": 1,
                            "question_seed": 0,
                            "row_slug": "row",
                        },
                    },
                    "broken": {
                        "question_text": {},
                        "recipe_info": {
                            "type_of_request": "direct",
                            "desc_level": "high",
                            "noise_level": "none",
                            "number_train_samples_per_class": 0,
                            "question_seed": 0,
                        },
                    },
                },
                "label_int_mapping": {"changed": 0, "no parameter change": 1},
                "ground_truth_by_path": {
                    "dataframes/sample.parquet": "changed",
                    "dataframes/train.parquet": "no parameter change",
                },
                "environment_name": path.name,
            }
        ),
        encoding="utf-8",
    )


def test_adapter_filters_requested_question_before_normalization(tmp_path: Path) -> None:
    bundle = tmp_path / "DemoModel"
    _write_bundle(bundle)

    adapter = IndustrialTimeSeriesAdapter(
        dataset_root=bundle,
        task_dir=tmp_path / "tasks",
        include_question_ids=["wanted"],
    )

    assert [item.relative_id for item in adapter.discover()] == ["wanted"]


def test_normalize_tsenv_question_prefers_question_allowed_labels() -> None:
    question = {
        "train_samples": [],
        "test_samples": ["dataframes/sample.parquet"],
        "question_text": {
            "sample_source": "Samples.",
            "label_space": 'Allowed labels:\n["class_1", "no parameter change"]',
            "allowed_labels": ["class_1", "no parameter change"],
            "ordered_field_agent_prompt": ["sample_source", "label_space"],
        },
        "recipe_info": {
            "type_of_request": "direct",
            "desc_level": "none",
            "noise_level": "none",
            "number_train_samples_per_class": 0,
            "question_seed": 0,
            "row_slug": "row",
        },
    }

    normalized = _normalize_tsenv_question(
        question_id="wanted",
        question=question,
        payload={"questions": {"wanted": question}, "label_int_mapping": {"real label": 0}},
        dataset_name="DemoModel",
        multiple_choices=["real label", "no parameter change"],
        ground_truth_by_path={"dataframes/sample.parquet": "class_1"},
    )

    assert normalized["multiple_choices"] == ["class_1", "no parameter change"]
    assert normalized["test_sample_labels"] == {"dataframes/sample.parquet": "class_1"}


def test_prompt_fields_use_documented_double_newline_separator() -> None:
    question_text = {
        "first_sentence": "First",
        "model_description": "Model",
        "shared_description": "",
        "task_instruction": "Question?\n\nData",
        "ordered_field_agent_prompt": [
            "first_sentence",
            "model_description",
            "shared_description",
            "task_instruction",
        ],
    }
    text = render_tsenv_agent_prompt(question_text)

    assert text == "First\n\nModel\n\nQuestion?\n\nData"
    assert _join_tsenv_question_text(question_text) == text


def test_prompt_fields_accept_documented_tuple_separator_entries() -> None:
    question_text = {
        "sample_source": "First",
        "environment_description": "Model",
        "observed_columns": "",
        "task_artifact": "Question?\n\nData",
        "ordered_field_agent_prompt": [
            ["sample_source", "\n\n"],
            ["environment_description", "\n\n"],
            ["observed_columns", "\n\n"],
            ["task_artifact", ""],
        ],
    }
    text = render_tsenv_agent_prompt(question_text)

    assert text == "First\n\nModel\n\nQuestion?\n\nData"
    assert _join_tsenv_question_text(question_text) == text


def test_prompt_renderer_uses_documented_default_order() -> None:
    text = render_tsenv_agent_prompt(
        {
            "first_sentence": "First",
            "model_description": "Model",
            "shared_description": "Shared",
            "task_instruction": "Question?\n\nData\n\nFormat\n\nGeneric",
        }
    )

    assert text == "First\n\nModel\n\nShared\n\nQuestion?\n\nData\n\nFormat\n\nGeneric"


def test_prompt_renderer_resolves_documented_question_text_placeholders() -> None:
    question_text = {
        "model_description": "Model {questions.<question_slug>.question_text.label_choices_json}",
        "label_choices_json": ["alpha", "beta"],
        "ordered_field_agent_prompt": ["model_description"],
    }

    assert (
        render_tsenv_agent_prompt(question_text, question_slug="wanted")
        == 'Model ["alpha", "beta"]'
    )


def test_adapter_generates_documented_task_layout(tmp_path: Path) -> None:
    bundle = tmp_path / "DemoModel"
    task_dir = tmp_path / "tasks"
    _write_bundle(bundle)

    adapter = IndustrialTimeSeriesAdapter(
        dataset_root=bundle,
        task_dir=task_dir,
        include_question_ids=["wanted"],
    )
    adapter.generate_task("wanted", "question_0")

    question_root = task_dir / "question_0"
    scenario = json.loads((question_root / "scenario_info.json").read_text(encoding="utf-8"))
    source_payload = json.loads((bundle / "questions.json").read_text(encoding="utf-8"))
    assert scenario["question_schema"] == source_payload["questions"]["wanted"]
    assert scenario["desc_level"] == "high"
    assert "context" not in scenario
    assert scenario["test_samples"] == ["test_samples/sample.parquet"]
    assert scenario["train_samples"] == ["train_samples/train.parquet"]
    assert (question_root / "test_samples" / "sample.parquet").exists()
    assert (question_root / "train_samples" / "train.parquet").exists()
    assert (question_root / "agent_payload" / "test_samples" / "sample.parquet").exists()
    assert (question_root / "agent_payload" / "train_samples" / "train.parquet").exists()
    proxy_config = (question_root / "proxy" / "squid.conf").read_text(encoding="utf-8")
    assert "acl allowed dstdomain .openrouter.ai" in proxy_config
    task_yaml = (question_root / "task.yaml").read_text(encoding="utf-8")
    assert (
        '  First\n\n  Model\n\n  Shared ["changed", "no parameter change"]\n\n'
        "  Question: determine which parameter changed.\n\n"
        "  Data\n\n  Format\n\n  Generic"
    ) in task_yaml


def test_adapter_passes_documented_baseline_to_materialize(tmp_path: Path) -> None:
    bundle = tmp_path / "DemoModel"
    task_dir = tmp_path / "tasks"
    _write_bundle(bundle)
    payload = json.loads((bundle / "questions.json").read_text(encoding="utf-8"))
    payload["questions"]["wanted"]["recipe_info"]["noise_level"] = "low"
    payload["ground_truth_information"] = {
        "interventions": {
            "sample": {"changed_parameter": "changed", "first_diff": [None, 4.0]},
            "train": {"changed_parameter": "", "first_diff": [None, 1.0]},
        }
    }
    (bundle / "questions.json").write_text(json.dumps(payload), encoding="utf-8")
    pd.DataFrame({"signal": [9.0, 10.0], "time": [0.0, 1.0]}).to_parquet(
        bundle / "dataframes" / "sample_baseline.parquet"
    )
    pd.DataFrame({"signal": [11.0, 12.0], "time": [0.0, 1.0]}).to_parquet(
        bundle / "dataframes" / "train_baseline.parquet"
    )
    (bundle / "sample_manifest.json").write_text(
        json.dumps(
            {
                "row": [
                    {
                        "train_samples": ["train"],
                        "train_samples_baselines": ["train_baseline"],
                        "test_samples": ["sample"],
                        "test_samples_baselines": ["sample_baseline"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "noise_adder.py").write_text(
        "\n".join(
            [
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "",
                "def quantify_noise(clean, noisy, reference):",
                "    return {'global': [float(len(reference))], 'local': [float(len(reference))]}",
                "",
                "def add_noise(src, seed=0, noise_level='low', ref=None):",
                "    out = src.copy()",
                "    out['signal'] = out['signal'] + float(len(ref))",
                "    return out, quantify_noise(src, out, ref)",
            ]
        ),
        encoding="utf-8",
    )

    adapter = IndustrialTimeSeriesAdapter(
        dataset_root=bundle,
        task_dir=task_dir,
        include_question_ids=["wanted"],
    )
    adapter.generate_task("wanted", "question_0")

    test_df = pd.read_parquet(task_dir / "question_0" / "test_samples" / "sample.parquet")
    train_df = pd.read_parquet(task_dir / "question_0" / "train_samples" / "train.parquet")
    assert list(test_df.columns) == ["col1", "col2"]
    assert list(train_df.columns) == ["col1", "col2"]
    assert test_df["col1"].tolist() == [3.0, 4.0]
    assert train_df["col1"].tolist() == [5.0, 6.0]


def test_adapter_treats_missing_manifest_baseline_as_optional(tmp_path: Path) -> None:
    bundle = tmp_path / "DemoModel"
    task_dir = tmp_path / "tasks"
    _write_bundle(bundle)
    payload = json.loads((bundle / "questions.json").read_text(encoding="utf-8"))
    payload["questions"]["wanted"]["recipe_info"]["noise_level"] = "low"
    (bundle / "questions.json").write_text(json.dumps(payload), encoding="utf-8")
    (bundle / "sample_manifest.json").write_text(
        json.dumps(
            {
                "row": [
                    {
                        "train_samples": ["train"],
                        "train_samples_baselines": ["missing_train_baseline"],
                        "test_samples": ["sample"],
                        "test_samples_baselines": ["missing_sample_baseline"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (bundle / "noise_adder.py").write_text(
        "\n".join(
            [
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "",
                "def quantify_noise(clean, noisy, reference):",
                "    value = None if reference is None else float(len(reference))",
                "    return {'global': [value], 'local': [value]}",
                "",
                "def add_noise(src, seed=0, noise_level='low', ref=None):",
                "    out = src.copy()",
                "    if ref is not None:",
                "        out['signal'] = out['signal'] + float(len(ref))",
                "    return out, quantify_noise(src, out, ref)",
            ]
        ),
        encoding="utf-8",
    )

    adapter = IndustrialTimeSeriesAdapter(
        dataset_root=bundle,
        task_dir=task_dir,
        include_question_ids=["wanted"],
    )

    with pytest.warns(RuntimeWarning, match="reference parquet not found"):
        adapter.generate_task("wanted", "question_0")

    test_df = pd.read_parquet(task_dir / "question_0" / "test_samples" / "sample.parquet")
    train_df = pd.read_parquet(task_dir / "question_0" / "train_samples" / "train.parquet")
    assert test_df["col1"].tolist() == [1.0, 2.0]
    assert train_df["col1"].tolist() == [3.0, 4.0]
    assert (task_dir / "question_0" / "scenario_info.json").exists()
