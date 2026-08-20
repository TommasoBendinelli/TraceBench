from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import pandas as pd

from shared.interface.model_record_json import (
    compute_child_parameters_hash,
    compute_parameters_hash,
    compute_time0_baseline_hash,
    dump_model_record_json,
)
from workflows.metrics import evaluate_rule


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2],
            "Ball_Center_Position": [1.0, 1.1, 1.2],
            "Ball_Center_Speed": [0.5, 0.4, 0.3],
        }
    ).to_parquet(path)


def _build_fixture(tmp_path: Path) -> tuple[Path, str]:
    models_root = tmp_path / "models" / "simulink"
    model_dir = models_root / "BounceBall"
    runs_root = model_dir / "runs"
    baseline_id = "11111111111111111111111111111111"
    child_id = "22222222222222222222222222222222"
    time0_id = "33333333333333333333333333333333"

    _write_text(
        model_dir / "experiment_config.json",
        json.dumps(
            {
                "sampling_rate_hz": 10.0,
                "end_time_input_s": 6.0,
                "detectability": {
                  "continuous": {
                    "min_srd_distance": 0.001,
                    "epsilon_SRD": 0.001,
                    "minimum_consecurive_below_SRD": 1,
                  },
                  "impulse_like": {
                    "min_srd_distance": 0.3,
                    "epsilon_SRD": 1.0,
                  },
                },
                "exposed_variables": {
                    "initial_state": {},
                    "parameters": {
                        "restitution_left": {
                            "allowed_intervals": [0.0, 1.0],
                            "min_srd_distance": 0.3,
                            "min_abs_dist": 0.0,
                            "sampling_strategy": "uniform",
                        },
                        "restitution_right": {
                            "allowed_intervals": [0.0, 1.0],
                            "min_srd_distance": 0.3,
                            "min_abs_dist": 0.0,
                            "sampling_strategy": "uniform",
                        },
                        "viscous_damping": {
                            "allowed_intervals": [0.0, 1.0],
                            "min_srd_distance": 0.3,
                            "min_abs_dist": 0.0,
                            "sampling_strategy": "uniform",
                        },
                    }
                },
                "observable_signals": {
                    "observable_signals": ["Ball_Center_Position", "Ball_Center_Speed"],
                    "signal_type": {
                        "Ball_Center_Position": {
                            "type": "continuous",
                        },
                        "Ball_Center_Speed": {
                            "type": "continuous",
                        },
                    },
                },
            },
            indent=2,
        ),
    )
    _write_text(
        model_dir / "description_levels.json",
        json.dumps(
            {
                "internal_naming_to_agent_facing_parameter": {
                    "restitution_left": "left restitution",
                    "restitution_right": "right restitution",
                    "viscous_damping": "viscous damping",
                }
            },
            indent=2,
        ),
    )
    _write_text(
        model_dir / "noise_adder.py",
        "\n".join(
            [
                "import pandas as pd",
                "",
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "",
                "def quantify_noise(clean, noisy, reference):",
                "    return {'global': [0.0], 'local': [None]}",
                "",
                "def add_noise(src: pd.DataFrame, seed: int = 0, noise_level: str = 'low', ref: pd.DataFrame | None = None):",
                "    out = src.copy()",
                "    out['noise_profile_marker'] = noise_level",
                "    out['noise_seed_marker'] = seed",
                "    out['baseline_rows_marker'] = len(ref)",
                "    return out, quantify_noise(src, out, ref)",
            ]
        ),
    )
    baseline_parameters = {
        "restitution_left": 0.9,
        "restitution_right": 0.7,
        "viscous_damping": 0.4,
    }
    intervention_time = 1.0
    baseline_hash = compute_parameters_hash(parameters=baseline_parameters)
    child_parameters = {"restitution_left": 0.5}
    child_hash = compute_child_parameters_hash(
        parent_parameters_hash=baseline_hash,
        child_parameters=child_parameters,
        intervention_time=intervention_time,
    )
    _write_text(
        model_dir / "model_run_specs.json",
        json.dumps(
            {
                baseline_id: {
                    "baseline_parameters": baseline_parameters,
                    "baseline_parameters_hash": baseline_hash,
                    "children": {
                        child_id: {
                            "intervention_time": intervention_time,
                            "parameters": child_parameters,
                            "parameter_hash": child_hash,
                            "time0_baseline_uuid": time0_id,
                            "time0_baseline_hash": compute_time0_baseline_hash(
                                child_parameters_hash=child_hash
                            ),
                        }
                    },
                }
            },
            indent=2,
        ),
    )
    dump_model_record_json(
        runs_root / "model_record.json",
        {
            baseline_id: {
                "parameters_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "run_type": "baseline",
                "class_internal": "baseline",
                "class_agent_facing_name": "baseline",
                "status": "success",
            },
            child_id: {
                "parameters_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "run_type": "intervention",
                "class_internal": "restitution_left",
                "class_agent_facing_name": "restitution_left",
                "status": "success",
            },
            time0_id: {
                "parameters_hash": "cccccccccccccccccccccccccccccccc",
                "run_type": "time0_baseline",
                "class_internal": "nothing_happened",
                "class_agent_facing_name": "no parameter change",
                "status": "success",
            },
        },
    )
    _write_parquet(runs_root / baseline_id / "data.parquet")
    _write_parquet(runs_root / child_id / "data.parquet")
    _write_parquet(runs_root / time0_id / "data.parquet")
    return models_root, child_id


def test_evaluate_rule_function_returns_correct_payload(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'restitution_left'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
    )

    assert set(payload) == {"ground_truth", "predicted_label", "is_correct"}
    assert payload["ground_truth"] == "restitution_left"
    assert payload["predicted_label"] == "restitution_left"
    assert payload["is_correct"] is True


def test_evaluate_rule_low_context_uses_agent_facing_label(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'left restitution'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        context_level="low",
    )

    assert payload == {
        "ground_truth": "left restitution",
        "predicted_label": "left restitution",
        "is_correct": True,
    }


def test_evaluate_rule_baseline_run_uses_nothing_happened_label(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    baseline_id = "11111111111111111111111111111111"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'nothing_happened'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / baseline_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
    )

    assert payload == {
        "ground_truth": "no_parameter_change",
        "predicted_label": "no_parameter_change",
        "is_correct": True,
    }


def test_evaluate_rule_baseline_run_uses_agent_facing_no_change_label(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    baseline_id = "11111111111111111111111111111111"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'Not sure'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / baseline_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        context_level="low",
    )

    assert payload == {
        "ground_truth": "no_parameter_change",
        "predicted_label": "no_parameter_change",
        "is_correct": True,
    }


def test_evaluate_rule_non_none_context_maps_nothing_happened_label(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    time0_id = "33333333333333333333333333333333"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'Not sure'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / time0_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        context_level="high",
    )

    assert payload == {
        "ground_truth": "no_parameter_change",
        "predicted_label": "no_parameter_change",
        "is_correct": True,
    }


def test_evaluate_rule_materializes_anonymized_columns(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "\n".join(
            [
                "def predict(df):",
                "    assert list(df.columns[:3]) == ['col1', 'col2', 'col3']",
                "    assert float(df['col2'].iloc[0]) == 1.0",
                "    return 'left restitution'",
            ]
        ),
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
    )

    assert payload["is_correct"] is True


def test_evaluate_rule_cli_prints_json(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return {'final_answer': 'restitution_left'}\n",
    )

    result = CliRunner().invoke(
        evaluate_rule.main,
        [
            str(model_dir / "runs" / child_id),
            "--rule_path",
            str(rule_path),
            "--model_record",
            str(model_dir / "runs" / "model_record.json"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"ground_truth", "predicted_label", "is_correct"}
    assert payload["is_correct"] is True
    assert payload["predicted_label"] == "restitution_left"


def test_evaluate_rule_cli_accepts_documented_context_values(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'left restitution'\n",
    )

    result = CliRunner().invoke(
        evaluate_rule.main,
        [
            str(model_dir / "runs" / child_id),
            "--rule_path",
            str(rule_path),
            "--model_record",
            str(model_dir / "runs" / "model_record.json"),
            "--context_level",
            "low",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "ground_truth": "left restitution",
        "predicted_label": "left restitution",
        "is_correct": True,
    }


def test_evaluate_rule_cli_rejects_undocumented_context_values(
    tmp_path: Path,
) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(rule_path, "def predict(df):\n    return 'restitution_left'\n")

    result = CliRunner().invoke(
        evaluate_rule.main,
        [
            str(model_dir / "runs" / child_id),
            "--rule_path",
            str(rule_path),
            "--model_record",
            str(model_dir / "runs" / "model_record.json"),
            "--context_level",
            "env_agnostic",
        ],
    )

    assert result.exit_code != 0
    assert "Invalid value for '--context_level'" in result.output


def test_evaluate_rule_applies_noise_profile_before_rule(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "\n".join(
            [
                "def predict(df):",
                "    profile = str(df['col4'].iloc[0])",
                "    seed = int(df['col5'].iloc[0])",
                "    if profile == 'high' and seed == 7:",
                "        return 'restitution_left'",
                "    return 'right restitution'",
            ]
        ),
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        noise_level="high",
        seed=7,
        noise_adder_path=model_dir / "noise_adder.py",
    )

    assert payload["is_correct"] is True
    assert payload["ground_truth"] == "restitution_left"
    assert payload["predicted_label"] == "restitution_left"


def test_evaluate_rule_passes_documented_baseline_to_noise_adder(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "\n".join(
            [
                "def predict(df):",
                "    assert float(df['col6'].iloc[0]) == len(df)",
                "    return 'restitution_left'",
            ]
        ),
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        noise_level="low",
        seed=0,
        noise_adder_path=model_dir / "noise_adder.py",
    )

    assert payload["is_correct"] is True


def test_evaluate_rule_supports_documented_shortlist_score_evaluation_type(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return ['restitution_left', 'restitution_right']\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
        evaluation_type="shortlist_score",
    )

    assert payload == {
        "ground_truth": "restitution_left",
        "predicted_label": ["restitution_left", "restitution_right"],
        "top1_correct": True,
        "shortlist_score": 0.5,
        "num_answers": 2,
    }


def test_evaluate_rule_on_dataframe_scores_anonymized_allowed_choice_label(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return ['label_2']\n",
    )

    payload = evaluate_rule.evaluate_rule_on_dataframe(
        model_id="BounceBall",
        child_df=pd.DataFrame({"value": [1.0]}),
        truth_label="label_2",
        rule_path=rule_path,
        models_root=models_root,
        run_id="demo",
        allowed_choices=["label_0", "label_1", "label_2", "label_3", "label_4"],
    )

    assert payload["error"] is None
    assert payload["ground_truth"] == "label_2"
    assert payload["predicted_label"] == "label_2"
    assert payload["top1_correct"] is True
    assert payload["is_correct"] is True
    assert payload["shortlist_score"] == 1.0
    assert payload["answer_count"] == 1


def test_evaluate_rule_on_dataframe_scores_anonymized_shortlist(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return ['label_0', 'label_2']\n",
    )

    payload = evaluate_rule.evaluate_rule_on_dataframe(
        model_id="BounceBall",
        child_df=pd.DataFrame({"value": [1.0]}),
        truth_label="label_2",
        rule_path=rule_path,
        models_root=models_root,
        run_id="demo",
        allowed_choices=["label_0", "label_1", "label_2", "label_3", "label_4"],
    )

    assert payload["error"] is None
    assert payload["ground_truth"] == "label_2"
    assert payload["predicted_label"] == ["label_0", "label_2"]
    assert payload["top1_correct"] is False
    assert payload["is_correct"] is False
    assert payload["shortlist_score"] == 0.5
    assert payload["answer_count"] == 2


def test_evaluate_rule_on_dataframe_rejects_anonymized_label_outside_choices(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return ['label_9']\n",
    )

    payload = evaluate_rule.evaluate_rule_on_dataframe(
        model_id="BounceBall",
        child_df=pd.DataFrame({"value": [1.0]}),
        truth_label="label_2",
        rule_path=rule_path,
        models_root=models_root,
        run_id="demo",
        allowed_choices=["label_0", "label_1", "label_2", "label_3", "label_4"],
    )

    assert payload["error"] == (
        "ValueError: Predictions ['label_9'] are not in multiple_choices "
        "['label_0', 'label_1', 'label_2', 'label_3', 'label_4']"
    )
    assert payload["top1_correct"] is False
    assert payload["shortlist_score"] == 0.0
    assert payload["answer_count"] == 0


def test_evaluate_rule_on_dataframe_accepts_no_change_as_valid_wrong_choice(
    tmp_path: Path,
) -> None:
    models_root, _child_id = _build_fixture(tmp_path)
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'no parameter change'\n",
    )

    payload = evaluate_rule.evaluate_rule_on_dataframe(
        model_id="BounceBall",
        child_df=pd.DataFrame({"value": [1.0]}),
        truth_label="left restitution",
        rule_path=rule_path,
        models_root=models_root,
        run_id="demo",
        allowed_choices=[
            "left restitution",
            "right restitution",
            "viscous damping",
            "no parameter change",
        ],
    )

    assert payload["error"] is None
    assert payload["is_correct"] is False
    assert payload["predicted_label"] == "no parameter change"
    assert payload["shortlist_score"] == 0.0
    assert payload["answer_count"] == 1


def test_evaluate_rule_requires_noise_adder_for_noise_profile(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(rule_path, "def predict(df):\n    return 'restitution_left'\n")

    try:
        evaluate_rule.evaluate_rule(
            model_dir / "runs" / child_id,
            rule_path,
            model_dir / "runs" / "model_record.json",
            noise_level="high",
        )
    except ValueError as exc:
        assert "noise_adder_path is required" in str(exc)
    else:
        raise AssertionError("evaluate_rule should require noise_adder_path")


def test_evaluate_rule_returns_error_for_invalid_rule_label(tmp_path: Path) -> None:
    models_root, child_id = _build_fixture(tmp_path)
    model_dir = models_root / "BounceBall"
    rule_path = tmp_path / "rule.py"
    _write_text(
        rule_path,
        "def predict(df):\n"
        "    return 'not_a_valid_label'\n",
    )

    payload = evaluate_rule.evaluate_rule(
        model_dir / "runs" / child_id,
        rule_path,
        model_dir / "runs" / "model_record.json",
    )

    assert set(payload) == {"ground_truth", "predicted_label", "is_correct"}
    assert payload["is_correct"] is False
    assert payload["predicted_label"] is None
