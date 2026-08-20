from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from shared.interface.similarity_metrics_json import (
    SimilarityMetricsJson,
    validate_similarity_metrics_schema,
    validate_similarity_metrics_semantics,
)
from workflows.metrics import compute_metrics


_REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeRunGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    @property
    def nodes_by_id(self):
        return {str(node["run_id"]): node for node in self.nodes}


def _child(
    *,
    vs_baseline: str = "yes",
    vs_time0_baseline: str = "yes",
    rule_correct: bool | None = None,
) -> dict[str, object]:
    baseline_first_diff = (
        [0.5] if vs_baseline == "yes" else ([None] if vs_baseline == "no" else [])
    )
    time0_first_diff = (
        [0.5]
        if vs_time0_baseline == "yes"
        else ([None] if vs_time0_baseline == "no" else [])
    )

    def status_payload(status: str, first_diff: list[float | None]) -> dict[str, object]:
        env_status = "yes" if status == "yes" else "no"
        return {
            "environment_specific_detectability": env_status,
            "max_SRD_detectability": status,
            "detectability": "yes" if status == "yes" and env_status == "yes" else "no",
            "detectable": "yes" if status == "yes" and env_status == "yes" else "no",
            "max_SRD": [1.0] if status != "error" else [],
            "euclidean_distance": [1.0] if status != "error" else [],
            "first_diff": first_diff,
        }

    child: dict[str, object] = {
        "detectability": {
            "vs_baseline": status_payload(vs_baseline, baseline_first_diff),
            "vs_time0_baseline": status_payload(
                vs_time0_baseline,
                time0_first_diff,
            ),
        },
    }
    if rule_correct is not None:
        child["evaluate_rule"] = {
            "ground_truth": "gravity",
            "is_perfect_correct": rule_correct,
            "is_perfect_predicted_label": "gravity" if rule_correct else "mass",
        }
    return child


def test_child_eligible_uses_detectability_and_optional_rule() -> None:
    assert compute_metrics._child_is_eligible(_child()) is True
    assert (
        compute_metrics._child_is_eligible(_child(), intervention_time=0.4)
        is True
    )
    assert (
        compute_metrics._child_is_eligible(_child(), intervention_time=0.5)
        is True
    )
    assert compute_metrics._child_is_eligible(_child(vs_time0_baseline="no")) is False
    assert (
        compute_metrics._child_is_eligible(_child(rule_correct=False))
        is True
    )
    assert (
        compute_metrics._child_is_eligible(_child(rule_correct=True))
        is True
    )


def test_compute_child_outputs_compute_ml_does_not_gate_eligibility(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        compute_metrics,
        "_compute_detectability_metrics",
        lambda **_: _child()["detectability"],
    )

    _run_id, child, summary = compute_metrics._compute_child_outputs(
        model_id="BallDrop",
        models_root=tmp_path,
        base_df=object(),
        entry={
            "run_id": "child-run",
            "df": object(),
            "raw": {"time0_baseline_uuid": "time0"},
            "truth_parameter": "mass",
        },
        min_srd_distance=0.001,
        epsilon_SRD=0.001,
        time0_df=object(),
        include_rule_eval=True,
    )

    assert "evaluate_rule" not in child
    assert child["eligible"] is True
    assert summary["eligible"] is True


def test_compute_child_outputs_rule_correctness_does_not_gate_eligibility(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        compute_metrics,
        "_compute_detectability_metrics",
        lambda **_: _child()["detectability"],
    )

    _run_id, child, summary = compute_metrics._compute_child_outputs(
        model_id="BallDrop",
        models_root=tmp_path,
        base_df=object(),
        entry={
            "run_id": "child-run",
            "df": object(),
            "raw": {"time0_baseline_uuid": "time0"},
            "truth_parameter": "mass",
        },
        min_srd_distance=0.001,
        epsilon_SRD=0.001,
        time0_df=object(),
        include_rule_eval=True,
    )

    assert "evaluate_rule" not in child
    assert child["eligible"] is True
    assert summary["eligible"] is True


def test_compute_child_outputs_builds_documented_run_parameters(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_compute_detectability_metrics(**kwargs):
        captured.update(kwargs)
        return _child()["detectability"]

    monkeypatch.setattr(
        compute_metrics,
        "_compute_detectability_metrics",
        _fake_compute_detectability_metrics,
    )

    compute_metrics._compute_child_outputs(
        model_id="BallDrop",
        models_root=tmp_path,
        base_df=object(),
        entry={
            "run_id": "child-run",
            "df": object(),
            "baseline_parameters": {"mass": 1.0, "gravity": 9.81},
            "raw": {
                "time0_baseline_uuid": "time0",
                "parameter": "mass",
                "set_value": 2.0,
                "intervention_time": 1.5,
            },
            "truth_parameter": "mass",
        },
        min_srd_distance=0.001,
        epsilon_SRD=0.001,
        time0_df=object(),
        include_rule_eval=False,
    )

    assert captured["run_parameters"] == {
        "mass": 2.0,
        "gravity": 9.81,
        "intervention_parameter": "mass",
        "intervention_time": 1.5,
    }


def test_baseline_family_eligible_requires_three_distinct_eligible_parameters() -> None:
    assert (
        compute_metrics._baseline_family_is_eligible(
            {
                "child-a": {"eligible": True},
                "child-b": {"eligible": True},
                "child-c": {"eligible": True},
                "child-e": {"eligible": False},
            },
            child_parameters={
                "child-a": "a",
                "child-b": "b",
                "child-c": "c",
                "child-e": "e",
            },
            expected_parameters=("a", "b", "c", "d", "e"),
        )
        is True
    )
    assert (
        compute_metrics._baseline_family_is_eligible(
            {
                "child-a-1": {"eligible": True},
                "child-a-2": {"eligible": True},
                "child-b": {"eligible": True},
                "child-c": {"eligible": False},
                "child-d": {"eligible": False},
            },
            child_parameters={
                "child-a-1": "a",
                "child-a-2": "a",
                "child-b": "b",
                "child-c": "c",
                "child-d": "d",
            },
            expected_parameters=("a", "b", "c", "d"),
        )
        is False
    )
    assert (
        compute_metrics._baseline_family_is_eligible(
            {
                "child-a": {"eligible": True},
                "child-b": {"eligible": True},
                "child-outside": {"eligible": True},
            },
            child_parameters={
                "child-a": "a",
                "child-b": "b",
                "child-outside": "outside",
            },
            expected_parameters=("a", "b", "c", "d"),
        )
        is False
    )


def test_summary_accuracy_payload_matches_documented_compute_ml_shape() -> None:
    payload = {
        "baselines": {
            "baseline-uuid": {
                "family_eligible": True,
                "children": {
                    "detectable-correct": {
                        "url": "http://localhost:3001/?model=DemoModel&run=detectable-correct&compare=none",
                        "detectability": {
                            "vs_baseline": {"detectable": "yes"},
                            "vs_time0_baseline": {"detectable": "yes"},
                        },
                        "evaluate_rule": {
                            "is_perfect_correct": True,
                            "noise_analysis": {
                                "avg_chance_corrected_accuracy_low": 1.0,
                                "avg_chance_corrected_accuracy_high": 0.8,
                            },
                        },
                    },
                    "detectable-wrong": {
                        "url": "http://localhost:3001/?model=DemoModel&run=detectable-wrong&compare=none",
                        "detectability": {
                            "vs_baseline": {"detectable": "yes"},
                            "vs_time0_baseline": {"detectable": "yes"},
                        },
                        "evaluate_rule": {
                            "is_perfect_correct": False,
                            "noise_analysis": {
                                "avg_chance_corrected_accuracy_low": 0.4,
                                "avg_chance_corrected_accuracy_high": 1.0,
                            },
                        },
                    },
                    "not-detectable-correct": {
                        "url": "http://localhost:3001/?model=DemoModel&run=not-detectable-correct&compare=none",
                        "detectability": {
                            "vs_baseline": {"detectable": "yes"},
                            "vs_time0_baseline": {"detectable": "no"},
                        },
                        "evaluate_rule": {
                            "is_perfect_correct": True,
                            "noise_analysis": {
                                "avg_chance_corrected_accuracy_low": 1.0,
                                "avg_chance_corrected_accuracy_high": 1.0,
                            },
                        },
                    },
                }
            },
            "baseline-not-family-eligible": {
                "family_eligible": False,
                "children": {
                    "detectable-correct-excluded-by-family": {
                        "url": "http://localhost:3001/?model=DemoModel&run=detectable-correct-excluded-by-family&compare=none",
                        "detectability": {
                            "vs_baseline": {"detectable": "yes"},
                            "vs_time0_baseline": {"detectable": "yes"},
                        },
                        "evaluate_rule": {
                            "is_perfect_correct": True,
                            "noise_analysis": {
                                "avg_chance_corrected_accuracy_low": 0.0,
                                "avg_chance_corrected_accuracy_high": 0.0,
                            },
                        },
                    }
                },
            }
        }
    }

    summary = compute_metrics._build_summary_accuracy_payload(payload)

    assert summary == {
        "detectability_ok": 4,
        "is_perfect_correct": 3,
        "avg_chance_corrected_accuracy_low_1": 2,
        "avg_chance_corrected_accuracy_high_1": 2,
        "family_eligibility_and_detectability_ok_but_not_perfect_correct": [
            "http://localhost:3001/?model=DemoModel&run=detectable-wrong&compare=none"
        ],
    }


def test_summary_accuracy_file_is_written_only_for_compute_ml(tmp_path) -> None:
    out_path = tmp_path / "eligibility_metrics.json"
    results = {"baselines": {}}

    skipped = compute_metrics._write_summary_accuracy_if_requested(
        compute_ml=False,
        out_path=out_path,
        results=results,
        model_id="DemoModel",
    )

    assert skipped is None
    assert not (tmp_path / "summary_accuracy.json").exists()

    written = compute_metrics._write_summary_accuracy_if_requested(
        compute_ml=True,
        out_path=out_path,
        results=results,
        model_id="DemoModel",
    )

    assert written == tmp_path / "summary_accuracy.json"
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "detectability_ok": 0,
        "is_perfect_correct": 0,
        "avg_chance_corrected_accuracy_low_1": 0,
        "avg_chance_corrected_accuracy_high_1": 0,
        "family_eligibility_and_detectability_ok_but_not_perfect_correct": [],
    }


def test_documented_cheap_filter_metric_run_writes_manifest_and_jsonl(
    tmp_path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "shared_data" / "BallDrop_v2"
    runs_root = model_dir / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / "model_record.json").write_text("{}", encoding="utf-8")
    (model_dir / "experiment_config.json").write_text("{}", encoding="utf-8")

    baseline_recipe = {
        "model": "DemoModel",
        "baseline_parameters": {"mass": 1.0},
        "intervention": {"parameter": None, "value": None, "time": None},
    }
    child_recipe = {
        "model": "DemoModel",
        "baseline_parameters": {"mass": 1.0},
        "intervention": {"parameter": "mass", "value": 2.0, "time": 1.0},
    }
    time0_recipe = {
        "model": "DemoModel",
        "baseline_parameters": {"mass": 1.0},
        "intervention": {"parameter": "mass", "value": 2.0, "time": 0.0},
    }
    graph = _FakeRunGraph(
        nodes=[
            {
                "run_id": "baseline",
                "kind": "baseline",
                "family_id": "fam_demo",
                "recipe": baseline_recipe,
            },
            {
                "run_id": "child",
                "kind": "intervention",
                "family_id": "fam_demo",
                "recipe": child_recipe,
            },
            {
                "run_id": "time0",
                "kind": "time0_baseline",
                "family_id": "fam_demo",
                "recipe": time0_recipe,
            },
            {
                "run_id": "failed-child",
                "kind": "intervention",
                "family_id": "fam_demo",
                "recipe": child_recipe,
            },
        ],
        edges=[
            {
                "edge_type": "baseline_to_intervention",
                "source_run_id": "baseline",
                "target_run_id": "child",
            },
            {
                "edge_type": "intervention_to_time0_baseline",
                "source_run_id": "child",
                "target_run_id": "time0",
            },
            {
                "edge_type": "baseline_to_intervention",
                "source_run_id": "baseline",
                "target_run_id": "failed-child",
            },
            {
                "edge_type": "intervention_to_time0_baseline",
                "source_run_id": "failed-child",
                "target_run_id": "time0",
            },
        ],
    )
    frames = {
        "baseline": pd.DataFrame({"signal": [0.0, 0.0], "time": [0.0, 1.0]}),
        "child": pd.DataFrame({"signal": [0.0, 2.0], "time": [0.0, 1.0]}),
        "failed-child": pd.DataFrame({"signal": [0.0, 2.0], "time": [0.0, 1.0]}),
        "time0": pd.DataFrame({"signal": [0.0, 1.0], "time": [0.0, 1.0]}),
    }

    monkeypatch.setattr(
        compute_metrics,
        "_load_cheap_filter_run_graph",
        lambda _path: graph,
    )
    monkeypatch.setattr(
        compute_metrics,
        "load_experiment_config_json",
        lambda _path: SimpleNamespace(
            min_srd_distance=0.001,
            epsilon_SRD=0.001,
            minimum_consecurive_below_SRD=1,
            detectability=SimpleNamespace(RMS_thresholds={"signal": 0.25}),
        ),
    )
    monkeypatch.setattr(
        compute_metrics,
        "_load_cheap_filter_model_record",
        lambda _path: {
            "child": {"status": "success"},
            "failed-child": {"status": "error"},
        },
    )
    monkeypatch.setattr(
        compute_metrics,
        "load_run_df",
        lambda run_dir: frames.get(Path(run_dir).name),
    )
    monkeypatch.setattr(
        compute_metrics,
        "_compute_detectability_metrics",
        lambda **_kwargs: {
            "vs_baseline": {
                "environment_specific_detectability": "yes",
                "max_SRD_detectability": "yes",
                "detectability": "yes",
                "detectable": "yes",
                "first_diff": [1.0],
                "mean_euclidean_distance_clean_dirty": [0.0],
                "mean_euclidean_distance_clean_baseline": [2.0],
                "mean_SNR": [None],
            },
            "vs_time0_baseline": {
                "environment_specific_detectability": "yes",
                "max_SRD_detectability": "yes",
                "detectability": "yes",
                "detectable": "yes",
                "first_diff": [1.0],
                "mean_euclidean_distance_clean_dirty": [0.0],
                "mean_euclidean_distance_clean_baseline": [1.0],
                "mean_SNR": [None],
            },
        },
    )

    result = compute_metrics.run_cheap_filter_metrics(
        model_dir=model_dir,
        policy="demo_policy",
        root_dir=runs_root,
        uuid="unit",
        thresholds={"signal": 3.0},
        jobs=1,
    )

    metric_run_dir = Path(result["metric_run_dir"])
    metric_run = json.loads((metric_run_dir / "metric_run.json").read_text())
    rows = [
        json.loads(line)
        for line in (metric_run_dir / "cheap_filter_metrics.jsonl").read_text().splitlines()
    ]

    assert metric_run["metric_run_id"] == "cheap_filter_unit"
    assert metric_run["model"] == "BallDrop_v2"
    assert metric_run["model_dir"] == str(model_dir.resolve())
    assert metric_run["run_root"] == str(runs_root.resolve())
    assert metric_run["thresholds"] == {"signal": 3.0}
    assert metric_run["inputs"]["run_nodes"].endswith(
        "shared_data/BallDrop_v2/plans/demo_policy/run_nodes.jsonl"
    )
    assert metric_run_dir == model_dir / "metrics" / "demo_policy" / "cheap_filter_unit"
    assert result["model_id"] == "BallDrop_v2"
    assert result["records"] == 2
    assert rows[0]["baseline_run_id"] == "baseline"
    assert rows[0]["time0_run_id"] == "time0"
    assert rows[0]["cheap_filter_pass"] is True
    assert rows[0]["detectability"]["vs_baseline"] == {
        "generic_numeric_detectability": True,
        "environment_specific_detectability": True,
        "detectability_output": {
            "mean_euclidean_distance_clean_dirty": [0.0],
            "mean_euclidean_distance_clean_baseline": [2.0],
            "mean_SNR": [None],
            "first_diff": [1.0],
        },
    }
    assert rows[0]["detectability"]["vs_time0_baseline"] == {
        "generic_numeric_detectability": True,
        "environment_specific_detectability": True,
        "detectability_output": {
            "mean_euclidean_distance_clean_dirty": [0.0],
            "mean_euclidean_distance_clean_baseline": [1.0],
            "mean_SNR": [None],
            "first_diff": [1.0],
        },
    }
    assert "detectable" not in rows[0]["detectability"]["vs_baseline"]
    assert "detectable" not in rows[0]["detectability"]["vs_time0_baseline"]
    assert math.isclose(rows[0]["baseline_rms_difference"]["signal"], math.sqrt(2.0))
    assert rows[1]["run_id"] == "failed-child"
    assert rows[1]["cheap_filter_pass"] is False
    assert "run_status_not_success" in rows[1]["failure_reasons"]


@pytest.mark.parametrize(
    ("vs_baseline", "vs_time0", "expected"),
    [
        (
            {
                "generic_numeric_detectability": True,
                "environment_specific_detectability": True,
            },
            {
                "generic_numeric_detectability": True,
                "environment_specific_detectability": True,
            },
            True,
        ),
        (
            {
                "generic_numeric_detectability": False,
                "environment_specific_detectability": True,
            },
            {
                "generic_numeric_detectability": True,
                "environment_specific_detectability": True,
            },
            False,
        ),
        (
            {
                "generic_numeric_detectability": True,
                "environment_specific_detectability": True,
            },
            {
                "generic_numeric_detectability": True,
                "environment_specific_detectability": False,
            },
            True,
        ),
    ],
)
def test_cheap_filter_pass_uses_generic_numeric_detectability_booleans(
    vs_baseline,
    vs_time0,
    expected,
) -> None:
    assert (
        compute_metrics._cheap_filter_pass(
            status_success=True,
            baseline_df=object(),
            run_df=object(),
            time0_df=object(),
            detectability={
                "vs_baseline": vs_baseline,
                "vs_time0_baseline": vs_time0,
            },
        )
        is expected
    )


def test_cheap_filter_optional_environment_hook_defaults_to_true() -> None:
    payload = compute_metrics._normalize_detectability_for_cheap_filter(
        {
            "vs_baseline": {
                "max_SRD_detectability": "yes",
                "environment_specific_detectability": "error",
                "first_diff": [1.0],
            },
            "vs_time0_baseline": {
                "max_SRD_detectability": "yes",
                "environment_specific_detectability": "error",
                "first_diff": [1.0],
            },
        },
        has_environment_hook=False,
    )

    assert payload["vs_baseline"]["generic_numeric_detectability"] is True
    assert payload["vs_baseline"]["environment_specific_detectability"] is True
    assert payload["vs_time0_baseline"]["generic_numeric_detectability"] is True
    assert payload["vs_time0_baseline"]["environment_specific_detectability"] is True


def test_detectability_metrics_use_documented_intervention_time() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    time0 = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    before_intervention = pd.DataFrame(
        {"signal": [2.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]}
    )
    after_intervention = pd.DataFrame(
        {"signal": [1.0, 1.0, 2.0], "time": [0.0, 1.0, 2.0]}
    )

    before = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=before_intervention,
        time0_df=time0,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        RMS_thresholds={"signal": 10.0},
    )
    after = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=after_intervention,
        time0_df=time0,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        RMS_thresholds={"signal": 10.0},
    )

    assert before["vs_baseline"]["max_SRD_detectability"] == "no"
    assert before["vs_baseline"]["environment_specific_detectability"] == "no"
    assert before["vs_baseline"]["detectability"] == "no"
    assert before["vs_baseline"]["first_diff"] == [None]
    assert before["vs_time0_baseline"]["max_SRD_detectability"] == "no"
    assert before["vs_time0_baseline"]["environment_specific_detectability"] == "no"
    assert before["vs_time0_baseline"]["detectability"] == "no"
    assert before["vs_time0_baseline"]["first_diff"] == [None]
    assert after["vs_baseline"]["max_SRD_detectability"] == "no"
    assert after["vs_baseline"]["environment_specific_detectability"] == "error"
    assert after["vs_baseline"]["detectability"] == "error"
    assert after["vs_baseline"]["first_diff"] == [2.0]
    assert after["vs_time0_baseline"]["max_SRD_detectability"] == "no"
    assert after["vs_time0_baseline"]["environment_specific_detectability"] == "no"
    assert after["vs_time0_baseline"]["detectability"] == "no"
    assert after["vs_time0_baseline"]["first_diff"] == [2.0]
    assert after["vs_time0_baseline"]["mean_euclidean_distance_clean_dirty"] == [0.0]
    assert after["vs_time0_baseline"]["mean_euclidean_distance_clean_baseline"] == [1.0]
    assert after["vs_time0_baseline"]["mean_SNR"] == [None]


def test_detectability_metrics_apply_high_noise_to_both_compared_runs(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    baseline.attrs["role"] = "baseline"
    time0 = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    time0.attrs["role"] = "time0"
    run = pd.DataFrame({"signal": [1.0, 2.0, 2.0], "time": [0.0, 1.0, 2.0]})
    run.attrs["role"] = "run"

    calls: list[tuple[str, str, int, str]] = []

    def fake_load_noise_adder(path):
        assert path == tmp_path / "noise_adder.py"
        return object()

    def fake_call_noise_adder(
        add_noise,
        df,
        *,
        baseline_df=None,
        seed,
        noise_level,
        **_,
    ):
        _ = add_noise
        role = str(df.attrs.get("role"))
        baseline_role = str(baseline_df.attrs.get("role"))
        calls.append((role, baseline_role, seed, noise_level))
        out = df.copy()
        if role == "run" and baseline_role == "baseline":
            out["signal"] = [1.0, 2.01, 2.01]
        elif role == "run" and baseline_role == "time0":
            out["signal"] = [2.0, 2.0, 2.0]
        else:
            out["signal"] = [1.0, 1.0, 1.0]
        return out, {"global": [1.0], "local": [1.0]}

    monkeypatch.setattr(
        compute_metrics,
        "load_noise_adder_from_path",
        fake_load_noise_adder,
    )
    monkeypatch.setattr(
        compute_metrics,
        "call_noise_adder",
        fake_call_noise_adder,
    )

    payload = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=run,
        time0_df=time0,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        noise_adder_path=tmp_path / "noise_adder.py",
        RMS_thresholds={"signal": 1.0},
    )

    assert calls == [
        *(
            call
            for seed in range(5)
            for call in (
                ("baseline", "baseline", seed, "high"),
                ("run", "baseline", seed, "high"),
            )
        ),
        *(
            call
            for seed in range(5)
            for call in (
                ("time0", "time0", seed, "high"),
                ("run", "time0", seed, "high"),
            )
        ),
    ]
    assert payload["vs_baseline"]["max_SRD_detectability"] == "yes"
    assert payload["vs_baseline"]["first_diff"] == [1.0]
    assert payload["vs_time0_baseline"]["max_SRD_detectability"] == "yes"
    assert payload["vs_time0_baseline"]["first_diff"] == [1.0]
    assert payload["vs_baseline"]["mean_euclidean_distance_clean_dirty"][0] > 0.0
    assert payload["vs_baseline"]["mean_euclidean_distance_clean_baseline"][0] > 0.0


def test_ball_drop_hard_stop_spike_without_velocity_rebound_is_preprocessed() -> None:
    baseline = pd.DataFrame(
        {
            "Position": [1.0, 1.0, 1.0, 1.0],
            "Velocity": [1.0, 1.0, 1.0, 1.0],
            "Hard_Stop_f": [0.0, 0.0, 0.0, 0.0],
            "time": [0.0, 0.5, 1.0, 1.5],
        }
    )
    run = baseline.copy()
    run["Hard_Stop_f"] = [20.0, 0.0, 0.0, 0.0]

    payload = compute_metrics._compute_detectability_metrics(
        model_id="BallDrop",
        baseline_df=baseline,
        run_df=run,
        time0_df=baseline,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        signal_detectability_specs={
            "Position": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
            "Velocity": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
            "Hard_Stop_f": {
                "min_srd_distance": 0.3,
                "epsilon_SRD": 1.0,
                "minimum_consecutive_srd_steps": 1,
            },
        },
        RMS_thresholds={"Hard_Stop_f": 30.0},
    )

    assert payload["vs_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_baseline"]["first_diff"] == [None, None, None]
    assert payload["vs_time0_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_time0_baseline"]["first_diff"] == [None, None, None]


def test_ball_drop_hard_stop_valid_rebound_peak_survives_preprocessing() -> None:
    baseline = pd.DataFrame(
        {
            "Position": [1.0, 1.0, 1.0, 1.0, 1.0],
            "Velocity": [-2.0, -1.0, -0.5, 1.0, 2.0],
            "Hard_Stop_f": [0.0, 0.0, 0.0, 0.0, 0.0],
            "time": [0.8, 0.95, 1.0, 1.05, 1.2],
        }
    )
    run = baseline.copy()
    run["Hard_Stop_f"] = [0.0, 0.0, 20.0, 0.0, 0.0]

    payload = compute_metrics._compute_detectability_metrics(
        model_id="BallDrop",
        baseline_df=baseline,
        run_df=run,
        time0_df=baseline,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        signal_detectability_specs={
            "Position": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
            "Velocity": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
            "Hard_Stop_f": {
                "min_srd_distance": 0.3,
                "epsilon_SRD": 1.0,
                "minimum_consecutive_srd_steps": 1,
            },
        },
        RMS_thresholds={"Hard_Stop_f": 30.0},
    )

    assert payload["vs_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_baseline"]["first_diff"] == [None, None, 1.0]
    assert payload["vs_time0_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_time0_baseline"]["first_diff"] == [None, None, 1.0]


def test_rms_thresholds_read_config() -> None:
    config = SimpleNamespace(
        detectability=SimpleNamespace(
            RMS_thresholds={
                "Position": 0.012,
                "Hard_Stop_f": 0.054,
            }
        )
    )

    assert compute_metrics._rms_thresholds(config) == {
        "Position": 0.012,
        "Hard_Stop_f": 0.054,
    }


def test_mean_snr_uses_mean_linear_ratio_before_db_conversion() -> None:
    payload = compute_metrics._mean_snr_db_by_channel(
        clean_dirty_by_seed=[[10.0], [10.0]],
        clean_baseline_by_seed=[[100.0], [10.0]],
    )
    mean_linear_ratio_db = 20.0 * math.log10(((100.0 / 10.0) + (10.0 / 10.0)) / 2.0)
    mean_per_seed_db = (
        (20.0 * math.log10(100.0 / 10.0))
        + (20.0 * math.log10(10.0 / 10.0))
    ) / 2.0

    assert payload == [pytest.approx(mean_linear_ratio_db)]
    assert payload != [pytest.approx(mean_per_seed_db)]


def test_ball_drop_multi_seed_snr_uses_preprocessed_clean_reference() -> None:
    model_dir = _REPO_ROOT / "models" / "simulink" / "BallDrop"
    runs_root = model_dir / "runs"
    baseline_id = "4d36b3bd1e524a49b337e2ac9a97206b"
    child_id = "61cd70ea73e1497f981510b030571fe5"
    time0_id = "1369f277876c4887a9cb7da986d19ddb"
    for run_id in (baseline_id, child_id, time0_id):
        if not (runs_root / run_id / "data.parquet").exists():
            pytest.skip(f"BallDrop fixture run is not present: {run_id}")

    payload = compute_metrics._compute_detectability_metrics(
        model_id="BallDrop",
        baseline_df=compute_metrics.load_run_df(runs_root / baseline_id),
        run_df=compute_metrics.load_run_df(runs_root / child_id),
        time0_df=compute_metrics.load_run_df(runs_root / time0_id),
        time0_expected=True,
        min_srd_distance=0.001,
        epsilon_SRD=0.001,
        noise_adder_path=model_dir / "noise_adder.py",
    )
    assert payload["vs_baseline"]["mean_SNR"] == pytest.approx(
        [18.543567138731547, 15.184073662815534, 30.503708447520324]
    )
    assert payload["vs_time0_baseline"]["mean_SNR"] == pytest.approx(
        [29.12467108392615, 20.672255263164125, 31.48964730208657]
    )


def test_detectability_metrics_report_error_when_noise_fails(
    tmp_path,
    monkeypatch,
) -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})

    monkeypatch.setattr(
        compute_metrics,
        "load_noise_adder_from_path",
        lambda _path: object(),
    )

    def failing_call_noise_adder(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("noise unavailable")

    monkeypatch.setattr(compute_metrics, "call_noise_adder", failing_call_noise_adder)

    payload = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=run,
        time0_df=baseline,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        noise_adder_path=tmp_path / "noise_adder.py",
    )

    assert payload["vs_baseline"]["max_SRD_detectability"] == "error"
    assert payload["vs_baseline"]["first_diff"] == []
    assert payload["vs_time0_baseline"]["max_SRD_detectability"] == "error"
    assert payload["vs_time0_baseline"]["first_diff"] == []


def test_detectability_metrics_use_environment_specific_hook(tmp_path) -> None:
    hook_path = tmp_path / "detectability_specific_environment.py"
    hook_path.write_text(
        "def is_detectable(df, clean_df, run_parameters, intervention_time, min_first_diff):\n"
        "    if run_parameters == {'mass': 2.0, 'intervention_time': 1.0} and intervention_time == 1.0 and min_first_diff == 1.0:\n"
        "        return 'yes'\n"
        "    return 'no'\n",
        encoding="utf-8",
    )
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [1.0, 2.0, 2.0], "time": [0.0, 1.0, 2.0]})

    payload = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=run,
        time0_df=baseline,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        run_parameters={"mass": 2.0, "intervention_time": 1.0},
        env_detectability_path=hook_path,
        RMS_thresholds={"signal": 10.0},
    )

    assert payload["vs_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_baseline"]["environment_specific_detectability"] == "yes"
    assert payload["vs_baseline"]["detectability"] == "no"
    assert payload["vs_baseline"]["first_diff"] == [1.0]
    assert payload["vs_baseline"]["mean_SNR"] == [None]


def test_time0_detectability_does_not_use_environment_specific_hook(tmp_path) -> None:
    hook_path = tmp_path / "detectability_specific_environment.py"
    hook_path.write_text(
        "def is_detectable(df, clean_df, run_parameters, intervention_time, min_first_diff):\n"
        "    return 'no'\n",
        encoding="utf-8",
    )
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    time0 = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [2.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})

    payload = compute_metrics._compute_detectability_metrics(
        baseline_df=baseline,
        run_df=run,
        time0_df=time0,
        time0_expected=True,
        min_srd_distance=0.1,
        epsilon_SRD=0.001,
        intervention_time=1.0,
        run_parameters={"mass": 2.0, "intervention_time": 1.0},
        env_detectability_path=hook_path,
        RMS_thresholds={"signal": 10.0},
    )

    assert payload["vs_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_baseline"]["environment_specific_detectability"] == "no"
    assert payload["vs_baseline"]["detectability"] == "no"
    assert payload["vs_time0_baseline"]["max_SRD_detectability"] == "no"
    assert payload["vs_time0_baseline"]["environment_specific_detectability"] == "no"
    assert payload["vs_time0_baseline"]["detectability"] == "no"


def test_evaluate_rule_summary_uses_documented_noise_analysis(
    tmp_path,
) -> None:
    models_root = tmp_path / "models" / "simulink"
    model_dir = models_root / "DemoModel"
    model_dir.mkdir(parents=True)
    (model_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "exposed_variables": {
                    "initial_state": {},
                    "parameters": {
                        "mass": {
                            "allowed_intervals": [1.0, 2.0],
                            "min_srd_distance": 0.3,
                            "min_abs_dist": 0.0,
                            "sampling_strategy": "uniform",
                        }
                    },
                },
                "sampling_rate_hz": 10.0,
                "end_time_input_s": 3.0,
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
                "observable_signals": {
                    "observable_signals": ["signal_a"],
                    "signal_type": {"signal_a": {"type": "continuous"}},
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "description_levels.json").write_text(
        json.dumps(
            {
                "internal_naming_to_agent_facing_parameter": {
                    "mass": "mass",
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "basic_rule.py").write_text(
        "def predict(df):\n    return 'mass'\n",
        encoding="utf-8",
    )
    (model_dir / "noise_adder.py").write_text(
        "\n".join(
            [
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "",
                "def quantify_noise(clean, noisy, reference):",
                "    return {'global': [100.0], 'local': [float(len(reference))]}",
                "",
                "def add_noise(src, seed=0, noise_level='low', ref=None):",
                "    out = src.copy()",
                "    return out, {'global': [float(seed)], 'local': [float(len(ref))]}",
            ]
        ),
        encoding="utf-8",
    )

    summary = compute_metrics._build_evaluate_rule_summary(
        model_id="DemoModel",
        run_id="child",
        truth_parameter="mass",
        child_df=pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]}),
        baseline_df=pd.DataFrame(
            {"signal": [10.0, 11.0, 12.0], "time": [0.0, 1.0, 2.0]}
        ),
        models_root=models_root,
        local_time=1.5,
    )

    assert summary is not None
    assert summary["is_perfect_correct"] is True
    assert summary["noise_analysis"]["global"]["low"] == {
        "mean_snr": [2.0],
        "std_snr": [pytest.approx(1.4142135623730951)],
    }
    assert summary["noise_analysis"]["local"]["high"] == {
        "mean_snr": [3.0],
        "std_snr": [0.0],
    }


def test_child_rule_summary_can_contain_only_noise_analysis_without_basic_rule(
    tmp_path,
) -> None:
    models_root = tmp_path / "models" / "simulink"
    model_dir = models_root / "DemoModel"
    model_dir.mkdir(parents=True)
    (model_dir / "noise_adder.py").write_text(
        "\n".join(
            [
                "NOISE_DICT = {'low': {}, 'high': {}}",
                "SNR_THR_DICT = {",
                "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
                "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
                "}",
                "def quantify_noise(clean, noisy, reference):",
                "    return {'global': [100.0], 'local': [100.0]}",
                "def add_noise(src, seed=0, noise_level='low', ref=None):",
                "    return src.copy(), {'global': [float(seed)], 'local': [1.0]}",
            ]
        ),
        encoding="utf-8",
    )

    summary = compute_metrics._build_child_evaluate_rule_summary(
        model_id="DemoModel",
        run_id="child",
        truth_parameter="mass",
        child_df=pd.DataFrame({"signal": [1.0], "time": [0.0]}),
        baseline_df=pd.DataFrame({"signal": [1.0], "time": [0.0]}),
        models_root=models_root,
    )

    assert summary == {
        "noise_analysis": {
            "global": {
                "low": {"mean_snr": [2.0], "std_snr": [pytest.approx(1.4142135623730951)]},
                "high": {"mean_snr": [2.0], "std_snr": [pytest.approx(1.4142135623730951)]},
            },
            "local": {
                "low": {"mean_snr": [1.0], "std_snr": [0.0]},
                "high": {"mean_snr": [1.0], "std_snr": [0.0]},
            },
        }
    }


def test_compute_metrics_skips_optional_basic_rule_when_absent(
    tmp_path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "DemoModel"
    model_dir.mkdir()
    (model_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model_run_specs.json").write_text("{}", encoding="utf-8")
    runs_dir = model_dir / "runs"
    runs_dir.mkdir()
    (runs_dir / "model_record.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        compute_metrics,
        "load_experiment_config_json",
        lambda _path: SimpleNamespace(
            min_srd_distance=0.001,
            epsilon_SRD=0.001,
            observable_signal_envelope_sizes={},
        ),
    )
    monkeypatch.setattr(compute_metrics, "load_model_record_json", lambda _path: {})
    monkeypatch.setattr(compute_metrics, "load_model_run_specs_json", lambda *_, **__: {})
    monkeypatch.setattr(
        compute_metrics,
        "build_model_record_registry",
        lambda **_: {"baselines": []},
    )

    result = compute_metrics._run_for_model(
        model_dir,
        jobs=1,
        compute_ml=True,
    )

    assert result["ok"] is True
    assert not (runs_dir / "summary_accuracy.json").exists()


def test_eligibility_based_on_basic_rule_flag_is_ignored_for_tex_contract(
    tmp_path,
    monkeypatch,
) -> None:
    model_dir = tmp_path / "DemoModel"
    model_dir.mkdir()
    (model_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model_run_specs.json").write_text("{}", encoding="utf-8")
    (model_dir / "basic_rule.py").write_text(
        "def predict(df):\n    return 'no_parameter_change'\n",
        encoding="utf-8",
    )
    runs_dir = model_dir / "runs"
    runs_dir.mkdir()
    (runs_dir / "model_record.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        compute_metrics,
        "load_experiment_config_json",
        lambda _path: SimpleNamespace(
            min_srd_distance=0.001,
            epsilon_SRD=0.001,
            observable_signal_envelope_sizes={},
        ),
    )
    monkeypatch.setattr(compute_metrics, "load_model_record_json", lambda _path: {})
    monkeypatch.setattr(compute_metrics, "load_model_run_specs_json", lambda *_, **__: {})
    monkeypatch.setattr(
        compute_metrics,
        "build_model_record_registry",
        lambda **_: {"baselines": []},
    )

    result = compute_metrics._run_for_model(
        model_dir,
        jobs=1,
        compute_ml=False,
        eligibility_based_on_basic_rule=True,
    )

    assert result["ok"] is True
    assert not (runs_dir / "summary_accuracy.json").exists()


def _metrics_payload() -> dict[str, object]:
    return {
        "timestamp": "2026-04-17T14:32:05Z",
        "noise_adder_md5": "fedcba9876543210fedcba9876543210",
        "eligible_baselines": 0,
        "total_baselines": 1,
        "baselines": {
            "baseline-uuid": {
                "url": "http://localhost:3001/?model=DemoModel&run=baseline-uuid&compare=none",
                "family_eligible": False,
                "eligible": False,
                "children": {},
            },
        },
    }


def test_similarity_metrics_accepts_strict_documented_shape() -> None:
    payload = _metrics_payload()

    validate_similarity_metrics_schema(payload)
    validate_similarity_metrics_semantics(payload)
    SimilarityMetricsJson.model_validate(payload)


def test_similarity_metrics_allows_run_eligible_to_differ_from_family_eligible() -> None:
    payload = _metrics_payload()
    payload["baselines"]["baseline-uuid"]["eligible"] = True

    validate_similarity_metrics_schema(payload)
    validate_similarity_metrics_semantics(payload)
    SimilarityMetricsJson.model_validate(payload)


def test_similarity_metrics_accepts_nullable_first_diff_entries() -> None:
    payload = _metrics_payload()
    baseline = payload["baselines"]["baseline-uuid"]
    baseline["children"] = {
        "child-uuid": {
            "url": "http://localhost:3001/?model=DemoModel&run=child-uuid&compare=none",
            "detectability": {
                "vs_baseline": {
                    "environment_specific_detectability": "yes",
                    "detectable": "yes",
                    "detectability_output": {
                        "mean_euclidean_distance_clean_dirty": [0.1, 0.0],
                        "mean_euclidean_distance_clean_baseline": [2.0, 0.0],
                        "mean_SNR": [26.020599913279625, None],
                        "first_diff": [1.2, None],
                    },
                },
                "vs_time0_baseline": {
                    "detectable": "no",
                    "detectability_output": {
                        "mean_euclidean_distance_clean_dirty": [0.0],
                        "mean_euclidean_distance_clean_baseline": [0.0],
                        "mean_SNR": [None],
                        "first_diff": [None],
                    },
                },
            },
            "eligible": False,
        }
    }

    validate_similarity_metrics_schema(payload)
    validate_similarity_metrics_semantics(payload)


def test_similarity_metrics_rejects_undocumented_rule_fields() -> None:
    payload = _metrics_payload()
    payload["basic_rule_md5"] = "0123456789abcdef0123456789abcdef"
    payload["baselines"]["baseline-uuid"]["evaluate_rule"] = {
        "ground_truth": "no_parameter_change",
    }

    with pytest.raises(Exception, match="Additional properties"):
        validate_similarity_metrics_schema(payload)
    with pytest.raises(ValueError, match="evaluate_rule|total_baselines|eligible_baselines"):
        validate_similarity_metrics_semantics(payload)
