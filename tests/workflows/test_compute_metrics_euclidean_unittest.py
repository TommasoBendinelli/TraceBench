from __future__ import annotations

import math

import pandas as pd

from workflows.metrics import compute_metrics_euclidean as euclid


def test_scale_normalized_rms_uses_pre_intervention_abs_max() -> None:
    reference = pd.DataFrame(
        {
            "x": [1.0, 2.0, 100.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    run = pd.DataFrame(
        {
            "x": [1.0, 4.0, 104.0],
            "time": [0.0, 1.0, 2.0],
        }
    )

    signals, scores, scales = euclid._scale_normalized_rms_by_signal(
        reference_df=reference,
        run_df=run,
        intervention_time=2.0,
    )

    assert signals == ["x"]
    assert scales == [4.0]
    expected_rms = math.sqrt((0.0**2 + 2.0**2 + 4.0**2) / 3.0)
    assert scores == [expected_rms / 4.0]


def test_scale_normalized_rms_falls_back_to_reference_scale() -> None:
    reference = pd.DataFrame(
        {
            "x": [3.0, 6.0, 9.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    run = pd.DataFrame(
        {
            "x": [0.0, 0.0, 3.0],
            "time": [0.0, 1.0, 2.0],
        }
    )

    _signals, _scores, scales = euclid._scale_normalized_rms_by_signal(
        reference_df=reference,
        run_df=run,
        intervention_time=2.0,
    )

    assert scales == [6.0]


def test_calibrate_threshold_retains_current_within_extra_budget() -> None:
    children = {
        "current": _child(scores={"x": 2.0, "y": 0.2}),
        "extra": _child(scores={"x": 2.1, "y": 0.1}),
        "rejected": _child(scores={"x": 1.0, "y": 3.0}),
    }

    calibration = euclid._calibrate_threshold(
        children=children,
        current_eligible={"current"},
        extra_budget_fraction=1.0,
    )

    assert calibration["ok"] is True
    assert calibration["dropped_child_ids"] == []
    assert calibration["added_child_count"] == 1
    assert calibration["initial_added_child_ids"] == ["extra"]
    assert set(calibration["thresholds"]) == {"x", "y"}


def test_calibrate_threshold_fails_when_extras_exceed_budget() -> None:
    children = {
        "current": _child(scores={"x": 2.0}),
        "extra": _child(scores={"x": 2.1}),
    }

    calibration = euclid._calibrate_threshold(
        children=children,
        current_eligible={"current"},
        extra_budget_fraction=0.0,
    )

    assert calibration["ok"] is False
    assert calibration["added_child_ids"] == ["extra"]


def test_calibrate_threshold_keeps_different_channels_independent() -> None:
    children = {
        "current_x": _child(scores={"x": 2.0, "y": 0.2}),
        "current_y": _child(scores={"x": 0.1, "y": 3.0}),
        "x_only": _child(scores={"x": 2.1, "y": 0.0}),
        "y_only": _child(scores={"x": 0.0, "y": 3.1}),
        "neither": _child(scores={"x": 1.0, "y": 2.0}),
    }

    calibration = euclid._calibrate_threshold(
        children=children,
        current_eligible={"current_x", "current_y"},
        extra_budget_fraction=1.0,
    )

    assert calibration["ok"] is True
    assert calibration["added_child_count"] == 2
    assert calibration["initial_added_child_ids"] == ["x_only", "y_only"]
    assert calibration["support_counts"] == {"x": 1, "y": 1}
    assert calibration["support_counts_by_side"]["vs_baseline"] == {"x": 1, "y": 1}
    assert set(calibration["thresholds"]) == {"x", "y"}


def test_apply_threshold_updates_child_and_family_eligibility() -> None:
    results = {
        "baselines": {
            "baseline": {
                "children": {
                    "a": _child(scores={"x": 2.0, "y": 0.1}),
                    "b": _child(scores={"x": 1.0, "y": 0.1}),
                    "c": _child(scores={"x": 0.1, "y": 3.0}),
                },
                "family_eligible": False,
                "eligible": False,
            }
        },
        "eligible_baselines": 0,
        "total_baselines": 1,
    }

    euclid._apply_threshold(
        results=results,
        thresholds={"x": 1.5, "y": 2.5},
        expected_parameters=("p1", "p2", "p3"),
        child_parameters_by_baseline={
            "baseline": {
                "a": "p1",
                "b": "p2",
                "c": "p3",
            }
        },
    )

    children = results["baselines"]["baseline"]["children"]
    assert children["a"]["eligible"] is True
    assert children["b"]["eligible"] is False
    assert children["c"]["eligible"] is True
    assert results["baselines"]["baseline"]["family_eligible"] is False
    baseline_output = children["c"]["detectability"]["vs_baseline"]["detectability_output"]
    assert baseline_output["scale_normalized_rms_thresholds"] == [1.5, 2.5]
    assert baseline_output["scale_normalized_rms_passed_by_signal"] == [False, True]


def test_apply_threshold_disables_unsupported_channels() -> None:
    side = _side(scores={"x": 0.0, "y": 100.0})

    out = euclid._apply_threshold_to_side(side, thresholds={"x": 1.0, "y": None})

    output = out["detectability_output"]
    assert output["scale_normalized_rms_thresholds"] == [1.0, None]
    assert output["scale_normalized_rms_passed_by_signal"] == [False, False]
    assert output["scale_normalized_rms_passed"] is False
    assert out["detectable"] == "no"


def _child(*, scores: dict[str, float]) -> dict:
    return {
        "detectability": {
            "vs_baseline": _side(scores=scores),
            "vs_time0_baseline": _side(scores=scores),
        },
        "eligible": False,
    }


def _side(*, scores: dict[str, float]) -> dict:
    signals = list(scores)
    values = [scores[signal] for signal in signals]
    return {
        "environment_specific_detectability": "yes",
        "detectable": "yes",
        "detectability_output": {
            "scale_normalized_rms_max": max(values),
            "scale_normalized_rms": values,
            "scale_normalized_rms_scales": [1.0 for _ in values],
            "scale_normalized_rms_signals": signals,
            "mean_euclidean_distance_clean_dirty": [0.0],
            "mean_euclidean_distance_clean_baseline": values,
            "mean_SNR": [None for _ in values],
            "first_diff": [1.0 for _ in values],
        },
    }
