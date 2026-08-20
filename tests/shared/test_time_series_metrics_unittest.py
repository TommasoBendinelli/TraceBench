from __future__ import annotations

import pandas as pd
import pytest

from shared.time_series_metrics import (
    compute_detectability,
    compute_detectability_baseline,
    compute_detectability_time0_baseline,
    compute_first_detectable_time,
)


def test_first_detectable_time_uses_srd_threshold() -> None:
    baseline = pd.DataFrame({"time": [0.0, 1.0, 2.0], "signal": [1.0, 1.0, 1.0]})
    run = pd.DataFrame({"time": [0.0, 1.0, 2.0], "signal": [1.0, 1.05, 2.0]})

    assert (
        compute_first_detectable_time(
            baseline_df=baseline,
            run_df=run,
            first_detectable_minimum_symmetric_distance=0.1,
        )
        == 2.0
    )


def test_first_detectable_time_accepts_equal_srd_threshold() -> None:
    baseline = pd.DataFrame({"time": [0.0, 1.0], "signal": [1.0, 1.0]})
    run = pd.DataFrame({"time": [0.0, 1.0], "signal": [2.0, 3.0]})
    first_row_srd = 2.0 * abs(1.0 - 2.0) / (1.0 + 2.0 + 0.001)

    assert (
        compute_first_detectable_time(
            baseline_df=baseline,
            run_df=run,
            first_detectable_minimum_symmetric_distance=first_row_srd,
        )
        == 0.0
    )


def test_first_detectable_time_uses_absolute_denominator_for_signed_values() -> None:
    baseline = pd.DataFrame({"time": [0.0], "signal": [10.0]})
    run = pd.DataFrame({"time": [0.0], "signal": [-9.0]})

    assert (
        compute_first_detectable_time(
            baseline_df=baseline,
            run_df=run,
            first_detectable_minimum_symmetric_distance=1.5,
        )
        == 0.0
    )


def test_compute_detectability_reports_no_when_srd_never_crosses() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0], "time": [0.0, 1.0]})
    run = pd.DataFrame({"signal": [1.01, 1.02], "time": [0.0, 1.0]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["environment_specific_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["detectable"] == "no"
    assert payload["first_diff"] == [None]
    assert payload["max_SRD"][0] > 0.0
    assert payload["euclidean_distance"][0] > 0.0


def test_compute_detectability_uses_documented_epsilon_denominator() -> None:
    baseline = pd.DataFrame({"signal": [0.0], "time": [0.0]})
    run = pd.DataFrame({"signal": [0.05], "time": [0.0]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        first_detectable_epsilon=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["first_diff"] == [None]
    assert payload["max_SRD"] == [2.0 * 0.05 / 1.05]


def test_compute_detectability_accepts_equal_srd_threshold() -> None:
    baseline = pd.DataFrame({"signal": [1.0], "time": [0.0]})
    run = pd.DataFrame({"signal": [2.0], "time": [0.0]})
    first_row_srd = 2.0 * abs(1.0 - 2.0) / (1.0 + 2.0 + 0.001)

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=first_row_srd,
    )

    assert payload["first_diff"] == [0.0]
    assert payload["max_SRD"] == [first_row_srd]


def test_compute_detectability_ignores_first_diff_before_intervention() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [2.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["detectable"] == "no"
    assert payload["first_diff"] == [None]


def test_compute_detectability_reports_no_without_snr_when_first_diff_equals_intervention() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [1.0, 2.0, 2.0], "time": [0.0, 1.0, 2.0]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["first_diff"] == [1.0]


def test_compute_detectability_reports_no_without_snr_when_first_diff_follows_intervention() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    run = pd.DataFrame({"signal": [1.0, 1.0, 2.0], "time": [0.0, 1.0, 2.0]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["environment_specific_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["detectable"] == "no"
    assert payload["first_diff"] == [2.0]


def test_compute_detectability_requires_documented_consecutive_srd_steps() -> None:
    baseline = pd.DataFrame(
        {"signal": [1.0, 1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )
    run = pd.DataFrame(
        {"signal": [2.0, 1.0, 2.0, 2.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        minimum_consecutive_srd_steps=2,
        intervention_time=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["first_diff"] == [2.0]


def test_compute_detectability_ignores_isolated_srd_spike_when_consecutive_required() -> None:
    baseline = pd.DataFrame(
        {"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]}
    )
    run = pd.DataFrame(
        {"signal": [1.0, 2.0, 1.0], "time": [0.0, 1.0, 2.0]}
    )

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        minimum_consecutive_srd_steps=2,
        intervention_time=1.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["first_diff"] == [None]
    assert payload["max_SRD"][0] > 0.1


def test_compute_detectability_uses_signal_specific_detectability_config() -> None:
    baseline = pd.DataFrame(
        {
            "continuous_signal": [1.0, 1.0, 1.0],
            "impulse_signal": [1.0, 1.0, 1.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    run = pd.DataFrame(
        {
            "continuous_signal": [1.0, 2.0, 1.0],
            "impulse_signal": [1.0, 2.0, 1.0],
            "time": [0.0, 1.0, 2.0],
        }
    )

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        minimum_consecutive_srd_steps=1,
        intervention_time=1.0,
        signal_detectability_specs={
            "continuous_signal": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 2,
            },
            "impulse_signal": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
        },
        require_signal_detectability_specs=True,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["first_diff"] == [None, 1.0]


def test_compute_detectability_errors_when_required_signal_config_is_missing() -> None:
    baseline = pd.DataFrame(
        {"known": [1.0], "missing": [1.0], "time": [0.0]}
    )
    run = pd.DataFrame(
        {"known": [2.0], "missing": [2.0], "time": [0.0]}
    )

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        signal_detectability_specs={
            "known": {
                "min_srd_distance": 0.1,
                "epsilon_SRD": 0.001,
                "minimum_consecutive_srd_steps": 1,
            },
        },
        require_signal_detectability_specs=True,
    )

    assert payload["max_SRD_detectability"] == "error"
    assert payload["first_diff"] == []


def test_compute_detectability_baseline_requires_rms_threshold_pass() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0], "time": [0.0, 1.0]})
    noisy_run = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})

    below_threshold = compute_detectability_baseline(
        baseline_df=baseline,
        run_df=noisy_run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
        mean_euclidean_distance_clean_dirty=[2.0],
        mean_euclidean_distance_clean_baseline=[0.5],
        RMS_thresholds={"signal": 1.0},
    )
    above_threshold = compute_detectability_baseline(
        baseline_df=baseline,
        run_df=noisy_run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
        mean_euclidean_distance_clean_dirty=[0.1],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )

    assert below_threshold["max_SRD_detectability"] == "no"
    assert below_threshold["first_diff"] == [1.0]
    assert below_threshold["mean_euclidean_distance_clean_dirty"] == [2.0]
    assert below_threshold["mean_euclidean_distance_clean_baseline"] == [0.5]
    assert below_threshold["mean_SNR"] == [pytest.approx(-12.041199826559248)]
    assert above_threshold["max_SRD_detectability"] == "yes"
    assert above_threshold["first_diff"] == [1.0]
    assert above_threshold["mean_SNR"] == [pytest.approx(26.020599913279625)]


def test_compute_detectability_reports_null_snr_for_zero_noise_distance() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0], "time": [0.0, 1.0]})
    noisy_run = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})

    payload = compute_detectability_baseline(
        baseline_df=baseline,
        run_df=noisy_run,
        first_detectable_minimum_symmetric_distance=0.1,
        intervention_time=1.0,
        mean_euclidean_distance_clean_dirty=[0.0],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )

    assert payload["max_SRD_detectability"] == "yes"
    assert payload["mean_SNR"] == [None]


def test_compute_detectability_ignores_consecutive_run_before_intervention() -> None:
    baseline = pd.DataFrame(
        {"signal": [1.0, 1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )
    run = pd.DataFrame(
        {"signal": [2.0, 2.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
        minimum_consecutive_srd_steps=2,
        intervention_time=2.0,
    )

    assert payload["max_SRD_detectability"] == "no"
    assert payload["detectability"] == "no"
    assert payload["first_diff"] == [None]


def test_compute_detectability_time0_uses_rms_even_without_post_intervention_srd() -> None:
    time0 = pd.DataFrame({"signal": [1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]})
    before_intervention = pd.DataFrame(
        {"signal": [2.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0]}
    )
    at_intervention = pd.DataFrame(
        {"signal": [1.0, 2.0, 1.0], "time": [0.0, 1.0, 2.0]}
    )

    before = compute_detectability_time0_baseline(
        baseline_df=time0,
        run_df=before_intervention,
        first_detectable_minimum_symmetric_distance=0.1,
        first_detectable_epsilon=0.001,
        intervention_time=1.0,
        mean_euclidean_distance_clean_dirty=[0.1],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )
    at = compute_detectability_time0_baseline(
        baseline_df=time0,
        run_df=at_intervention,
        first_detectable_minimum_symmetric_distance=0.1,
        first_detectable_epsilon=0.001,
        intervention_time=1.0,
        mean_euclidean_distance_clean_dirty=[0.1],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )

    assert before["max_SRD_detectability"] == "yes"
    assert before["detectability"] == "yes"
    assert before["first_diff"] == [None]
    assert at["max_SRD_detectability"] == "yes"
    assert at["detectability"] == "yes"
    assert at["first_diff"] == [1.0]


def test_compute_detectability_time0_uses_consecutive_srd_steps() -> None:
    time0 = pd.DataFrame(
        {"signal": [1.0, 1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )
    isolated_before = pd.DataFrame(
        {"signal": [2.0, 1.0, 1.0, 1.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )
    consecutive_after = pd.DataFrame(
        {"signal": [1.0, 1.0, 2.0, 2.0], "time": [0.0, 1.0, 2.0, 3.0]}
    )

    isolated = compute_detectability_time0_baseline(
        baseline_df=time0,
        run_df=isolated_before,
        first_detectable_minimum_symmetric_distance=0.1,
        first_detectable_epsilon=0.001,
        minimum_consecutive_srd_steps=2,
        intervention_time=2.0,
        mean_euclidean_distance_clean_dirty=[0.1],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )
    consecutive = compute_detectability_time0_baseline(
        baseline_df=time0,
        run_df=consecutive_after,
        first_detectable_minimum_symmetric_distance=0.1,
        first_detectable_epsilon=0.001,
        minimum_consecutive_srd_steps=2,
        intervention_time=2.0,
        mean_euclidean_distance_clean_dirty=[0.1],
        mean_euclidean_distance_clean_baseline=[2.0],
        RMS_thresholds={"signal": 1.0},
    )

    assert isolated["max_SRD_detectability"] == "yes"
    assert isolated["first_diff"] == [None]
    assert consecutive["max_SRD_detectability"] == "yes"
    assert consecutive["first_diff"] == [2.0]


def test_compute_detectability_errors_on_mismatched_timestamps() -> None:
    baseline = pd.DataFrame({"signal": [1.0, 1.0], "time": [0.0, 1.0]})
    run = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.5]})

    payload = compute_detectability(
        baseline_df=baseline,
        run_df=run,
        first_detectable_minimum_symmetric_distance=0.1,
    )

    assert payload == {
        "environment_specific_detectability": "error",
        "max_SRD_detectability": "error",
        "detectability": "no",
        "detectable": "error",
        "max_SRD": [],
        "euclidean_distance": [],
        "mean_euclidean_distance_clean_dirty": [],
        "mean_euclidean_distance_clean_baseline": [],
        "mean_SNR": [],
        "first_diff": [],
    }
