from __future__ import annotations

import math

import pandas as pd
import pytest

from shared.noise_analysis import quantify_analysis


def test_quantify_analysis_uses_effect_relative_to_reference_for_global_snr() -> None:
    clean = pd.DataFrame({"signal": [3.0, 5.0], "time": [0.0, 1.0]})
    noisy = pd.DataFrame({"signal": [4.0, 4.0], "time": [0.0, 1.0]})
    reference = pd.DataFrame({"signal": [1.0, 1.0], "time": [0.0, 1.0]})

    payload = quantify_analysis(clean, noisy, reference)

    effect_rms = math.sqrt(((3.0 - 1.0) ** 2 + (5.0 - 1.0) ** 2) / 2.0)
    noise_rms = math.sqrt(((4.0 - 3.0) ** 2 + (4.0 - 5.0) ** 2) / 2.0)
    expected = 20.0 * math.log10(effect_rms / noise_rms)
    assert payload["global"] == [pytest.approx(expected)]
    assert payload["local"] == [None]


def test_quantify_analysis_uses_reference_for_local_first_diff_window() -> None:
    clean = pd.DataFrame(
        {
            "signal": [1.0, 3.0, 5.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    noisy = pd.DataFrame(
        {
            "signal": [1.5, 2.5, 5.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    reference = pd.DataFrame(
        {
            "signal": [1.0, 1.0, 1.0],
            "time": [0.0, 1.0, 2.0],
        }
    )

    payload = quantify_analysis(
        clean,
        noisy,
        reference,
        1.0,
        local_radius_rows=0,
    )

    expected_local = 20.0 * math.log10(abs(3.0 - 1.0) / abs(2.5 - 3.0))
    assert payload["local"] == [pytest.approx(expected_local)]


def test_quantify_analysis_uses_raw_signal_as_effect_without_reference() -> None:
    clean = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})
    noisy = pd.DataFrame({"signal": [1.0, 2.1], "time": [0.0, 1.0]})

    payload = quantify_analysis(clean, noisy, None, None)

    effect_rms = math.sqrt((1.0**2 + 2.0**2) / 2.0)
    noise_rms = math.sqrt((0.0**2 + 0.1**2) / 2.0)
    expected = 20.0 * math.log10(effect_rms / noise_rms)
    assert payload["global"] == [pytest.approx(expected)]
    assert payload["local"] == [None]


def test_quantify_analysis_returns_none_when_noise_rms_is_zero() -> None:
    clean = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})
    noisy = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})

    payload = quantify_analysis(clean, noisy, None, None)

    assert payload == {"global": [None], "local": [None]}


def test_quantify_analysis_reports_negative_infinity_for_zero_effect_with_noise() -> None:
    clean = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})
    noisy = pd.DataFrame({"signal": [2.0, 2.0], "time": [0.0, 1.0]})
    reference = pd.DataFrame({"signal": [1.0, 2.0], "time": [0.0, 1.0]})

    payload = quantify_analysis(clean, noisy, reference)

    assert payload["global"] == ["-inf"]
    assert payload["local"] == [None]


def test_quantify_analysis_list_inputs_average_linear_ratios_before_db() -> None:
    clean_a = pd.DataFrame({"signal": [3.0]})
    noisy_a = pd.DataFrame({"signal": [4.0]})
    reference_a = pd.DataFrame({"signal": [1.0]})
    clean_b = pd.DataFrame({"signal": [9.0]})
    noisy_b = pd.DataFrame({"signal": [11.0]})
    reference_b = pd.DataFrame({"signal": [1.0]})

    payload = quantify_analysis(
        [clean_a, clean_b],
        [noisy_a, noisy_b],
        [reference_a, reference_b],
    )

    ratio_a = abs(3.0 - 1.0) / abs(4.0 - 3.0)
    ratio_b = abs(9.0 - 1.0) / abs(11.0 - 9.0)
    expected = 20.0 * math.log10((ratio_a + ratio_b) / 2.0)
    assert payload["global"] == [pytest.approx(expected)]
    assert payload["local"] == [None]


def test_quantify_analysis_rejects_mismatched_list_lengths() -> None:
    frame = pd.DataFrame({"signal": [1.0]})

    with pytest.raises(ValueError, match="equal length"):
        quantify_analysis([frame], [frame, frame], [frame])
