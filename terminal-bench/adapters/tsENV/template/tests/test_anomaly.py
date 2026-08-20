from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

try:
    from shared.prompts import FINAL_ANSWER_KEY
except ModuleNotFoundError:
    FINAL_ANSWER_KEY = "final_answer"

RESULTS_PATH = Path("results.json")
POST_AGENT_LOG = Path("post-agent.txt")


def _read_results_json(results_path: Path) -> Optional[dict]:
    if not results_path.exists():
        return None
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{results_path} is not valid JSON: {exc}") from exc





def _extract_anomaly_windows(results_path: Path, _log_path: Path) -> list:
    payload = _read_results_json(results_path)
    # if isinstance(payload, dict):
    candidate = payload[FINAL_ANSWER_KEY] # _extract_results_candidate(payload)
    return candidate



def _parse_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        pytest.fail(f"{label} must be an integer, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    pytest.fail(f"{label} must be an integer, got {value!r}")


def _parse_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        pytest.fail(f"{label} must be a float in [0, 1], got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            pass
    pytest.fail(f"{label} must be a float in [0, 1], got {value!r}")


def _extract_anomalies() -> list:
    try:
        anomalies = _extract_anomaly_windows(RESULTS_PATH, POST_AGENT_LOG)
    except ValueError as exc:
        pytest.fail(str(exc))
    if anomalies is None:
        pytest.fail(
            "Could not parse anomaly windows. "
            f"Write results.json with {{'{FINAL_ANSWER_KEY}': [[start_idx, end_idx, confidence], ...]}}."
        )
    return anomalies


def _parse_anomaly(entry: object, index: int) -> tuple[int, int, float]:

    start_val, end_val, score_val = entry[0], entry[1], entry[2]

    start = _parse_int(start_val, f"start_idx for entry {index}")
    end = _parse_int(end_val, f"end_idx for entry {index}")
    score = _parse_float(score_val, f"confidence for entry {index}")
    if score < 0.0 or score > 1.0:
        pytest.fail(f"confidence for entry {index} must be in [0, 1], got {score}")
    if start > end:
        pytest.fail(f"start_idx must be <= end_idx for entry {index}, got {start} > {end}")
    return start, end, score


def test_final_answer_format():
    anomalies = _extract_anomalies()
    assert isinstance(anomalies, list), "final_answer must be a list"
    for idx, entry in enumerate(anomalies):
        _parse_anomaly(entry, idx)
