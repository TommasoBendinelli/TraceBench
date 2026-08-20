from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

FINAL_ANSWER_KEY = "final_answer"
_REQUIRED_RESULTS_FORMAT = (
    "{\"predictions\": {\"test_samples/sample_0000.parquet\": "
    "{\"change_time\": <t>, \"final_answer\": \"<class_label>\"}, ...}}"
)

RESULTS_PATH = Path("results.json")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

def _extract_answer() -> str:
    payload = _read_json(RESULTS_PATH)
    if payload is None:
        pytest.fail(
            "Missing results.json. Save results.json with "
            f"{_REQUIRED_RESULTS_FORMAT}."
        )

    if not isinstance(payload, dict):
        pytest.fail("results.json must be a JSON object.")
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict) or not predictions:
        pytest.fail(
            "Missing predictions object in results.json. Save results.json with "
            f"{_REQUIRED_RESULTS_FORMAT}."
        )
    out: dict[str, dict] = {}
    for sample_path, prediction in predictions.items():
        path_text = str(sample_path or "").strip()
        if not path_text:
            pytest.fail("predictions keys must be non-empty sample paths.")
        if not isinstance(prediction, dict):
            pytest.fail("Each prediction entry must be a JSON object.")
        if "change_time" not in prediction:
            pytest.fail("Each prediction entry must include change_time.")
        change_time = prediction.get("change_time")
        if change_time is not None:
            if not isinstance(change_time, (int, float)):
                pytest.fail("change_time must be a number, -1, or null.")
            numeric = float(change_time)
            if numeric < 0 and numeric != -1:
                pytest.fail("change_time must be >= 0, -1, or null.")
        candidate: Any = prediction.get(FINAL_ANSWER_KEY)
        if not isinstance(candidate, str) or not candidate:
            pytest.fail("final_answer must be a non-empty string.")
        out[path_text] = prediction
    return out


def test_final_answer_format():
    answer = _extract_answer()
    if not answer.strip():
        pytest.fail("final_answer must be a non-empty string.")
