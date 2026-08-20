from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

FINAL_ANSWER_KEY = "final_answer"
_REQUIRED_RESULTS_FORMAT = "{\"final_answer\": \"<class_label>\"}"

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
    candidate: Any = payload.get(FINAL_ANSWER_KEY)
    if not isinstance(candidate, str) or not candidate:
        pytest.fail("final_answer must be a non-empty string.")
    return candidate


def test_final_answer_format():
    answer = _extract_answer()
    if not answer.strip():
        pytest.fail("final_answer must be a non-empty string.")
