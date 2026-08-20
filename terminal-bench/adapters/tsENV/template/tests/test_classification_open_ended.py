from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

try:
    from shared.prompts import tsenv_open_ended_results_payload
except ImportError:  # pragma: no cover
    def tsenv_open_ended_results_payload() -> str:
        return '{"<filename>.parquet": "<your answer>", ...}'


_REQUIRED_RESULTS_FORMAT = tsenv_open_ended_results_payload()

RESULTS_PATH = Path("results.json")
APP_ROOT = Path("/app")
TEST_SAMPLES_DIR = APP_ROOT / "test_samples"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_sample_paths() -> list[str]:
    if not TEST_SAMPLES_DIR.is_dir():
        pytest.fail("Missing /app/test_samples directory.")
    samples = sorted(
        path.name
        for path in TEST_SAMPLES_DIR.rglob("*.parquet")
        if path.is_file()
    )
    if not samples:
        pytest.fail("Expected at least one parquet file under /app/test_samples.")
    return samples


def _load_predictions(expected_paths: list[str]) -> dict[str, str]:
    payload = _read_json(RESULTS_PATH)
    if payload is None:
        pytest.fail(
            "Missing results.json. Save results.json with "
            f"{_REQUIRED_RESULTS_FORMAT}."
        )
    actual_paths = [str(key) for key in payload.keys()]
    if set(actual_paths) != set(expected_paths) or len(actual_paths) != len(expected_paths):
        pytest.fail(
            "results.json keys must exactly match the parquet filenames under test_samples/. "
            f"Expected {expected_paths!r}, got {actual_paths!r}."
        )
    predictions: dict[str, str] = {}
    for raw_path, raw_label in payload.items():
        sample_path = str(raw_path)
        if not isinstance(raw_label, str) or not raw_label.strip():
            pytest.fail(
                f"Prediction for sample {sample_path!r} must be a non-empty string answer, got {raw_label!r}."
            )
        predictions[sample_path] = raw_label.strip()
    return predictions


def test_final_answer_format() -> None:
    sample_paths = _load_sample_paths()
    predictions = _load_predictions(sample_paths)

    for sample_path in sample_paths:
        answer = predictions.get(sample_path, "")
        if not isinstance(answer, str) or not answer.strip():
            pytest.fail(f"Prediction for sample {sample_path!r} must be a non-empty string.")
