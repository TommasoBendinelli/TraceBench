from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

try:
    from shared.prompts import tsenv_direct_results_payload
except ImportError:  # pragma: no cover
    def tsenv_direct_results_payload() -> str:
        return '{"<filename>.parquet": ["<possible_answer_1>",...,"<possible_answer_n>"], ...}'


_REQUIRED_RESULTS_FORMAT = tsenv_direct_results_payload()

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


def _normalize_prediction(sample_path: str, raw_label: object) -> list[str]:
    if isinstance(raw_label, str) and raw_label.strip():
        return [raw_label.strip()]
    if isinstance(raw_label, list):
        labels = []
        for idx, item in enumerate(raw_label):
            if not isinstance(item, str) or not item.strip():
                pytest.fail(
                    f"Prediction for sample {sample_path!r}[{idx}] must be a non-empty string label, got {item!r}."
                )
            labels.append(item.strip())
        if not labels:
            pytest.fail(f"Prediction for sample {sample_path!r} must contain at least one label.")
        if len(set(labels)) != len(labels):
            pytest.fail(f"Prediction for sample {sample_path!r} must not contain duplicate labels.")
        return labels
    pytest.fail(
        f"Prediction for sample {sample_path!r} must be a non-empty string label or list of labels, got {raw_label!r}."
    )


def _load_predictions(expected_paths: list[str]) -> dict[str, list[str]]:
    payload = _read_json(RESULTS_PATH)
    if payload is None:
        pytest.fail(
            "Missing results.json. Save results.json with "
            f"{_REQUIRED_RESULTS_FORMAT}."
        )
    expected_by_key: dict[str, str] = {}
    for expected_path in expected_paths:
        expected_by_key[expected_path] = expected_path
        path = Path(expected_path)
        if path.suffix == ".parquet":
            expected_by_key[path.stem] = expected_path
    predictions: dict[str, list[str]] = {}
    actual_paths: list[str] = []
    unexpected_paths: list[str] = []
    duplicate_paths: list[str] = []
    for raw_path, raw_label in payload.items():
        actual_path = str(raw_path).strip()
        actual_paths.append(actual_path)
        sample_path = expected_by_key.get(actual_path)
        if sample_path is None:
            unexpected_paths.append(actual_path)
            continue
        if sample_path in predictions:
            duplicate_paths.append(actual_path)
            continue
        predictions[sample_path] = _normalize_prediction(sample_path, raw_label)
    if (
        unexpected_paths
        or duplicate_paths
        or set(predictions) != set(expected_paths)
        or len(predictions) != len(expected_paths)
    ):
        pytest.fail(
            "results.json keys must match the parquet filenames under test_samples/, "
            "with or without the .parquet suffix. Do not include both forms for one sample. "
            f"Expected {expected_paths!r}, got {actual_paths!r}."
        )
    return predictions


def test_final_answer_format() -> None:
    sample_paths = _load_sample_paths()
    predictions = _load_predictions(sample_paths)

    for sample_path in sample_paths:
        answer = predictions.get(sample_path, "")
        if not isinstance(answer, list) or not answer:
            pytest.fail(f"Prediction for sample {sample_path!r} must be a non-empty list.")
