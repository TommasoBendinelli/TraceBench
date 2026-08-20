from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import pytest

RULE_PATH = Path("rule.py")
TEST_SAMPLES_DIR = Path("/app/test_samples")


def _discover_code_samples() -> list[Path]:
    samples = sorted(path for path in TEST_SAMPLES_DIR.rglob("*.parquet") if path.is_file())
    if not samples:
        pytest.fail("Expected at least one parquet file under /app/test_samples.")
    return samples


def _resolve_rule_path(rule_file: str) -> Path:
    rule_path = Path(rule_file).expanduser()
    if not rule_path.is_absolute():
        rule_path = (Path(".") / rule_path).resolve()
    return rule_path


def _load_predict_fn(rule_path: Path) -> Callable[[Any], Any]:
    spec = importlib.util.spec_from_file_location("code_rule_module", rule_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Unable to import rule module from {rule_path}.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.fail(f"Failed to execute rule file {rule_path}: {exc}")
    predict = getattr(module, "predict", None)
    if not callable(predict):
        pytest.fail("rule_file must define a callable function predict(df).")
    return predict


def _extract_answer(prediction: Any, *, sample_idx: int) -> list[str]:
    if isinstance(prediction, str) and prediction.strip():
        return [prediction.strip()]
    if isinstance(prediction, list):
        out = []
        for idx, item in enumerate(prediction):
            if not isinstance(item, str) or not item.strip():
                pytest.fail(
                    f"predict output for sample {sample_idx}[{idx}] must be a non-empty string label, got {item!r}"
                )
            out.append(item.strip())
        if not out:
            pytest.fail(f"predict output for sample {sample_idx} must contain at least one label.")
        return out
    if isinstance(prediction, dict):
        answer = prediction.get("final_answer")
        if isinstance(answer, str) and answer.strip():
            return [answer.strip()]
    else:
        answer = prediction
    if not isinstance(answer, str) or not answer.strip():
        pytest.fail(
            f"predict output for sample {sample_idx} must be a list of labels, a string label, "
            f"or a dict containing non-empty 'final_answer', got {prediction!r}"
        )
    return [answer.strip()]


def test_code_rule_contract() -> None:
    rule_path = _resolve_rule_path(str(RULE_PATH))
    if not rule_path.exists():
        pytest.fail("Missing rule.py. Create `rule.py` with a function `predict(df) -> list[str]`.")

    predict = _load_predict_fn(rule_path)
    sample_paths = _discover_code_samples()

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        pytest.fail(f"pandas is required for code evaluation: {exc}")

    for idx, sample_path in enumerate(sample_paths):
        if not sample_path.exists():
            pytest.fail(f"test_samples[{idx}] path does not exist: {sample_path}")
        try:
            dataframe = pd.read_parquet(sample_path)
        except Exception as exc:
            pytest.fail(f"Failed to read sample parquet at {sample_path}: {exc}")
        try:
            prediction = predict(dataframe)
        except Exception as exc:
            pytest.fail(f"predict(df) failed on sample {idx} ({sample_path}): {exc}")
        _extract_answer(prediction, sample_idx=idx)
