from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

from shared.tsenv_task_materialization import materialize


DOCUMENTED_NOISE_ADDER = "\n".join(
    [
        "import pandas as pd",
        "",
        "NOISE_DICT = {'low': {}, 'high': {}}",
        "SNR_THR_DICT = {",
        "    'low': {'global': [1.0e9, 1.0e9], 'local': [1.0e9, 1.0e9]},",
        "    'high': {'global': [1.0e9, 1.0e9], 'local': [1.0e9, 1.0e9]},",
        "}",
        "",
        "def quantify_noise(clean, noisy, reference):",
        "    return {'global': [0.0, 0.0], 'local': [None, None]}",
        "",
        "def add_noise(src: pd.DataFrame, seed: int = 0, noise_level: str = 'low', ref: pd.DataFrame | None = None):",
        "    assert noise_level == 'low'",
        "    out = src.copy()",
        "    out.iloc[:, 0] = out.iloc[:, 0] + float(seed % 10)",
        "    return out, quantify_noise(src, out, ref)",
    ]
)
DOCUMENTED_FAIL_NOISE_ADDER = "\n".join(
    [
        "NOISE_DICT = {'low': {}, 'high': {}}",
        "SNR_THR_DICT = {",
        "    'low': {'global': [1.0e9], 'local': [1.0e9]},",
        "    'high': {'global': [1.0e9], 'local': [1.0e9]},",
        "}",
        "",
        "def quantify_noise(clean, noisy, reference):",
        "    return {'global': [0.0], 'local': [None]}",
        "",
        "def add_noise(src, seed=0, noise_level='low', ref=None):",
        "    raise AssertionError('should not be called')",
    ]
)
BASELINE_AWARE_NOISE_ADDER = "\n".join(
    [
        "NOISE_DICT = {'low': {}, 'high': {}}",
        "SNR_THR_DICT = {",
        "    'low': {'global': [0.0], 'local': [0.0]},",
        "    'high': {'global': [0.0], 'local': [0.0]},",
        "}",
        "",
        "def quantify_noise(clean, noisy, reference):",
        "    return {'global': [float(clean.iloc[0, 0])], 'local': [float(reference.iloc[0, 0])]}",
        "",
        "def add_noise(src, seed=0, noise_level='low', ref=None):",
        "    out = src.copy()",
        "    out.iloc[:, 0] = out.iloc[:, 0] + float(ref.iloc[0, 0])",
        "    return out, quantify_noise(src, out, ref)",
    ]
)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "signal_position": [1.0],
            "signal_velocity": [2.0],
            "time": [0.1],
        }
    ).to_parquet(path)


def _write_value_parquet(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"signal": [float(value)], "time": [0.1]}).to_parquet(path)


def test_materialize_reads_noise_and_anonymizes_columns(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    models_root = tmp_path / "models" / "simulink"
    run_uuid = "182039f0b58a4ad590d08a1e9158e272"

    _write_parquet(tsenv_model_root / run_uuid)
    _write_text(
        tsenv_model_root / "noise_adder.py",
        DOCUMENTED_NOISE_ADDER,
    )
    _write_text(
        models_root / "BounceBall" / "noise_adder.py",
        DOCUMENTED_FAIL_NOISE_ADDER,
    )

    out, noise_analysis = materialize(
        run_uuid,
        "low",
        7,
        tsenv_model_root=tsenv_model_root,
        models_root=models_root,
    )

    assert list(out.columns) == ["col1", "col2", "col3"]
    assert float(out["col1"].iloc[0]) == pytest.approx(1.0 + float(7 % 10))
    assert float(out["col2"].iloc[0]) == pytest.approx(2.0)
    assert float(out["col3"].iloc[0]) == pytest.approx(0.1)
    assert noise_analysis == {"global": [0.0, 0.0], "local": [None, None]}


def test_materialize_passes_documented_baseline_dataframe_from_manifest(
    tmp_path: Path,
) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    child_uuid = "child_sample"
    baseline_uuid = "baseline_sample"

    _write_value_parquet(tsenv_model_root / "dataframes" / f"{child_uuid}.parquet", 2.0)
    _write_value_parquet(tsenv_model_root / "dataframes" / f"{baseline_uuid}.parquet", 100.0)
    _write_text(tsenv_model_root / "noise_adder.py", BASELINE_AWARE_NOISE_ADDER)
    _write_text(
        tsenv_model_root / "sample_manifest.json",
        """{"row": [{"train_samples": [], "train_samples_baselines": [], "test_samples": ["child_sample"], "test_samples_baselines": ["baseline_sample"]}]}""",
    )

    out, noise_analysis = materialize(
        child_uuid,
        "low",
        0,
        tsenv_model_root=tsenv_model_root,
    )

    assert list(out.columns) == ["col1", "col2"]
    assert float(out["col1"].iloc[0]) == pytest.approx(102.0)
    assert noise_analysis == {"global": [2.0], "local": [100.0]}


def test_materialize_passes_explicit_uuid_baseline_path(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    child_uuid = "child_sample"
    baseline_uuid = "baseline_sample"

    _write_value_parquet(tsenv_model_root / "dataframes" / f"{child_uuid}.parquet", 3.0)
    _write_value_parquet(tsenv_model_root / "dataframes" / f"{baseline_uuid}.parquet", 20.0)
    _write_text(tsenv_model_root / "noise_adder.py", BASELINE_AWARE_NOISE_ADDER)

    out, noise_analysis = materialize(
        child_uuid,
        "low",
        0,
        tsenv_model_root=tsenv_model_root,
        uuid_baseline_path=f"dataframes/{baseline_uuid}.parquet",
    )

    assert float(out["col1"].iloc[0]) == pytest.approx(23.0)
    assert noise_analysis == {"global": [3.0], "local": [20.0]}


def test_materialize_treats_missing_explicit_baseline_as_optional(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    child_uuid = "child_sample"

    _write_value_parquet(tsenv_model_root / "dataframes" / f"{child_uuid}.parquet", 5.0)
    _write_text(tsenv_model_root / "noise_adder.py", DOCUMENTED_NOISE_ADDER)

    with pytest.warns(RuntimeWarning, match="reference parquet not found"):
        out, noise_analysis = materialize(
            child_uuid,
            "low",
            4,
            tsenv_model_root=tsenv_model_root,
            uuid_baseline_path="dataframes/missing_baseline.parquet",
        )

    assert list(out.columns) == ["col1", "col2"]
    assert float(out["col1"].iloc[0]) == pytest.approx(9.0)
    assert noise_analysis == {"global": [0.0, 0.0], "local": [None, None]}


def test_materialize_treats_missing_manifest_baseline_as_optional(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    child_uuid = "child_sample"

    _write_value_parquet(tsenv_model_root / "dataframes" / f"{child_uuid}.parquet", 6.0)
    _write_text(tsenv_model_root / "noise_adder.py", DOCUMENTED_NOISE_ADDER)
    _write_text(
        tsenv_model_root / "sample_manifest.json",
        """{"row": [{"train_samples": [], "train_samples_baselines": [], "test_samples": ["child_sample"], "test_samples_baselines": ["missing_baseline"]}]}""",
    )

    with pytest.warns(RuntimeWarning, match="reference parquet not found"):
        out, noise_analysis = materialize(
            child_uuid,
            "low",
            3,
            tsenv_model_root=tsenv_model_root,
        )

    assert list(out.columns) == ["col1", "col2"]
    assert float(out["col1"].iloc[0]) == pytest.approx(9.0)
    assert noise_analysis == {"global": [0.0, 0.0], "local": [None, None]}


def test_materialize_passes_documented_baseline_dataframe_from_model_run_specs(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models" / "simulink" / "BounceBall"
    runs_root = model_root / "runs"
    child_uuid = "child_sample"
    baseline_uuid = "baseline_sample"

    _write_value_parquet(runs_root / child_uuid / "data.parquet", 4.0)
    _write_value_parquet(runs_root / baseline_uuid / "data.parquet", 30.0)
    _write_text(model_root / "noise_adder.py", BASELINE_AWARE_NOISE_ADDER)
    _write_text(
        model_root / "model_run_specs.json",
        """{"parent": {"children": {"child_sample": {"time0_baseline_uuid": "baseline_sample"}}}}""",
    )

    out, noise_analysis = materialize(
        "BounceBall",
        child_uuid,
        "low",
        0,
        runs_root=runs_root,
        noise_adder_path=model_root / "noise_adder.py",
    )

    assert float(out["col1"].iloc[0]) == pytest.approx(34.0)
    assert noise_analysis == {"global": [4.0], "local": [30.0]}


def test_materialize_does_not_fall_back_to_models_root_noise_adder(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    models_root = tmp_path / "models" / "simulink"
    run_uuid = "182039f0b58a4ad590d08a1e9158e272"

    _write_parquet(tsenv_model_root / "dataframes" / f"{run_uuid}.parquet")
    _write_text(
        models_root / "BounceBall" / "noise_adder.py",
        DOCUMENTED_NOISE_ADDER,
    )

    with pytest.raises(FileNotFoundError):
        materialize(
            run_uuid,
            "low",
            7,
            tsenv_model_root=tsenv_model_root,
            models_root=models_root,
        )


def test_materialize_supports_none_without_noise(tmp_path: Path) -> None:
    tsenv_model_root = tmp_path / "tsENV_questions" / "BounceBall"
    models_root = tmp_path / "models" / "simulink"
    run_uuid = "44f04c1b06cb4685b9878649fd7d6835"

    _write_parquet(tsenv_model_root / "dataframes" / f"{run_uuid}.parquet")
    _write_text(
        models_root / "BounceBall" / "noise_adder.py",
        DOCUMENTED_FAIL_NOISE_ADDER,
    )

    out, noise_analysis = materialize(
        run_uuid,
        "none",
        0,
        tsenv_model_root=tsenv_model_root,
        models_root=models_root,
    )

    assert list(out.columns) == ["col1", "col2", "col3"]
    assert float(out["col1"].iloc[0]) == pytest.approx(1.0)
    assert float(out["col2"].iloc[0]) == pytest.approx(2.0)
    assert float(out["col3"].iloc[0]) == pytest.approx(0.1)
    assert noise_analysis == {}


def test_materialize_supports_runs_root_layout(tmp_path: Path) -> None:
    model = "BounceBall"
    run_uuid = "182039f0b58a4ad590d08a1e9158e272"
    runs_root = tmp_path / "models" / "simulink" / model / "runs"

    _write_parquet(runs_root / run_uuid / "data.parquet")

    out, noise_analysis = materialize(
        run_uuid,
        "none",
        0,
        model=model,
        runs_root=runs_root,
    )

    assert list(out.columns) == ["col1", "col2", "col3"]
    assert float(out["col1"].iloc[0]) == pytest.approx(1.0)
    assert noise_analysis == {}
