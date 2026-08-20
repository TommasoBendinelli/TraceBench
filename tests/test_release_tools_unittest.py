from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.download_release as release_download
from scripts.download_release import ALLOW_PATTERNS, FORBIDDEN_RESULT_COLUMNS
from scripts.reproduce_results import METRICS, reproduce, summarize_results
from shared.release_paths import DATASET_REPO_ID, DATASET_REVISION, default_questions_root


ROOT = Path(__file__).resolve().parents[1]


def _row(*, agent_id: str, accuracy: float) -> dict[str, object]:
    row: dict[str, object] = {
        "tag": "PAPER",
        "status": "DONE",
        "agentic_run_id": f"run-{agent_id}-{accuracy}",
        "question_slug": "frost_01234-anchor_0",
        "row_slug": "frost_01234-anchor",
        "agent_id": agent_id,
        "model": "BallDrop",
    }
    for metric in METRICS:
        row[metric] = accuracy if metric in {"test_top1_accuracy", "test_shortlist_score"} else 1
    return row


def test_release_manifest_matches_public_constants() -> None:
    manifest = json.loads((ROOT / "reproduction" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["dataset"]["repo_id"] == DATASET_REPO_ID
    assert manifest["dataset"]["revision"] == DATASET_REVISION
    assert len(manifest["dataset"]["release_manifest_sha256"]) == 64
    assert len(manifest["dataset"]["results_csv_sha256"]) == 64
    assert len(manifest["dataset"]["results_parquet_sha256"]) == 64
    assert manifest["dataset"]["expected_columns"] == 47
    assert manifest["dataset"]["expected_models"] == [
        "BallDrop",
        "BounceBall",
        "MassSlide",
    ]


def test_download_release_selects_questions_and_results() -> None:
    assert ALLOW_PATTERNS == (
        "questions/**",
        "release_manifest.json",
        "results.csv",
        "results.parquet",
    )


def test_download_release_rejects_machine_local_result_columns() -> None:
    assert FORBIDDEN_RESULT_COLUMNS == {
        "path_to_the_run",
        "scores_path",
        "trajectory_evaluation_path",
        "atif_trajectory_path",
        "atif_pdf_trajectory_path",
        "is_correct_format",
    }


def _write_small_release(root: Path) -> Path:
    models = ["BallDrop", "BounceBall", "MassSlide"]
    frame = pd.DataFrame(
        [
            {"tag": "PAPER", "status": "DONE", "model": model, "value": index}
            for index, model in enumerate(models)
        ]
    )
    frame.to_csv(root / "results.csv", index=False)
    frame.to_parquet(root / "results.parquet", index=False)
    for model in models:
        model_root = root / "questions" / model
        model_root.mkdir(parents=True)
        (model_root / "questions.json").write_text(
            json.dumps({"environment_name": model, "questions": {"q": {}}}),
            encoding="utf-8",
        )
    release_manifest_path = root / "release_manifest.json"
    release_manifest_path.write_text(
        json.dumps(
            {
                "results": {
                    "rows": 3,
                    "columns": 4,
                    "results_csv_sha256": release_download.sha256_file(
                        root / "results.csv"
                    ),
                    "results_parquet_sha256": release_download.sha256_file(
                        root / "results.parquet"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "release_manifest_sha256": release_download.sha256_file(
                        release_manifest_path
                    ),
                    "results_csv_sha256": release_download.sha256_file(root / "results.csv"),
                    "results_parquet_sha256": release_download.sha256_file(
                        root / "results.parquet"
                    ),
                    "expected_rows": 3,
                    "expected_columns": 4,
                    "expected_rows_by_tag": {"PAPER": 3},
                    "expected_rows_by_model": {model: 1 for model in models},
                    "expected_models": models,
                    "expected_questions_by_model": {model: 1 for model in models},
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_verify_pinned_release_checks_both_formats_and_layout(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _write_small_release(tmp_path)
    monkeypatch.setattr(release_download, "MANIFEST_PATH", manifest_path)
    release_download.verify_pinned_release(tmp_path, DATASET_REVISION)


def test_verify_pinned_release_rejects_private_paths(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_small_release(tmp_path)
    monkeypatch.setattr(release_download, "MANIFEST_PATH", manifest_path)
    (tmp_path / "questions" / "BallDrop" / "model_record.json").write_text(
        '{"path": "/home/example-user/private"}',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="Private path pattern"):
        release_download.verify_pinned_release(tmp_path, DATASET_REVISION)


def test_default_questions_root_prefers_public_layout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRACEBENCH_QUESTIONS_ROOT", raising=False)
    monkeypatch.delenv("TSENV_QUESTIONS_ROOT", raising=False)
    assert default_questions_root(tmp_path) == (tmp_path / "data" / "questions").resolve()


def test_default_questions_root_retains_legacy_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TRACEBENCH_QUESTIONS_ROOT", raising=False)
    monkeypatch.delenv("TSENV_QUESTIONS_ROOT", raising=False)
    legacy = tmp_path / "tsENV_questions"
    legacy.mkdir()
    assert default_questions_root(tmp_path) == legacy.resolve()


def test_summarize_results_groups_released_rows() -> None:
    frame = pd.DataFrame([_row(agent_id="agent-a", accuracy=1.0), _row(agent_id="agent-a", accuracy=0.0)])
    by_condition, by_agent = summarize_results(frame, tag="PAPER")
    assert by_condition.loc[0, "run_count"] == 2
    assert by_agent.loc[0, "test_top1_accuracy_mean"] == 0.5


def test_reproduce_writes_public_outputs(tmp_path: Path) -> None:
    source = tmp_path / "results.csv"
    pd.DataFrame([_row(agent_id="agent-a", accuracy=1.0)]).to_csv(source, index=False)
    outputs = reproduce(source, tmp_path / "out", verify_pinned=False)
    assert set(outputs) == {"summary_by_condition", "summary_by_agent", "accuracy_plot", "provenance"}
    assert all(path.is_file() for path in outputs.values())
