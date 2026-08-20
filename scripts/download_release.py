#!/usr/bin/env python3
"""Download the immutable public TraceBench question and result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from shared.release_paths import DATASET_REPO_ID, DATASET_REVISION


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reproduction" / "manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data"
ALLOW_PATTERNS = (
    "questions/**",
    "release_manifest.json",
    "results.csv",
    "results.parquet",
)
FORBIDDEN_RESULT_COLUMNS = frozenset(
    {
        "path_to_the_run",
        "scores_path",
        "trajectory_evaluation_path",
        "atif_trajectory_path",
        "atif_pdf_trajectory_path",
        "is_correct_format",
    }
)
PRIVATE_PATH_PATTERNS = (
    re.compile(rb"/home/[^/\s\"']+"),
    re.compile(rb"/Users/[^/\s\"']+"),
    re.compile(rb"/csem/divr/users/[^/\s\"']+"),
)
LEGACY_OR_INTERMEDIATE_PATHS = (
    "questions/raw",
    "questions/BounceBall.zip",
    "questions/README.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{MANIFEST_PATH} must contain a JSON object")
    return payload


def download_release(output_dir: Path, revision: str = DATASET_REVISION) -> Path:
    destination = output_dir.expanduser().resolve()
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=list(ALLOW_PATTERNS),
        local_dir=destination,
        token=False,
    )
    return destination


def _verify_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Missing required release file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def _verify_no_private_paths(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if not path.is_file() or ".cache" in path.relative_to(output_dir).parts:
            continue
        payload = path.read_bytes()
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(payload):
                raise SystemExit(f"Private path pattern {pattern.pattern!r} found in {path}")


def verify_pinned_release(output_dir: Path, revision: str) -> None:
    if revision != DATASET_REVISION:
        return
    manifest = load_manifest()
    dataset_manifest = manifest["dataset"]
    release_manifest_path = output_dir / "release_manifest.json"
    csv_path = output_dir / "results.csv"
    parquet_path = output_dir / "results.parquet"
    _verify_sha256(
        release_manifest_path,
        str(dataset_manifest["release_manifest_sha256"]),
    )
    _verify_sha256(csv_path, str(dataset_manifest["results_csv_sha256"]))
    _verify_sha256(parquet_path, str(dataset_manifest["results_parquet_sha256"]))

    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    released_results = release_manifest.get("results")
    if not isinstance(released_results, dict):
        raise SystemExit("release_manifest.json lacks results provenance")
    expected_release_values = {
        "rows": int(dataset_manifest["expected_rows"]),
        "columns": int(dataset_manifest["expected_columns"]),
        "results_csv_sha256": str(dataset_manifest["results_csv_sha256"]),
        "results_parquet_sha256": str(dataset_manifest["results_parquet_sha256"]),
    }
    actual_release_values = {
        key: released_results.get(key) for key in expected_release_values
    }
    if actual_release_values != expected_release_values:
        raise SystemExit(
            "release_manifest.json does not match the pinned reproduction manifest: "
            f"expected {expected_release_values}, got {actual_release_values}"
        )

    csv_frame = pd.read_csv(csv_path)
    parquet_frame = pd.read_parquet(parquet_path)
    if list(csv_frame.columns) != list(parquet_frame.columns):
        raise SystemExit("results.csv and results.parquet have different columns or column order")
    try:
        pd.testing.assert_frame_equal(csv_frame, parquet_frame, check_dtype=False)
    except AssertionError as error:
        raise SystemExit(f"results.csv and results.parquet differ: {error}") from error

    expected_rows = int(dataset_manifest["expected_rows"])
    expected_columns = int(dataset_manifest["expected_columns"])
    if csv_frame.shape != (expected_rows, expected_columns):
        raise SystemExit(
            f"Expected results shape {(expected_rows, expected_columns)}, found {csv_frame.shape}"
        )
    leaked_columns = sorted(FORBIDDEN_RESULT_COLUMNS.intersection(csv_frame.columns))
    if leaked_columns:
        raise SystemExit(f"Forbidden result columns found: {', '.join(leaked_columns)}")
    actual_by_tag = csv_frame.groupby("tag", dropna=False).size().to_dict()
    if actual_by_tag != dataset_manifest["expected_rows_by_tag"]:
        raise SystemExit(
            f"Released row counts by tag differ: expected "
            f"{dataset_manifest['expected_rows_by_tag']}, got {actual_by_tag}"
        )
    actual_by_model = csv_frame.groupby("model", dropna=False).size().to_dict()
    if actual_by_model != dataset_manifest["expected_rows_by_model"]:
        raise SystemExit(
            f"Released row counts by model differ: expected "
            f"{dataset_manifest['expected_rows_by_model']}, got {actual_by_model}"
        )

    questions_root = output_dir / "questions"
    expected_models = list(dataset_manifest["expected_models"])
    actual_models = sorted(path.name for path in questions_root.iterdir() if path.is_dir())
    if actual_models != sorted(expected_models):
        raise SystemExit(
            f"Expected canonical question folders {sorted(expected_models)}, got {actual_models}"
        )
    expected_question_counts = dataset_manifest["expected_questions_by_model"]
    for model in expected_models:
        questions_path = questions_root / model / "questions.json"
        payload = json.loads(questions_path.read_text(encoding="utf-8"))
        actual_count = len(payload["questions"])
        expected_count = int(expected_question_counts[model])
        if actual_count != expected_count:
            raise SystemExit(
                f"Expected {expected_count} questions for {model}, found {actual_count}"
            )
        if payload.get("environment_name") != model:
            raise SystemExit(
                f"Expected environment_name {model!r} in {questions_path}, "
                f"got {payload.get('environment_name')!r}"
            )

    for relative_path in LEGACY_OR_INTERMEDIATE_PATHS:
        path = output_dir / relative_path
        if path.exists():
            raise SystemExit(f"Legacy or intermediate release artifact found: {path}")
    generated_files = list(questions_root.rglob("__pycache__")) + list(
        questions_root.rglob("*.pyc")
    )
    if generated_files:
        raise SystemExit(f"Generated Python artifacts found: {generated_files[0]}")
    _verify_no_private_paths(output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--revision", default=DATASET_REVISION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = download_release(args.output_dir, revision=args.revision)
    verify_pinned_release(output_dir, args.revision)
    print(f"Question bundle: {output_dir / 'questions'}")
    print(f"Released CSV results: {output_dir / 'results.csv'}")
    print(f"Released Parquet results: {output_dir / 'results.parquet'}")
    print(f"Dataset revision: {args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
