#!/usr/bin/env python3
"""Recompute released TraceBench aggregates from the pinned results table."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reproduction" / "manifest.json"
DEFAULT_RESULTS_PATH = ROOT / "data" / "results.csv"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "reproduced"
GROUP_COLUMNS = ("row_slug", "agent_id", "model")
METRICS = (
    "test_top1_accuracy",
    "test_shortlist_score",
    "cost_usd",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_steps",
    "python_calls",
)
HELD_OUT_MAIN_SLUGS = {
    "low": "gentle_01234-flame",
    "high": "gentle_01234-orbit",
}
EXPECTED_PAPER_AGENTS = (
    "gpt_5_5_codex_high",
    "gemini_3_1_pro_high",
    "claude_4_opus_high",
    "minimax-m2.7",
)
HELD_OUT_REQUIRED_COLUMNS = {
    "type_of_request",
    "noise_level",
    "other_top1_accuracy",
}
REQUIRED_COLUMNS = {
    "tag",
    "status",
    "agentic_run_id",
    "question_slug",
    *GROUP_COLUMNS,
    *METRICS,
}


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


def load_results(path: Path, *, verify_pinned: bool = True) -> tuple[pd.DataFrame, str]:
    source = path.expanduser().resolve()
    actual_hash = sha256_file(source)
    manifest = load_manifest()
    expected_hash = str(manifest["dataset"]["results_csv_sha256"])
    if verify_pinned and actual_hash != expected_hash:
        raise ValueError(
            f"results.csv checksum mismatch: expected {expected_hash}, got {actual_hash}"
        )
    frame = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"results.csv is missing required columns: {', '.join(missing)}")
    if verify_pinned:
        expected_rows = int(manifest["dataset"]["expected_rows"])
        if len(frame) != expected_rows:
            raise ValueError(f"Expected {expected_rows} released rows, found {len(frame)}")
        expected_by_tag = manifest["dataset"]["expected_rows_by_tag"]
        actual_by_tag = frame.groupby("tag", dropna=False).size().to_dict()
        if actual_by_tag != expected_by_tag:
            raise ValueError(
                f"Released row counts by tag differ: expected {expected_by_tag}, got {actual_by_tag}"
            )
    return frame, actual_hash


def summarize_main_held_out(frame: pd.DataFrame, *, tag: str) -> pd.DataFrame:
    """Reproduce the eight held-out accuracy cells in the main paper table."""

    missing = sorted(HELD_OUT_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"results.csv is missing held-out columns: {', '.join(missing)}"
        )
    selected = frame.loc[
        frame["tag"].eq(tag)
        & frame["status"].eq("DONE")
        & frame["type_of_request"].eq("code")
    ].copy()
    selected["other_top1_accuracy"] = pd.to_numeric(
        selected["other_top1_accuracy"], errors="raise"
    )
    rows: list[dict[str, object]] = []
    expected_identities = {
        (model, seed)
        for model in ("BallDrop", "BounceBall", "MassSlide")
        for seed in range(5)
    }
    for noise_level, row_slug in HELD_OUT_MAIN_SLUGS.items():
        condition = selected.loc[
            selected["noise_level"].eq(noise_level)
            & selected["row_slug"].eq(row_slug)
        ]
        actual_agents = set(condition["agent_id"])
        if actual_agents != set(EXPECTED_PAPER_AGENTS):
            raise ValueError(
                f"Held-out {noise_level} agents differ: expected "
                f"{sorted(EXPECTED_PAPER_AGENTS)}, got {sorted(actual_agents)}"
            )
        for agent_id in EXPECTED_PAPER_AGENTS:
            agent_rows = condition.loc[condition["agent_id"].eq(agent_id)].copy()
            identities = set(
                zip(
                    agent_rows["model"],
                    pd.to_numeric(agent_rows["question_seed"], errors="raise").astype(int),
                    strict=True,
                )
            )
            if identities != expected_identities or len(agent_rows) != 15:
                raise ValueError(
                    f"Expected three simulators and seeds 0--4 for "
                    f"{noise_level}/{agent_id}, found {len(agent_rows)} rows"
                )
            values_by_simulator = {
                model: agent_rows.loc[
                    agent_rows["model"].eq(model), "other_top1_accuracy"
                ].tolist()
                for model in ("BallDrop", "BounceBall", "MassSlide")
            }
            values = [
                value
                for model_values in values_by_simulator.values()
                for value in model_values
            ]
            residuals = [
                value - statistics.fmean(model_values)
                for model_values in values_by_simulator.values()
                for value in model_values
            ]
            rows.append(
                {
                    "noise_level": noise_level,
                    "agent_id": agent_id,
                    "run_count": len(agent_rows),
                    "other_top1_accuracy_mean": statistics.fmean(values),
                    "other_top1_accuracy_within_simulator_residual_std": (
                        statistics.stdev(residuals)
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_results(frame: pd.DataFrame, *, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = frame.loc[frame["tag"].eq(tag) & frame["status"].eq("DONE")].copy()
    if selected.empty:
        raise ValueError(f"No DONE rows found for tag {tag!r}")
    for metric in METRICS:
        selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    aggregations: dict[str, tuple[str, str]] = {
        "run_count": ("agentic_run_id", "nunique"),
        "question_count": ("question_slug", "nunique"),
    }
    for metric in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    by_condition = (
        selected.groupby(list(GROUP_COLUMNS), dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values(list(GROUP_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    by_agent = (
        selected.groupby("agent_id", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("agent_id", kind="stable")
        .reset_index(drop=True)
    )
    if tag == "PAPER" and HELD_OUT_REQUIRED_COLUMNS.issubset(selected.columns):
        held_out = summarize_main_held_out(selected, tag=tag)
        for noise_level in HELD_OUT_MAIN_SLUGS:
            noise_rows = held_out.loc[held_out["noise_level"].eq(noise_level)].set_index(
                "agent_id"
            )
            by_agent[f"held_out_{noise_level}_top1_accuracy_mean"] = by_agent[
                "agent_id"
            ].map(noise_rows["other_top1_accuracy_mean"])
            by_agent[
                f"held_out_{noise_level}_top1_accuracy_within_simulator_residual_std"
            ] = by_agent["agent_id"].map(
                noise_rows[
                    "other_top1_accuracy_within_simulator_residual_std"
                ]
            )
    return by_condition, by_agent


def write_accuracy_plot(by_agent: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(by_agent["agent_id"], by_agent["test_top1_accuracy_mean"])
    axis.set_ylabel("Mean test top-1 accuracy")
    axis.set_xlabel("Agent profile")
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def reproduce(
    results_csv: Path,
    output_dir: Path,
    *,
    tag: str = "PAPER",
    verify_pinned: bool = True,
) -> dict[str, Path]:
    frame, source_hash = load_results(results_csv, verify_pinned=verify_pinned)
    by_condition, by_agent = summarize_results(frame, tag=tag)
    held_out = (
        summarize_main_held_out(frame, tag=tag)
        if tag == "PAPER" and HELD_OUT_REQUIRED_COLUMNS.issubset(frame.columns)
        else pd.DataFrame()
    )
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    condition_path = destination / "summary_by_condition.csv"
    agent_path = destination / "summary_by_agent.csv"
    plot_path = destination / "accuracy_by_agent.png"
    provenance_path = destination / "reproduction.json"
    by_condition.to_csv(condition_path, index=False)
    by_agent.to_csv(agent_path, index=False)
    write_accuracy_plot(by_agent, plot_path)
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_results_csv": str(results_csv.expanduser().resolve()),
                "source_sha256": source_hash,
                "tag": tag,
                "selected_rows": int(by_agent["run_count"].sum()),
                "condition_rows": len(by_condition),
                "agent_rows": len(by_agent),
                "held_out_main_rows": (
                    int(held_out["run_count"].sum()) if not held_out.empty else 0
                ),
                "held_out_main_cells": (
                    json.loads(held_out.to_json(orient="records"))
                    if not held_out.empty
                    else []
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary_by_condition": condition_path,
        "summary_by_agent": agent_path,
        "accuracy_plot": plot_path,
        "provenance": provenance_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="PAPER")
    parser.add_argument(
        "--allow-unpinned-input",
        action="store_true",
        help="Skip the release checksum and row-count checks for exploratory inputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = reproduce(
        args.results_csv,
        args.output_dir,
        tag=args.tag,
        verify_pinned=not args.allow_unpinned_input,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
