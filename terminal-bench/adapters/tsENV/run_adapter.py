#!/usr/bin/env python3
"""CLI entry point to generate Terminal-Bench tasks from time-series runs."""

from __future__ import annotations

import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List
import click

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from adapter import IndustrialTimeSeriesAdapter
from shared.run_artifacts import QUESTIONS_FILENAME

T_BENCH_ROOT = SCRIPT_DIR.parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("sim-bench-adapter")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip("/"))
    slug = slug.strip("-").lower()
    return slug or "timeseries-task"


def filter_instances(instances: Iterable, include: List[str] | None, contains: str | None) -> list:
    items = list(instances)
    if include:
        targets = {entry.strip("/") for entry in include}
        items = [inst for inst in items if inst.relative_id in targets]
    if contains:
        items = [inst for inst in items if contains in inst.relative_id]
    return items


DEFAULT_SRC_ROOT = T_BENCH_ROOT.parent / "exam_questions_complete"


def _is_model_root(root: Path) -> bool:
    return (
        (root / QUESTIONS_FILENAME).exists()
        or (root / "metadata.json").exists()
        or (root / "questions_summary.csv").exists()
    )


def _iter_model_roots(root: Path) -> list[Path]:
    """Return list of model roots. If root is an exam_questions dir, return its children."""
    if _is_model_root(root):
        return [root]

    children: list[Path] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if _is_model_root(child):
                children.append(child)
    return children or [root]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("dataset_root", type=click.Path(path_type=Path), required=False)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    show_default="terminal-bench/tasks/<dataset-name>",
    help="Where to place generated tasks.",
)
@click.option(
    "--skip-if-existing/--no-skip-if-existing",
    default=True,
    show_default=True,
    help="Skip conversion for a model if its output directory already exists.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Delete the output task folder before generating tasks.",
)
@click.option(
    "--prefix-model/--no-prefix-model",
    default=True,
    show_default=True,
    help="Reserved flag (currently no effect).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of questions to convert per model root.",
)
@click.option(
    "--only-question-id",
    "only_question_ids",
    multiple=True,
    help="Convert only the provided question_id values.",
)
@click.option(
    "--contains",
    type=str,
    default=None,
    help="Convert only question IDs containing this substring.",
)

def main(
    dataset_root: Path | None,
    output_dir: Path,
    skip_if_existing: bool,
    overwrite: bool,
    prefix_model: bool,
    limit: int | None,
    only_question_ids: tuple[str, ...],
    contains: str | None,
) -> None:
    """Generate Terminal-Bench tasks from the time-series simulation"""
    default_root = DEFAULT_SRC_ROOT.resolve()
    dataset_root = (dataset_root or default_root).resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    dataset_slug = slugify(dataset_root.name)
    if output_dir is None:
        output_dir = T_BENCH_ROOT / "tasks" / dataset_slug
    output_dir = output_dir.expanduser().resolve()
    if overwrite and output_dir.exists():
        LOGGER.info("Overwriting output at %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = prefix_model  # compatibility flag retained for CLI stability

    model_roots = _iter_model_roots(dataset_root)
    single_root = len(model_roots) == 1 and model_roots[0].resolve() == dataset_root
    LOGGER.info("Found %d model roots to process", len(model_roots))
    for model_root in model_roots:
        if single_root:
            model_output_dir = output_dir
        else:
            model_output_dir = output_dir / model_root.name
        if skip_if_existing and model_output_dir.exists() and any(model_output_dir.iterdir()):
            LOGGER.info("Skipping %s because output already exists at %s", model_root.name, model_output_dir)
            continue
        if not skip_if_existing and model_output_dir.exists():
            LOGGER.info("Removing existing output at %s", model_output_dir)
            shutil.rmtree(model_output_dir)

        model_output_dir.mkdir(parents=True, exist_ok=True)
        adapter = IndustrialTimeSeriesAdapter(
            dataset_root=model_root,
            task_dir=model_output_dir,
            include_question_ids=list(only_question_ids) if only_question_ids else None,
        )

        instances = filter_instances(
            adapter.discover(),
            include=list(only_question_ids) if only_question_ids else None,
            contains=contains,
        )
        instances = sorted(instances, key=lambda inst: inst.relative_id)
        if limit is not None and limit >= 0:
            instances = instances[:limit]


        LOGGER.info("[%s] Found %d candidate tasks", adapter.model_name, len(instances))

        for idx, inst in enumerate(instances):
            folder_name = f"question_{idx}"
            adapter.generate_task(inst.relative_id, folder_name)

    LOGGER.info("All tasks written to %s", output_dir)


if __name__ == "__main__":
    main()
