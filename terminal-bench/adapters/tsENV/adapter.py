"""Industrial time-series multiple-choice adapter for Terminal-Bench.

This adapter converts simulation runs that live under ``dataset_root`` into
Terminal-Bench task directories. Each source run is expected to contain a
``dataframe.parquet`` (pandas ``DataFrame``) alongside ``metadata_task.json`` with a
multiple-choice question and ground-truth answer.

Only answer-neutral context is made visible to the agent. The ground truth is
stored inside the test harness (``tests/ground_truth.json``).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import Any, ClassVar, Iterable, Optional, Sequence
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

from shared.benchmark_utils import (
    is_anomaly_run,
    is_classification_run,
)
from shared.prompts import render_tsenv_agent_prompt
from shared.tsenv_metadata import (
    ground_truth_by_path_from_payload,
    label_choices_from_payload,
    load_metadata_payload,
    metadata_questions_by_id,
    question_sample_paths,
    resolve_tsenv_payload_path,
)
from shared.tsenv_eval_mode import normalize_tsenv_eval_mode
from shared.tsenv_task_materialization import materialize

LOGGER = logging.getLogger(__name__)

ADAPTER_NAME = "INDUSTRIAL_TIME_SERIES"
TEMPLATE_DIR = Path(__file__).parent / "template"


def _tsenv_eval_mode(raw_mode: object) -> str:
    return normalize_tsenv_eval_mode(raw_mode)


def _tsenv_question_text_value(question_text: dict, field: str) -> str:
    aliases = {
        "description": ("description", "model_description"),
        "model_description": ("model_description", "description"),
    }
    for candidate in aliases.get(field, (field,)):
        value = str(question_text.get(candidate) or "").strip()
        if value:
            return value
    return ""


def _tsenv_recipe_value(recipe_info: dict, *keys: str, default: object = None) -> object:
    for key in keys:
        if key in recipe_info and recipe_info.get(key) is not None:
            return recipe_info.get(key)
    return default


def _tsenv_shot(train_samples_per_class: object) -> str:
    count = int(train_samples_per_class or 0)
    if count == 0:
        return "zero_shot"
    if count == 1:
        return "one_shot"
    if count == 3:
        return "few_shot"
    if count > 3:
        return "many_shots"
    raise ValueError(
        "Unsupported train_samples_per_class="
        f"{train_samples_per_class!r} in tsENV questions payload."
    )


def _tsenv_context(desc_level: object) -> str:
    return str(desc_level or "none").strip().lower() or "none"


def _tsenv_noise_resolved(recipe_info: dict) -> dict:
    noise_name = str(
        _tsenv_recipe_value(recipe_info, "noise_level", "noise", default="none") or "none"
    ).strip().lower()
    seed = int(_tsenv_recipe_value(recipe_info, "question_seed", "seed", default=0) or 0)
    train_count = int(
        _tsenv_recipe_value(
            recipe_info,
            "number_train_samples_per_class",
            "train_samples_per_class",
            "num_examples",
            default=0,
        )
        or 0
    )
    shot = _tsenv_shot(train_count)
    return {
        "context": _tsenv_context(
            _tsenv_recipe_value(recipe_info, "desc_level", "context", default="none")
        ),
        "shot": shot,
        "eval_mode": _tsenv_eval_mode(
            _tsenv_recipe_value(recipe_info, "type_of_request", "eval_mode", default="direct")
        ),
        "noise_profile": noise_name or "none",
        "noise_seed": seed,
        "train_count": train_count,
        "test_sample_count": 0,
    }


def _normalize_model_noise_profile(raw_profile: object) -> str:
    normalized = str(raw_profile or "").strip().lower()
    aliases = {
        "": "none",
        "noise_none": "none",
        "noise_low": "low",
        "noise_medium": "medium",
        "noise_high": "high",
    }
    return aliases.get(normalized, normalized)


def _join_tsenv_question_text(
    question_text: dict,
    *,
    question_slug: str | None = None,
    questions_by_id: dict | None = None,
    questions_metadata: dict | None = None,
) -> str:
    return render_tsenv_agent_prompt(
        question_text,
        question_slug=question_slug,
        questions_by_id=questions_by_id,
        questions_metadata=questions_metadata,
    )


def _resolve_adapter_payload_path(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)
    try:
        return resolve_tsenv_payload_path(dataset_root)
    except FileNotFoundError:
        legacy_path = dataset_root / "metadata.json"
        if legacy_path.exists():
            return legacy_path
        raise


def _question_allowed_labels(question: dict) -> list[str]:
    question_text = question.get("question_text")
    if not isinstance(question_text, dict):
        return []
    labels = question_text.get("allowed_labels")
    if not isinstance(labels, list):
        return []
    return [str(label).strip() for label in labels if str(label).strip()]


def _normalize_tsenv_question(
    *,
    question_id: str,
    question: dict,
    payload: dict,
    dataset_name: str,
    multiple_choices: Sequence[str] | None = None,
    ground_truth_by_path: dict[str, str] | None = None,
) -> dict:
    question_schema = {key: value for key, value in question.items() if key != "question_id"}
    if "benchmark" in question:
        normalized = dict(question)
        normalized.setdefault("question_schema", question_schema)
        return normalized
    recipe_info = question.get("recipe_info")
    question_text = question.get("question_text")
    if not isinstance(recipe_info, dict) or not isinstance(question_text, dict):
        return dict(question)
    resolved_choices = list(
        _question_allowed_labels(question)
        or multiple_choices
        or label_choices_from_payload(payload)
    )
    if not resolved_choices:
        raise ValueError(
            f"questions payload missing label_int_mapping for question_id={question_id!r}"
        )
    train_samples = question_sample_paths(payload, question=question, subset="train")
    test_samples = question_sample_paths(payload, question=question, subset="test")
    if not test_samples:
        raise ValueError(f"questions payload question {question_id!r} missing test_samples")
    eval_mode = _tsenv_eval_mode(
        _tsenv_recipe_value(recipe_info, "type_of_request", "eval_mode", default="direct")
    )
    shot = _tsenv_shot(
        _tsenv_recipe_value(
            recipe_info,
            "number_train_samples_per_class",
            "train_samples_per_class",
            "num_examples",
            default=0,
        )
    )
    if ground_truth_by_path is None:
        normalized_payload = dict(payload)
        normalized_payload["questions"] = {question_id: question}
        ground_truth_by_path = ground_truth_by_path_from_payload(normalized_payload)
    test_sample_labels = {
        sample: str(ground_truth_by_path.get(sample) or "").strip()
        for sample in test_samples
        if str(ground_truth_by_path.get(sample) or "").strip()
    }
    if len(test_sample_labels) != len(test_samples):
        missing = [sample for sample in test_samples if sample not in test_sample_labels]
        raise ValueError(
            f"ground_truth_by_path missing labels for question_id={question_id!r}: {missing!r}"
        )
    instruction = _join_tsenv_question_text(
        question_text,
        question_slug=question_id,
        questions_by_id=payload.get("questions") if isinstance(payload.get("questions"), dict) else None,
        questions_metadata=payload,
    )
    metadata = {
        "sample_id": question.get("sample_id"),
        "recipe_id": _tsenv_recipe_value(recipe_info, "recipe_id", "row_slug", default=question_id),
        "test_sample_labels": test_sample_labels,
    }
    question_schema = dict(question)
    question_schema.pop("question_id", None)
    context_level = _tsenv_context(
        _tsenv_recipe_value(recipe_info, "desc_level", "context", default="none")
    )
    return {
        "question_id": question_id,
        "question_schema": question_schema,
        "benchmark": "tsenv_cls",
        "context": context_level,
        "desc_level": context_level,
        "shot": shot,
        "dataset": dataset_name,
        "question_hash": str(question.get("question_hash") or question_id).strip(),
        "eval_mode": eval_mode,
        "instruction_agent_format": instruction,
        "instruction_human_format": "\n\n".join(
            part
            for part in (
                _tsenv_question_text_value(question_text, "model_description"),
                str(question_text.get("task_instruction") or "").strip(),
            )
            if part
        ).strip(),
        "question": str(question_text.get("task_instruction") or "").strip(),
        "multiple_choices": resolved_choices,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "test_sample_labels": test_sample_labels,
        "metadata": metadata,
        "recipe": {
            "name": "tsenv-default",
            "resolved": {
                **_tsenv_noise_resolved(recipe_info),
                "test_sample_count": len(test_samples),
            },
        },
    }


def is_context_run(question: dict) -> bool:
    context = str(question.get("desc_level") or question.get("context") or "").strip().lower()
    if context:
        return context != "none"
    variant = str(question.get("variant") or "").strip().lower()
    if not variant:
        return False
    return not variant.startswith("none")


def is_code_run(question: dict) -> bool:
    return normalize_tsenv_eval_mode(question.get("eval_mode")) == "code"


def is_tsenv_direct_run(question: dict) -> bool:
    if normalize_tsenv_eval_mode(question.get("eval_mode")) != "direct":
        return False
    return str(question.get("benchmark") or "").strip().lower() == "tsenv_cls"


def is_tsenv_open_ended_run(question: dict) -> bool:
    if normalize_tsenv_eval_mode(question.get("eval_mode")) != "open-ended":
        return False
    return str(question.get("benchmark") or "").strip().lower() == "tsenv_cls"


def is_tsenv_run(question: dict) -> bool:
    return str(question.get("benchmark") or "").strip().lower() == "tsenv_cls"


def is_few_shot_run(question: dict) -> bool:
    shot = str(question.get("shot") or "").strip().lower()
    return shot in {"one_shot", "few_shot", "many_shots", "many_shot"}


class RequireNameMeta(type):
    def __init__(cls, name, bases, namespace) -> None:  # type: ignore[override]
        super().__init__(name, bases, namespace)
        if name != "BaseAdapter" and not hasattr(cls, "NAME"):
            raise TypeError(f"Class {name} must define a class attribute NAME")


class BaseAdapter(metaclass=RequireNameMeta):
    NAME: ClassVar[str]

    def generate_task(self, task_id: str, local_task_id: str) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TimeSeriesInstance:
    """Container for a discovered dataset instance."""

    relative_id: str  # e.g. "20251102_163236/scenario_0001/rec_00003"
    source_dir: Path
    metadata_path: Path
    question: Optional[dict] = None


@dataclass(frozen=True)
class ModelNoiseSpec:
    profile_name: str
    noise_seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.profile_name not in {"", "none"}


class IndustrialTimeSeriesAdapter(BaseAdapter):
    NAME = ADAPTER_NAME

    def __init__(
        self,
        dataset_root: Path,
        task_dir: Path,
        *,
        include_question_ids: Iterable[str] | None = None,
        include_preview: bool = False,
        include_diagrams: bool = False,
        csv_sample_rows: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.task_dir = Path(task_dir)
        self.include_preview = include_preview
        self.include_diagrams = include_diagrams
        self.csv_sample_rows = csv_sample_rows if csv_sample_rows and csv_sample_rows > 0 else None
        self.model_name = self.dataset_root.name
        self.include_question_ids = {
            str(item).strip("/")
            for item in (include_question_ids or [])
            if str(item).strip("/")
        }

        # Exam-question export format
        self.exam_questions: list[dict] = []
        self.exam_index: dict[str, dict] = {}
        self.metadata_payload: dict = {}
        self.exam_payload: dict[str, Any] = {}
        self._current_benchmark_name: Optional[str] = None

        payload_path = _resolve_adapter_payload_path(self.dataset_root)
        self._load_exam_assets(payload_path)

    def _load_exam_assets(self, payload_path: Path) -> None:
        payload = load_metadata_payload(payload_path)
        self.metadata_payload = payload
        questions_by_id = metadata_questions_by_id(payload)
        if self.include_question_ids:
            questions_by_id = {
                question_id: question
                for question_id, question in questions_by_id.items()
                if question_id in self.include_question_ids
            }
        scoped_payload = dict(payload)
        scoped_payload["questions"] = questions_by_id
        self.exam_payload = scoped_payload
        multiple_choices = label_choices_from_payload(scoped_payload)
        ground_truth_by_path = (
            ground_truth_by_path_from_payload(scoped_payload)
            if questions_by_id
            else {}
        )
        questions = [
            _normalize_tsenv_question(
                question_id=question_id,
                question=question,
                payload=scoped_payload,
                dataset_name=self.model_name,
                multiple_choices=multiple_choices,
                ground_truth_by_path=ground_truth_by_path,
            )
            for question_id, question in questions_by_id.items()
        ]
        self.exam_questions = questions
        self.exam_index = {q["question_id"]: q for q in self.exam_questions if "question_id" in q}
    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def discover(self, include_failed: bool = False) -> Iterable[TimeSeriesInstance]:
        payload_path = _resolve_adapter_payload_path(self.dataset_root)
        for question in self.exam_questions:
            qid = question.get("question_id")
            if not qid:
                continue
            yield TimeSeriesInstance(
                relative_id=str(qid),
                source_dir=self.dataset_root,
                metadata_path=payload_path,
                question=question,
            )

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------
    def generate_task(self, task_id: str, folder_name: str) -> None:
        question = self.exam_index.get(task_id)

        self._generate_exam_task(question, folder_name)


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_directory(self, output_dir: Path) -> None:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.copytree(TEMPLATE_DIR, output_dir)

    def _maybe_export_csv(self, data_path: Path, output_dir: Path) -> None:
        if not self.csv_sample_rows:
            return
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.warning("Skipping CSV export because pandas is unavailable: %s", exc)
            return

        try:
            df = self._load_dataframe(data_path)
        except Exception as exc:  # pragma: no cover - convert failure is non-critical
            LOGGER.warning("Failed to read %s for CSV export: %s", data_path, exc)
            return

        preview = df.head(self.csv_sample_rows)
        preview_path = output_dir / "data_preview.csv"
        preview.to_csv(preview_path, index=False)

    def _write_instruction(
        self,
        metadata: dict,
        output_dir: Path,
        *,
        extra_context: Optional[Sequence[str]] = None,
    ) -> None:
        task_yaml = output_dir / "task.yaml"
        template = task_yaml.read_text(encoding="utf-8")

        scenario_info_path = output_dir / "scenario_info.json"
        scenario_meta: dict = {}
        if scenario_info_path.exists():
            try:
                scenario_meta = json.loads(scenario_info_path.read_text(encoding="utf-8"))
            except Exception:
                scenario_meta = {}

        metadata = {**scenario_meta, **metadata}
        instruction_text = metadata["instruction_agent_format"]
        if not instruction_text:
            raise ValueError("Missing instruction_agent_format for task instruction")

        instruction_block = indent(instruction_text, "  ")
        task_yaml.write_text(
            template.replace("{instruction_block}", instruction_block),
            encoding="utf-8",
        )

    def _patch_tests_scripts(self, output_dir: Path, question) -> None:
        tests_dir = output_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        source_name = self._select_test_template(question)
        if not source_name:
            raise ValueError(
                f"Unsupported benchmark/task type for tests in model {self.model_name}."
            )

        source_path = tests_dir / source_name
        target_path = tests_dir / "test.py"

        if not source_path.exists():
            raise FileNotFoundError(f"Expected test template missing: {source_path}")

        # Remove any other test_* files so only the relevant test is copied into the container.
        for test_file in tests_dir.glob("test_*.py"):
            if test_file != source_path:
                test_file.unlink()

        if target_path.exists():
            target_path.unlink()
        source_path.rename(target_path)

    @staticmethod
    def _refresh_agent_payload(output_dir: Path) -> None:
        payload_dir = output_dir / "agent_payload"
        shutil.rmtree(payload_dir, ignore_errors=True)
        payload_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train_samples", "test_samples", "train_labels.json"):
            source = output_dir / name
            if not source.exists():
                continue
            destination = payload_dir / name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    @staticmethod
    def _select_test_template(question) -> Optional[str]:
        if is_code_run(question) and is_classification_run(question):
            return "test_code_classification.py"
        if is_classification_run(question):
            if is_tsenv_direct_run(question):
                return "test_classification.py"
            if is_tsenv_open_ended_run(question):
                return "test_classification_open_ended.py"
            benchmark = str(question.get("benchmark") or "").strip().lower()
            if benchmark in {"simbench_cls", "tsenv_cls"}:
                return "test_classification_simbench.py"
            return "test_classification_ucr.py"
        if is_anomaly_run(question):
            return "test_anomaly.py"
        return None
     
    # ------------------------------------------------------------------
    # Exam-question export support (flattened exam_questions/<model>/)
    # ------------------------------------------------------------------
    def _generate_exam_task(self, question: dict, folder_name: str) -> None:
        if question is None:
            raise ValueError("Unknown question id for task generation.")
        output_dir = self.task_dir / folder_name
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Preparing question %s -> %s", question["question_id"], output_dir)
        self._prepare_directory(output_dir)

        question = dict(question)
        question_schema = question.get("question_schema")
        if not isinstance(question_schema, dict):
            question_schema = {key: value for key, value in question.items() if key != "question_id"}
        benchmark = question.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("Question missing benchmark.")
        variant = question.get("variant")
        if not isinstance(variant, str) or not variant:
            context = str(question.get("desc_level") or question.get("context") or "").strip().lower()
            shot = str(question.get("shot") or "").strip().lower()
            if context and shot:
                variant = f"{context}_{shot}"
            elif context:
                variant = context
            elif shot:
                variant = shot
            else:
                variant = "unknown"
        benchmark_id = f"{benchmark}_{variant}"
        self._current_benchmark_name = benchmark_id
        code_mode = is_code_run(question)
        tsenv_mode = is_tsenv_run(question)
        multisample_mode = code_mode or tsenv_mode
        source_train_samples = [
            str(sample).strip()
            for sample in (question.get("train_samples") or [])
            if str(sample).strip()
        ]

        if not code_mode and not tsenv_mode:
            dataframe_path = question.get("dataframe_path")
            if not dataframe_path:
                test_samples = question.get("test_samples")
                if isinstance(test_samples, list) and test_samples:
                    dataframe_path = test_samples[0]
            if not dataframe_path:
                raise FileNotFoundError("Question missing dataframe_path/test_samples.")
            data_path = self._resolve_data_path(f"{dataframe_path}")
            self._write_anonymized_parquet(
                data_path,
                output_dir / "dataframe.parquet",
                    question=question,
                    source_ref=str(dataframe_path),
                )

        rewritten_train_samples: list[str] = []
        if not multisample_mode and is_few_shot_run(question):
            few_shots = question.get("train_samples") or []
            for idx, few_shot in enumerate(few_shots):
                few_shot_path = self._resolve_data_path(few_shot)
                target_name = Path(str(few_shot)).name or f"train_{idx:04d}.parquet"
                self._write_anonymized_parquet(
                    few_shot_path,
                    output_dir / target_name,
                    question=question,
                    source_ref=str(few_shot),
                )
                rewritten_train_samples.append(target_name)

        def _copy_sample_group(*, source_paths: object, output_subdir: str) -> list[str]:
            if not isinstance(source_paths, list):
                return []
            output_dir_path = output_dir / output_subdir
            output_dir_path.mkdir(parents=True, exist_ok=True)
            basename_to_sources: dict[str, list[str]] = {}
            normalized_sources: list[tuple[str, Path]] = []
            for raw_path in source_paths:
                source_ref = str(raw_path).strip()
                if not source_ref:
                    raise ValueError(f"{output_subdir} contains an empty source path.")
                basename = Path(source_ref).name
                if not basename:
                    raise ValueError(
                        f"{output_subdir} source path must have a basename: {source_ref!r}"
                    )
                basename_to_sources.setdefault(basename, []).append(source_ref)
                normalized_sources.append((source_ref, self._resolve_data_path(raw_path)))
            duplicates = {
                basename: refs
                for basename, refs in basename_to_sources.items()
                if len(refs) > 1
            }
            if duplicates:
                rendered = ", ".join(
                    f"{name}: {refs!r}" for name, refs in sorted(duplicates.items())
                )
                raise ValueError(
                    f"Duplicate output filenames under {output_subdir}: {rendered}"
                )
            rewritten: list[str] = []
            for source_ref, source in normalized_sources:
                target = output_dir_path / Path(source_ref).name
                self._write_anonymized_parquet(
                    source,
                    target,
                    question=question,
                    source_ref=source_ref,
                )
                rewritten.append(str(Path(output_subdir) / target.name))
            return rewritten

        if code_mode:
            source_test_samples = question.get("test_samples")
            if isinstance(source_test_samples, list):
                source_test_samples = [str(sample) for sample in source_test_samples]
            else:
                source_test_samples = []
            rewritten_test = _copy_sample_group(
                source_paths=question.get("test_samples"),
                output_subdir="test_samples",
            )
            rewritten_train = _copy_sample_group(
                source_paths=question.get("train_samples"),
                output_subdir="train_samples",
            )
            question["test_samples"] = rewritten_test
            question["train_samples"] = rewritten_train
            question["test_samples_source_paths"] = source_test_samples
        elif tsenv_mode:
            source_test_samples = question.get("test_samples")
            if isinstance(source_test_samples, list):
                source_test_samples = [str(sample) for sample in source_test_samples]
            else:
                source_test_samples = []
            question["test_samples"] = _copy_sample_group(
                source_paths=question.get("test_samples"),
                output_subdir="test_samples",
            )
            question["train_samples"] = _copy_sample_group(
                source_paths=question.get("train_samples"),
                output_subdir="train_samples",
            )
            question["test_samples_source_paths"] = source_test_samples
        else:
            question["test_samples"] = ["dataframe.parquet"]
            question["train_samples"] = rewritten_train_samples
            question["test_samples_source_paths"] = []

        if not code_mode and not tsenv_mode:
            self._maybe_export_csv(output_dir / "dataframe.parquet", output_dir)
        # Patch question to make it anonymous
        question.pop("question_id", None)
        question.pop("question_index", None)
        question.pop("scenario_dir", None)
        question.pop("dataframe_path", None)
        question.pop("dataset", None)
        self._write_exam_metadata(
            question,
            output_dir,
            question_schema=question_schema,
        )
        instruction_meta = dict(question)
        instruction_meta["model"] = None
        question_text = (
            question.get("instruction_human_format")
            or question.get("question")
            or question.get("instruction_agent_format")
        )
        if isinstance(question_text, str) and question_text.strip():
            instruction_meta.setdefault("question", question_text)
        self._write_instruction(instruction_meta, output_dir) #, extra_context=context_lines)
        self._patch_tests_scripts(output_dir, question)
        self._write_train_labels(
            source_train_samples=source_train_samples,
            materialized_train_samples=question.get("train_samples"),
            output_dir=output_dir,
        )
        self._refresh_agent_payload(output_dir)
        self._current_benchmark_name = None

    def _resolve_data_path(self, raw_path: object) -> Path:
        candidate = Path(str(raw_path)) if raw_path is not None else None
        if candidate is None:
            raise FileNotFoundError("No test_samples entry provided for this question")
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (self.dataset_root / candidate).resolve()
        if candidate.exists():
            return candidate
        alt = self.dataset_root / candidate.name
        if alt.exists():
            return alt
        raise FileNotFoundError(f"Dataframe not found at {candidate}")

    @staticmethod
    def _load_dataframe(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def _resolve_recipe_payload(question: object) -> dict:
        if not isinstance(question, dict):
            return {}
        recipe = question.get("recipe")
        if isinstance(recipe, dict):
            return dict(recipe)
        recipe_info = question.get("recipe_info")
        if isinstance(recipe_info, dict):
            return {
                "name": "tsenv-default",
                "resolved": _tsenv_noise_resolved(recipe_info),
            }
        meta = question.get("metadata")
        if isinstance(meta, dict):
            candidate = meta.get("recipe")
            if isinstance(candidate, dict):
                return dict(candidate)
        return {}

    @classmethod
    def _resolve_noise_spec(cls, question: object) -> ModelNoiseSpec:
        recipe = cls._resolve_recipe_payload(question)
        resolved = recipe.get("resolved") if isinstance(recipe, dict) else None
        if not isinstance(resolved, dict):
            return ModelNoiseSpec(profile_name="none", noise_seed=0)
        try:
            seed = int(resolved.get("noise_seed", 0))
        except (TypeError, ValueError):
            seed = 0
        profile_name = _normalize_model_noise_profile(
            resolved.get("noise_profile") or resolved.get("noise")
        )
        if profile_name == "medium":
            raise ValueError("Unsupported noise profile 'medium' for task materialization.")
        if profile_name not in {"", "none", "low", "high"}:
            raise ValueError(f"Unsupported noise profile '{profile_name}'.")
        return ModelNoiseSpec(profile_name=profile_name or "none", noise_seed=seed)

    def _first_diff_for_sample(self, sample_uuid: str) -> float | None:
        ground_truth = self.exam_payload.get("ground_truth_information")
        if not isinstance(ground_truth, dict):
            return None
        interventions = ground_truth.get("interventions")
        if not isinstance(interventions, dict):
            return None
        entry = interventions.get(str(sample_uuid))
        if not isinstance(entry, dict):
            return None
        value = entry.get("first_diff")
        if value is None:
            return None
        if isinstance(value, list):
            parsed_values: list[float] = []
            for item in value:
                if item is None:
                    continue
                try:
                    parsed = float(item)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0.0:
                    parsed_values.append(parsed)
            return min(parsed_values) if parsed_values else None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0.0 else None

    def _write_anonymized_parquet(
        self,
        source_path: Path,
        dest_path: Path,
        *,
        question: Optional[dict] = None,
        source_ref: Optional[str] = None,
    ) -> None:
        spec = self._resolve_noise_spec(question)
        sample_uuid = Path(str(source_ref or source_path)).stem
        baseline_ref = self._baseline_ref_for_sample(sample_uuid)
        df, _noise_analysis = materialize(
            source_path.stem,
            spec.profile_name,
            spec.noise_seed,
            tsenv_model_root=self.dataset_root,
            uuid_baseline_path=baseline_ref,
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest_path, index=False)

    def _baseline_ref_for_sample(self, sample_uuid: str) -> Optional[str]:
        manifest_path = self.dataset_root / "sample_manifest.json"
        if not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(manifest, dict):
            return None
        sample_id = str(sample_uuid).strip()
        for raw_entries in manifest.values():
            if not isinstance(raw_entries, list):
                continue
            for raw_entry in raw_entries:
                if not isinstance(raw_entry, dict):
                    continue
                for samples_key, baselines_key in (
                    ("train_samples", "train_samples_baselines"),
                    ("test_samples", "test_samples_baselines"),
                ):
                    samples = raw_entry.get(samples_key)
                    baselines = raw_entry.get(baselines_key)
                    if not isinstance(samples, list) or not isinstance(baselines, list):
                        continue
                    for idx, sample in enumerate(samples):
                        if str(sample).strip() != sample_id or idx >= len(baselines):
                            continue
                        baseline_id = str(baselines[idx] or "").strip()
                        if not baseline_id or baseline_id == sample_id:
                            return None
                        return f"dataframes/{baseline_id}.parquet"
        return None
        

    def _first_diff_for_sample(self, sample_uuid: str) -> Optional[float]:
        ground_truth = self.metadata_payload.get("ground_truth_information")
        if not isinstance(ground_truth, dict):
            return None
        interventions = ground_truth.get("interventions")
        if not isinstance(interventions, dict):
            return None
        entry = interventions.get(str(sample_uuid).strip())
        if not isinstance(entry, dict):
            return None
        value = entry.get("first_diff")
        if value is None:
            return None
        if isinstance(value, list):
            candidates: list[float] = []
            for item in value:
                if item is None:
                    continue
                try:
                    parsed = float(item)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(parsed) and parsed >= 0.0:
                    candidates.append(parsed)
            return min(candidates) if candidates else None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) and parsed >= 0.0 else None


    def _write_pickle_copy(self, source: Path, dest: Path) -> None:
        df = self._load_dataframe(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(dest)

    def _write_train_labels(
        self,
        *,
        source_train_samples: Sequence[str],
        materialized_train_samples: object,
        output_dir: Path,
    ) -> None:
        if not source_train_samples:
            return
        if not isinstance(materialized_train_samples, list):
            raise ValueError("materialized train_samples must be a list to export train_labels.json")
        ground_truth_by_path = ground_truth_by_path_from_payload(self.exam_payload)
        basename_to_label: dict[str, str] = {}
        for source_ref in source_train_samples:
            basename = Path(source_ref).name
            label = str(ground_truth_by_path.get(source_ref) or "").strip()
            if not label:
                raise ValueError(
                    f"Missing ground-truth label for training sample {source_ref!r}"
                )
            if basename in basename_to_label:
                raise ValueError(
                    f"Duplicate training sample basename {basename!r} prevents label export."
                )
            basename_to_label[basename] = label

        train_labels: dict[str, str] = {}
        for raw_sample in materialized_train_samples:
            rel_path = Path(str(raw_sample).strip())
            if rel_path.is_absolute() or any(part == ".." for part in rel_path.parts):
                raise ValueError(f"Invalid materialized training sample path {raw_sample!r}.")
            label = basename_to_label.get(rel_path.name)
            if not label:
                raise ValueError(
                    f"Could not map materialized training sample {rel_path.name!r} back to a label."
                )
            sample_key = rel_path.name
            if sample_key in train_labels:
                raise ValueError(
                    f"Duplicate materialized training sample filename {sample_key!r}."
                )
            train_labels[sample_key] = label

        if train_labels:
            (output_dir / "train_labels.json").write_text(
                json.dumps(train_labels, indent=2, sort_keys=True),
                encoding="utf-8",
            )


    def _write_exam_metadata(
        self,
        question: dict,
        output_dir: Path,
        *,
        question_schema: dict | None = None,
    ) -> None:
        eval_mode = normalize_tsenv_eval_mode(question.get("eval_mode"))
        payload = {
            "instruction_agent_format": question.get("instruction_agent_format"),
            "multiple_choices": question.get("multiple_choices", []),
            "run_type": question.get("run_type"),
            "num_samples": question.get("num_samples"),
            "question_id": question.get("question_id") or question.get("question_hash"),
            "question_schema": question_schema if question_schema is not None else question.get("question_schema"),
            "hash": question.get("question_hash"),
            "question_hash": question.get("question_hash"),
            "desc_level": question.get("desc_level") or question.get("context"),
            "shot": question.get("shot"),
            "eval_mode": eval_mode,
            "test_samples": question.get("test_samples"),
            "train_samples": question.get("train_samples"),
            "test_samples_source_paths": question.get("test_samples_source_paths", []),
        }
        if question.get("question_schema") is not None:
            payload["question_schema"] = question.get("question_schema")
        recipe_payload = self._resolve_recipe_payload(question)
        noise_spec = self._resolve_noise_spec(question)
        payload["noise_level"] = noise_spec.profile_name or "none"
        if recipe_payload:
            recipe_out = dict(recipe_payload)
            resolved = recipe_out.get("resolved")
            if isinstance(resolved, dict):
                resolved_out = dict(resolved)
                resolved_out["noise_profile"] = noise_spec.profile_name
                resolved_out["noise_seed"] = int(noise_spec.noise_seed)
                resolved_out.pop("noise_local", None)
                resolved_out.pop("noise_global", None)
                resolved_out.pop("noise_abs", None)
                recipe_out["resolved"] = resolved_out
            payload["recipe"] = recipe_out
        labels = question.get("test_sample_labels")
        if not isinstance(labels, dict):
            meta = question.get("metadata")
            if isinstance(meta, dict):
                labels = meta.get("test_sample_labels")
        if isinstance(labels, dict):
            payload["test_sample_labels"] = labels
        if eval_mode != "code" and not is_tsenv_run(question):
            payload["data_file"] = "dataframe.parquet"

        scenario_info_path = output_dir / "scenario_info.json"
        scenario_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
