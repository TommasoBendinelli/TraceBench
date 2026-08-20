from __future__ import annotations

import concurrent.futures
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shared.benchmark_utils import ALLOWED_TSENV_MODELS  # noqa: E402
from shared.interface.distribution_json import (  # noqa: E402
    ValidationError as DistributionValidationError,
    load_experiment_config_json,
)
from shared.interface.model_record_json import (  # noqa: E402
    load_model_record_json,
    load_model_run_specs_json,
)
from shared.model_run_specs_runtime import build_model_record_registry  # noqa: E402
from shared.question_eligibility import is_success_status  # noqa: E402
from shared.run_artifacts import (  # noqa: E402
    resolve_model_record_path,
    resolve_runs_root,
    resolve_similarity_metrics_path,
)
from shared.time_series_metrics import _compute_detectability_values, load_run_df  # noqa: E402
from workflows.metrics import compute_metrics as cm  # noqa: E402

_EPS = 1.0e-12
_DEFAULT_EXTRA_BUDGET_FRACTION = 0.05
_SIDE_NAMES = ("vs_baseline", "vs_time0_baseline")


def _resolve_models_root() -> Path:
    cwd_models_root = Path(os.getcwd()).resolve() / "models" / "simulink"
    if cwd_models_root.exists():
        return cwd_models_root
    return root_dir / "models" / "simulink"


def _finite_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _signal_columns(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns if str(column) != "time"]


def _pre_intervention_abs_max(
    df: pd.DataFrame,
    *,
    signal: str,
    intervention_time: object,
) -> Optional[float]:
    if signal not in df.columns or "time" not in df.columns:
        return None
    intervention = _finite_float(intervention_time)
    if intervention is None:
        intervention = 0.0
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[signal], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(values) & (times < float(intervention))
    if not np.any(finite):
        return None
    scale = float(np.max(np.abs(values[finite])))
    return scale if math.isfinite(scale) and scale > _EPS else None


def _scale_for_signal(
    *,
    run_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    signal: str,
    intervention_time: object,
) -> float:
    scale = _pre_intervention_abs_max(
        run_df,
        signal=signal,
        intervention_time=intervention_time,
    )
    if scale is None:
        scale = _pre_intervention_abs_max(
            reference_df,
            signal=signal,
            intervention_time=intervention_time,
        )
    return float(scale) if scale is not None else 1.0


def _validate_aligned_frames(reference_df: pd.DataFrame, run_df: pd.DataFrame) -> list[str]:
    if "time" not in reference_df.columns or "time" not in run_df.columns:
        raise ValueError("reference_df and run_df must contain a time column")
    if [str(column) for column in reference_df.columns] != [
        str(column) for column in run_df.columns
    ]:
        raise ValueError("reference_df and run_df columns must match")
    reference_time = pd.to_numeric(reference_df["time"], errors="coerce").to_numpy(dtype=float)
    run_time = pd.to_numeric(run_df["time"], errors="coerce").to_numpy(dtype=float)
    if reference_time.shape != run_time.shape or not np.allclose(
        reference_time,
        run_time,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("reference_df and run_df time columns must be identical")
    signals = _signal_columns(reference_df)
    if not signals:
        raise ValueError("reference_df and run_df must contain signal columns")
    return signals


def _scale_normalized_rms_by_signal(
    *,
    reference_df: pd.DataFrame,
    run_df: pd.DataFrame,
    intervention_time: object,
) -> tuple[list[str], list[float], list[float]]:
    signals = _validate_aligned_frames(reference_df, run_df)
    scores: list[float] = []
    scales: list[float] = []
    for signal in signals:
        reference_values = pd.to_numeric(reference_df[signal], errors="coerce").to_numpy(dtype=float)
        run_values = pd.to_numeric(run_df[signal], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(reference_values) & np.isfinite(run_values)
        scale = _scale_for_signal(
            run_df=run_df,
            reference_df=reference_df,
            signal=signal,
            intervention_time=intervention_time,
        )
        scales.append(float(scale))
        if not np.any(valid):
            scores.append(0.0)
            continue
        delta = run_values[valid] - reference_values[valid]
        rms = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
        scores.append(float(rms / max(scale, _EPS)) if math.isfinite(rms) else 0.0)
    return signals, scores, scales


def _max_score(scores: Sequence[float]) -> Optional[float]:
    finite = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite:
        return None
    return float(max(finite))


def _side_payload(
    *,
    model_id: str,
    reference_df: pd.DataFrame,
    run_df: pd.DataFrame,
    intervention_time: object,
    min_srd_distance: float,
    epsilon_srd: float,
    minimum_consecutive_srd_steps: int,
    signal_detectability_specs: Mapping[str, Mapping[str, object]],
    env_detectability_path: Optional[Path] = None,
    run_parameters: Optional[Mapping[str, Any]] = None,
    include_environment_hook: bool = False,
) -> Dict[str, Any]:
    clean_reference_df = cm._preprocess_detectability_frame(reference_df, model_id=model_id)
    clean_run_df = cm._preprocess_detectability_frame(run_df, model_id=model_id)
    max_srd, first_diff, euclidean_distance = _compute_detectability_values(
        baseline_df=clean_reference_df,
        run_df=clean_run_df,
        first_detectable_minimum_symmetric_distance=min_srd_distance,
        first_detectable_epsilon=epsilon_srd,
        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
        intervention_time=intervention_time,
        signal_detectability_specs=signal_detectability_specs,
        require_signal_detectability_specs=bool(signal_detectability_specs),
    )
    signals, normalized_rms, scales = _scale_normalized_rms_by_signal(
        reference_df=clean_reference_df,
        run_df=clean_run_df,
        intervention_time=intervention_time,
    )
    base_payload = cm._detectability_summary_entry(
        detectable="yes",
        max_srd=max_srd,
        first_diff=first_diff,
        environment_specific_detectability="yes",
        max_srd_detectability="yes",
        euclidean_distance=euclidean_distance,
        mean_euclidean_distance_clean_dirty=[0.0 for _ in first_diff],
        mean_euclidean_distance_clean_baseline=euclidean_distance,
        mean_SNR=[None for _ in first_diff],
    )
    if include_environment_hook:
        base_payload = cm._with_environment_specific_detectability(
            base_payload,
            run_df=clean_run_df,
            clean_df=clean_reference_df,
            run_parameters=run_parameters or {},
            intervention_time=intervention_time,
            env_detectability_path=env_detectability_path,
        )
    else:
        base_payload["environment_specific_detectability"] = "yes"
        base_payload["detectability"] = "yes"
        base_payload["detectable"] = "yes"

    output = dict(base_payload)
    output["scale_normalized_rms"] = normalized_rms
    output["scale_normalized_rms_scales"] = scales
    output["scale_normalized_rms_signals"] = signals
    output["scale_normalized_rms_max"] = _max_score(normalized_rms)
    return output


def _error_side_payload(*, include_environment_hook: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "environment_specific_detectability": "error" if include_environment_hook else "yes",
        "max_SRD_detectability": "error",
        "detectability": "error",
        "detectable": "error",
        "max_SRD": [],
        "euclidean_distance": [],
        "mean_euclidean_distance_clean_dirty": [],
        "mean_euclidean_distance_clean_baseline": [],
        "mean_SNR": [],
        "first_diff": [],
        "scale_normalized_rms": [],
        "scale_normalized_rms_scales": [],
        "scale_normalized_rms_signals": [],
        "scale_normalized_rms_max": None,
    }
    return out


def _side_scores_by_signal(side: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    status = cm._detectability_status(side)
    env_status = str(side.get("environment_specific_detectability") or "yes").strip().lower()
    if status == "error" or env_status in {"error", "no"}:
        return None
    output = side.get("detectability_output")
    if not isinstance(output, Mapping):
        return None
    signals = output.get("scale_normalized_rms_signals")
    scores = output.get("scale_normalized_rms")
    if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
        return None
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        return None
    out: Dict[str, float] = {}
    for signal, score in zip(signals, scores):
        signal_name = str(signal)
        score_value = _finite_float(score)
        if signal_name and score_value is not None:
            out.setdefault(signal_name, score_value)
    return out if out else None


def _child_side_scores(
    child: Mapping[str, Any],
    side_name: str,
) -> Optional[Dict[str, float]]:
    detectability = child.get("detectability")
    if not isinstance(detectability, Mapping):
        return None
    side = detectability.get(side_name)
    if not isinstance(side, Mapping):
        return None
    return _side_scores_by_signal(side)


def _threshold_value(threshold: object) -> float:
    value = _finite_float(threshold)
    return float(value) if value is not None else math.inf


def _child_passes_thresholds(
    child: Mapping[str, Any],
    thresholds: Mapping[str, object],
) -> bool:
    for side_name in _SIDE_NAMES:
        scores = _child_side_scores(child, side_name)
        if not scores:
            return False
        if not any(
            score > _threshold_value(thresholds.get(signal))
            for signal, score in scores.items()
        ):
            return False
    return True


def _thresholds_from_side_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    return {str(signal): float(score) for signal, score in scores.items()}


def _threshold_score_sum(thresholds: Mapping[str, object]) -> float:
    values = [
        float(value)
        for value in thresholds.values()
        if _finite_float(value) is not None
    ]
    return float(sum(values))


def _evaluate_thresholds(
    *,
    children: Mapping[str, Mapping[str, Any]],
    current_eligible: set[str],
    thresholds: Mapping[str, object],
) -> tuple[set[str], list[str], list[str]]:
    predicted = {
        child_id
        for child_id, child in children.items()
        if _child_passes_thresholds(child, thresholds)
    }
    dropped = sorted(current_eligible - predicted)
    added = sorted(predicted - current_eligible)
    return predicted, dropped, added


def _refine_thresholds_from_extra_children(
    *,
    children: Mapping[str, Mapping[str, Any]],
    current_eligible: set[str],
    thresholds: Mapping[str, object],
    max_extra: int,
    initial_added: Sequence[str],
) -> tuple[Dict[str, object], set[str], list[str], list[str]]:
    best_thresholds: Dict[str, object] = dict(thresholds)
    best_predicted, best_dropped, best_added = _evaluate_thresholds(
        children=children,
        current_eligible=current_eligible,
        thresholds=best_thresholds,
    )
    initial_added_set = set(initial_added)
    best_key = (
        len(best_added),
        len(initial_added_set & set(best_added)),
        -_threshold_score_sum(best_thresholds),
    )
    for child_id in initial_added:
        child = children.get(child_id)
        if not isinstance(child, Mapping):
            continue
        for side_name in _SIDE_NAMES:
            scores = _child_side_scores(child, side_name)
            if not scores:
                continue
            candidate_thresholds = dict(best_thresholds)
            candidate_thresholds.update(_thresholds_from_side_scores(scores))
            predicted, dropped, added = _evaluate_thresholds(
                children=children,
                current_eligible=current_eligible,
                thresholds=candidate_thresholds,
            )
            if dropped or len(added) > max_extra:
                continue
            key = (
                len(added),
                len(initial_added_set & set(added)),
                -_threshold_score_sum(candidate_thresholds),
            )
            if key < best_key:
                best_key = key
                best_thresholds = candidate_thresholds
                best_predicted = predicted
                best_dropped = dropped
                best_added = added
    exact = _refine_thresholds_exact(
        children=children,
        current_eligible=current_eligible,
        initial_added=initial_added,
    )
    if exact is not None:
        exact_thresholds, exact_predicted, exact_dropped, exact_added = exact
        exact_key = (
            len(exact_added),
            len(initial_added_set & set(exact_added)),
            -_threshold_score_sum(exact_thresholds),
        )
        if not exact_dropped and exact_key < best_key:
            return exact_thresholds, exact_predicted, exact_dropped, exact_added
    return best_thresholds, best_predicted, best_dropped, best_added


def _refine_thresholds_exact(
    *,
    children: Mapping[str, Mapping[str, Any]],
    current_eligible: set[str],
    initial_added: Sequence[str],
) -> Optional[tuple[Dict[str, object], set[str], list[str], list[str]]]:
    if not initial_added:
        return None
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception:
        return None

    child_ids = sorted(current_eligible | set(initial_added))
    signal_order: list[str] = []
    side_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    for child_id in child_ids:
        child = children.get(child_id)
        if not isinstance(child, Mapping):
            return None
        side_scores[child_id] = {}
        for side_name in _SIDE_NAMES:
            scores = _child_side_scores(child, side_name)
            if not scores:
                return None
            side_scores[child_id][side_name] = scores
            for signal in scores:
                if signal not in signal_order:
                    signal_order.append(signal)
    if not signal_order:
        return None

    threshold_candidates: list[list[float]] = []
    for signal in signal_order:
        values = sorted(
            {
                float(side_scores[child_id][side_name][signal])
                for child_id in child_ids
                for side_name in _SIDE_NAMES
                if signal in side_scores[child_id][side_name]
                and math.isfinite(float(side_scores[child_id][side_name][signal]))
            }
        )
        candidates = sorted(
            {
                candidate
                for value in values
                for candidate in (float(value), float(np.nextafter(value, -math.inf)))
                if math.isfinite(candidate)
            }
        )
        if not candidates:
            return None
        threshold_candidates.append(candidates)

    z_offsets: list[int] = []
    variable_count = 0
    for candidates in threshold_candidates:
        z_offsets.append(variable_count)
        variable_count += len(candidates)
    p_offsets: list[tuple[int, int]] = []
    y_offsets: list[int] = []
    extra_ids = [child_id for child_id in child_ids if child_id in set(initial_added)]
    for _child_id in extra_ids:
        p_offsets.append((variable_count, variable_count + 1))
        variable_count += 2
        y_offsets.append(variable_count)
        variable_count += 1

    rows: list[Dict[int, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for signal_index, candidates in enumerate(threshold_candidates):
        rows.append(
            {
                z_offsets[signal_index] + candidate_index: 1.0
                for candidate_index in range(len(candidates))
            }
        )
        lower_bounds.append(1.0)
        upper_bounds.append(1.0)

    def pass_coefficients(child_id: str, side_name: str) -> Dict[int, float]:
        coefficients: Dict[int, float] = {}
        scores = side_scores[child_id][side_name]
        for signal_index, signal in enumerate(signal_order):
            score = scores.get(signal)
            if score is None:
                continue
            for candidate_index, threshold in enumerate(threshold_candidates[signal_index]):
                if float(threshold) < float(score):
                    coefficients[z_offsets[signal_index] + candidate_index] = 1.0
        return coefficients

    for child_id in child_ids:
        if child_id not in current_eligible:
            continue
        for side_name in _SIDE_NAMES:
            rows.append(pass_coefficients(child_id, side_name))
            lower_bounds.append(1.0)
            upper_bounds.append(math.inf)

    side_count = float(len(signal_order))
    for extra_index, child_id in enumerate(extra_ids):
        for side_offset, side_name in enumerate(_SIDE_NAMES):
            p_var = p_offsets[extra_index][side_offset]
            coefficients = pass_coefficients(child_id, side_name)
            row = dict(coefficients)
            row[p_var] = row.get(p_var, 0.0) - 1.0
            rows.append(row)
            lower_bounds.append(0.0)
            upper_bounds.append(math.inf)

            row = {key: -value / side_count for key, value in coefficients.items()}
            row[p_var] = row.get(p_var, 0.0) + 1.0
            rows.append(row)
            lower_bounds.append(0.0)
            upper_bounds.append(math.inf)

        y_var = y_offsets[extra_index]
        p0_var, p1_var = p_offsets[extra_index]
        rows.append({y_var: 1.0, p0_var: -1.0, p1_var: -1.0})
        lower_bounds.append(-1.0)
        upper_bounds.append(math.inf)

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    for row_index, row in enumerate(rows):
        for variable_index, value in row.items():
            matrix[row_index, variable_index] = value
    objective = np.zeros(variable_count, dtype=float)
    for y_var in y_offsets:
        objective[y_var] = 1.0
    try:
        result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=float),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(
                matrix.tocsr(),
                np.array(lower_bounds, dtype=float),
                np.array(upper_bounds, dtype=float),
            ),
            options={"time_limit": 120.0, "mip_rel_gap": 0.0},
        )
    except Exception:
        return None
    if result.x is None or result.status not in {0, 1}:
        return None

    thresholds: Dict[str, object] = {}
    for signal_index, signal in enumerate(signal_order):
        offset = z_offsets[signal_index]
        candidate_count = len(threshold_candidates[signal_index])
        selected = int(np.argmax(result.x[offset : offset + candidate_count]))
        thresholds[signal] = float(threshold_candidates[signal_index][selected])
    predicted, dropped, added = _evaluate_thresholds(
        children=children,
        current_eligible=current_eligible,
        thresholds=thresholds,
    )
    return thresholds, predicted, dropped, added


def _current_eligible_children(payload: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    baselines = payload.get("baselines")
    if not isinstance(baselines, Mapping):
        return out
    for baseline in baselines.values():
        if not isinstance(baseline, Mapping):
            continue
        children = baseline.get("children")
        if not isinstance(children, Mapping):
            continue
        for child_id, child in children.items():
            if isinstance(child, Mapping) and child.get("eligible") is True:
                out.add(str(child_id))
    return out


def _calibrate_threshold(
    *,
    children: Mapping[str, Mapping[str, Any]],
    current_eligible: set[str],
    extra_budget_fraction: float,
) -> Dict[str, Any]:
    side_scores = {
        side_name: {
            child_id: _child_side_scores(child, side_name)
            for child_id, child in children.items()
        }
        for side_name in _SIDE_NAMES
    }
    missing_current_by_side = {
        side_name: sorted(
            child_id
            for child_id in current_eligible
            if not side_scores[side_name].get(child_id)
        )
        for side_name in _SIDE_NAMES
    }
    signal_order: list[str] = []
    assigned_scores: Dict[str, list[float]] = {}
    assigned_children: Dict[str, list[str]] = {}
    support_child_ids_by_side: Dict[str, Dict[str, list[str]]] = {
        side_name: {} for side_name in _SIDE_NAMES
    }
    for side_name in _SIDE_NAMES:
        for child_id, scores in side_scores[side_name].items():
            if not scores:
                continue
            for signal in scores:
                if signal not in signal_order:
                    signal_order.append(signal)
            if child_id not in current_eligible:
                continue
            strongest_signal, strongest_score = max(scores.items(), key=lambda item: item[1])
            assigned_scores.setdefault(strongest_signal, []).append(float(strongest_score))
            assigned_children.setdefault(strongest_signal, []).append(str(child_id))
            support_child_ids_by_side[side_name].setdefault(strongest_signal, []).append(
                str(child_id)
            )
    thresholds: Dict[str, Optional[float]] = {}
    support_child_ids: Dict[str, list[str]] = {}
    min_current_scores: Dict[str, Optional[float]] = {}
    for signal in signal_order:
        support = assigned_scores.get(signal, [])
        support_child_ids[signal] = sorted(set(assigned_children.get(signal, [])))
        if support:
            min_score = float(min(support))
            thresholds[signal] = float(np.nextafter(min_score, -math.inf))
            min_current_scores[signal] = min_score
        else:
            thresholds[signal] = None
            min_current_scores[signal] = None
    max_extra = int(math.floor(max(0, len(current_eligible)) * float(extra_budget_fraction)))
    predicted, dropped, added = _evaluate_thresholds(
        children=children,
        current_eligible=current_eligible,
        thresholds=thresholds,
    )
    initial_thresholds = dict(thresholds)
    initial_added = list(added)
    thresholds, predicted, dropped, added = _refine_thresholds_from_extra_children(
        children=children,
        current_eligible=current_eligible,
        thresholds=thresholds,
        max_extra=max_extra,
        initial_added=initial_added,
    )
    missing_current = sorted(
        {
            child_id
            for child_ids in missing_current_by_side.values()
            for child_id in child_ids
        }
    )
    ok = not missing_current and not dropped and len(added) <= max_extra
    return {
        "ok": bool(ok),
        "thresholds": thresholds,
        "initial_thresholds": initial_thresholds,
        "initial_added_child_ids": initial_added,
        "support_child_ids": support_child_ids,
        "support_counts": {
            signal: len(child_ids)
            for signal, child_ids in support_child_ids.items()
        },
        "support_child_ids_by_side": {
            side_name: {
                signal: sorted(set(child_ids))
                for signal, child_ids in side_support.items()
            }
            for side_name, side_support in support_child_ids_by_side.items()
        },
        "support_counts_by_side": {
            side_name: {
                signal: len(set(child_ids))
                for signal, child_ids in side_support.items()
            }
            for side_name, side_support in support_child_ids_by_side.items()
        },
        "current_eligible_count": len(current_eligible),
        "euclidean_eligible_count": len(predicted),
        "retained_current_eligible_count": len(current_eligible - set(dropped)),
        "added_child_count": len(added),
        "max_extra_child_count": max_extra,
        "missing_current_score_child_ids": missing_current,
        "missing_current_score_child_ids_by_side": missing_current_by_side,
        "dropped_child_ids": dropped,
        "added_child_ids": added,
        "min_current_scores": min_current_scores,
    }


def _apply_threshold_to_side(
    side: Mapping[str, Any],
    *,
    thresholds: Mapping[str, object],
) -> Dict[str, Any]:
    out = dict(side)
    output = dict(out.get("detectability_output") or {})
    signals = output.get("scale_normalized_rms_signals")
    scores = output.get("scale_normalized_rms")
    if not isinstance(signals, Sequence) or isinstance(signals, (str, bytes)):
        signals = []
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        scores = []
    threshold_values: list[Optional[float]] = []
    passed_by_signal: list[bool] = []
    for signal, score in zip(signals, scores):
        threshold = thresholds.get(str(signal))
        threshold_value = _threshold_value(threshold)
        score_value = _finite_float(score)
        threshold_values.append(None if math.isinf(threshold_value) else threshold_value)
        passed_by_signal.append(bool(score_value is not None and score_value > threshold_value))
    passed = any(passed_by_signal)
    output["scale_normalized_rms_thresholds"] = threshold_values
    output["scale_normalized_rms_passed_by_signal"] = passed_by_signal
    output["scale_normalized_rms_passed"] = passed
    out["detectability_output"] = output
    env_status = str(out.get("environment_specific_detectability") or "yes").strip().lower()
    if env_status == "error" or cm._detectability_status(out) == "error":
        status = "error"
    elif env_status == "no":
        status = "no"
    else:
        status = "yes" if passed else "no"
    out["max_SRD_detectability"] = status
    out["detectability"] = status
    out["detectable"] = status
    return out


def _apply_threshold(
    *,
    results: Dict[str, Any],
    thresholds: Mapping[str, object],
    expected_parameters: Sequence[str],
    child_parameters_by_baseline: Mapping[str, Mapping[str, str]],
) -> None:
    baselines = results.get("baselines")
    if not isinstance(baselines, dict):
        return
    for baseline_id, baseline in baselines.items():
        if not isinstance(baseline, dict):
            continue
        children = baseline.get("children")
        if not isinstance(children, dict):
            continue
        for child in children.values():
            if not isinstance(child, dict):
                continue
            detectability = child.get("detectability")
            if not isinstance(detectability, dict):
                child["eligible"] = False
                continue
            for side_name in _SIDE_NAMES:
                side = detectability.get(side_name)
                if isinstance(side, dict):
                    detectability[side_name] = _apply_threshold_to_side(
                        side,
                        thresholds=thresholds,
                    )
            child["eligible"] = cm._child_is_eligible(child)
        family_eligible = cm._baseline_family_is_eligible(
            children,
            child_parameters=dict(child_parameters_by_baseline.get(str(baseline_id), {})),
            expected_parameters=tuple(expected_parameters),
        )
        baseline["family_eligible"] = bool(family_eligible)
        baseline["eligible"] = bool(family_eligible)
    results["eligible_baselines"] = sum(
        1
        for baseline in baselines.values()
        if isinstance(baseline, dict) and baseline.get("family_eligible") is True
    )
    results["total_baselines"] = len(baselines)


def _documented_side(side: Mapping[str, Any], *, baseline_side: bool) -> Dict[str, Any]:
    output = {
        "mean_euclidean_distance_clean_dirty": list(
            side.get("mean_euclidean_distance_clean_dirty") or []
        ),
        "mean_euclidean_distance_clean_baseline": list(
            side.get("mean_euclidean_distance_clean_baseline") or []
        ),
        "mean_SNR": list(side.get("mean_SNR") or []),
        "first_diff": list(side.get("first_diff") or []),
        "scale_normalized_rms": list(side.get("scale_normalized_rms") or []),
        "scale_normalized_rms_scales": list(side.get("scale_normalized_rms_scales") or []),
        "scale_normalized_rms_signals": list(side.get("scale_normalized_rms_signals") or []),
        "scale_normalized_rms_max": side.get("scale_normalized_rms_max"),
        "scale_normalized_rms_thresholds": list(
            side.get("scale_normalized_rms_thresholds") or []
        ),
        "scale_normalized_rms_passed_by_signal": list(
            side.get("scale_normalized_rms_passed_by_signal") or []
        ),
        "scale_normalized_rms_passed": bool(side.get("scale_normalized_rms_passed") is True),
    }
    out = {
        "detectable": cm._detectability_status(side) or "no",
        "detectability_output": output,
    }
    if baseline_side:
        out["environment_specific_detectability"] = str(
            side.get("environment_specific_detectability") or "error"
        ).strip().lower()
    return out


def _documented_child(child: Mapping[str, Any]) -> Dict[str, Any]:
    detectability = child.get("detectability") if isinstance(child, Mapping) else None
    if not isinstance(detectability, Mapping):
        return copy.deepcopy(child) if isinstance(child, dict) else {}
    return {
        "url": str(child.get("url") or ""),
        "detectability": {
            "vs_baseline": _documented_side(
                detectability.get("vs_baseline") or {},
                baseline_side=True,
            ),
            "vs_time0_baseline": _documented_side(
                detectability.get("vs_time0_baseline") or {},
                baseline_side=False,
            ),
        },
        "eligible": bool(child.get("eligible") is True),
    }


def _collect_children(results: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    baselines = results.get("baselines")
    if not isinstance(baselines, Mapping):
        return out
    for baseline in baselines.values():
        if not isinstance(baseline, Mapping):
            continue
        children = baseline.get("children")
        if not isinstance(children, Mapping):
            continue
        for child_id, child in children.items():
            if isinstance(child, Mapping):
                out[str(child_id)] = child
    return out


def _load_current_eligibility(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise click.ClickException(f"Missing current eligibility metrics at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_df_cached(cache: Dict[str, Any], runs_root: Path, run_id: str):
    if run_id in cache:
        return cache[run_id]
    df = load_run_df(runs_root / run_id)
    cache[run_id] = df
    return df


def _compute_child_raw(
    *,
    model_id: str,
    base_df: Any,
    entry: Mapping[str, Any],
    time0_df: Any,
    min_srd_distance: float,
    epsilon_srd: float,
    minimum_consecutive_srd_steps: int,
    signal_detectability_specs: Mapping[str, Mapping[str, object]],
    env_detectability_path: Optional[Path],
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    run_id = str(entry["run_id"])
    raw_iv = entry["raw"]
    baseline_parameters = (
        entry.get("baseline_parameters")
        if isinstance(entry.get("baseline_parameters"), Mapping)
        else {}
    )
    run_parameters = cm._build_run_parameters(
        baseline_parameters=baseline_parameters,
        raw_intervention=raw_iv if isinstance(raw_iv, Mapping) else {},
    )
    try:
        baseline_side = _side_payload(
            model_id=model_id,
            reference_df=base_df,
            run_df=entry["df"],
            intervention_time=raw_iv.get("intervention_time"),
            min_srd_distance=min_srd_distance,
            epsilon_srd=epsilon_srd,
            minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
            signal_detectability_specs=signal_detectability_specs,
            env_detectability_path=env_detectability_path,
            run_parameters=run_parameters,
            include_environment_hook=True,
        )
    except Exception:
        baseline_side = _error_side_payload(include_environment_hook=True)

    time0_expected = bool(str(raw_iv.get("time0_baseline_uuid") or "").strip())
    if time0_expected and time0_df is not None:
        try:
            time0_side = _side_payload(
                model_id=model_id,
                reference_df=time0_df,
                run_df=entry["df"],
                intervention_time=raw_iv.get("intervention_time"),
                min_srd_distance=min_srd_distance,
                epsilon_srd=epsilon_srd,
                minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                signal_detectability_specs=signal_detectability_specs,
                include_environment_hook=False,
            )
        except Exception:
            time0_side = _error_side_payload(include_environment_hook=False)
    elif time0_expected:
        time0_side = _error_side_payload(include_environment_hook=False)
    else:
        time0_side = _error_side_payload(include_environment_hook=False)
        time0_side["detectability"] = "no"
        time0_side["detectable"] = "no"

    child = {
        "url": cm._webapp_run_url(model_id=model_id, run_id=run_id),
        "detectability": {
            "vs_baseline": baseline_side,
            "vs_time0_baseline": time0_side,
        },
        "eligible": False,
    }
    return run_id, child, _documented_child(child)


def _run_child_tasks(
    *,
    tasks: Iterable[tuple[str, Mapping[str, Any], Any, Any]],
    model_id: str,
    min_srd_distance: float,
    epsilon_srd: float,
    minimum_consecutive_srd_steps: int,
    signal_detectability_specs: Mapping[str, Mapping[str, object]],
    env_detectability_path: Optional[Path],
    jobs: int,
) -> Dict[str, Dict[str, Any]]:
    task_list = list(tasks)

    def run_one(task: tuple[str, Mapping[str, Any], Any, Any]) -> tuple[str, str, Dict[str, Any]]:
        baseline_id, entry, base_df, time0_df = task
        run_id, child, _documented = _compute_child_raw(
            model_id=model_id,
            base_df=base_df,
            entry=entry,
            time0_df=time0_df,
            min_srd_distance=min_srd_distance,
            epsilon_srd=epsilon_srd,
            minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
            signal_detectability_specs=signal_detectability_specs,
            env_detectability_path=env_detectability_path,
        )
        return str(baseline_id), str(run_id), child

    out: Dict[str, Dict[str, Any]] = {}
    progress_kwargs = {
        "total": len(task_list),
        "desc": f"{model_id}: euclidean eligibility",
        "unit": "child",
    }
    if jobs <= 1:
        for task in tqdm(task_list, **progress_kwargs):
            baseline_id, run_id, child = run_one(task)
            out.setdefault(baseline_id, {})[run_id] = child
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {executor.submit(run_one, task): task for task in task_list}
        for future in tqdm(concurrent.futures.as_completed(future_map), **progress_kwargs):
            baseline_id, run_id, child = future.result()
            out.setdefault(baseline_id, {})[run_id] = child
    return out


def run_for_model(
    model_dir: Path,
    *,
    jobs: Optional[int] = None,
    extra_budget_fraction: float = _DEFAULT_EXTRA_BUDGET_FRACTION,
) -> Dict[str, Any]:
    model_record_path = resolve_model_record_path(model_dir)
    if not model_record_path.exists():
        raise click.ClickException(f"Missing model_record.json at {model_record_path}")
    config_path = model_dir / "experiment_config.json"
    if not config_path.exists():
        raise click.ClickException(f"Missing experiment_config.json at {config_path}")
    try:
        experiment_config = load_experiment_config_json(config_path)
    except DistributionValidationError as exc:
        raise click.ClickException(f"Invalid experiment_config.json at {config_path}: {exc}") from exc

    runs_root = resolve_runs_root(model_dir)
    specs_path = model_dir / "model_run_specs.json"
    model_record = load_model_record_json(model_record_path)
    specs = load_model_run_specs_json(
        specs_path,
        enforce_baseline_pair_diversity=False,
    )
    registry = build_model_record_registry(
        model_id=model_dir.name,
        specs=specs,
        runtime_map=model_record,
        experiment_config=experiment_config,
    )
    current_path = resolve_similarity_metrics_path(model_dir)
    current_payload = _load_current_eligibility(current_path)
    current_eligible = _current_eligible_children(current_payload)

    min_srd_distance = float(experiment_config.min_srd_distance)
    epsilon_srd = float(experiment_config.epsilon_SRD)
    minimum_consecutive_srd_steps = max(
        1,
        int(getattr(experiment_config, "minimum_consecurive_below_SRD", 1)),
    )
    expected_parameters = cm._expected_intervention_parameters(experiment_config)
    signal_detectability_specs = cm._signal_detectability_specs(experiment_config)
    env_detectability_path = model_dir / "detectability_specific_environment.py"
    resolved_env_detectability_path = (
        env_detectability_path if env_detectability_path.exists() else None
    )
    resolved_jobs = cm._resolve_jobs(jobs)

    df_cache: Dict[str, Any] = {}
    results: Dict[str, Any] = {
        "timestamp": cm._now_iso8601_utc(),
        "metric": "scale_normalized_rms",
        "threshold_scope": "per_environment_per_channel",
        "calibration_source": str(current_path),
        "extra_budget_fraction": float(extra_budget_fraction),
        "noise_adder_md5": cm._md5_hex_or_none(model_dir / "noise_adder.py"),
        "eligible_baselines": 0,
        "total_baselines": 0,
        "baselines": {},
    }
    child_parameters_by_baseline: Dict[str, Dict[str, str]] = {}
    tasks: list[tuple[str, Mapping[str, Any], Any, Any]] = []
    baselines = registry.get("baselines", [])
    if not isinstance(baselines, list):
        raise click.ClickException("derived registry baselines must be a list")
    for baseline in baselines:
        if not isinstance(baseline, Mapping):
            continue
        baseline_id = str(baseline.get("run_id") or "").strip()
        if not baseline_id:
            continue
        children_out: Dict[str, Any] = {}
        child_parameters: Dict[str, str] = {}
        interventions = baseline.get("interventions")
        if not isinstance(interventions, list):
            interventions = []
        for iv in interventions:
            if not isinstance(iv, Mapping):
                continue
            child_id = str(iv.get("name") or "").strip()
            if child_id:
                children_out[child_id] = cm._default_child_summary(
                    url=cm._webapp_run_url(model_id=model_dir.name, run_id=child_id)
                )
        results["baselines"][baseline_id] = {
            "url": cm._webapp_run_url(model_id=model_dir.name, run_id=baseline_id),
            "family_eligible": False,
            "eligible": False,
            "children": children_out,
        }
        if not is_success_status(baseline.get("status")):
            child_parameters_by_baseline[baseline_id] = child_parameters
            continue
        try:
            base_df = _load_df_cached(df_cache, runs_root, baseline_id)
        except Exception:
            child_parameters_by_baseline[baseline_id] = child_parameters
            continue
        for iv in interventions:
            if not isinstance(iv, Mapping):
                continue
            child_id = str(iv.get("name") or "").strip()
            if not child_id:
                continue
            truth_parameter = str(iv.get("parameter") or iv.get("variable") or "").strip()
            if truth_parameter:
                child_parameters[child_id] = truth_parameter
            if not is_success_status(iv.get("status")):
                continue
            try:
                child_df = _load_df_cached(df_cache, runs_root, child_id)
            except Exception:
                continue
            time0_id = str(iv.get("time0_baseline_uuid") or "").strip()
            time0_df = None
            if time0_id and is_success_status(iv.get("time0_baseline_status")):
                try:
                    time0_df = _load_df_cached(df_cache, runs_root, time0_id)
                except Exception:
                    time0_df = None
            entry = {
                "run_id": child_id,
                "df": child_df,
                "raw": iv,
                "baseline_parameters": dict(baseline.get("parameters") or {}),
                "truth_parameter": truth_parameter,
            }
            tasks.append((baseline_id, entry, base_df, time0_df))
        child_parameters_by_baseline[baseline_id] = child_parameters

    click.echo(
        f"{model_dir.name}: computing euclidean eligibility for "
        f"{len(results['baselines'])} baselines, {len(tasks)} children, jobs={resolved_jobs}"
    )
    child_results = _run_child_tasks(
        tasks=tasks,
        model_id=model_dir.name,
        min_srd_distance=min_srd_distance,
        epsilon_srd=epsilon_srd,
        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
        signal_detectability_specs=signal_detectability_specs,
        env_detectability_path=resolved_env_detectability_path,
        jobs=resolved_jobs,
    )
    for baseline_id, children in child_results.items():
        baseline = results["baselines"].get(baseline_id)
        if not isinstance(baseline, dict):
            continue
        for child_id, child in children.items():
            baseline["children"][child_id] = _documented_child(child)

    raw_children = _collect_children(results)
    calibration = _calibrate_threshold(
        children=raw_children,
        current_eligible=current_eligible,
        extra_budget_fraction=extra_budget_fraction,
    )
    calibrated_thresholds = calibration.get("thresholds")
    _apply_threshold(
        results=results,
        thresholds=calibrated_thresholds if isinstance(calibrated_thresholds, Mapping) else {},
        expected_parameters=expected_parameters,
        child_parameters_by_baseline=child_parameters_by_baseline,
    )
    calibration["post_threshold_eligible_child_ids"] = sorted(
        child_id
        for child_id, child in _collect_children(results).items()
        if isinstance(child, Mapping) and child.get("eligible") is True
    )

    out_path = current_path.with_name("eligibility_metrics_euclidean.json")
    calibration_path = current_path.with_name("eligibility_metrics_euclidean_calibration.json")
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    calibration_path.write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    click.echo(f"{model_dir.name}: wrote {out_path}")
    click.echo(f"{model_dir.name}: wrote {calibration_path}")
    if not calibration.get("ok"):
        raise click.ClickException(
            f"{model_dir.name}: euclidean calibration did not satisfy retention/extra budget; "
            f"see {calibration_path}"
        )
    return {
        "model_id": model_dir.name,
        "ok": True,
        "path": str(out_path),
        "calibration_path": str(calibration_path),
    }


@click.command()
@click.option(
    "--model",
    type=str,
    default=None,
    help="Optional tsENV model name under models/simulink/. If omitted, iterate over all allowed models.",
)
@click.option(
    "--jobs",
    type=int,
    default=None,
    help="Number of worker threads. Default: min(8, CPU count).",
)
@click.option(
    "--extra-budget-fraction",
    type=float,
    default=_DEFAULT_EXTRA_BUDGET_FRACTION,
    show_default=True,
    help="Maximum added eligible children as a fraction of current eligible children.",
)
def cli(
    model: Optional[str],
    jobs: Optional[int],
    extra_budget_fraction: float,
) -> None:
    models_root = _resolve_models_root()
    if model is None:
        model_ids = list(ALLOWED_TSENV_MODELS)
    else:
        model_id = str(model).strip()
        if "/" in model_id or "\\" in model_id:
            raise click.ClickException(f"Expected a model id, got path-like value: {model!r}")
        if model_id not in ALLOWED_TSENV_MODELS:
            raise click.ClickException(f"Model {model_id!r} is not an allowed tsENV model")
        model_ids = [model_id]
    computed: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    for model_id in sorted(model_ids):
        model_dir = models_root / model_id
        if not model_dir.exists():
            skipped.append({"model_id": model_id, "ok": False, "reason": "missing_model_dir"})
            continue
        try:
            computed.append(
                run_for_model(
                    model_dir,
                    jobs=jobs,
                    extra_budget_fraction=extra_budget_fraction,
                )
            )
        except Exception as exc:
            skipped.append({"model_id": model_id, "ok": False, "error": str(exc)})
            if model is not None:
                raise
    click.echo(json.dumps({"computed": computed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    cli()
