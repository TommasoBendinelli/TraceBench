from __future__ import annotations

import concurrent.futures
import copy
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import weakref
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode

import click
import numpy as np
from tqdm import tqdm

# Add repo root to sys.path
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
from shared.child_only_predictor import parameter_key_to_display_label  # noqa: E402
from shared.model_intervention_interface import load_allowed_interventions  # noqa: E402
from shared.model_noise_adder import call_noise_adder, load_noise_adder_from_path  # noqa: E402
from shared.model_run_specs_runtime import build_model_record_registry  # noqa: E402
from shared.question_eligibility import is_success_status  # noqa: E402
from shared.tsenv_combinations import TIME0_BASELINE_AGENT_FACING_LABEL  # noqa: E402
from shared.interface.similarity_metrics_json import (  # noqa: E402
    dump_similarity_metrics_json,
    validate_similarity_metrics_semantics,
)
from shared.run_artifacts import (  # noqa: E402
    resolve_model_record_path,
    resolve_runs_root,
    resolve_similarity_metrics_path,
)
from shared.time_series_metrics import (  # noqa: E402
    compute_detectability_baseline,
    compute_detectability_time0_baseline,
    load_run_df,
)
from workflows.metrics._evaluate_rule_helpers import (  # noqa: E402
    _build_display_label_to_parameter_key_map,
    _extract_rule_prediction,
    _load_rule_predict_fn,
)
from workflows.metrics import evaluate_rule as evaluate_rule_workflow  # noqa: E402

_NOISE_SWEEP_SEEDS = tuple(range(5))
_GROUND_TRUTH_RULE_PROFILE_SEEDS = {
    "low": _NOISE_SWEEP_SEEDS,
    "high": _NOISE_SWEEP_SEEDS,
}
_WEBAPP_BASE_URL = "http://localhost:3001/"
_PROCESS_WORKER_MODEL_ID: Optional[str] = None
_PROCESS_WORKER_MODELS_ROOT: Optional[Path] = None
_PROCESS_WORKER_BASELINE_CONTEXTS: Dict[str, Dict[str, Any]] = {}
_PROCESS_WORKER_ENTRY_BY_RUN_ID: Dict[str, Dict[str, Any]] = {}
_PROCESS_WORKER_REQUIRED_MINIMUM_SRD = 0.0
_PROCESS_WORKER_EPSILON_SRD = 0.001
_PROCESS_WORKER_MINIMUM_CONSECUTIVE_SRD_STEPS = 1
_PROCESS_WORKER_INCLUDE_RULE_EVAL = False
_PROCESS_WORKER_SIGNAL_ENVELOPE_SIZES: Dict[str, int] = {}
_PROCESS_WORKER_SIGNAL_DETECTABILITY_SPECS: Dict[str, Dict[str, Any]] = {}
_PROCESS_WORKER_RMS_THRESHOLDS: Dict[str, float] = {}
_PROCESS_WORKER_ENV_DETECTABILITY_PATH: Optional[Path] = None
_PROCESS_WORKER_DETECTABILITY_NOISE_ADDER_PATH: Optional[Path] = None
_ENV_DETECTABILITY_FN_CACHE: Dict[str, Any] = {}
_PREPROCESSED_DETECTABILITY_FRAME_CACHE: "OrderedDict[Any, Any]" = OrderedDict()
_BALL_DROP_ENV_FEATURE_CACHE: "OrderedDict[int, Any]" = OrderedDict()
_DETECTABILITY_ARRAY_CACHE: "OrderedDict[int, Any]" = OrderedDict()
_PREPROCESSED_DETECTABILITY_FRAME_CACHE_MAX_ITEMS = 2048
_BALL_DROP_ENV_FEATURE_CACHE_MAX_ITEMS = 4096
_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS = 4096
_CACHE_MISS = object()
_BALL_DROP_MODEL_ID = "balldrop"
_BALL_DROP_FORCE_SIGNAL = "Hard_Stop_f"
_BALL_DROP_VELOCITY_SIGNAL = "Velocity"
_BALL_DROP_REBOUND_WINDOW_S = 0.1
_BALL_DROP_FORCE_THRESHOLD_FRACTION = 0.05
_BALL_DROP_APEX_HEIGHT_THRESHOLD = 0.1
_BALL_DROP_MAX_BOUNCES = 25
_BALL_DROP_IMPACT_FORCE_THRESHOLD = 1.0
_BALL_DROP_MIN_INCOMING_BOUNCE_SPEED = 0.5
_BALL_DROP_MIN_REBOUND_SPEED = 0.1
_BALL_DROP_DRAG_DTW_EUCLIDEAN_WINDOW_S = 0.2
_BALL_DROP_DRAG_DTW_EUCLIDEAN_THRESHOLD = 0.1
_BALL_DROP_DRAG_DTW_EUCLIDEAN_TOLERANCE = 1.0e-12
_BALL_DROP_DRAG_DTW_EUCLIDEAN_TIME_TOLERANCE = 1.0e-9
_EPS = 1.0e-12
_NO_CHANGE_CLASS = "no_parameter_change"
_LEGACY_NO_CHANGE_ALIASES = {
    _NO_CHANGE_CLASS,
    "No_parameter_change",
    "nothing_happened",
    "Nothing happened",
    TIME0_BASELINE_AGENT_FACING_LABEL,
}


def _resolve_models_root() -> Path:
    cwd_models_root = Path(os.getcwd()).resolve() / "models" / "simulink"
    if cwd_models_root.exists():
        return cwd_models_root
    return root_dir / "models" / "simulink"


def _md5_hex_or_none(path: Path) -> Optional[str]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return None
    return hashlib.md5(resolved.read_bytes()).hexdigest()


def _basic_rule_path(model_dir: Path) -> Path:
    return model_dir / "basic_rule.py"


def _basic_rule_path_for_model(*, models_root: Path, model_id: str) -> Path:
    return _basic_rule_path(models_root / model_id)


def _now_iso8601_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _webapp_run_url(*, model_id: str, run_id: str) -> str:
    query = urlencode(
        {
            "model": str(model_id or "").strip(),
            "run": str(run_id or "").strip(),
            "compare": "none",
        }
    )
    return f"{_WEBAPP_BASE_URL}?{query}"


def _normalize_label(value: object) -> str:
    return " ".join(str(value or "").strip().split()).lower()


_DF_CACHE_MAX_ITEMS = 512


def _load_df_cached(cache: Dict[str, Any], runs_root: Path, run_id: str):
    if run_id in cache:
        df = cache[run_id]
        move_to_end = getattr(cache, "move_to_end", None)
        if callable(move_to_end):
            move_to_end(run_id)
        return df
    df = load_run_df(runs_root / run_id)
    cache[run_id] = df
    popitem = getattr(cache, "popitem", None)
    while len(cache) > _DF_CACHE_MAX_ITEMS and callable(popitem):
        try:
            popitem(last=False)
        except TypeError:
            break
    return df


def _identity_cache_get(cache: OrderedDict, key: object, owner: object) -> Any:
    entry = cache.get(key, _CACHE_MISS)
    if entry is _CACHE_MISS:
        return _CACHE_MISS
    owner_ref, value = entry
    if owner_ref() is owner:
        cache.move_to_end(key)
        return value
    cache.pop(key, None)
    return _CACHE_MISS


def _identity_cache_set(
    cache: OrderedDict,
    key: object,
    owner: object,
    value: Any,
    *,
    max_items: int,
) -> Any:
    try:
        owner_ref = weakref.ref(owner)
    except TypeError:
        return value
    cache[key] = (owner_ref, value)
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)
    return value


def _apply_high_noise_pair_for_detectability(
    *,
    baseline_df: Any,
    run_df: Any,
    noise_baseline_df: Any,
    noise_adder_path: Optional[Path],
    seed: int = 0,
) -> tuple[Any, Any]:
    if noise_adder_path is None:
        return baseline_df, run_df
    add_noise = load_noise_adder_from_path(noise_adder_path)
    noisy_baseline_df, _baseline_noise_analysis = call_noise_adder(
        add_noise,
        baseline_df,
        baseline_df=noise_baseline_df,
        seed=seed,
        noise_level="high",
    )
    noisy_run_df, _run_noise_analysis = call_noise_adder(
        add_noise,
        run_df,
        baseline_df=noise_baseline_df,
        seed=seed,
        noise_level="high",
    )
    return noisy_baseline_df, noisy_run_df


def _mean_finite_by_channel(values_by_seed: List[List[float]]) -> List[float]:
    if not values_by_seed:
        return []
    max_len = max(len(values) for values in values_by_seed)
    means: List[float] = []
    for idx in range(max_len):
        values: List[float] = []
        for seed_values in values_by_seed:
            if idx >= len(seed_values):
                continue
            value = float(seed_values[idx])
            if np.isfinite(value):
                values.append(value)
        means.append(float(np.mean(values)) if values else 0.0)
    return means


def _mean_snr_db_by_channel(
    *,
    clean_dirty_by_seed: List[List[float]],
    clean_baseline_by_seed: List[List[float]],
) -> List[Optional[float]]:
    max_len = max(
        [
            0,
            *(len(values) for values in clean_dirty_by_seed),
            *(len(values) for values in clean_baseline_by_seed),
        ]
    )
    out: List[Optional[float]] = []
    for idx in range(max_len):
        ratios: List[float] = []
        for dirty_values, baseline_values in zip(
            clean_dirty_by_seed,
            clean_baseline_by_seed,
        ):
            if idx >= len(dirty_values) or idx >= len(baseline_values):
                continue
            noise = float(dirty_values[idx])
            signal_norm = float(baseline_values[idx])
            if not (np.isfinite(noise) and np.isfinite(signal_norm)):
                continue
            if noise <= 0.0 or signal_norm <= 0.0:
                continue
            ratios.append(signal_norm / noise)
        if not ratios:
            out.append(None)
            continue
        mean_ratio = float(np.mean(ratios))
        if mean_ratio <= 0.0 or not np.isfinite(mean_ratio):
            out.append(None)
            continue
        snr_db = float(20.0 * np.log10(mean_ratio))
        out.append(snr_db if np.isfinite(snr_db) else None)
    return out


def _detectability_noise_distance_summary(
    *,
    model_id: Optional[str],
    baseline_df: Any,
    run_df: Any,
    noise_baseline_df: Any,
    clean_baseline_df: Any,
    noise_adder_path: Optional[Path],
) -> tuple[Any, Any, List[float], List[float], List[Optional[float]]]:
    clean_dirty_by_seed: List[List[float]] = []
    clean_baseline_by_seed: List[List[float]] = []
    seed0_baseline_df: Any = None
    seed0_run_df: Any = None
    clean_run_df = _preprocess_detectability_frame(
        run_df,
        model_id=model_id,
    )
    clean_baseline_reference_df = _preprocess_detectability_frame(
        clean_baseline_df,
        model_id=model_id,
    )
    for seed in _NOISE_SWEEP_SEEDS:
        noisy_baseline_df, noisy_run_df = _apply_high_noise_pair_for_detectability(
            baseline_df=baseline_df,
            run_df=run_df,
            noise_baseline_df=noise_baseline_df,
            noise_adder_path=noise_adder_path,
            seed=seed,
        )
        noisy_baseline_df = _preprocess_detectability_frame(
            noisy_baseline_df,
            model_id=model_id,
        )
        noisy_run_df = _preprocess_detectability_frame(
            noisy_run_df,
            model_id=model_id,
        )
        if seed == _NOISE_SWEEP_SEEDS[0]:
            seed0_baseline_df = noisy_baseline_df
            seed0_run_df = noisy_run_df
        clean_dirty_by_seed.append(
            _per_signal_euclidean_distance(
                reference_df=clean_run_df,
                target_df=noisy_run_df,
            )
        )
        clean_baseline_by_seed.append(
            _per_signal_euclidean_distance(
                reference_df=clean_run_df,
                target_df=clean_baseline_reference_df,
            )
        )
    return (
        seed0_baseline_df,
        seed0_run_df,
        _mean_finite_by_channel(clean_dirty_by_seed),
        _mean_finite_by_channel(clean_baseline_by_seed),
        _mean_snr_db_by_channel(
            clean_dirty_by_seed=clean_dirty_by_seed,
            clean_baseline_by_seed=clean_baseline_by_seed,
        ),
    )


def _uses_ball_drop_hard_stop_preprocessing(model_id: Optional[str]) -> bool:
    return str(model_id or "").strip().lower() == _BALL_DROP_MODEL_ID


def _has_ball_drop_velocity_rebound(
    *,
    peak_time: float,
    velocity_times: np.ndarray,
    velocity_values: np.ndarray,
) -> bool:
    finite = np.isfinite(velocity_times) & np.isfinite(velocity_values)
    if not np.any(finite):
        return False
    times = velocity_times[finite]
    values = velocity_values[finite]
    window = _BALL_DROP_REBOUND_WINDOW_S
    pre_mask = (times >= float(peak_time) - window) & (times <= float(peak_time))
    post_mask = (times >= float(peak_time)) & (times <= float(peak_time) + window)
    if not np.any(pre_mask) or not np.any(post_mask):
        return False
    return bool(
        float(np.median(values[pre_mask])) < 0.0
        and float(np.median(values[post_mask])) > 0.0
    )


def _preprocess_ball_drop_hard_stop_for_detectability(df: Any) -> Any:
    columns = [str(column) for column in getattr(df, "columns", [])]
    required = {
        "time",
        _BALL_DROP_FORCE_SIGNAL,
        _BALL_DROP_VELOCITY_SIGNAL,
    }
    if not required.issubset(set(columns)):
        return df

    out = df.copy()
    times = np.asarray(out["time"], dtype=float)
    forces = np.asarray(out[_BALL_DROP_FORCE_SIGNAL], dtype=float)
    velocities = np.asarray(out[_BALL_DROP_VELOCITY_SIGNAL], dtype=float)
    if times.shape != forces.shape or times.shape != velocities.shape:
        raise ValueError("BallDrop Hard_Stop_f preprocessing requires aligned columns")

    finite_force = np.isfinite(forces)
    if not np.any(finite_force):
        out[_BALL_DROP_FORCE_SIGNAL] = np.zeros(forces.shape, dtype=float)
        return out

    scale = float(np.max(np.abs(forces[finite_force])))
    if not np.isfinite(scale) or scale <= _EPS:
        out[_BALL_DROP_FORCE_SIGNAL] = np.zeros(forces.shape, dtype=float)
        return out

    threshold = _BALL_DROP_FORCE_THRESHOLD_FRACTION * scale
    candidate_indices = np.flatnonzero(
        np.isfinite(times) & finite_force & (np.abs(forces) > threshold)
    )
    kept = np.zeros(forces.shape, dtype=float)
    if candidate_indices.size == 0:
        out[_BALL_DROP_FORCE_SIGNAL] = kept
        return out

    split_points = np.flatnonzero(np.diff(candidate_indices) > 1) + 1
    for group in np.split(candidate_indices, split_points):
        if group.size == 0:
            continue
        local_idx = int(group[np.argmax(np.abs(forces[group]))])
        peak_time = float(times[local_idx])
        if _has_ball_drop_velocity_rebound(
            peak_time=peak_time,
            velocity_times=times,
            velocity_values=velocities,
        ):
            kept[local_idx] = float(forces[local_idx])

    out[_BALL_DROP_FORCE_SIGNAL] = kept
    return out


def _preprocess_detectability_frame(
    df: Any,
    *,
    model_id: Optional[str],
) -> Any:
    if _uses_ball_drop_hard_stop_preprocessing(model_id):
        key = (str(model_id or "").strip().lower(), id(df))
        cached = _identity_cache_get(
            _PREPROCESSED_DETECTABILITY_FRAME_CACHE,
            key,
            df,
        )
        if cached is not _CACHE_MISS:
            return cached
        out = _preprocess_ball_drop_hard_stop_for_detectability(df)
        return _identity_cache_set(
            _PREPROCESSED_DETECTABILITY_FRAME_CACHE,
            key,
            df,
            out,
            max_items=_PREPROCESSED_DETECTABILITY_FRAME_CACHE_MAX_ITEMS,
        )
    return df


def _per_signal_euclidean_distance(
    *,
    reference_df: Any,
    target_df: Any,
) -> List[float]:
    if str(reference_df.columns[-1]) != "time" or str(target_df.columns[-1]) != "time":
        raise ValueError("last column for reference_df and target_df must be time")
    if [str(column) for column in reference_df.columns] != [
        str(column) for column in target_df.columns
    ]:
        raise ValueError("reference_df and target_df columns must match")
    reference_time = np.asarray(reference_df["time"], dtype=float)
    target_time = np.asarray(target_df["time"], dtype=float)
    if reference_time.shape != target_time.shape or not np.allclose(
        reference_time,
        target_time,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("reference_df and target_df time columns must be identical")
    distances: List[float] = []
    for column in reference_df.columns[:-1]:
        reference_values = np.asarray(reference_df[column], dtype=float)
        target_values = np.asarray(target_df[column], dtype=float)
        valid = np.isfinite(reference_values) & np.isfinite(target_values)
        if not np.any(valid):
            distances.append(0.0)
            continue
        delta = target_values[valid] - reference_values[valid]
        distances.append(float(np.sqrt(np.sum(np.square(delta, dtype=np.float64)))))
    return distances


def _detectability_arrays(
    df: Any,
) -> Optional[tuple[np.ndarray, List[str], np.ndarray]]:
    key = id(df)
    cached = _identity_cache_get(_DETECTABILITY_ARRAY_CACHE, key, df)
    if cached is not _CACHE_MISS:
        return cached
    try:
        columns = [str(column) for column in df.columns]
    except Exception:
        return _identity_cache_set(
            _DETECTABILITY_ARRAY_CACHE,
            key,
            df,
            None,
            max_items=_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS,
        )
    if not columns or columns[-1] != "time":
        return _identity_cache_set(
            _DETECTABILITY_ARRAY_CACHE,
            key,
            df,
            None,
            max_items=_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS,
        )
    signal_columns = columns[:-1]
    if not signal_columns:
        return _identity_cache_set(
            _DETECTABILITY_ARRAY_CACHE,
            key,
            df,
            None,
            max_items=_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS,
        )
    try:
        time = np.asarray(df["time"], dtype=float)
        values = np.asarray(df[signal_columns], dtype=float)
    except Exception:
        return _identity_cache_set(
            _DETECTABILITY_ARRAY_CACHE,
            key,
            df,
            None,
            max_items=_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS,
        )
    out = (
        (time, signal_columns, values)
        if time.ndim == 1
        and values.ndim == 2
        and values.shape[0] == time.shape[0]
        and np.all(np.isfinite(time))
        else None
    )
    return _identity_cache_set(
        _DETECTABILITY_ARRAY_CACHE,
        key,
        df,
        out,
        max_items=_DETECTABILITY_ARRAY_CACHE_MAX_ITEMS,
    )


def _fast_first_consecutive_true_index(
    mask: np.ndarray,
    *,
    minimum_steps: int,
) -> Optional[int]:
    mask = np.asarray(mask, dtype=bool)
    if minimum_steps <= 1:
        if not np.any(mask):
            return None
        return int(np.argmax(mask))
    if mask.size < minimum_steps:
        return None
    window = np.ones(int(minimum_steps), dtype=np.int16)
    counts = np.convolve(mask.astype(np.int16, copy=False), window, mode="valid")
    hits = np.flatnonzero(counts == int(minimum_steps))
    return int(hits[0]) if hits.size else None


def _fast_detectability_payload_from_clean_frames(
    *,
    baseline_df: Any,
    run_df: Any,
    first_detectable_minimum_symmetric_distance: float,
    first_detectable_epsilon: float,
    minimum_consecutive_srd_steps: int,
    intervention_time: object,
    signal_detectability_specs: Mapping[str, Mapping[str, object]],
    RMS_thresholds: Mapping[str, float],
) -> Dict[str, Any]:
    baseline_arrays = _detectability_arrays(baseline_df)
    run_arrays = _detectability_arrays(run_df)
    if baseline_arrays is None or run_arrays is None:
        return _detectability_summary_entry(detectable="error", max_srd=[], first_diff=[])
    baseline_time, baseline_signals, baseline_values = baseline_arrays
    run_time, run_signals, run_values = run_arrays
    if (
        baseline_signals != run_signals
        or baseline_time.shape != run_time.shape
        or baseline_values.shape != run_values.shape
        or not np.array_equal(baseline_time, run_time)
    ):
        return _detectability_summary_entry(detectable="error", max_srd=[], first_diff=[])

    threshold = float(first_detectable_minimum_symmetric_distance)
    epsilon = float(first_detectable_epsilon)
    try:
        intervention_time_value = float(intervention_time)
    except (TypeError, ValueError):
        intervention_time_value = 0.0
    if not np.isfinite(intervention_time_value):
        intervention_time_value = 0.0
    time_after_intervention = baseline_time >= intervention_time_value

    max_srd: List[float] = []
    first_diff: List[Optional[float]] = []
    euclidean_distance: List[float] = []
    for col_idx, signal_name in enumerate(baseline_signals):
        signal_spec = signal_detectability_specs.get(signal_name)
        signal_threshold = (
            float(signal_spec.get("min_srd_distance"))
            if isinstance(signal_spec, Mapping) and "min_srd_distance" in signal_spec
            else threshold
        )
        signal_epsilon = (
            float(signal_spec.get("epsilon_SRD"))
            if isinstance(signal_spec, Mapping) and "epsilon_SRD" in signal_spec
            else epsilon
        )
        signal_consecutive_steps = (
            max(1, int(signal_spec.get("minimum_consecutive_srd_steps")))
            if isinstance(signal_spec, Mapping)
            and "minimum_consecutive_srd_steps" in signal_spec
            else max(1, int(minimum_consecutive_srd_steps))
        )
        baseline_column = baseline_values[:, col_idx]
        run_column = run_values[:, col_idx]
        denominator = np.abs(baseline_column) + np.abs(run_column) + signal_epsilon
        with np.errstate(divide="ignore", invalid="ignore"):
            column_distance = 2.0 * np.abs(baseline_column - run_column) / denominator
        valid = (
            np.isfinite(baseline_column)
            & np.isfinite(run_column)
            & np.isfinite(column_distance)
        )
        finite_distance = column_distance[valid]
        max_srd.append(
            0.0 if finite_distance.size == 0 else float(np.max(finite_distance))
        )
        delta = run_column - baseline_column
        finite_delta = delta[valid]
        euclidean_distance.append(
            0.0
            if finite_delta.size == 0
            else float(np.sqrt(np.sum(np.square(finite_delta, dtype=np.float64))))
        )
        detectable_mask = (
            valid
            & time_after_intervention
            & (column_distance >= signal_threshold)
        )
        first_idx = _fast_first_consecutive_true_index(
            detectable_mask,
            minimum_steps=signal_consecutive_steps,
        )
        first_diff.append(None if first_idx is None else float(baseline_time[first_idx]))

    rms_passes = False
    for signal_name, distance in zip(baseline_signals, euclidean_distance):
        if signal_name not in RMS_thresholds:
            continue
        try:
            rms_passes = float(distance) > float(RMS_thresholds[signal_name])
        except (TypeError, ValueError):
            rms_passes = False
        if rms_passes:
            break
    detectable = "yes" if rms_passes else "no"
    return _detectability_summary_entry(
        detectable=detectable,
        max_srd=max_srd,
        first_diff=first_diff,
        euclidean_distance=euclidean_distance,
        mean_euclidean_distance_clean_dirty=[0.0 for _ in euclidean_distance],
        mean_euclidean_distance_clean_baseline=euclidean_distance,
        mean_SNR=[None for _ in euclidean_distance],
    )


def _compute_detectability_metrics(
    *,
    model_id: Optional[str] = None,
    baseline_df: Any,
    run_df: Any,
    time0_df: Optional[Any],
    time0_expected: bool,
    min_srd_distance: float,
    epsilon_SRD: float,
    minimum_consecutive_srd_steps: int = 1,
    intervention_time: object = 0.0,
    run_parameters: Optional[Mapping[str, Any]] = None,
    env_detectability_path: Optional[Path] = None,
    noise_adder_path: Optional[Path] = None,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    RMS_thresholds: Optional[Mapping[str, float]] = None,
    compute_noise_summary: bool = True,
    skip_env_when_not_detectable: bool = False,
) -> Dict[str, Any]:
    try:
        clean_baseline_df = _preprocess_detectability_frame(
            baseline_df,
            model_id=model_id,
        )
        clean_run_df = _preprocess_detectability_frame(
            run_df,
            model_id=model_id,
        )
        if compute_noise_summary:
            (
                _noisy_baseline_seed0,
                _noisy_run_seed0,
                mean_dirty_baseline,
                mean_clean_baseline,
                mean_snr_baseline,
            ) = _detectability_noise_distance_summary(
                model_id=model_id,
                baseline_df=baseline_df,
                run_df=run_df,
                noise_baseline_df=baseline_df,
                clean_baseline_df=baseline_df,
                noise_adder_path=noise_adder_path,
            )
        if compute_noise_summary:
            baseline_detectability = compute_detectability_baseline(
                baseline_df=clean_baseline_df,
                run_df=clean_run_df,
                first_detectable_minimum_symmetric_distance=(
                    min_srd_distance
                ),
                first_detectable_epsilon=epsilon_SRD,
                minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                intervention_time=intervention_time,
                signal_detectability_specs=signal_detectability_specs,
                require_signal_detectability_specs=bool(signal_detectability_specs),
                mean_euclidean_distance_clean_dirty=mean_dirty_baseline,
                mean_euclidean_distance_clean_baseline=mean_clean_baseline,
                mean_SNR=mean_snr_baseline,
                RMS_thresholds=RMS_thresholds,
            )
        else:
            baseline_detectability = _fast_detectability_payload_from_clean_frames(
                baseline_df=clean_baseline_df,
                run_df=clean_run_df,
                first_detectable_minimum_symmetric_distance=min_srd_distance,
                first_detectable_epsilon=epsilon_SRD,
                minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                intervention_time=intervention_time,
                signal_detectability_specs=signal_detectability_specs or {},
                RMS_thresholds=RMS_thresholds or {},
            )
        baseline_detectability = _with_environment_specific_detectability(
            baseline_detectability,
            run_df=clean_run_df,
            clean_df=clean_baseline_df,
            run_parameters=run_parameters or {},
            intervention_time=intervention_time,
            env_detectability_path=env_detectability_path,
            skip_when_not_detectable=skip_env_when_not_detectable,
        )
    except Exception:
        baseline_detectability = _with_environment_specific_detectability(
            _detectability_summary_entry(
                detectable="error",
                max_srd=[],
                first_diff=[],
            ),
            run_df=run_df,
            clean_df=baseline_df,
            run_parameters=run_parameters or {},
            intervention_time=intervention_time,
            env_detectability_path=env_detectability_path,
            skip_when_not_detectable=skip_env_when_not_detectable,
        )

    time0_detectability: Dict[str, Any] = {
        "environment_specific_detectability": "error",
        "max_SRD_detectability": "error",
        "detectability": "no",
        "detectable": "no",
        "max_SRD": [],
        "euclidean_distance": [],
        "mean_euclidean_distance_clean_dirty": [],
        "mean_euclidean_distance_clean_baseline": [],
        "mean_SNR": [],
        "first_diff": [],
    }
    if time0_expected:
        if time0_df is None:
            time0_detectability = {
                "environment_specific_detectability": "error",
                "max_SRD_detectability": "error",
                "detectability": "no",
                "detectable": "no",
                "max_SRD": [],
                "euclidean_distance": [],
                "mean_euclidean_distance_clean_dirty": [],
                "mean_euclidean_distance_clean_baseline": [],
                "mean_SNR": [],
                "first_diff": [],
            }
        else:
            try:
                clean_time0_df = _preprocess_detectability_frame(
                    time0_df,
                    model_id=model_id,
                )
                clean_run_df = _preprocess_detectability_frame(
                    run_df,
                    model_id=model_id,
                )
                if compute_noise_summary:
                    (
                        _noisy_time0_seed0,
                        _noisy_run_time0_seed0,
                        mean_dirty_time0,
                        mean_clean_time0,
                        mean_snr_time0,
                    ) = _detectability_noise_distance_summary(
                        model_id=model_id,
                        baseline_df=time0_df,
                        run_df=run_df,
                        noise_baseline_df=time0_df,
                        clean_baseline_df=time0_df,
                        noise_adder_path=noise_adder_path,
                    )
                if compute_noise_summary:
                    time0_detectability = compute_detectability_time0_baseline(
                        baseline_df=clean_time0_df,
                        run_df=clean_run_df,
                        first_detectable_minimum_symmetric_distance=(
                            min_srd_distance
                        ),
                        first_detectable_epsilon=epsilon_SRD,
                        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                        intervention_time=intervention_time,
                        signal_detectability_specs=signal_detectability_specs,
                        require_signal_detectability_specs=bool(signal_detectability_specs),
                        mean_euclidean_distance_clean_dirty=mean_dirty_time0,
                        mean_euclidean_distance_clean_baseline=mean_clean_time0,
                        mean_SNR=mean_snr_time0,
                        RMS_thresholds=RMS_thresholds,
                    )
                else:
                    time0_detectability = _fast_detectability_payload_from_clean_frames(
                        baseline_df=clean_time0_df,
                        run_df=clean_run_df,
                        first_detectable_minimum_symmetric_distance=min_srd_distance,
                        first_detectable_epsilon=epsilon_SRD,
                        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                        intervention_time=intervention_time,
                        signal_detectability_specs=signal_detectability_specs or {},
                        RMS_thresholds=RMS_thresholds or {},
                    )
            except Exception:
                time0_detectability = {
                    "environment_specific_detectability": "error",
                    "max_SRD_detectability": "error",
                    "detectability": "no",
                    "detectable": "error",
                    "max_SRD": [],
                    "euclidean_distance": [],
                    "mean_euclidean_distance_clean_dirty": [],
                    "mean_euclidean_distance_clean_baseline": [],
                    "mean_SNR": [],
                    "first_diff": [],
                }

    return {
        "vs_baseline": baseline_detectability,
        "vs_time0_baseline": time0_detectability,
    }


def _detectability_summary_entry(
    *,
    detectable: object,
    max_srd: Optional[object],
    first_diff: Optional[object],
    environment_specific_detectability: object = None,
    max_srd_detectability: object = None,
    detectability: object = None,
    euclidean_distance: Optional[object] = None,
    mean_euclidean_distance_clean_dirty: Optional[object] = None,
    mean_euclidean_distance_clean_baseline: Optional[object] = None,
    mean_SNR: Optional[object] = None,
) -> Dict[str, Any]:
    max_status = str(
        max_srd_detectability
        if max_srd_detectability is not None
        else detectable
    ).strip().lower() or "error"
    env_status = str(
        environment_specific_detectability
        if environment_specific_detectability is not None
        else ("yes" if max_status == "yes" else "no")
    ).strip().lower() or "error"
    final_status = str(
        detectability
        if detectability is not None
        else ("yes" if max_status == "yes" and env_status == "yes" else "no")
    ).strip().lower() or "no"
    return {
        "environment_specific_detectability": env_status,
        "max_SRD_detectability": max_status,
        "detectability": "yes" if final_status == "yes" else "no",
        "detectable": "yes" if final_status == "yes" else "no",
        "max_SRD": list(max_srd or []),
        "euclidean_distance": list(euclidean_distance or []),
        "mean_euclidean_distance_clean_dirty": list(
            mean_euclidean_distance_clean_dirty or []
        ),
        "mean_euclidean_distance_clean_baseline": list(
            mean_euclidean_distance_clean_baseline or []
        ),
        "mean_SNR": list(mean_SNR)
        if isinstance(mean_SNR, list)
        else _signal_to_noise_ratio_db_values(
            mean_euclidean_distance_clean_dirty=mean_euclidean_distance_clean_dirty,
            mean_euclidean_distance_clean_baseline=mean_euclidean_distance_clean_baseline,
            length=len(list(first_diff or [])),
        ),
        "first_diff": list(first_diff or []),
    }


def _signal_to_noise_ratio_db_values(
    *,
    mean_euclidean_distance_clean_dirty: Optional[object],
    mean_euclidean_distance_clean_baseline: Optional[object],
    length: Optional[int] = None,
) -> List[Optional[float]]:
    dirty_distances = (
        list(mean_euclidean_distance_clean_dirty)
        if isinstance(mean_euclidean_distance_clean_dirty, list)
        else []
    )
    baseline_distances = (
        list(mean_euclidean_distance_clean_baseline)
        if isinstance(mean_euclidean_distance_clean_baseline, list)
        else []
    )
    if length is None:
        length = max(len(dirty_distances), len(baseline_distances))
    out: List[Optional[float]] = []
    for idx in range(length):
        try:
            noise = float(dirty_distances[idx])
            signal_norm = float(baseline_distances[idx])
        except (IndexError, TypeError, ValueError):
            out.append(None)
            continue
        if not (np.isfinite(noise) and np.isfinite(signal_norm)):
            out.append(None)
            continue
        if noise <= 0.0 or signal_norm <= 0.0:
            out.append(None)
            continue
        snr_db = float(20.0 * np.log10(signal_norm / noise))
        out.append(snr_db if np.isfinite(snr_db) else None)
    return out


def _detectability_output_from_status(status: Mapping[str, Any]) -> Dict[str, Any]:
    first_diff = list(status.get("first_diff") or [])
    clean_dirty = list(status.get("mean_euclidean_distance_clean_dirty") or [])
    clean_baseline = list(status.get("mean_euclidean_distance_clean_baseline") or [])
    snr = list(status.get("mean_SNR") or [])
    if first_diff and not clean_dirty:
        clean_dirty = [0.0 for _ in first_diff]
    if first_diff and not clean_baseline:
        clean_baseline = list(status.get("euclidean_distance") or [])
    if len(clean_baseline) < len(first_diff):
        clean_baseline = [
            *clean_baseline,
            *([0.0] * (len(first_diff) - len(clean_baseline))),
        ]
    if len(clean_dirty) < len(first_diff):
        clean_dirty = [
            *clean_dirty,
            *([0.0] * (len(first_diff) - len(clean_dirty))),
        ]
    if len(snr) < len(first_diff):
        computed_snr = _signal_to_noise_ratio_db_values(
            mean_euclidean_distance_clean_dirty=clean_dirty,
            mean_euclidean_distance_clean_baseline=clean_baseline,
            length=len(first_diff),
        )
        snr = [*(snr or []), *computed_snr[len(snr) :]]
    return {
        "mean_euclidean_distance_clean_dirty": clean_dirty[: len(first_diff)],
        "mean_euclidean_distance_clean_baseline": clean_baseline[: len(first_diff)],
        "mean_SNR": snr[: len(first_diff)],
        "first_diff": first_diff,
    }


def _documented_baseline_detectability_entry(
    status: Mapping[str, Any],
) -> Dict[str, Any]:
    env_status = str(
        status.get("environment_specific_detectability") or "error"
    ).strip().lower()
    if env_status not in {"yes", "no", "error"}:
        env_status = "error"
    detectable = str(status.get("detectable") or "error").strip().lower()
    if str(status.get("max_SRD_detectability") or "").strip().lower() == "error":
        detectable = "error"
    if detectable not in {"yes", "no", "error"}:
        detectable = "error"
    return {
        "environment_specific_detectability": env_status,
        "detectable": detectable,
        "detectability_output": _detectability_output_from_status(status),
    }


def _documented_time0_detectability_entry(
    status: Mapping[str, Any],
) -> Dict[str, Any]:
    detectable = str(status.get("detectable") or "error").strip().lower()
    if str(status.get("max_SRD_detectability") or "").strip().lower() == "error":
        detectable = "error"
    if detectable not in {"yes", "no", "error"}:
        detectable = "error"
    return {
        "detectable": detectable,
        "detectability_output": _detectability_output_from_status(status),
    }


def _documented_detectability_summary(
    detectability_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    vs_baseline = detectability_metrics.get("vs_baseline")
    if not isinstance(vs_baseline, Mapping):
        vs_baseline = _detectability_summary_entry(
            detectable="error",
            max_srd=[],
            first_diff=[],
        )
    vs_time0 = detectability_metrics.get("vs_time0_baseline")
    if not isinstance(vs_time0, Mapping):
        vs_time0 = _detectability_summary_entry(
            detectable="error",
            max_srd=[],
            first_diff=[],
        )
    return {
        "vs_baseline": _documented_baseline_detectability_entry(vs_baseline),
        "vs_time0_baseline": _documented_time0_detectability_entry(vs_time0),
    }


def _detectability_status(value: object) -> str:
    if not isinstance(value, dict):
        return str(value or "").strip().lower()
    for key in (
        "detectability",
        "detectable",
        "max_SRD_detectability",
        "max_simmetric_detectability",
    ):
        text = str(value.get(key) or "").strip().lower()
        if text:
            return text
    return ""


def _max_symmetric_status(value: Mapping[str, Any]) -> str:
    return str(
        value.get("max_SRD_detectability")
        or value.get("max_simmetric_detectability")
        or value.get("detectable")
        or value.get("detectability")
        or ""
    ).strip().lower()


def _coerce_env_detectability_status(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, dict):
        for key in (
            "environment_specific_detectability",
            "enviroment_specific_detectability",
            "detectability",
            "detectable",
        ):
            if key in value:
                return _coerce_env_detectability_status(value.get(key))
        return "error"
    text = str(value or "").strip().lower()
    return text if text in {"yes", "no", "error"} else "error"


def _load_env_detectability_fn(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return None
    key = str(resolved)
    if key in _ENV_DETECTABILITY_FN_CACHE:
        return _ENV_DETECTABILITY_FN_CACHE[key]
    module_name = f"_tsenv_detectability_{hashlib.md5(key.encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "is_detectable", None)
    if not callable(fn):
        return None
    _ENV_DETECTABILITY_FN_CACHE[key] = fn
    return fn


def _ball_drop_env_column(df: Any, *names: str) -> Optional[str]:
    columns = {str(column): str(column) for column in getattr(df, "columns", [])}
    for name in names:
        if name in columns:
            return columns[name]
    non_time = [
        str(column)
        for column in getattr(df, "columns", [])
        if str(column).strip().lower() != "time"
    ]
    if names and names[-1].startswith("col"):
        try:
            idx = int(names[-1][3:]) - 1
        except ValueError:
            idx = -1
        if 0 <= idx < len(non_time):
            return non_time[idx]
    return None


def _finite_array_or_none(df: Any, column: Optional[str]) -> Optional[np.ndarray]:
    if column is None or column not in getattr(df, "columns", []):
        return None
    try:
        values = np.asarray(df[column], dtype=float)
    except Exception:
        return None
    return values if np.all(np.isfinite(values)) else None


def _ball_drop_env_features(df: Any) -> Optional[Dict[str, np.ndarray]]:
    key = id(df)
    cached = _identity_cache_get(_BALL_DROP_ENV_FEATURE_CACHE, key, df)
    if cached is not _CACHE_MISS:
        return cached
    time = _finite_array_or_none(df, "time")
    height = _finite_array_or_none(df, _ball_drop_env_column(df, "Position", "col1"))
    velocity = _finite_array_or_none(df, _ball_drop_env_column(df, "Velocity", "col2"))
    force = _finite_array_or_none(df, _ball_drop_env_column(df, "Hard_Stop_f", "col3"))
    features = (
        {"time": time, "height": height, "velocity": velocity, "force": force}
        if time is not None
        and height is not None
        and velocity is not None
        and force is not None
        and len(time) == len(height) == len(velocity) == len(force)
        else None
    )
    return _identity_cache_set(
        _BALL_DROP_ENV_FEATURE_CACHE,
        key,
        df,
        features,
        max_items=_BALL_DROP_ENV_FEATURE_CACHE_MAX_ITEMS,
    )


def _fast_ball_drop_first_apex_height_after(
    features: Dict[str, np.ndarray],
    target_time: float,
) -> Optional[float]:
    time = features["time"]
    height = features["height"]
    velocity = features["velocity"]
    if len(height) < 3:
        return None
    row_idx = int(np.argmin(np.abs(time - target_time)))
    start = max(1, row_idx + 1)
    if start < len(velocity):
        idx = np.arange(start, len(velocity))
        mask = (time[idx] > target_time) & (velocity[idx - 1] > 0.0) & (velocity[idx] <= 0.0)
        hits = idx[mask]
        if hits.size:
            hit = int(hits[0])
            return float(max(height[hit - 1], height[hit]))
    if start < len(height) - 1:
        idx = np.arange(start, len(height) - 1)
        mask = (
            (time[idx] > target_time)
            & (height[idx] >= height[idx - 1])
            & (height[idx] >= height[idx + 1])
        )
        hits = idx[mask]
        if hits.size:
            return float(height[int(hits[0])])
    return None


def _fast_ball_drop_bounce_times(features: Dict[str, np.ndarray]) -> np.ndarray:
    time = features["time"]
    velocity = features["velocity"]
    force = features["force"]
    if len(time) < 2:
        return np.asarray([], dtype=float)
    idx = np.flatnonzero(
        (force[1:] > _BALL_DROP_IMPACT_FORCE_THRESHOLD)
        & (velocity[:-1] < -_BALL_DROP_MIN_INCOMING_BOUNCE_SPEED)
        & (velocity[1:] > _BALL_DROP_MIN_REBOUND_SPEED)
    ) + 1
    if idx.size == 0:
        return np.asarray([], dtype=float)
    out: List[float] = []
    for raw_time in time[idx]:
        value = float(raw_time)
        if out and value - out[-1] < 0.03:
            continue
        out.append(value)
    return np.asarray(out, dtype=float)


def _fast_ball_drop_windowed_values(
    features: Dict[str, np.ndarray],
    *,
    column: str,
    start_time: float,
    end_time: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    time = features["time"]
    values = features[column]
    if (
        time.size < 2
        or start_time < float(time[0]) - _BALL_DROP_DRAG_DTW_EUCLIDEAN_TIME_TOLERANCE
        or end_time > float(time[-1]) + _BALL_DROP_DRAG_DTW_EUCLIDEAN_TIME_TOLERANCE
        or end_time <= start_time
    ):
        return None
    clipped_start = float(time[0]) if start_time < float(time[0]) else start_time
    clipped_end = float(time[-1]) if end_time > float(time[-1]) else end_time
    inside = time[(time > clipped_start) & (time < clipped_end)]
    grid = np.unique(np.concatenate(([clipped_start, clipped_end], inside)))
    if grid.size < 2:
        return None
    return grid, np.interp(grid, time, values)


def _fast_ball_drop_normalized_dtw_euclidean(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> Optional[float]:
    if left_values.size == 0 or right_values.size == 0:
        return None
    cols = int(right_values.size)
    previous_costs = np.full(cols + 1, np.inf, dtype=float)
    previous_lengths = np.zeros(cols + 1, dtype=np.int32)
    previous_costs[0] = 0.0
    for left in left_values:
        current_costs = np.full(cols + 1, np.inf, dtype=float)
        current_lengths = np.zeros(cols + 1, dtype=np.int32)
        for col_idx, right in enumerate(right_values, start=1):
            candidates = (
                (previous_costs[col_idx], previous_lengths[col_idx]),
                (current_costs[col_idx - 1], current_lengths[col_idx - 1]),
                (previous_costs[col_idx - 1], previous_lengths[col_idx - 1]),
            )
            previous_cost, previous_length = min(candidates, key=lambda item: (item[0], item[1]))
            local_cost = abs(float(left) - float(right))
            if not np.isfinite(local_cost):
                return None
            current_costs[col_idx] = previous_cost + local_cost
            current_lengths[col_idx] = int(previous_length) + 1
        previous_costs = current_costs
        previous_lengths = current_lengths
    path_length = int(previous_lengths[cols])
    total_cost = float(previous_costs[cols])
    if path_length <= 0 or not np.isfinite(total_cost):
        return None
    return total_cost / float(path_length)


def _fast_ball_drop_drag_dtw_rule(
    *,
    run_features: Dict[str, np.ndarray],
    clean_features: Dict[str, np.ndarray],
    target_time: float,
) -> Optional[bool]:
    start_time = target_time
    end_time = start_time + _BALL_DROP_DRAG_DTW_EUCLIDEAN_WINDOW_S
    run_window = _fast_ball_drop_windowed_values(
        run_features,
        column="velocity",
        start_time=start_time,
        end_time=end_time,
    )
    clean_window = _fast_ball_drop_windowed_values(
        clean_features,
        column="velocity",
        start_time=start_time,
        end_time=end_time,
    )
    if run_window is None or clean_window is None:
        return None
    grid = np.unique(np.concatenate((run_window[0], clean_window[0])))
    run_values = np.interp(grid, run_window[0], run_window[1])
    clean_values = np.interp(grid, clean_window[0], clean_window[1])
    distance = _fast_ball_drop_normalized_dtw_euclidean(run_values, clean_values)
    if distance is None:
        return None
    return bool(
        distance
        > _BALL_DROP_DRAG_DTW_EUCLIDEAN_THRESHOLD
        + _BALL_DROP_DRAG_DTW_EUCLIDEAN_TOLERANCE
    )


def _fast_ball_drop_terminal_velocity_intervals(
    features: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    time = features["time"]
    height = features["height"]
    velocity = features["velocity"]
    if len(time) != len(height) or len(time) != len(velocity) or len(time) < 5:
        return None
    diffs = np.diff(time)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if positive_diffs.size == 0:
        return None
    dt = float(np.median(positive_diffs))
    if not np.isfinite(dt) or dt <= 0.0:
        return None
    height_span = float(np.max(height) - np.min(height)) if height.size else 0.0
    ground_tol = max(1.0e-3, 0.01 * height_span)
    acceleration = np.empty_like(velocity, dtype=float)
    acceleration[0] = (
        (velocity[1] - velocity[0]) / (time[1] - time[0])
        if time[1] > time[0]
        else np.nan
    )
    acceleration[-1] = (
        (velocity[-1] - velocity[-2]) / (time[-1] - time[-2])
        if time[-1] > time[-2]
        else np.nan
    )
    denom = time[2:] - time[:-2]
    middle = np.full(len(time) - 2, np.nan, dtype=float)
    valid_denom = denom > 0.0
    middle[valid_denom] = (velocity[2:] - velocity[:-2])[valid_denom] / denom[valid_denom]
    acceleration[1:-1] = middle

    finite_acceleration = np.abs(
        acceleration[
            np.isfinite(acceleration)
            & (height > ground_tol)
            & (velocity < -1.0e-6)
        ]
    )
    if finite_acceleration.size == 0:
        return np.empty((0, 2), dtype=float)
    finite_acceleration.sort()
    percentile_idx = min(
        finite_acceleration.size - 1,
        max(0, int(round(0.9 * (finite_acceleration.size - 1)))),
    )
    scale = max(1.0e-3, 0.05 * float(finite_acceleration[percentile_idx]))
    min_run_len = max(8, int(np.ceil(0.05 / dt)))
    candidate = (
        np.isfinite(time)
        & np.isfinite(height)
        & np.isfinite(velocity)
        & np.isfinite(acceleration)
        & (height > ground_tol)
        & (velocity < -1.0e-6)
        & (np.abs(acceleration) <= scale)
        & (np.abs(velocity) >= 0.25)
    )
    if not np.any(candidate):
        return np.empty((0, 2), dtype=float)
    padded = np.concatenate(([False], candidate, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops_exclusive = np.flatnonzero(padded[:-1] & ~padded[1:])
    intervals = [
        (float(time[start]), float(time[stop - 1]))
        for start, stop in zip(starts, stops_exclusive)
        if stop - start >= min_run_len
    ]
    return np.asarray(intervals, dtype=float) if intervals else np.empty((0, 2), dtype=float)


def _fast_ball_drop_terminal_velocity_rule(
    features: Dict[str, np.ndarray],
) -> Optional[bool]:
    intervals = _fast_ball_drop_terminal_velocity_intervals(features)
    if intervals is None:
        return None
    bounce_times = _fast_ball_drop_bounce_times(features)
    after_bounce = (
        bool(np.any(intervals[:, 0][:, None] > bounce_times[None, :]))
        if intervals.size and bounce_times.size
        else False
    )
    if after_bounce:
        return True
    time = features["time"]
    height = features["height"]
    if time.size == 0:
        return False
    end_time = float(np.max(time))
    for _start, stop in intervals:
        if float(stop) >= end_time:
            continue
        stop_height = float(np.interp(float(stop), time, height))
        if stop_height > _BALL_DROP_APEX_HEIGHT_THRESHOLD:
            return True
    return False


def _fast_combine_any(results: List[Optional[bool]]) -> Optional[bool]:
    if any(result is True for result in results):
        return True
    if any(result is None for result in results):
        return None
    return False


def _fast_ball_drop_environment_detectability(
    *,
    run_df: Any,
    clean_df: Any,
    run_parameters: Mapping[str, Any],
    min_first_diff: object,
) -> Optional[str]:
    parameter = str(
        run_parameters.get("intervention_parameter")
        or run_parameters.get("changed_parameter")
        or run_parameters.get("parameter")
        or run_parameters.get("variable")
        or ""
    ).strip().lower()
    requires_drag_rules = parameter in {"drag_coeff", "drag coefficient", "drag"}
    try:
        target_time = float(min_first_diff)
    except (TypeError, ValueError):
        return "error"
    if not np.isfinite(target_time):
        return "error"

    frames = [
        features
        for features in (
            _ball_drop_env_features(run_df),
            _ball_drop_env_features(clean_df),
        )
        if features is not None
    ]
    if not frames:
        return "error"

    apex_results: List[Optional[bool]] = []
    for features in frames:
        apex = _fast_ball_drop_first_apex_height_after(features, target_time)
        apex_results.append(
            None if apex is None else bool(apex > _BALL_DROP_APEX_HEIGHT_THRESHOLD)
        )
    if any(result is True for result in apex_results):
        apex_rule: Optional[bool] = True
    elif any(result is False for result in apex_results):
        apex_rule = False
    else:
        apex_rule = None

    bounce_times = [_fast_ball_drop_bounce_times(features) for features in frames]
    bounce_count_rule = (
        None if not bounce_times else all(len(times) < _BALL_DROP_MAX_BOUNCES for times in bounce_times)
    )
    results: List[Optional[bool]] = [apex_rule, bounce_count_rule]
    if requires_drag_rules:
        post_difference_bounce_rule = (
            None
            if not bounce_times
            else any(bool(np.any(times > target_time)) for times in bounce_times)
        )
        drag_terminal_rule = _fast_combine_any(
            [_fast_ball_drop_terminal_velocity_rule(features) for features in frames]
        )
        if drag_terminal_rule is True:
            drag_specific_rule = True
        else:
            drag_dtw_rule = (
                _fast_ball_drop_drag_dtw_rule(
                    run_features=frames[0],
                    clean_features=frames[1],
                    target_time=target_time,
                )
                if len(frames) >= 2
                else None
            )
            drag_specific_rule = _fast_combine_any(
                [drag_dtw_rule, drag_terminal_rule]
            )
        results.extend(
            [
                post_difference_bounce_rule,
                drag_specific_rule,
            ]
        )
    if parameter in {
        "drag_coeff",
        "drag coefficient",
        "drag",
        "mass",
        "restitution",
        "coefficient of restitution",
    }:
        before_after = any(
            bool(np.any(times < target_time)) and bool(np.any(times > target_time))
            for times in bounce_times
        )
        results.append(before_after)
    if any(result is None for result in results):
        return "error"
    if any(result is False for result in results):
        return "no"
    return "yes"


def _with_environment_specific_detectability(
    payload: Mapping[str, Any],
    *,
    run_df: Any,
    clean_df: Any,
    run_parameters: Mapping[str, Any],
    intervention_time: object,
    env_detectability_path: Optional[Path],
    skip_when_not_detectable: bool = False,
) -> Dict[str, Any]:
    out = dict(payload)
    max_status = _max_symmetric_status(out)
    min_first_diff = _min_first_diff(out.get("first_diff"))
    if max_status == "error":
        env_status = "error"
    elif skip_when_not_detectable and max_status == "no":
        env_status = "no"
    elif min_first_diff is None:
        env_status = "no"
    else:
        try:
            fast_status = (
                _fast_ball_drop_environment_detectability(
                    run_df=run_df,
                    clean_df=clean_df,
                    run_parameters=run_parameters,
                    min_first_diff=min_first_diff,
                )
                if skip_when_not_detectable
                else None
            )
            if fast_status is not None:
                env_status = fast_status
            else:
                fn = _load_env_detectability_fn(env_detectability_path)
                if fn is None:
                    env_status = "error"
                else:
                    env_status = _coerce_env_detectability_status(
                        fn(
                            run_df,
                            clean_df,
                            dict(run_parameters),
                            intervention_time,
                            min_first_diff,
                        )
                    )
        except Exception:
            env_status = "error"
    out["environment_specific_detectability"] = env_status
    out["max_SRD_detectability"] = (
        max_status if max_status in {"yes", "no", "error"} else "error"
    )
    if out["max_SRD_detectability"] == "error" or env_status == "error":
        final_status = "error"
    elif env_status == "no":
        final_status = "no"
    else:
        final_status = "yes" if out["max_SRD_detectability"] == "yes" else "no"
    out["detectability"] = final_status
    out["detectable"] = final_status
    return out


def _build_run_parameters(
    *,
    baseline_parameters: Mapping[str, Any],
    raw_intervention: Mapping[str, Any],
) -> Dict[str, Any]:
    run_parameters = dict(baseline_parameters)
    parameter = str(
        raw_intervention.get("parameter")
        or raw_intervention.get("variable")
        or ""
    ).strip()
    if parameter:
        run_parameters["intervention_parameter"] = parameter
        if "set_value" in raw_intervention:
            run_parameters[parameter] = raw_intervention.get("set_value")
        elif "value" in raw_intervention:
            run_parameters[parameter] = raw_intervention.get("value")
    if "intervention_time" in raw_intervention:
        run_parameters["intervention_time"] = raw_intervention.get("intervention_time")
    return run_parameters


def _empty_rule_outcome_summary() -> Dict[str, Any]:
    return {
        "ground_truth": None,
    }


def _rule_result_from_payload(payload: object) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {
            "ground_truth": None,
            "predicted_label": None,
            "is_correct": False,
        }
    ground_truth = payload.get("ground_truth")
    if ground_truth is None:
        ground_truth = payload.get("truth_label")
    predicted_label = payload.get("predicted_label")
    return {
        "ground_truth": None if ground_truth is None else str(ground_truth),
        "predicted_label": None if predicted_label is None else str(predicted_label),
        "is_correct": bool(payload.get("is_correct") is True),
    }


def _add_perfect_rule_fields(
    summary: Dict[str, Any],
    clean_result: object,
) -> Dict[str, Any]:
    if not isinstance(clean_result, dict):
        summary["is_perfect_correct"] = False
        summary["is_perfect_predicted_label"] = None
        return summary
    predicted_label = clean_result.get("predicted_label")
    summary["is_perfect_correct"] = clean_result.get("is_correct") is True
    summary["is_perfect_predicted_label"] = (
        None if predicted_label is None else str(predicted_label)
    )
    return summary


def _rule_accuracy_from_payloads(payloads: List[object]) -> float:
    if not payloads:
        return 0.0
    correct = sum(
        1 for payload in payloads
        if isinstance(payload, dict) and payload.get("is_correct") is True
    )
    return float(correct) / float(len(payloads))


def _chance_corrected_accuracy(*, accuracy: float, num_classes: int) -> float:
    k = int(num_classes)
    if k <= 1:
        return float(accuracy)
    chance = 1.0 / float(k)
    return float((float(accuracy) - chance) / (1.0 - chance))


def _numeric_or_none(value: object) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _aggregate_seed_noise_values(values_by_seed: List[object]) -> Dict[str, List[Optional[float]]]:
    sequences: List[List[Optional[float]]] = []
    for raw_values in values_by_seed:
        if isinstance(raw_values, (list, tuple)):
            sequences.append([_numeric_or_none(value) for value in raw_values])
    if not sequences:
        return {"mean_snr": [], "std_snr": []}
    width = max(len(values) for values in sequences)
    mean_values: List[Optional[float]] = []
    std_values: List[Optional[float]] = []
    for idx in range(width):
        samples = [
            values[idx]
            for values in sequences
            if idx < len(values) and values[idx] is not None
        ]
        if not samples:
            mean_values.append(None)
            std_values.append(None)
            continue
        arr = np.asarray(samples, dtype=np.float64)
        mean_values.append(float(np.mean(arr)))
        std_values.append(float(np.std(arr)))
    return {
        "mean_snr": mean_values,
        "std_snr": std_values,
    }


def _aggregate_noise_analysis_by_profile(
    profile_payloads: Mapping[str, List[object]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for scope in ("global", "local"):
        scope_payload: Dict[str, Any] = {}
        for profile in ("low", "high"):
            seed_payloads = profile_payloads.get(profile) or []
            values_by_seed = [
                payload.get(scope)
                for payload in seed_payloads
                if isinstance(payload, dict) and scope in payload
            ]
            if values_by_seed:
                scope_payload[profile] = _aggregate_seed_noise_values(values_by_seed)
        if scope_payload:
            out[scope] = scope_payload
    return out


def _build_noise_analysis_summary(
    *,
    child_df: Any,
    baseline_df: Any = None,
    noise_adder_path: Optional[Path],
    local_time: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if noise_adder_path is None:
        return None
    try:
        add_noise = load_noise_adder_from_path(noise_adder_path)
    except Exception:
        return None
    noise_payloads_by_profile: Dict[str, List[object]] = {"low": [], "high": []}
    for profile in ("low", "high"):
        for seed in _NOISE_SWEEP_SEEDS:
            try:
                _noisy_df, seed_noise_analysis = call_noise_adder(
                    add_noise,
                    child_df,
                    baseline_df=baseline_df,
                    first_diff=float(local_time) if local_time is not None else -1.0,
                    seed=int(seed),
                    noise_level=profile,
                )
                noise_payloads_by_profile[profile].append(seed_noise_analysis)
            except Exception:
                continue
    aggregated = _aggregate_noise_analysis_by_profile(noise_payloads_by_profile)
    return aggregated or None


def _anonymize_columns_for_rule(df: Any) -> Any:
    if not hasattr(df, "copy") or not hasattr(df, "columns"):
        return df
    out = df.copy()
    out.columns = [f"col{idx + 1}" for idx, _ in enumerate(out.columns)]
    return out


def _local_noise_analysis_time(detectability_metrics: Dict[str, Any]) -> Optional[float]:
    entry = detectability_metrics.get("vs_baseline")
    if not isinstance(entry, dict):
        return None
    if _detectability_status(entry) != "yes":
        return None
    output = entry.get("detectability_output")
    raw_values = (
        output.get("first_diff")
        if isinstance(output, dict)
        else entry.get("first_diff")
    )
    values = raw_values if isinstance(raw_values, list) else []
    candidates: List[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed >= 0.0:
            candidates.append(parsed)
    if candidates:
        return min(candidates)
    return None


def _min_first_diff(first_diff: object) -> Optional[float]:
    values = first_diff if isinstance(first_diff, list) else [first_diff]
    candidates: List[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed >= 0.0:
            candidates.append(parsed)
    return min(candidates) if candidates else None


def _build_high_context_rule_labels(
    *,
    model_id: str,
    truth_parameter: str,
    models_root: Path,
) -> Dict[str, Any]:
    allowed_params = set(
        load_allowed_interventions(model_id=str(model_id), models_root=models_root)
    )
    truth_internal, truth_display = evaluate_rule_workflow._resolve_truth_labels(
        model_id=str(model_id),
        truth_label=str(truth_parameter),
        models_root=models_root,
        allowed_params=allowed_params,
    )
    param_to_display = {
        parameter: str(
            parameter_key_to_display_label(
                model_id=str(model_id),
                parameter_key=parameter,
                models_root=models_root,
            )
            or parameter
        ).strip()
        for parameter in allowed_params
    }
    param_to_display[_NO_CHANGE_CLASS] = _NO_CHANGE_CLASS
    display_to_param = _build_display_label_to_parameter_key_map(
        model_id=str(model_id),
        models_root=models_root,
        allowed_params=allowed_params,
    )
    for label in _LEGACY_NO_CHANGE_ALIASES:
        display_to_param[_normalize_label(label)] = _NO_CHANGE_CLASS
    return {
        "truth_internal": truth_internal,
        "truth_display": truth_display,
        "param_to_display": param_to_display,
        "display_to_param": display_to_param,
    }


def _canonical_prediction_for_high_context(
    *,
    prediction_label: object,
    labels: Dict[str, Any],
) -> str:
    label = str(prediction_label or "").strip()
    if not label:
        raise ValueError("basic_rule.py returned an empty canonical prediction.")
    display_to_param = labels.get("display_to_param")
    param_to_display = labels.get("param_to_display")
    if not isinstance(display_to_param, dict) or not isinstance(param_to_display, dict):
        raise ValueError("Internal error: rule label mapping is not initialized.")

    normalized_label = _normalize_label(label)
    if normalized_label in display_to_param:
        key = display_to_param[normalized_label]
        return str(param_to_display.get(key) or label).strip()
    if label in param_to_display:
        return str(param_to_display[label]).strip()

    truth_display = str(labels.get("truth_display") or "").strip()
    if normalized_label == _normalize_label(truth_display):
        return truth_display
    allowed = sorted(str(value) for value in param_to_display.values())
    raise ValueError(f"Prediction '{label}' is not in allowed labels {allowed}")


def _evaluate_rule_on_loaded_dataframe(
    *,
    model_id: str,
    run_id: str,
    child_df: Any,
    baseline_df: Any = None,
    rule_path: Path,
    noise_adder_path: Optional[Path],
    models_root: Path,
    noise_level: str,
    seed: int,
    labels: Dict[str, Any],
    first_diff: float = -1.0,
) -> Dict[str, Any]:
    truth_display = str(labels.get("truth_display") or "").strip()
    try:
        noisy_df = evaluate_rule_workflow._apply_noise_profile(
            child_df=child_df,
            baseline_df=baseline_df,
            model_id=str(model_id),
            models_root=models_root,
            noise_level=str(noise_level),
            seed=int(seed),
            noise_adder_path=noise_adder_path,
            first_diff=float(first_diff),
        )
        rule_df = _anonymize_columns_for_rule(noisy_df)
        predict = _load_rule_predict_fn(rule_path)
        raw_prediction = predict(rule_df)
        predicted_label_raw, _change_time = _extract_rule_prediction(raw_prediction)
        predicted_label = _canonical_prediction_for_high_context(
            prediction_label=predicted_label_raw,
            labels=labels,
        )
        return {
            "ground_truth": truth_display,
            "predicted_label": predicted_label,
            "is_correct": _normalize_label(predicted_label) == _normalize_label(truth_display),
        }
    except Exception:
        return {
            "ground_truth": truth_display or None,
            "predicted_label": None,
            "is_correct": False,
        }


def _build_evaluate_rule_summary(
    *,
    model_id: str,
    run_id: str,
    truth_parameter: str,
    child_df: Any,
    baseline_df: Any = None,
    models_root: Path,
    local_time: Optional[float] = None,
    signal_envelope_sizes: Optional[Mapping[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    rule_path = _basic_rule_path_for_model(models_root=models_root, model_id=model_id)
    noise_adder_path = models_root / model_id / "noise_adder.py"
    resolved_noise_adder_path = noise_adder_path if noise_adder_path.exists() else None
    out: Dict[str, Any] = {}
    if not rule_path.exists():
        noise_analysis = _build_noise_analysis_summary(
            child_df=child_df,
            baseline_df=baseline_df,
            noise_adder_path=resolved_noise_adder_path,
            local_time=local_time,
        )
        if noise_analysis is not None:
            out["noise_analysis"] = noise_analysis
        return out or None
    out = _empty_rule_outcome_summary()
    if not str(truth_parameter or "").strip():
        noise_analysis = _build_noise_analysis_summary(
            child_df=child_df,
            baseline_df=baseline_df,
            noise_adder_path=resolved_noise_adder_path,
            local_time=local_time,
        )
        if noise_analysis is not None:
            out["noise_analysis"] = noise_analysis
        return out
    try:
        labels = _build_high_context_rule_labels(
            model_id=model_id,
            truth_parameter=truth_parameter,
            models_root=models_root,
        )
    except Exception:
        return out
    clean_result = _rule_result_from_payload(
        _evaluate_rule_on_loaded_dataframe(
            model_id=model_id,
            run_id=run_id,
            child_df=child_df,
            baseline_df=baseline_df,
            rule_path=rule_path,
            noise_adder_path=resolved_noise_adder_path,
            models_root=models_root,
            noise_level="none",
            seed=0,
            labels=labels,
        )
    )
    if isinstance(clean_result, dict):
        out["ground_truth"] = clean_result.get("ground_truth")
    _add_perfect_rule_fields(out, clean_result)
    noise_analysis = _build_noise_analysis_summary(
        child_df=child_df,
        baseline_df=baseline_df,
        noise_adder_path=resolved_noise_adder_path,
        local_time=local_time,
    )
    if noise_analysis is not None:
        out["noise_analysis"] = noise_analysis
    return out


def _build_baseline_evaluate_rule_summary(
    *,
    model_id: str,
    run_id: str,
    base_df: Any,
    models_root: Path,
    signal_envelope_sizes: Optional[Mapping[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    basic_summary = _build_evaluate_rule_summary(
        model_id=model_id,
        run_id=run_id,
        truth_parameter=_NO_CHANGE_CLASS,
        child_df=base_df,
        baseline_df=base_df,
        models_root=models_root,
        signal_envelope_sizes=signal_envelope_sizes,
    )
    if basic_summary is None:
        return None
    ground_truth = None
    if isinstance(basic_summary, dict):
        ground_truth = basic_summary.get("ground_truth")
    if ground_truth is None:
        try:
            ground_truth = _build_high_context_rule_labels(
                model_id=model_id,
                truth_parameter=_NO_CHANGE_CLASS,
                models_root=models_root,
            ).get("truth_display")
        except Exception:
            ground_truth = None
    out: Dict[str, Any] = {}
    if ground_truth is not None:
        out["ground_truth"] = str(ground_truth)
    if isinstance(basic_summary, dict):
        if "is_perfect_correct" in basic_summary:
            out["is_perfect_correct"] = basic_summary["is_perfect_correct"]
        if "is_perfect_predicted_label" in basic_summary:
            out["is_perfect_predicted_label"] = basic_summary["is_perfect_predicted_label"]
        if "noise_analysis" in basic_summary:
            out["noise_analysis"] = dict(basic_summary["noise_analysis"])
    return out or None


def _build_child_evaluate_rule_summary(
    *,
    model_id: str,
    run_id: str,
    truth_parameter: str,
    child_df: Any,
    baseline_df: Any = None,
    models_root: Path,
    local_time: Optional[float] = None,
    signal_envelope_sizes: Optional[Mapping[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    basic_summary = _build_evaluate_rule_summary(
        model_id=model_id,
        run_id=run_id,
        truth_parameter=truth_parameter,
        child_df=child_df,
        baseline_df=baseline_df,
        models_root=models_root,
        local_time=local_time,
        signal_envelope_sizes=signal_envelope_sizes,
    )
    if basic_summary is None:
        return None
    ground_truth = None
    if isinstance(basic_summary, dict):
        ground_truth = basic_summary.get("ground_truth")
    if ground_truth is None:
        try:
            ground_truth = _build_high_context_rule_labels(
                model_id=model_id,
                truth_parameter=truth_parameter,
                models_root=models_root,
            ).get("truth_display")
        except Exception:
            ground_truth = None
    out: Dict[str, Any] = {}
    if ground_truth is not None:
        out["ground_truth"] = str(ground_truth)
    if isinstance(basic_summary, dict):
        if "is_perfect_correct" in basic_summary:
            out["is_perfect_correct"] = basic_summary["is_perfect_correct"]
        if "is_perfect_predicted_label" in basic_summary:
            out["is_perfect_predicted_label"] = basic_summary["is_perfect_predicted_label"]
        if "noise_analysis" in basic_summary:
            out["noise_analysis"] = dict(basic_summary["noise_analysis"])
    return out or None


def _build_ground_truth_rule_summary(**kwargs: Any) -> Dict[str, Any]:
    return _build_evaluate_rule_summary(**kwargs)


def _default_child_summary(*, url: Optional[str] = None) -> Dict[str, Any]:
    out = {
        "eligible": False,
        "detectability": _documented_detectability_summary({}),
    }
    if url is not None:
        out["url"] = str(url)
    return out


def _expected_intervention_parameters(experiment_config: Any) -> tuple[str, ...]:
    names = getattr(experiment_config, "intervention_parameter_names", None)
    if names is None:
        exposed_variables = getattr(experiment_config, "exposed_variables", None)
        parameters = getattr(exposed_variables, "parameters", None)
        if isinstance(parameters, dict):
            names = parameters.keys()
        else:
            names = ()
    return tuple(
        sorted(
            {
                str(name).strip()
                for name in names
                if str(name).strip()
            },
            key=str.casefold,
        )
    )


def _signal_detectability_specs(experiment_config: Any) -> Dict[str, Dict[str, Any]]:
    signal_types = getattr(experiment_config, "observable_signal_types", None)
    detectability = getattr(experiment_config, "detectability", None)
    if not isinstance(signal_types, Mapping) or detectability is None:
        return {}
    specs: Dict[str, Dict[str, Any]] = {}
    for signal, signal_type in signal_types.items():
        normalized_type = str(signal_type or "").strip().lower()
        if normalized_type == "continuous":
            continuous = getattr(detectability, "continuous", None)
            if continuous is None:
                continue
            specs[str(signal)] = {
                "min_srd_distance": float(continuous.min_srd_distance),
                "epsilon_SRD": float(continuous.epsilon_SRD),
                "minimum_consecutive_srd_steps": max(
                    1,
                    int(continuous.minimum_consecurive_below_SRD),
                ),
            }
        elif normalized_type == "impulse_like":
            impulse_like = getattr(detectability, "impulse_like", None)
            if impulse_like is None:
                continue
            specs[str(signal)] = {
                "min_srd_distance": float(impulse_like.min_srd_distance),
                "epsilon_SRD": float(impulse_like.epsilon_SRD),
                "minimum_consecutive_srd_steps": 1,
            }
    return specs


def _rms_thresholds(experiment_config: Any) -> Dict[str, float]:
    detectability = getattr(experiment_config, "detectability", None)
    raw_thresholds = getattr(detectability, "RMS_thresholds", None)
    if not isinstance(raw_thresholds, Mapping):
        return {}
    thresholds: Dict[str, float] = {}
    for signal, value in raw_thresholds.items():
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(threshold):
            thresholds[str(signal)] = threshold
    return thresholds


def _child_is_detectable(child: Dict[str, Any]) -> bool:
    detectability = child.get("detectability")
    if not isinstance(detectability, dict):
        return False
    baseline_detectability = detectability.get("vs_baseline")
    if not isinstance(baseline_detectability, dict):
        return False
    if _detectability_status(baseline_detectability) != "yes":
        return False
    time0_detectability = detectability.get("vs_time0_baseline")
    if not isinstance(time0_detectability, dict):
        return False
    return _detectability_status(time0_detectability) == "yes"


def _child_vs_baseline_is_detectable(child: Mapping[str, Any]) -> bool:
    detectability = child.get("detectability")
    if not isinstance(detectability, dict):
        return False
    baseline_detectability = detectability.get("vs_baseline")
    if not isinstance(baseline_detectability, dict):
        return False
    return _detectability_status(baseline_detectability) == "yes"


def _child_noise_metric_equals_one(
    child: Mapping[str, Any],
    *,
    profile: str,
) -> bool:
    evaluate_rule = child.get("evaluate_rule")
    if not isinstance(evaluate_rule, dict):
        return False
    noise_analysis = evaluate_rule.get("noise_analysis")
    if not isinstance(noise_analysis, dict):
        return False
    raw_value = noise_analysis.get(f"avg_chance_corrected_accuracy_{profile}")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and np.isclose(value, 1.0))


def _child_has_correct_clean_rule(child: Dict[str, Any]) -> bool:
    evaluate_rule = child.get("evaluate_rule")
    if evaluate_rule is None:
        return False
    if not isinstance(evaluate_rule, dict):
        return False
    return evaluate_rule.get("is_perfect_correct") is True


def _child_is_eligible(
    child: Dict[str, Any],
    *,
    intervention_time: object = None,
) -> bool:
    _ = intervention_time
    if not _child_is_detectable(child):
        return False
    return True


def _baseline_family_is_eligible(
    children: Dict[str, Dict[str, Any]],
    *,
    child_parameters: Optional[Dict[str, str]] = None,
    expected_parameters: Optional[tuple[str, ...]] = None,
) -> bool:
    if not children or not expected_parameters:
        return False
    child_parameters = child_parameters or {}
    satisfied_parameters: set[str] = set()
    expected = {
        str(parameter).strip()
        for parameter in expected_parameters
        if str(parameter).strip()
    }
    if not expected:
        return False
    for child_id, child in children.items():
        if not isinstance(child, dict):
            continue
        if child.get("eligible") is not True:
            continue
        parameter = str(child_parameters.get(str(child_id).strip()) or "").strip()
        if parameter not in expected:
            continue
        satisfied_parameters.add(parameter)
    return len(satisfied_parameters) >= 3


def _build_summary_accuracy_payload(results: Mapping[str, Any]) -> Dict[str, Any]:
    baselines = results.get("baselines")
    if not isinstance(baselines, dict):
        baselines = {}

    detectability_ok = 0
    is_perfect_correct = 0
    avg_low_one = 0
    avg_high_one = 0
    family_eligible_detectable_and_not_perfect: List[str] = []

    for baseline in baselines.values():
        if not isinstance(baseline, dict):
            continue
        baseline_family_eligible = baseline.get("family_eligible") is True
        children = baseline.get("children")
        if not isinstance(children, dict):
            continue
        for child in children.values():
            if not isinstance(child, dict):
                continue
            child_detectability_ok = _child_vs_baseline_is_detectable(child)
            child_rule_correct = _child_has_correct_clean_rule(child)
            if child_detectability_ok:
                detectability_ok += 1
                if baseline_family_eligible and not child_rule_correct:
                    url = str(child.get("url") or "").strip()
                    if url:
                        family_eligible_detectable_and_not_perfect.append(url)
            if child_rule_correct:
                is_perfect_correct += 1
            if _child_noise_metric_equals_one(child, profile="low"):
                avg_low_one += 1
            if _child_noise_metric_equals_one(child, profile="high"):
                avg_high_one += 1

    return {
        "detectability_ok": detectability_ok,
        "is_perfect_correct": is_perfect_correct,
        "avg_chance_corrected_accuracy_low_1": avg_low_one,
        "avg_chance_corrected_accuracy_high_1": avg_high_one,
        "family_eligibility_and_detectability_ok_but_not_perfect_correct": (
            family_eligible_detectable_and_not_perfect
        ),
    }


def _write_summary_accuracy_if_requested(
    *,
    compute_ml: bool,
    out_path: Path,
    results: Mapping[str, Any],
    model_id: str,
) -> Optional[Path]:
    if not compute_ml:
        return None
    summary_accuracy_path = out_path.with_name("summary_accuracy.json")
    summary_accuracy_path.write_text(
        json.dumps(_build_summary_accuracy_payload(results), indent=2),
        encoding="utf-8",
    )
    click.echo(f"{model_id}: wrote {summary_accuracy_path}")
    return summary_accuracy_path


def _resolve_jobs(jobs: Optional[int]) -> int:
    if jobs is None:
        return max(1, min(8, os.cpu_count() or 1))
    resolved = int(jobs)
    if resolved < 1:
        raise click.UsageError("--jobs must be >= 1")
    return resolved


def _resolve_process_pool_context():
    if sys.platform.startswith("linux"):
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def _initialize_process_worker_context(context: Dict[str, Any]) -> None:
    _configure_process_worker_context(**context)


def _configure_process_worker_context(
    *,
    model_id: str,
    models_root: Path,
    baseline_contexts: Dict[str, Dict[str, Any]],
    entry_by_run_id: Dict[str, Dict[str, Any]],
    min_srd_distance: float,
    epsilon_SRD: float,
    minimum_consecutive_srd_steps: int,
    include_rule_eval: bool,
    signal_envelope_sizes: Optional[Dict[str, int]] = None,
    signal_detectability_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    RMS_thresholds: Optional[Dict[str, float]] = None,
    env_detectability_path: Optional[Path] = None,
    detectability_noise_adder_path: Optional[Path] = None,
) -> None:
    global _PROCESS_WORKER_MODEL_ID
    global _PROCESS_WORKER_MODELS_ROOT
    global _PROCESS_WORKER_BASELINE_CONTEXTS
    global _PROCESS_WORKER_ENTRY_BY_RUN_ID
    global _PROCESS_WORKER_REQUIRED_MINIMUM_SRD
    global _PROCESS_WORKER_EPSILON_SRD
    global _PROCESS_WORKER_MINIMUM_CONSECUTIVE_SRD_STEPS
    global _PROCESS_WORKER_INCLUDE_RULE_EVAL
    global _PROCESS_WORKER_SIGNAL_ENVELOPE_SIZES
    global _PROCESS_WORKER_SIGNAL_DETECTABILITY_SPECS
    global _PROCESS_WORKER_RMS_THRESHOLDS
    global _PROCESS_WORKER_ENV_DETECTABILITY_PATH
    global _PROCESS_WORKER_DETECTABILITY_NOISE_ADDER_PATH
    _PROCESS_WORKER_MODEL_ID = model_id
    _PROCESS_WORKER_MODELS_ROOT = models_root
    _PROCESS_WORKER_BASELINE_CONTEXTS = baseline_contexts
    _PROCESS_WORKER_ENTRY_BY_RUN_ID = entry_by_run_id
    _PROCESS_WORKER_REQUIRED_MINIMUM_SRD = float(min_srd_distance)
    _PROCESS_WORKER_EPSILON_SRD = float(epsilon_SRD)
    _PROCESS_WORKER_MINIMUM_CONSECUTIVE_SRD_STEPS = max(
        1,
        int(minimum_consecutive_srd_steps),
    )
    _PROCESS_WORKER_INCLUDE_RULE_EVAL = bool(include_rule_eval)
    _PROCESS_WORKER_SIGNAL_ENVELOPE_SIZES = dict(signal_envelope_sizes or {})
    _PROCESS_WORKER_SIGNAL_DETECTABILITY_SPECS = {
        str(signal): dict(spec)
        for signal, spec in (signal_detectability_specs or {}).items()
    }
    _PROCESS_WORKER_RMS_THRESHOLDS = {
        str(signal): float(threshold)
        for signal, threshold in (RMS_thresholds or {}).items()
    }
    _PROCESS_WORKER_ENV_DETECTABILITY_PATH = (
        Path(env_detectability_path).expanduser().resolve()
        if env_detectability_path is not None
        else None
    )
    _PROCESS_WORKER_DETECTABILITY_NOISE_ADDER_PATH = (
        Path(detectability_noise_adder_path).expanduser().resolve()
        if detectability_noise_adder_path is not None
        else None
    )


def _compute_child_outputs(
    *,
    model_id: str,
    models_root: Path,
    base_df: Any,
    entry: Dict[str, Any],
    min_srd_distance: float,
    epsilon_SRD: float,
    time0_df: Any,
    include_rule_eval: bool,
    minimum_consecutive_srd_steps: int = 1,
    signal_envelope_sizes: Optional[Mapping[str, int]] = None,
    signal_detectability_specs: Optional[Mapping[str, Mapping[str, object]]] = None,
    RMS_thresholds: Optional[Mapping[str, float]] = None,
    env_detectability_path: Optional[Path] = None,
    detectability_noise_adder_path: Optional[Path] = None,
) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    run_id = str(entry["run_id"])
    run_df = entry["df"]
    raw_iv = entry["raw"]
    truth_parameter = str(entry.get("truth_parameter") or "").strip()
    time0_id = str(raw_iv.get("time0_baseline_uuid") or "").strip()
    raw_baseline_parameters = entry.get("baseline_parameters")
    baseline_parameters = (
        raw_baseline_parameters
        if isinstance(raw_baseline_parameters, Mapping)
        else {}
    )
    run_parameters = _build_run_parameters(
        baseline_parameters=baseline_parameters,
        raw_intervention=raw_iv if isinstance(raw_iv, Mapping) else {},
    )

    detectability_metrics = _compute_detectability_metrics(
        model_id=model_id,
        baseline_df=base_df,
        run_df=run_df,
        time0_df=time0_df,
        time0_expected=bool(time0_id),
        min_srd_distance=min_srd_distance,
        epsilon_SRD=epsilon_SRD,
        minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
        intervention_time=raw_iv.get("intervention_time"),
        run_parameters=run_parameters,
        env_detectability_path=env_detectability_path,
        noise_adder_path=detectability_noise_adder_path,
        signal_detectability_specs=signal_detectability_specs,
        RMS_thresholds=RMS_thresholds,
    )
    child_summary = {
        "url": _webapp_run_url(model_id=model_id, run_id=run_id),
        "detectability": _documented_detectability_summary(detectability_metrics),
    }
    _ = (include_rule_eval, truth_parameter, signal_envelope_sizes, models_root)
    child_summary["eligible"] = _child_is_eligible(
        child_summary,
        intervention_time=raw_iv.get("intervention_time"),
    )
    return run_id, child_summary, copy.deepcopy(child_summary)


def _compute_baseline_outputs(
    *,
    model_id: str,
    models_root: Path,
    baseline_id: str,
    base_df: Any,
    include_rule_eval: bool,
    signal_envelope_sizes: Optional[Mapping[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    _ = (
        model_id,
        models_root,
        baseline_id,
        base_df,
        include_rule_eval,
        signal_envelope_sizes,
    )
    return None


def _compute_metrics_task_from_context(
    task_kind: str,
    baseline_id: str,
    run_id: str,
) -> tuple[str, str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
    model_id = _PROCESS_WORKER_MODEL_ID
    models_root = _PROCESS_WORKER_MODELS_ROOT
    if model_id is None or models_root is None:
        raise RuntimeError("Process worker context is not configured")
    baseline_ctx = _PROCESS_WORKER_BASELINE_CONTEXTS.get(baseline_id)
    if baseline_ctx is None:
        raise KeyError(f"Missing worker context for baseline={baseline_id}")
    if task_kind == "baseline":
        baseline_summary = _compute_baseline_outputs(
            model_id=model_id,
            models_root=models_root,
            baseline_id=baseline_id,
            base_df=baseline_ctx.get("base_df"),
            include_rule_eval=_PROCESS_WORKER_INCLUDE_RULE_EVAL,
            signal_envelope_sizes=_PROCESS_WORKER_SIGNAL_ENVELOPE_SIZES,
        )
        return task_kind, baseline_id, run_id, baseline_summary, None
    if task_kind != "child":
        raise ValueError(f"Unknown metrics task kind: {task_kind}")
    entry = _PROCESS_WORKER_ENTRY_BY_RUN_ID.get(run_id)
    if entry is None:
        raise KeyError(f"Missing worker context for child run={run_id}")
    child_run_id, out, child_summary = _compute_child_outputs(
        model_id=model_id,
        models_root=models_root,
        base_df=baseline_ctx.get("base_df"),
        entry=entry,
        min_srd_distance=_PROCESS_WORKER_REQUIRED_MINIMUM_SRD,
        epsilon_SRD=_PROCESS_WORKER_EPSILON_SRD,
        minimum_consecutive_srd_steps=(
            _PROCESS_WORKER_MINIMUM_CONSECUTIVE_SRD_STEPS
        ),
        time0_df=(baseline_ctx.get("time0_df_by_run_id") or {}).get(run_id),
        include_rule_eval=_PROCESS_WORKER_INCLUDE_RULE_EVAL,
        signal_envelope_sizes=_PROCESS_WORKER_SIGNAL_ENVELOPE_SIZES,
        signal_detectability_specs=_PROCESS_WORKER_SIGNAL_DETECTABILITY_SPECS,
        RMS_thresholds=_PROCESS_WORKER_RMS_THRESHOLDS,
        env_detectability_path=_PROCESS_WORKER_ENV_DETECTABILITY_PATH,
        detectability_noise_adder_path=(
            _PROCESS_WORKER_DETECTABILITY_NOISE_ADDER_PATH
        ),
    )
    return task_kind, baseline_id, child_run_id, out, child_summary


def _compute_child_outputs_from_context(
    baseline_id: str,
    run_id: str,
) -> tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    _task_kind, result_baseline_id, child_run_id, out, child_summary = (
        _compute_metrics_task_from_context("child", baseline_id, run_id)
    )
    if child_summary is None:
        raise RuntimeError("Child metrics task returned no child summary")
    return result_baseline_id, child_run_id, out, child_summary


def _json_dumpable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_dumpable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_dumpable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_dumpable(item())
        except Exception:
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parse_threshold_overrides(thresholds: Optional[tuple[str, ...]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in thresholds or ():
        text = str(raw or "").strip()
        if not text:
            continue
        if "=" not in text:
            raise click.UsageError("--thresholds entries must use SIGNAL=VALUE")
        signal, raw_value = text.split("=", 1)
        signal = signal.strip()
        if not signal:
            raise click.UsageError("--thresholds signal names must be non-empty")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise click.UsageError(
                f"Invalid threshold for {signal!r}: {raw_value!r}"
            ) from exc
        if not np.isfinite(value):
            raise click.UsageError(f"Invalid non-finite threshold for {signal!r}")
        out[signal] = value
    return out


def _metric_run_id(uuid: Optional[str]) -> str:
    text = str(uuid or "").strip()
    if text:
        return text if text.startswith("cheap_filter_") else f"cheap_filter_{text}"
    digest = hashlib.md5(_now_iso8601_utc().encode("utf-8")).hexdigest()[:8]
    return f"cheap_filter_{digest}"


def _metric_run_dir(
    *,
    model_dir: Path,
    policy_id: str,
    metric_run_id: str,
) -> Path:
    return model_dir / "metrics" / policy_id / metric_run_id


class _CheapFilterRunGraph:
    def __init__(self, *, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.nodes = nodes
        self.edges = edges

    @property
    def nodes_by_id(self) -> Dict[str, Dict[str, Any]]:
        return {str(node.get("run_id") or ""): node for node in self.nodes}


def _read_jsonl_objects(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing run graph JSONL at {path}")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {path} line {line_number}: {exc}"
                ) from exc
            if isinstance(item, Mapping):
                out.append(dict(item))
    return out


def _load_cheap_filter_run_graph(plan_dir: Path) -> _CheapFilterRunGraph:
    return _CheapFilterRunGraph(
        nodes=_read_jsonl_objects(plan_dir / "run_nodes.jsonl"),
        edges=_read_jsonl_objects(plan_dir / "run_edges.jsonl"),
    )


def _load_cheap_filter_model_record(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid model_record.json at {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SystemExit(f"model_record.json at {path} must contain an object")
    return dict(raw)


def _build_run_graph_links(graph: Any) -> Dict[str, Dict[str, str]]:
    by_intervention: Dict[str, Dict[str, str]] = {}
    for edge in graph.edges:
        edge_type = str(edge.get("edge_type") or "").strip()
        source = str(edge.get("source_run_id") or "").strip()
        target = str(edge.get("target_run_id") or "").strip()
        if edge_type == "baseline_to_intervention":
            by_intervention.setdefault(target, {})["baseline_run_id"] = source
        elif edge_type == "intervention_to_time0_baseline":
            by_intervention.setdefault(source, {})["time0_run_id"] = target
    return by_intervention


def _signal_rms_difference(
    *,
    reference_df: Any,
    target_df: Any,
) -> Dict[str, float]:
    reference_arrays = _detectability_arrays(reference_df)
    target_arrays = _detectability_arrays(target_df)
    if reference_arrays is not None and target_arrays is not None:
        reference_time, reference_signals, reference_values = reference_arrays
        target_time, target_signals, target_values = target_arrays
        if (
            reference_signals == target_signals
            and reference_time.shape == target_time.shape
            and reference_values.shape == target_values.shape
            and np.array_equal(reference_time, target_time)
        ):
            delta = target_values - reference_values
            valid = np.isfinite(reference_values) & np.isfinite(target_values)
            out: Dict[str, float] = {}
            for idx, signal in enumerate(reference_signals):
                column_valid = valid[:, idx]
                if not np.any(column_valid):
                    out[str(signal)] = 0.0
                    continue
                column_delta = delta[column_valid, idx]
                out[str(signal)] = float(
                    np.sqrt(np.mean(np.square(column_delta, dtype=np.float64)))
                )
            return out

    if str(reference_df.columns[-1]) != "time" or str(target_df.columns[-1]) != "time":
        raise ValueError("last column for reference_df and target_df must be time")
    if [str(column) for column in reference_df.columns] != [
        str(column) for column in target_df.columns
    ]:
        raise ValueError("reference_df and target_df columns must match")
    reference_time = np.asarray(reference_df["time"], dtype=float)
    target_time = np.asarray(target_df["time"], dtype=float)
    if reference_time.shape != target_time.shape or not np.allclose(
        reference_time,
        target_time,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("reference_df and target_df time columns must be identical")

    out: Dict[str, float] = {}
    for column in reference_df.columns[:-1]:
        signal = str(column)
        reference_values = np.asarray(reference_df[column], dtype=float)
        target_values = np.asarray(target_df[column], dtype=float)
        valid = np.isfinite(reference_values) & np.isfinite(target_values)
        if not np.any(valid):
            out[signal] = 0.0
            continue
        delta = target_values[valid] - reference_values[valid]
        out[signal] = float(np.sqrt(np.mean(np.square(delta, dtype=np.float64))))
    return out


def _normalize_detectability_for_cheap_filter(
    detectability_metrics: Mapping[str, Any],
    *,
    has_environment_hook: bool,
) -> Dict[str, Any]:
    def _entry(raw: Any) -> Dict[str, Any]:
        status = raw if isinstance(raw, Mapping) else {}
        generic_status = str(
            status.get("max_SRD_detectability") or ""
        ).strip().lower()
        env_status = str(
            status.get("environment_specific_detectability") or ""
        ).strip().lower()
        return {
            "generic_numeric_detectability": generic_status == "yes",
            "environment_specific_detectability": (
                True if not has_environment_hook else env_status == "yes"
            ),
            "detectability_output": _detectability_output_from_status(status),
        }

    return {
        "vs_baseline": _entry(detectability_metrics.get("vs_baseline")),
        "vs_time0_baseline": _entry(
            detectability_metrics.get("vs_time0_baseline")
        ),
    }


def _cheap_filter_pass(
    *,
    status_success: bool,
    baseline_df: Any,
    run_df: Any,
    time0_df: Any,
    detectability: Mapping[str, Any],
) -> bool:
    if not status_success or baseline_df is None or run_df is None or time0_df is None:
        return False
    vs_baseline = detectability.get("vs_baseline")
    vs_time0 = detectability.get("vs_time0_baseline")
    if not isinstance(vs_baseline, Mapping) or not isinstance(vs_time0, Mapping):
        return False
    return (
        vs_baseline.get("generic_numeric_detectability") is True
        and vs_time0.get("generic_numeric_detectability") is True
    )


def _compute_cheap_filter_record(
    *,
    model_id: str,
    model_dir: Path,
    runs_root: Path,
    model_record: Mapping[str, Any],
    node: Mapping[str, Any],
    links: Mapping[str, str],
    metric_run_id: str,
    policy_id: str,
    min_srd_distance: float,
    epsilon_SRD: float,
    minimum_consecutive_srd_steps: int,
    signal_detectability_specs: Mapping[str, Mapping[str, object]],
    RMS_thresholds: Mapping[str, float],
    env_detectability_path: Optional[Path],
    detectability_noise_adder_path: Optional[Path],
    df_cache: Dict[str, Any],
    compute_noise_summary: bool,
) -> Dict[str, Any]:
    run_id = str(node.get("run_id") or "").strip()
    recipe = node.get("recipe") if isinstance(node.get("recipe"), Mapping) else {}
    baseline_parameters = (
        recipe.get("baseline_parameters") if isinstance(recipe, Mapping) else {}
    )
    if not isinstance(baseline_parameters, Mapping):
        baseline_parameters = {}
    intervention = recipe.get("intervention") if isinstance(recipe, Mapping) else {}
    if not isinstance(intervention, Mapping):
        intervention = {}
    baseline_run_id = str(links.get("baseline_run_id") or "").strip()
    time0_run_id = str(links.get("time0_run_id") or "").strip()
    changed_parameter = str(intervention.get("parameter") or "").strip()
    failure_reasons: List[str] = []
    raw_record = model_record.get(run_id)
    runtime_record = raw_record if isinstance(raw_record, Mapping) else {}
    status_success = is_success_status(runtime_record.get("status"))
    if not status_success:
        failure_reasons.append("run_status_not_success")
    if not baseline_run_id:
        failure_reasons.append("missing_baseline_edge")
    if not time0_run_id:
        failure_reasons.append("missing_time0_edge")

    baseline_df = _load_df_cached(df_cache, runs_root, baseline_run_id) if baseline_run_id else None
    run_df = _load_df_cached(df_cache, runs_root, run_id) if run_id else None
    time0_df = _load_df_cached(df_cache, runs_root, time0_run_id) if time0_run_id else None
    if baseline_df is None:
        failure_reasons.append("missing_baseline_data")
    if run_df is None:
        failure_reasons.append("missing_run_data")
    if time0_df is None:
        failure_reasons.append("missing_time0_data")

    baseline_rms_difference: Dict[str, float] = {}
    time0_rms_difference: Dict[str, float] = {}
    detectability_metrics: Dict[str, Any] = {}
    if baseline_df is not None and run_df is not None and time0_df is not None:
        try:
            baseline_rms_difference = _signal_rms_difference(
                reference_df=baseline_df,
                target_df=run_df,
            )
            time0_rms_difference = _signal_rms_difference(
                reference_df=time0_df,
                target_df=run_df,
            )
            run_parameters = _build_run_parameters(
                baseline_parameters=baseline_parameters,
                raw_intervention=intervention,
            )
            if "intervention_time" not in run_parameters and "time" in intervention:
                run_parameters["intervention_time"] = intervention.get("time")
            detectability_metrics = _compute_detectability_metrics(
                model_id=model_id,
                baseline_df=baseline_df,
                run_df=run_df,
                time0_df=time0_df,
                time0_expected=bool(time0_run_id),
                min_srd_distance=min_srd_distance,
                epsilon_SRD=epsilon_SRD,
                minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                intervention_time=intervention.get("time"),
                run_parameters=run_parameters,
                env_detectability_path=env_detectability_path,
                noise_adder_path=detectability_noise_adder_path,
                signal_detectability_specs=signal_detectability_specs,
                RMS_thresholds=RMS_thresholds,
                compute_noise_summary=compute_noise_summary,
                skip_env_when_not_detectable=not compute_noise_summary,
            )
        except Exception as exc:
            failure_reasons.append(f"metric_error:{type(exc).__name__}")
    detectability = _normalize_detectability_for_cheap_filter(
        detectability_metrics,
        has_environment_hook=env_detectability_path is not None,
    )
    passed = _cheap_filter_pass(
        status_success=status_success,
        baseline_df=baseline_df,
        run_df=run_df,
        time0_df=time0_df,
        detectability=detectability,
    )
    if not passed and not failure_reasons:
        failure_reasons.append("detectability_not_passed")

    return _json_dumpable(
        {
            "record_type": "run",
            "metric_run_id": metric_run_id,
            "policy_id": policy_id,
            "family_id": str(node.get("family_id") or ""),
            "run_id": run_id,
            "kind": "intervention",
            "ground_truth_label": changed_parameter,
            "baseline_run_id": baseline_run_id,
            "time0_run_id": time0_run_id,
            "changed_parameter": changed_parameter,
            "cheap_filter_pass": bool(passed),
            "baseline_rms_difference": baseline_rms_difference,
            "time0_rms_difference": time0_rms_difference,
            "detectability": detectability,
            "failure_reasons": failure_reasons,
        }
    )


def run_cheap_filter_metrics(
    *,
    model_dir: Path,
    policy: str,
    root_dir: Path,
    uuid: Optional[str] = None,
    thresholds: Optional[Mapping[str, float]] = None,
    jobs: Optional[int] = None,
    limit_records: Optional[int] = None,
    compute_noise_summary: bool = False,
) -> Dict[str, Any]:
    _ = jobs
    policy_id = str(policy or "").strip()
    resolved_model_dir = Path(model_dir).expanduser().resolve()
    model_id = resolved_model_dir.name
    if not model_id:
        raise click.UsageError("--model_dir must name a simulator directory")
    if not policy_id:
        raise click.UsageError("--policy is required")
    if not resolved_model_dir.exists():
        raise SystemExit(f"Model directory {resolved_model_dir} missing")
    runs_root = Path(root_dir).expanduser().resolve()
    model_record_path = runs_root / "model_record.json"
    if not model_record_path.exists():
        raise SystemExit(f"Missing model_record.json at {model_record_path}")
    config_path = resolved_model_dir / "experiment_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing experiment_config.json at {config_path}")
    plan_dir = resolved_model_dir / "plans" / policy_id
    graph = _load_cheap_filter_run_graph(plan_dir)

    try:
        experiment_config = load_experiment_config_json(config_path)
    except DistributionValidationError as exc:
        raise SystemExit(f"Invalid experiment_config.json at {config_path}: {exc}")
    resolved_thresholds = dict(thresholds or _rms_thresholds(experiment_config))
    min_srd_distance = float(experiment_config.min_srd_distance)
    epsilon_SRD = float(experiment_config.epsilon_SRD)
    minimum_consecutive_srd_steps = max(
        1,
        int(getattr(experiment_config, "minimum_consecurive_below_SRD", 1)),
    )
    signal_detectability_specs = _signal_detectability_specs(experiment_config)
    metric_run_id = _metric_run_id(uuid)
    out_dir = _metric_run_dir(
        model_dir=resolved_model_dir,
        policy_id=policy_id,
        metric_run_id=metric_run_id,
    )
    env_path = resolved_model_dir / "detectability_specific_environment.py"
    resolved_env_path = env_path if env_path.exists() else None
    noise_adder_path = resolved_model_dir / "noise_adder.py"
    resolved_noise_adder_path = noise_adder_path if noise_adder_path.exists() else None
    model_record = _load_cheap_filter_model_record(model_record_path)
    links_by_intervention = _build_run_graph_links(graph)
    nodes_by_id = graph.nodes_by_id
    intervention_nodes = [
        node for node in graph.nodes if str(node.get("kind") or "") == "intervention"
    ]
    if limit_records is not None:
        intervention_nodes = intervention_nodes[: max(0, int(limit_records))]

    metric_run = _json_dumpable(
        {
            "record_type": "metric_run",
            "metric_run_id": metric_run_id,
            "metric_type": "cheap_filter",
            "created_at": _now_iso8601_utc(),
            "script": "workflows/metrics/compute_metrics.py",
            "model": model_id,
            "policy_id": policy_id,
            "model_dir": str(resolved_model_dir),
            "run_root": str(runs_root),
            "inputs": {
                "run_nodes": str(plan_dir / "run_nodes.jsonl"),
                "run_edges": str(plan_dir / "run_edges.jsonl"),
                "model_record": str(model_record_path),
            },
            "thresholds": resolved_thresholds,
            "detectability_specific_environment": (
                str(resolved_env_path) if resolved_env_path is not None else None
            ),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_run_path = out_dir / "metric_run.json"
    metrics_path = out_dir / "cheap_filter_metrics.jsonl"
    metric_run_path.write_text(
        json.dumps(metric_run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    df_cache: Dict[str, Any] = OrderedDict()
    record_count = 0
    pass_count = 0
    with metrics_path.open("w", encoding="utf-8") as handle:
        for node in tqdm(
            intervention_nodes,
            desc=f"{model_id}: cheap filter",
            mininterval=5.0,
            unit="run",
        ):
            if str(node.get("run_id") or "") not in nodes_by_id:
                continue
            record = _compute_cheap_filter_record(
                model_id=model_id,
                model_dir=resolved_model_dir,
                runs_root=runs_root,
                model_record=model_record,
                node=node,
                links=links_by_intervention.get(str(node.get("run_id") or ""), {}),
                metric_run_id=metric_run_id,
                policy_id=policy_id,
                min_srd_distance=min_srd_distance,
                epsilon_SRD=epsilon_SRD,
                minimum_consecutive_srd_steps=minimum_consecutive_srd_steps,
                signal_detectability_specs=signal_detectability_specs,
                RMS_thresholds=resolved_thresholds,
                env_detectability_path=resolved_env_path,
                detectability_noise_adder_path=resolved_noise_adder_path,
                df_cache=df_cache,
                compute_noise_summary=compute_noise_summary,
            )
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            record_count += 1
            if record.get("cheap_filter_pass") is True:
                pass_count += 1
    click.echo(f"{model_id}: wrote {metrics_path}")
    return {
        "model_id": model_id,
        "policy_id": policy_id,
        "metric_run_id": metric_run_id,
        "metric_run_dir": str(out_dir),
        "metric_run_path": str(metric_run_path),
        "metrics_path": str(metrics_path),
        "records": record_count,
        "cheap_filter_pass": pass_count,
        "ok": True,
    }


def _run_for_model(
    model_dir: Path,
    *,
    jobs: Optional[int] = None,
    compute_ml: bool = False,
    eligibility_based_on_basic_rule: bool = False,
) -> Dict[str, Any]:
    model_record_path = resolve_model_record_path(model_dir)
    if not model_record_path.exists():
        raise SystemExit(f"Missing model_record.json at {model_record_path}")

    config_path = model_dir / "experiment_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing experiment_config.json at {config_path}")
    try:
        experiment_config = load_experiment_config_json(config_path)
    except DistributionValidationError as exc:
        raise SystemExit(f"Invalid experiment_config.json at {config_path}: {exc}")
    min_srd_distance = float(experiment_config.min_srd_distance)
    epsilon_SRD = float(experiment_config.epsilon_SRD)
    minimum_consecutive_srd_steps = max(
        1,
        int(getattr(experiment_config, "minimum_consecurive_below_SRD", 1)),
    )
    signal_envelope_sizes: Dict[str, int] = {}

    runs_root = resolve_runs_root(model_dir)
    if not runs_root.exists():
        raise SystemExit(f"Missing runs directory at {runs_root}")

    specs_path = model_dir / "model_run_specs.json"
    if not specs_path.exists():
        raise SystemExit(f"Missing model_run_specs.json at {specs_path}")
    model_record = load_model_record_json(model_record_path)
    merged_record = build_model_record_registry(
        model_id=model_dir.name,
        specs=load_model_run_specs_json(
            specs_path,
            enforce_baseline_pair_diversity=False,
        ),
        runtime_map=model_record,
        experiment_config=experiment_config,
    )
    baselines = merged_record.get("baselines", [])
    if not isinstance(baselines, list):
        raise SystemExit("derived registry baselines must be a list")
    expected_parameters = _expected_intervention_parameters(experiment_config)
    signal_detectability_specs = _signal_detectability_specs(experiment_config)
    RMS_thresholds = _rms_thresholds(experiment_config)
    resolved_jobs = _resolve_jobs(jobs)
    _ = (compute_ml, eligibility_based_on_basic_rule)
    noise_adder_md5 = _md5_hex_or_none(model_dir / "noise_adder.py")
    include_rule_eval = False
    env_detectability_path = model_dir / "detectability_specific_environment.py"
    resolved_env_detectability_path = (
        env_detectability_path if env_detectability_path.exists() else None
    )
    noise_adder_path = model_dir / "noise_adder.py"
    resolved_detectability_noise_adder_path = (
        noise_adder_path if noise_adder_path.exists() else None
    )
    include_baseline_rule_eval = include_rule_eval

    df_cache: Dict[str, Any] = {}
    results: Dict[str, Any] = {
        "timestamp": _now_iso8601_utc(),
        "noise_adder_md5": noise_adder_md5,
        "eligible_baselines": 0,
        "baselines": {},
    }

    baseline_contexts: Dict[str, Dict[str, Any]] = {}
    children_by_baseline: Dict[str, List[Dict[str, Any]]] = {}
    summary_children_by_baseline_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    child_parameters_by_baseline_id: Dict[str, Dict[str, str]] = {}
    summary_by_baseline_id: Dict[str, Dict[str, Any]] = {}
    models_root = model_dir.expanduser().resolve().parent
    for baseline in baselines:
        if not isinstance(baseline, dict):
            continue
        baseline_id = str(baseline.get("run_id") or "").strip()
        if not baseline_id:
            continue
        ivs = baseline.get("interventions", [])
        if not isinstance(ivs, list):
            ivs = []
        baseline_children_summary: Dict[str, Dict[str, Any]] = {}
        for iv in ivs:
            if not isinstance(iv, dict):
                continue
            run_id = str(iv.get("name") or "").strip()
            if run_id:
                baseline_children_summary[run_id] = _default_child_summary(
                    url=_webapp_run_url(model_id=model_dir.name, run_id=run_id)
                )
        summary_children_by_baseline_id[baseline_id] = baseline_children_summary

        if not is_success_status(baseline.get("status")):
            summary_by_baseline_id[baseline_id] = {
                "family_eligible": False,
                "eligible": False,
                "children": baseline_children_summary,
            }
            continue
        base_df = _load_df_cached(df_cache, runs_root, baseline_id)
        if base_df is None:
            summary_by_baseline_id[baseline_id] = {
                "family_eligible": False,
                "eligible": False,
                "children": baseline_children_summary,
            }
            continue
        baseline_contexts[baseline_id] = {
            "base_df": base_df,
            "baseline_parameters": dict(baseline.get("parameters") or {}),
        }

        if not ivs:
            children_by_baseline[baseline_id] = []
            continue

        children: List[Dict[str, Any]] = []
        child_parameters: Dict[str, str] = {}
        for iv in ivs:
            if not isinstance(iv, dict):
                continue
            run_id = str(iv.get("name") or "").strip()
            if not run_id:
                continue
            if not is_success_status(iv.get("status")):
                continue
            df = _load_df_cached(df_cache, runs_root, run_id)
            if df is None:
                continue
            entry = {
                "run_id": run_id,
                "baseline_id": baseline_id,
                "df": df,
                "raw": iv,
                "baseline_parameters": dict(baseline.get("parameters") or {}),
                "truth_parameter": str(
                    iv.get("parameter")
                    or iv.get("variable")
                    or ""
                ).strip(),
            }
            children.append(entry)
            if entry["truth_parameter"]:
                child_parameters[run_id] = entry["truth_parameter"]

            time0_id = str(iv.get("time0_baseline_uuid") or "").strip()
            if time0_id:
                _load_df_cached(df_cache, runs_root, time0_id)

        children_by_baseline[baseline_id] = children
        child_parameters_by_baseline_id[baseline_id] = child_parameters

    total_children = sum(len(children) for children in children_by_baseline.values())
    click.echo(
        f"{model_dir.name}: computing eligibility metrics for {len(baselines)} baselines, "
        f"{total_children} children, jobs={resolved_jobs}"
    )

    baseline_out_by_id: Dict[str, Dict[str, Any]] = {}
    metrics_task_ids: List[tuple[str, str, str]] = []
    child_entry_by_run_id: Dict[str, Dict[str, Any]] = {}
    process_baseline_contexts: Dict[str, Dict[str, Any]] = {}
    for baseline_id, children in children_by_baseline.items():
        ctx = baseline_contexts.get(baseline_id) or {}
        base_df = ctx.get("base_df")
        if base_df is None:
            continue
        baseline_children_summary = summary_children_by_baseline_id.get(baseline_id, {})

        baseline_out: Dict[str, Any] = {
            "url": _webapp_run_url(model_id=model_dir.name, run_id=baseline_id),
            "family_eligible": False,
            "eligible": False,
            "children": copy.deepcopy(baseline_children_summary),
        }
        baseline_out_by_id[baseline_id] = baseline_out

        time0_df_by_run_id: Dict[str, Any] = {}
        process_baseline_contexts[baseline_id] = {
            "base_df": base_df,
            "baseline_parameters": dict(ctx.get("baseline_parameters") or {}),
            "time0_df_by_run_id": time0_df_by_run_id,
        }
        if include_baseline_rule_eval:
            metrics_task_ids.append(("baseline", baseline_id, baseline_id))

        if not children:
            summary_by_baseline_id[baseline_id] = {
                "family_eligible": False,
                "eligible": False,
                "children": baseline_children_summary,
            }
            continue
        for entry in children:
            raw_iv = entry["raw"]
            time0_id = str(raw_iv.get("time0_baseline_uuid") or "").strip()
            run_id = str(entry["run_id"])
            child_entry_by_run_id[run_id] = entry
            time0_df_by_run_id[run_id] = (
                df_cache.get(time0_id)
                if time0_id and is_success_status(raw_iv.get("time0_baseline_status"))
                else None
            )
            metrics_task_ids.append(("child", baseline_id, run_id))

    baseline_rule_results_by_id: Dict[str, Dict[str, Any]] = {}
    child_results_by_baseline: Dict[str, Dict[str, tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    if metrics_task_ids:
        worker_context = {
            "model_id": model_dir.name,
            "models_root": models_root,
            "baseline_contexts": process_baseline_contexts,
            "entry_by_run_id": child_entry_by_run_id,
            "min_srd_distance": min_srd_distance,
            "epsilon_SRD": epsilon_SRD,
            "minimum_consecutive_srd_steps": minimum_consecutive_srd_steps,
            "include_rule_eval": include_rule_eval,
            "signal_envelope_sizes": signal_envelope_sizes,
            "signal_detectability_specs": signal_detectability_specs,
            "RMS_thresholds": RMS_thresholds,
            "env_detectability_path": resolved_env_detectability_path,
            "detectability_noise_adder_path": (
                resolved_detectability_noise_adder_path
            ),
        }
        _configure_process_worker_context(**worker_context)
        progress_kwargs = {
            "total": len(metrics_task_ids),
            "desc": f"{model_dir.name}: eligibility",
            "unit": "task",
        }
        if resolved_jobs == 1:
            for task_kind, baseline_id, run_id in tqdm(metrics_task_ids, **progress_kwargs):
                result_kind, result_baseline_id, result_run_id, out, child_summary = (
                    _compute_metrics_task_from_context(
                        task_kind,
                        baseline_id,
                        run_id,
                    )
                )
                if result_kind == "baseline":
                    if out is not None:
                        baseline_rule_results_by_id[result_baseline_id] = out
                    continue
                if child_summary is None:
                    raise RuntimeError("Child metrics task returned no child summary")
                child_results_by_baseline.setdefault(result_baseline_id, {})[
                    result_run_id
                ] = (
                    out,
                    child_summary,
                )
        else:
            process_pool_context = _resolve_process_pool_context()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=resolved_jobs,
                mp_context=process_pool_context,
                initializer=_initialize_process_worker_context,
                initargs=(worker_context,),
            ) as executor:
                future_map = {
                    executor.submit(
                        _compute_metrics_task_from_context,
                        task_kind,
                        baseline_id,
                        run_id,
                    ): (
                        task_kind,
                        baseline_id,
                        run_id,
                    )
                    for task_kind, baseline_id, run_id in metrics_task_ids
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_map),
                    **progress_kwargs,
                ):
                    result_kind, result_baseline_id, result_run_id, out, child_summary = future.result()
                    if result_kind == "baseline":
                        if out is not None:
                            baseline_rule_results_by_id[result_baseline_id] = out
                        continue
                    if child_summary is None:
                        raise RuntimeError("Child metrics task returned no child summary")
                    child_results_by_baseline.setdefault(
                        result_baseline_id,
                        {},
                    )[result_run_id] = (out, child_summary)

    for baseline_id, children in children_by_baseline.items():
        baseline_out = baseline_out_by_id.get(baseline_id)
        if baseline_out is None:
            continue
        if baseline_id in baseline_rule_results_by_id:
            baseline_out["evaluate_rule"] = baseline_rule_results_by_id[baseline_id]
        baseline_children_summary = summary_children_by_baseline_id.get(baseline_id, {})
        if children:
            baseline_child_results = child_results_by_baseline.get(baseline_id, {})
            for entry in children:
                run_id = str(entry["run_id"])
                result = baseline_child_results.get(run_id)
                if result is None:
                    continue
                out, child_summary = result
                baseline_out["children"][run_id] = out
                baseline_children_summary[run_id] = child_summary
            family_eligible = _baseline_family_is_eligible(
                baseline_children_summary,
                child_parameters=child_parameters_by_baseline_id.get(baseline_id, {}),
                expected_parameters=expected_parameters,
            )
            summary_by_baseline_id[baseline_id] = {
                "family_eligible": family_eligible,
                "eligible": family_eligible,
                "children": baseline_children_summary,
            }
            baseline_out["family_eligible"] = bool(family_eligible)
            baseline_out["eligible"] = bool(
                summary_by_baseline_id[baseline_id]["eligible"]
            )
        results["baselines"][baseline_id] = baseline_out

    results["eligible_baselines"] = sum(
        1
        for baseline in results["baselines"].values()
        if isinstance(baseline, dict) and baseline.get("family_eligible") is True
    )
    results["total_baselines"] = len(results["baselines"])

    out_path = resolve_similarity_metrics_path(model_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validate_similarity_metrics_semantics(
        results,
        is_baseline_a_class=True,
    )
    dump_similarity_metrics_json(out_path, results, indent=2)
    click.echo(f"{model_dir.name}: wrote {out_path}")
    return {
        "model_id": model_dir.name,
        "path": str(out_path),
        "ok": True,
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
    help="Number of eligibility worker processes. Default: min(8, CPU count).",
)
@click.option(
    "--policy",
    type=str,
    default=None,
    help="Resolved run graph policy id. Enables the documented cheap-filter metric-run output.",
)
@click.option(
    "--model_dir",
    "model_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Simulator directory containing experiment_config.json and plans/<policy_id>/.",
)
@click.option(
    "--root_dir",
    "root_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Run-artifact root containing model_record.json and per-run directories.",
)
@click.option(
    "--uuid",
    "uuid",
    type=str,
    default=None,
    help="Optional metric run id suffix. If omitted, a cheap_filter_<hash> id is generated.",
)
@click.option(
    "--thresholds",
    multiple=True,
    help="Override RMS thresholds as SIGNAL=VALUE. May be passed multiple times.",
)
@click.option(
    "--limit_records",
    type=int,
    default=None,
    hidden=True,
    help="Internal profiling aid: process only the first N intervention records.",
)
@click.option(
    "--compute_noise_summary",
    is_flag=True,
    default=False,
    hidden=True,
    help="Internal compatibility aid: compute expensive noise summary fields.",
)
def cli(
    model: Optional[str],
    jobs: Optional[int],
    policy: Optional[str],
    model_dir: Optional[Path],
    root_dir: Optional[Path],
    uuid: Optional[str],
    thresholds: tuple[str, ...],
    limit_records: Optional[int],
    compute_noise_summary: bool,
) -> None:
    """Compute tsENV metrics."""
    if policy is not None or model_dir is not None or root_dir is not None:
        if model_dir is None:
            raise click.UsageError("--model_dir is required with --policy/--root_dir")
        if policy is None:
            raise click.UsageError("--policy is required with --root_dir")
        if root_dir is None:
            raise click.UsageError("--root_dir is required with --policy")
        result = run_cheap_filter_metrics(
            model_dir=model_dir,
            policy=policy,
            root_dir=root_dir,
            uuid=uuid,
            thresholds=(
                _parse_threshold_overrides(thresholds) if thresholds else None
            ),
            jobs=jobs,
            limit_records=limit_records,
            compute_noise_summary=compute_noise_summary,
        )
        click.echo(json.dumps(result, indent=2))
        return
    main(model, jobs=jobs)


def main(
    target_model: Optional[str] = None,
    *,
    jobs: Optional[int] = None,
    compute_ml: bool = False,
    eligibility_based_on_basic_rule: bool = False,
) -> None:
    """If target_model is None, process all ALLOWED_TSENV_MODELS. Else target_model is a model id."""
    _ = (compute_ml, eligibility_based_on_basic_rule)
    models_root = _resolve_models_root()

    if target_model is None:
        model_ids = list(ALLOWED_TSENV_MODELS)
    else:
        model_id = str(target_model).strip()
        if "/" in model_id or "\\" in model_id:
            raise SystemExit(
                f"Expected a model id (e.g. BallDrop), got a path-like value: {target_model!r}"
            )
        if model_id not in ALLOWED_TSENV_MODELS:
            raise SystemExit(
                f"Model '{model_id}' is not an allowed tsENV model. "
                "Update shared/benchmark_utils.py (ALLOWED_TSENV_MODELS) to add it."
            )
        model_ids = [model_id]
    resolved_jobs = _resolve_jobs(jobs)

    computed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for model_id in sorted(model_ids):
        model_dir = models_root / model_id
        if not model_dir.exists():
            if target_model is not None:
                raise SystemExit(f"Model directory {model_dir} missing")
            skipped.append(
                {"model_id": model_id, "ok": False, "reason": "missing_model_dir"}
            )
            continue
        try:
            computed.append(
                _run_for_model(
                    model_dir,
                    jobs=resolved_jobs,
                )
            )
        except Exception as exc:
            skipped.append({"model_id": model_id, "ok": False, "error": str(exc)})

    click.echo(json.dumps({"computed": computed, "skipped": skipped}, indent=2))


if __name__ == "__main__":
    cli()
