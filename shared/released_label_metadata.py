"""Parameter labels published with the pinned question release."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Mapping

from shared.release_paths import DATASET_REVISION


_LABEL_MAPPINGS_PATH = Path(__file__).resolve().parent / "config" / "released_label_mappings.json"


@functools.lru_cache(maxsize=1)
def _released_label_mappings() -> dict[str, Any]:
    payload = json.loads(_LABEL_MAPPINGS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{_LABEL_MAPPINGS_PATH} must contain an object")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("Released label mappings do not match the pinned dataset revision")
    return payload


def released_questions_version(model_id: str) -> int | None:
    versions = _released_label_mappings().get("questions_versions")
    if not isinstance(versions, Mapping):
        raise TypeError("Released questions versions must contain an object")
    value = versions.get(str(model_id))
    return int(value) if value is not None else None


def released_parameter_display_mapping(model_id: str) -> dict[str, str]:
    mappings = _released_label_mappings().get("parameter_display_mappings")
    if not isinstance(mappings, Mapping):
        raise TypeError("Released parameter display mappings must contain an object")
    model_mapping = mappings.get(str(model_id))
    if not isinstance(model_mapping, Mapping):
        return {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in model_mapping.items()
        if str(key).strip() and str(value).strip()
    }
