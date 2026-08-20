"""Paths and immutable release identifiers used by the public interface."""

from __future__ import annotations

import os
from pathlib import Path


DATASET_REPO_ID = "eth-siplab/tracebench"
DATASET_REVISION = "ec08d06b0731315a188bc723e668ab68234da029"


def default_questions_root(repo_root: Path, explicit: Path | None = None) -> Path:
    """Resolve the public question bundle while retaining the legacy location."""

    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    for variable in ("TRACEBENCH_QUESTIONS_ROOT", "TSENV_QUESTIONS_ROOT"):
        value = str(os.environ.get(variable) or "").strip()
        if value:
            return Path(value).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()
    public_root = root / "data" / "questions"
    legacy_root = root / "tsENV_questions"
    if legacy_root.exists() and not public_root.exists():
        return legacy_root
    return public_root
