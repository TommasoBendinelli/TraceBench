from __future__ import annotations

import json
import re
from typing import Any, Optional


class SubscriptionError(RuntimeError):
    """Raised when a Gemini subscription or quota error is detected."""


QUOTA_EXCEEDED_MARKER = "You have exhausted your capacity on this model"
FALLBACK_SUCCESS_MARKER = "GEMINI_API_FALLBACK_SUCCESS"


def parse_first_json(text: str) -> Optional[Any]:
    """Return the first JSON object/array embedded in text, if any."""
    if not isinstance(text, str):
        return None

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def is_subscription_error_payload(payload: Any) -> bool:
    """Check whether a decoded JSON payload matches the Gemini error shape."""
    if not isinstance(payload, dict):
        return False
    error_block = payload.get("error")
    return isinstance(error_block, dict) and error_block.get("type") == "Error"


def has_quota_exceeded(text: Optional[str]) -> bool:
    """Check text content for a Gemini quota exceeded message."""
    return bool(text) and QUOTA_EXCEEDED_MARKER in text


def has_subscription_error(text: Optional[str]) -> bool:
    """Check text content for a Gemini subscription error JSON payload."""
    if not text:
        return False
    if FALLBACK_SUCCESS_MARKER in text:
        return False
    if has_quota_exceeded(text):
        return True
    parsed = parse_first_json(text or "")
    return is_subscription_error_payload(parsed)


_AGENT_TIMEOUT_PATTERN = re.compile(
    r"Agent timed out after\s+(?P<seconds>\d+(?:\.\d+)?)s\s+for task\s+(?P<task_id>question_\d+)",
    flags=re.IGNORECASE,
)


def find_agent_timeout_entries(text: Optional[str]) -> list[tuple[str, float]]:
    """Extract agent timeout lines like 'Agent timed out after 900.0s for task question_68'."""
    if not text:
        return []

    entries: list[tuple[str, float]] = []
    for match in _AGENT_TIMEOUT_PATTERN.finditer(text):
        seconds_raw = match.group("seconds")
        try:
            seconds = float(seconds_raw)
        except (TypeError, ValueError):
            continue
        entries.append((match.group("task_id"), seconds))
    return entries


def has_agent_timeout(text: Optional[str]) -> bool:
    """Convenience helper to check if any agent timeout entries exist in a log blob."""
    return bool(find_agent_timeout_entries(text))
