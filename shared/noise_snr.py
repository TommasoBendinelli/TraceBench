"""Deterministic string hashing used by the released noise adders."""

from __future__ import annotations


def hash_string(value: str) -> int:
    """Match the hash used when generating the released question data."""
    h = 2166136261
    for ch in str(value):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)
