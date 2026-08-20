from __future__ import annotations

import pytest

from shared.shortlist_score import sample_score


def test_sample_score_rewards_retrieval_and_penalizes_extra_answers() -> None:
    assert sample_score(
        ["a", "b"],
        "a",
    ) == pytest.approx(0.5)


def test_sample_score_returns_zero_when_correct_label_is_absent() -> None:
    assert (
        sample_score(
            ["b"],
            "a",
        )
        == 0.0
    )
