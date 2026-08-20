from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.benchmark_utils import CONTEXT_LEVELS, parse_context_and_shot
from shared.context_levels import (
    BENCHMARK_CONTEXT_VALUES,
    PARSEABLE_CONTEXT_VALUES,
    QUESTION_CONTEXT_VALUES,
    TSENV_CONTEXT_VALUES,
)


class TestContextLevels(unittest.TestCase):
    def test_benchmark_utils_reexports_shared_benchmark_contexts(self) -> None:
        self.assertEqual(CONTEXT_LEVELS, BENCHMARK_CONTEXT_VALUES)

    def test_shared_context_sets_have_expected_scope(self) -> None:
        self.assertEqual(
            TSENV_CONTEXT_VALUES,
            ("none", "low", "high", "ground_truth"),
        )
        self.assertEqual(
            QUESTION_CONTEXT_VALUES,
            ("none", "low", "high", "ground_truth", "unknown"),
        )
        self.assertEqual(
            PARSEABLE_CONTEXT_VALUES,
            ("ground_truth", "high", "medium", "low", "none"),
        )

    def test_parse_context_and_shot_rejects_removed_context_token(self) -> None:
        self.assertEqual(
            parse_context_and_shot("ucr_removed_few_shot"),
            (None, "few_shot", True),
        )

    def test_parse_context_and_shot_rejects_removed_tsenv_alias(self) -> None:
        self.assertEqual(
            parse_context_and_shot("tsenv_cls_removed_zero_shot"),
            (None, "zero_shot", False),
        )


if __name__ == "__main__":
    unittest.main()
