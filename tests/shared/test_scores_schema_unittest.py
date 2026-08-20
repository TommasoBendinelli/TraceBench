from __future__ import annotations

import unittest

from shared.scores_schema import validate_scores_payload


class TestScoresSchema(unittest.TestCase):
    def test_validate_scores_payload_accepts_documented_schema(self) -> None:
        payload = validate_scores_payload(
            {
                "agent_run_id": "demo_run",
                "is_correct_format": True,
                "final_metric_test": {
                    "average_top1_accuracy": 1.0,
                    "average_shortlist_score": 0.5,
                    "average_num_answers": 2.0,
                },
                "sample_results": {
                    "sample_a": {
                        "predictions": ["class_a", "class_b"],
                        "top1_correct": False,
                        "shortlist_score": 0.5,
                        "num_answers": 2,
                        "sample_type": "test",
                    }
                },
            }
        )

        self.assertEqual(payload.final_metric_test.average_top1_accuracy, 1.0)
        self.assertEqual(payload.is_correct_format, True)
        self.assertEqual(payload.final_metric_test.average_shortlist_score, 0.5)
        self.assertEqual(payload.final_metric_test.average_num_answers, 2.0)
        self.assertEqual(
            payload.sample_results["sample_a"].predictions,
            ["class_a", "class_b"],
        )

    def test_validate_scores_payload_accepts_code_other_metric(self) -> None:
        payload = validate_scores_payload(
            {
                "agent_run_id": "demo_run",
                "final_metric_test": {
                    "average_top1_accuracy": 0.75,
                    "average_shortlist_score": 0.43,
                    "average_num_answers": 1.2,
                },
                "final_metric_other": {
                    "average_top1_accuracy": 0.25,
                    "average_shortlist_score": 0.2,
                    "average_num_answers": 3.0,
                },
                "sample_results": {
                    "sample_a": {
                        "predictions": ["class_a"],
                        "top1_correct": True,
                        "shortlist_score": 1.0,
                        "num_answers": 1,
                        "sample_type": "test",
                    },
                    "sample_b": {
                        "predictions": ["class_b"],
                        "top1_correct": False,
                        "shortlist_score": 0.0,
                        "num_answers": None,
                        "sample_type": "other",
                        "error": "ValueError: demo",
                    },
                },
            }
        )

        self.assertEqual(payload.final_metric_other.average_shortlist_score, 0.2)
        self.assertIsNone(payload.sample_results["sample_b"].num_answers)

    def test_validate_scores_payload_rejects_old_final_metric_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_metric_test"):
            validate_scores_payload(
                {
                    "agent_run_id": "demo_run",
                    "final_metric": {"accuracy": 1.0},
                    "sample_a": {
                        "prediction": ["class_a", "class_b"],
                        "is_correct": "maybe",
                        "ground_truth": "class_a",
                    },
                }
            )

    def test_validate_scores_payload_rejects_legacy_boolean_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "top1_correct"):
            validate_scores_payload(
                {
                    "agent_run_id": "demo_run",
                    "final_metric_test": {
                        "average_top1_accuracy": 1.0,
                        "average_shortlist_score": 1.0,
                        "average_num_answers": 1.0,
                    },
                    "sample_results": {
                        "sample_a": {
                            "predictions": ["class_a"],
                            "top1_correct": "yes",
                            "shortlist_score": 1.0,
                            "num_answers": 1,
                            "sample_type": "test",
                        },
                    },
                }
            )

    def test_validate_scores_payload_rejects_duplicate_prediction_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            validate_scores_payload(
                {
                    "agent_run_id": "demo_run",
                    "final_metric_test": {
                        "average_top1_accuracy": 1.0,
                        "average_shortlist_score": 1.0,
                        "average_num_answers": 2.0,
                    },
                    "sample_results": {
                        "sample_a": {
                            "predictions": ["class_a", "class_a"],
                            "top1_correct": False,
                            "shortlist_score": 1.0,
                            "num_answers": 2,
                            "sample_type": "test",
                        },
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
