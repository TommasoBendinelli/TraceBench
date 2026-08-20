from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.interface.sample_manifest_json import compute_train_test_sample_hash
from shared import tsenv_metadata
from shared.tsenv_metadata import label_for_question_sample, load_metadata_payload, question_sample_paths


_ACCURACY = {
    "euclidean_knn": 0.0,
    "euclidean_centroid": 0.0,
    "correlation_nn": 0.0,
}


def test_load_metadata_payload_accepts_documented_sample_manifest_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "BallDrop"
    model_dir.mkdir(parents=True, exist_ok=True)

    questions_payload = {
        "questions": {
            "q_demo": {
                "recipe_info": {
                    "shot_slug": "maple_01234",
                    "question_seed": 3,
                "test_set_slug": "home",
                }
            }
        },
        "label_int_mapping": {
            "drag coefficient": 0,
            "no parameter change": 1,
        },
    }
    (model_dir / "questions.json").write_text(json.dumps(questions_payload), encoding="utf-8")

    sample_manifest_payload = {
        "maple_01234": [
            {
                "train_samples_baselines": [],
                "train_samples": [],
                "train_samples_hashes": [],
                "test_samples_baselines": [
                    "baseline_test_a",
                    "baseline_test_b",
                ],
                "test_samples": [
                    "test_a",
                    "test_b",
                ],
                "test_samples_hashes": [
                    "hash_test_a",
                    "hash_test_b",
                ],
                "other_samples": [
                    "other_a",
                ],
                "seed": 3,
                "test_set_slug": "home",
                "shot_slug_recipe": {
                    "is_adversarial": None,
                    "number_train_samples_per_class": 0,
                    "number_test_samples": 2,
                },
                "accuracy_with_baselines": dict(_ACCURACY),
                "accuracy_with_baselines_all_samples": dict(_ACCURACY),
                "train_test_sample_hash": compute_train_test_sample_hash(
                    train_samples_hashes=[],
                    test_samples_hashes=["hash_test_a", "hash_test_b"],
                ),
            }
        ]
    }
    (model_dir / "sample_manifest.json").write_text(
        json.dumps(sample_manifest_payload),
        encoding="utf-8",
    )

    payload = load_metadata_payload(model_dir / "questions.json")

    assert isinstance(payload.get("_sample_manifest_payload"), dict)
    question = payload["questions"]["q_demo"]
    assert question_sample_paths(payload, question=question, subset="test") == [
        "dataframes/test_a.parquet",
        "dataframes/test_b.parquet",
    ]
    assert question_sample_paths(payload, question=question, subset="other") == [
        "dataframes/other_a.parquet",
    ]


def test_none_description_labels_map_to_display_sorted_class_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_root = tmp_path / "models"
    model_dir = models_root / "DemoModel"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "exposed_variables": {
                    "parameters": {
                        "internal_z": {},
                        "internal_a": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tsenv_metadata, "_MODELS_ROOT", models_root)
    payload = {
        "environment_name": "DemoModel",
        "ground_truth_information": {
            "parameter_display_mapping": {
                "internal_z": "alpha display",
                "internal_a": "zeta display",
            },
            "interventions": {
                "sample_alpha": {"changed_parameter": "internal_z"},
                "sample_zeta": {"changed_parameter": "internal_a"},
                "sample_none": {},
            },
        },
    }
    question = {
        "recipe_info": {"desc_level": "none"},
        "question_text": {
            "allowed_labels": [
                "class_0",
                "class_1",
                "class_2",
            ]
        },
    }

    assert label_for_question_sample(
        payload,
        question=question,
        sample_path="dataframes/sample_alpha.parquet",
    ) == "class_0"
    assert label_for_question_sample(
        payload,
        question=question,
        sample_path="dataframes/sample_zeta.parquet",
    ) == "class_1"
    assert label_for_question_sample(
        payload,
        question=question,
        sample_path="dataframes/sample_none.parquet",
    ) == "class_2"


def test_load_metadata_payload_derives_other_samples_for_legacy_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "BallDrop"
    dataframes_dir = model_dir / "dataframes"
    dataframes_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in ("test_a", "test_b", "other_a"):
        (dataframes_dir / f"{sample_id}.parquet").write_text("", encoding="utf-8")
    (model_dir / "questions.json").write_text(
        json.dumps(
            {
                "questions": {
                    "q_demo": {
                        "recipe_info": {
                            "shot_slug": "maple_01234",
                            "question_seed": 3,
                "test_set_slug": "home",
                        }
                    }
                },
                "label_int_mapping": {
                    "drag coefficient": 0,
                    "no parameter change": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "sample_manifest.json").write_text(
        json.dumps(
            {
                "maple_01234": [
                    {
                        "train_samples_baselines": [],
                        "train_samples": [],
                        "train_samples_hashes": [],
                        "test_samples_baselines": ["baseline_test_a", "baseline_test_b"],
                        "test_samples": ["test_a", "test_b"],
                        "test_samples_hashes": ["hash_test_a", "hash_test_b"],
                        "seed": 3,
                        "test_set_slug": "home",
                        "shot_slug_recipe": {
                            "is_adversarial": None,
                            "number_train_samples_per_class": 0,
                            "number_test_samples": 2,
                        },
                        "accuracy_with_baselines": dict(_ACCURACY),
                        "accuracy_with_baselines_all_samples": dict(_ACCURACY),
                        "train_test_sample_hash": compute_train_test_sample_hash(
                            train_samples_hashes=[],
                            test_samples_hashes=["hash_test_a", "hash_test_b"],
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    payload = load_metadata_payload(model_dir / "questions.json")

    assert question_sample_paths(
        payload,
        question=payload["questions"]["q_demo"],
        subset="other",
    ) == ["dataframes/other_a.parquet"]


def test_load_metadata_payload_rejects_legacy_list_manifest(tmp_path: Path) -> None:
    model_dir = tmp_path / "BallDrop"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "questions.json").write_text(json.dumps({"questions": {}}), encoding="utf-8")
    (model_dir / "sample_manifest.json").write_text(
        json.dumps(
            [
                {
                    "train_samples_baselines": [],
                    "train_samples": [],
                    "test_samples_baselines": ["baseline_test_a"],
                    "test_samples": ["test_a"],
                    "seed": 0,
                    "id": {
                        "shot_slug": "legacy_0",
                        "is_adversarial": True,
                        "number_train_samples_per_class": 0,
                        "number_test_samples": 1,
                    },
                    "hash": "legacy_hash",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must contain a JSON object keyed by shot_slug"):
        load_metadata_payload(model_dir / "questions.json")
