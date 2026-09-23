from __future__ import annotations

from pathlib import Path

import pytest

from shared.child_only_predictor import parameter_key_to_display_label


@pytest.mark.parametrize(
    "model,internal,display",
    [
        ("BallDrop", "drag_coeff", "drag coefficient"),
        ("BounceBall", "restitution_left", "coefficient of restitution of the left wall"),
        ("MassSlide", "plane_inclination", "plane inclination angle"),
    ],
)
def test_public_label_conversion_uses_pinned_release_mapping(
    model: str, internal: str, display: str
) -> None:
    assert parameter_key_to_display_label(
        model_id=model, parameter_key=internal
    ) == display


def test_explicit_model_root_does_not_use_pinned_release_mapping(tmp_path: Path) -> None:
    assert parameter_key_to_display_label(
        model_id="BallDrop",
        parameter_key="drag_coeff",
        models_root=tmp_path,
    ) == "drag_coeff"
