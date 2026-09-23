from __future__ import annotations

from shared.noise_snr import hash_string


def test_hash_string_matches_released_noise_seed_values() -> None:
    assert hash_string("") == 2166136261
    assert hash_string("a") == 3826002220
