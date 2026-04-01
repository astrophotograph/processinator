"""Shared fixtures: synthetic test images with ground truth."""

import pytest

from processinator.synthetic import SyntheticImage, make_test_image


@pytest.fixture(scope="session")
def mono_field() -> SyntheticImage:
    """Plain mono star field with noise."""
    return make_test_image(seed=42)


@pytest.fixture(scope="session")
def rgb_field() -> SyntheticImage:
    """RGB star field with a color cast."""
    return make_test_image(rgb=True, seed=42)


@pytest.fixture(scope="session")
def gradient_field() -> SyntheticImage:
    """Mono field with a strong sky gradient and some nebulosity."""
    return make_test_image(
        gradient_amplitude=400.0,
        nebula_amplitude=200.0,
        seed=7,
    )


@pytest.fixture(scope="session")
def edged_field() -> SyntheticImage:
    """Mono field with dark stacking edges on three sides."""
    return make_test_image(dark_edges=(12, 8, 20, 0), seed=3)
