"""Tests for stretching algorithms."""

import numpy as np
import pytest

from processinator.stretching.algorithms import StretchAlgorithm, stretch


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """Simulate a typical FITS image: mostly dark with a few bright pixels."""
    rng = np.random.default_rng(42)
    # Background noise around 100, a few bright stars
    data = rng.normal(100, 10, (256, 256)).astype(np.float64)
    data[50, 50] = 50000  # bright star
    data[100, 150] = 30000  # medium star
    data[200, 200] = 65000  # very bright star
    return np.clip(data, 0, 65535)


@pytest.fixture
def rgb_image(grayscale_image: np.ndarray) -> np.ndarray:
    """RGB version of the test image."""
    return np.stack([grayscale_image, grayscale_image * 0.8, grayscale_image * 0.6], axis=2)


class TestStretchOutputRange:
    """All algorithms should produce output in [0, 1]."""

    @pytest.mark.parametrize("algo", list(StretchAlgorithm))
    def test_output_range_grayscale(self, grayscale_image: np.ndarray, algo: StretchAlgorithm):
        result = stretch(grayscale_image, algorithm=algo)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("algo", list(StretchAlgorithm))
    def test_output_range_rgb(self, rgb_image: np.ndarray, algo: StretchAlgorithm):
        result = stretch(rgb_image, algorithm=algo)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestStretchShape:
    """Output shape should match input shape."""

    @pytest.mark.parametrize("algo", list(StretchAlgorithm))
    def test_shape_preserved_grayscale(self, grayscale_image: np.ndarray, algo: StretchAlgorithm):
        result = stretch(grayscale_image, algorithm=algo)
        assert result.shape == grayscale_image.shape

    @pytest.mark.parametrize("algo", list(StretchAlgorithm))
    def test_shape_preserved_rgb(self, rgb_image: np.ndarray, algo: StretchAlgorithm):
        result = stretch(rgb_image, algorithm=algo)
        assert result.shape == rgb_image.shape


class TestStretchBehavior:
    """Verify that stretching actually changes the data distribution."""

    def test_mtf_reveals_background(self, grayscale_image: np.ndarray):
        result = stretch(grayscale_image, algorithm=StretchAlgorithm.MTF)
        # After MTF stretch, the median should be closer to bg_percent (0.15)
        median = np.median(result)
        assert 0.01 < median < 0.5

    def test_linear_uses_percentiles(self, grayscale_image: np.ndarray):
        result = stretch(grayscale_image, algorithm=StretchAlgorithm.LINEAR)
        # Most values should be in the middle range after linear stretch
        assert np.median(result) > 0.1

    def test_constant_image_produces_zeros(self):
        data = np.ones((64, 64)) * 1000.0
        result = stretch(data, algorithm=StretchAlgorithm.MTF)
        assert np.allclose(result, 0.0)

    def test_default_algorithm_is_mtf(self, grayscale_image: np.ndarray):
        default_result = stretch(grayscale_image)
        mtf_result = stretch(grayscale_image, algorithm=StretchAlgorithm.MTF)
        np.testing.assert_array_equal(default_result, mtf_result)
