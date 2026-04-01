"""Tests for background gradient removal."""

import numpy as np

from processinator.gradient import remove_gradient
from processinator.synthetic import SyntheticImage, make_test_image


def _background_tilt(img: np.ndarray) -> float:
    """Spread of row/column medians — a star-robust flatness metric."""
    row_med = np.median(img, axis=1)
    col_med = np.median(img, axis=0)
    return float(np.ptp(row_med) + np.ptp(col_med))


class TestGradientRemoval:
    def test_flattens_background(self, gradient_field: SyntheticImage):
        data = gradient_field.data / gradient_field.data.max()
        result = remove_gradient(data)
        assert _background_tilt(result) < 0.35 * _background_tilt(data)

    def test_no_gradient_is_gentle(self, mono_field: SyntheticImage):
        """On an already-flat image the fitted surface should be near-constant."""
        data = mono_field.data / mono_field.data.max()
        result = remove_gradient(data)
        # The output is background-shifted toward 0 but should stay flat
        assert _background_tilt(result) < _background_tilt(data) + 0.01

    def test_stars_survive(self, gradient_field: SyntheticImage):
        """Star contrast above local background must be preserved."""
        scale = gradient_field.data.max()
        data = gradient_field.data / scale
        result = remove_gradient(data)

        bright = gradient_field.stars[gradient_field.stars[:, 2] > 5000]
        assert len(bright) > 0
        for x, y, amp in bright:
            xi, yi = int(round(x)), int(round(y))
            local_bg = np.median(result[max(0, yi - 12) : yi + 12, max(0, xi - 12) : xi + 12])
            contrast = result[yi, xi] - local_bg
            # Expected contrast in normalized units (clipping may trim the top)
            expected = amp / scale
            assert contrast > 0.5 * min(expected, 1.0 - local_bg)

    def test_shape_preserved_mono_and_rgb(self):
        mono = make_test_image((128, 128), gradient_amplitude=300.0, seed=6)
        rgb = make_test_image((128, 128), rgb=True, gradient_amplitude=300.0, seed=6)
        for img in (mono, rgb):
            data = img.data / img.data.max()
            result = remove_gradient(data)
            assert result.shape == data.shape

    def test_output_clipped_to_unit_range(self, gradient_field: SyntheticImage):
        data = gradient_field.data / gradient_field.data.max()
        result = remove_gradient(data)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
