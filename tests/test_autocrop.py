"""Tests for dark stacking-edge detection and cropping."""

import numpy as np

from processinator.autocrop import autocrop
from processinator.synthetic import SyntheticImage, make_test_image


class TestEdgeDetection:
    def test_detects_configured_edges(self, edged_field: SyntheticImage):
        _, crop = autocrop(edged_field.data)
        assert crop == edged_field.dark_edges

    def test_cropped_shape_matches(self, edged_field: SyntheticImage):
        cropped, (top, bottom, left, right) = autocrop(edged_field.data)
        h, w = edged_field.data.shape
        assert cropped.shape == (h - top - bottom, w - left - right)

    def test_clean_image_not_cropped(self, mono_field: SyntheticImage):
        cropped, crop = autocrop(mono_field.data)
        assert crop == (0, 0, 0, 0)
        assert cropped.shape == mono_field.data.shape

    def test_rgb_edges_detected(self):
        img = make_test_image(rgb=True, dark_edges=(10, 0, 0, 16), seed=9)
        cropped, crop = autocrop(img.data)
        assert crop == (10, 0, 0, 16)
        assert cropped.ndim == 3

    def test_subthreshold_edge_ignored(self):
        """Edges thinner than min_crop_fraction (1%) should not trigger a crop."""
        img = make_test_image((256, 256), dark_edges=(1, 0, 0, 0), seed=4)
        _, crop = autocrop(img.data)
        assert crop == (0, 0, 0, 0)

    def test_interior_content_unchanged(self, edged_field: SyntheticImage):
        cropped, (top, bottom, left, right) = autocrop(edged_field.data)
        h, w = edged_field.data.shape
        np.testing.assert_array_equal(
            cropped, edged_field.data[top : h - bottom, left : w - right]
        )
