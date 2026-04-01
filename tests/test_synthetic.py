"""Sanity tests for the synthetic image generator itself."""

import numpy as np

from processinator.stretching.fits_io import read_fits
from processinator.synthetic import make_test_image, save_fits


class TestGenerator:
    def test_mono_shape(self):
        img = make_test_image((128, 200), seed=1)
        assert img.data.shape == (128, 200)
        assert img.clean.shape == (128, 200)

    def test_rgb_shape(self):
        img = make_test_image((128, 200), rgb=True, seed=1)
        assert img.data.shape == (128, 200, 3)

    def test_deterministic(self):
        a = make_test_image(seed=99)
        b = make_test_image(seed=99)
        np.testing.assert_array_equal(a.data, b.data)

    def test_stars_present_and_recorded(self):
        img = make_test_image(n_stars=30, seed=1)
        assert img.stars.shape == (30, 3)
        # Brightest star should tower over the background
        x, y, amp = img.stars[np.argmax(img.stars[:, 2])]
        assert img.clean[int(round(y)), int(round(x))] > img.background + 0.8 * amp

    def test_noise_matches_config(self):
        img = make_test_image(noise_sigma=25.0, n_stars=0, seed=1)
        residual_std = float(np.std(img.data - img.clean))
        assert 22.0 < residual_std < 28.0

    def test_dark_edges_are_dark(self):
        img = make_test_image(dark_edges=(15, 0, 0, 0), seed=1)
        edge_median = float(np.median(img.data[:15]))
        interior_median = float(np.median(img.data[30:-30]))
        assert edge_median < 0.1 * interior_median

    def test_data_non_negative(self):
        img = make_test_image(noise_sigma=100.0, seed=1)
        assert img.data.min() >= 0.0


class TestFitsRoundTrip:
    def test_mono_roundtrip(self, tmp_path):
        img = make_test_image((96, 128), seed=8)
        path = save_fits(img.data, tmp_path / "mono.fits")
        loaded, _header = read_fits(path)
        assert loaded.shape == (96, 128)
        np.testing.assert_allclose(loaded, img.data, rtol=1e-6)

    def test_rgb_roundtrip(self, tmp_path):
        img = make_test_image((96, 128), rgb=True, seed=8)
        path = save_fits(img.data, tmp_path / "rgb.fits")
        loaded, _header = read_fits(path)
        # Stored channels-first, read back as (H, W, 3)
        assert loaded.shape == (96, 128, 3)
        np.testing.assert_allclose(loaded, img.data, rtol=1e-6)
