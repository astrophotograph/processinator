"""Tests for starlet wavelet denoising.

The key "astronomically safe" properties: noise goes down, star photometry
and large-scale structure survive.

Note on modes: hard thresholding (the default) preserves point-source
photometry exactly, which is why it is the astronomy standard. Soft
thresholding gives a smoother background but systematically dims stars,
so its quality tests use a starless field.
"""

import numpy as np
import pytest

from processinator.denoise import denoise
from processinator.synthetic import SyntheticImage, make_test_image


@pytest.fixture(scope="module")
def noisy() -> SyntheticImage:
    """Star field with nebulosity and noise."""
    return make_test_image(nebula_amplitude=300.0, seed=11)


@pytest.fixture(scope="module")
def starless() -> SyntheticImage:
    """Pure background + nebulosity — no point sources."""
    return make_test_image(n_stars=0, nebula_amplitude=300.0, seed=11)


class TestNoiseReduction:
    def test_mse_against_clean_improves(self, noisy: SyntheticImage):
        """Denoised image must be much closer to the noise-free truth."""
        result = denoise(noisy.data)
        mse_before = float(np.mean((noisy.data - noisy.clean) ** 2))
        mse_after = float(np.mean((result - noisy.clean) ** 2))
        assert mse_after < 0.5 * mse_before

    def test_background_noise_suppressed(self, starless: SyntheticImage):
        """In a starless field the residual noise should shrink a lot."""
        result = denoise(starless.data)
        std_before = float(np.std(starless.data - starless.clean))
        std_after = float(np.std(result - starless.clean))
        assert std_after < 0.5 * std_before

    def test_soft_mode_on_starless_field(self, starless: SyntheticImage):
        """Soft thresholding shines when there are no point sources."""
        result = denoise(starless.data, mode="soft")
        mse_before = float(np.mean((starless.data - starless.clean) ** 2))
        mse_after = float(np.mean((result - starless.clean) ** 2))
        assert mse_after < 0.2 * mse_before

    def test_higher_threshold_smooths_more(self, starless: SyntheticImage):
        gentle = denoise(starless.data, threshold=2.0)
        strong = denoise(starless.data, threshold=5.0)
        resid_gentle = float(np.std(gentle - starless.clean))
        resid_strong = float(np.std(strong - starless.clean))
        # Stronger threshold removes more noise in a starless field
        assert resid_strong <= resid_gentle


class TestSignalPreservation:
    def test_star_peaks_preserved_exactly_by_default(self, noisy: SyntheticImage):
        """Hard thresholding (default) keeps large coefficients bit-for-bit."""
        result = denoise(noisy.data)
        bright = noisy.stars[noisy.stars[:, 2] > 100 * noisy.noise_sigma]
        assert len(bright) > 0
        for x, y, _amp in bright:
            xi, yi = int(round(x)), int(round(y))
            assert result[yi, xi] == pytest.approx(float(noisy.data[yi, xi]), rel=1e-5)

    def test_soft_mode_star_loss_is_bounded(self, noisy: SyntheticImage):
        """Soft mode dims stars by ~the threshold sum — small for bright stars."""
        result = denoise(noisy.data, mode="soft")
        bright = noisy.stars[noisy.stars[:, 2] > 200 * noisy.noise_sigma]
        assert len(bright) > 0
        for x, y, _amp in bright:
            xi, yi = int(round(x)), int(round(y))
            assert result[yi, xi] > 0.9 * noisy.clean[yi, xi]

    def test_nebulosity_preserved(self, noisy: SyntheticImage):
        """Large-scale structure lives in the untouched residual."""
        result = denoise(noisy.data)
        block = 32
        h, w = noisy.clean.shape
        coarse_clean = noisy.clean[: h // block * block, : w // block * block]
        coarse_clean = coarse_clean.reshape(h // block, block, w // block, block).mean(axis=(1, 3))
        coarse_result = result[: h // block * block, : w // block * block]
        coarse_result = coarse_result.reshape(h // block, block, w // block, block).mean(
            axis=(1, 3)
        )
        np.testing.assert_allclose(coarse_result, coarse_clean, rtol=0.05)

    def test_noise_free_image_nearly_unchanged(self):
        img = make_test_image(noise_sigma=0.0, seed=5)
        result = denoise(img.data)
        # Nothing to remove — output should track the input closely
        assert float(np.median(np.abs(result - img.data))) < 1e-4 * float(img.data.max())


class TestInterface:
    def test_shape_and_dtype_mono(self, noisy: SyntheticImage):
        result = denoise(noisy.data)
        assert result.shape == noisy.data.shape
        assert result.dtype == np.float64

    def test_rgb(self):
        img = make_test_image(rgb=True, seed=11)
        result = denoise(img.data)
        assert result.shape == img.data.shape
        mse_before = float(np.mean((img.data - img.clean) ** 2))
        mse_after = float(np.mean((result - img.clean) ** 2))
        assert mse_after < mse_before

    def test_small_image_clamps_scales(self):
        img = make_test_image((48, 48), n_stars=5, seed=2)
        result = denoise(img.data, n_scales=8)
        assert result.shape == img.data.shape

    def test_invalid_mode_raises(self, noisy: SyntheticImage):
        with pytest.raises(ValueError, match="mode"):
            denoise(noisy.data, mode="banana")

    def test_deterministic(self, noisy: SyntheticImage):
        a = denoise(noisy.data)
        b = denoise(noisy.data)
        np.testing.assert_array_equal(a, b)
