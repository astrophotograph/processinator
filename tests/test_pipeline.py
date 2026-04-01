"""End-to-end tests for the processing pipeline."""

import numpy as np
import pytest

from processinator.pipeline import PipelineConfig, process
from processinator.stretching.algorithms import StretchAlgorithm
from processinator.synthetic import SyntheticImage, make_test_image


@pytest.fixture(scope="module")
def full_frame() -> SyntheticImage:
    """The works: gradient, nebula, dark edges, hot pixels, RGB color cast."""
    return make_test_image(
        rgb=True,
        gradient_amplitude=300.0,
        nebula_amplitude=150.0,
        dark_edges=(10, 0, 14, 0),
        hot_pixels=20,
        seed=13,
    )


class TestPipeline:
    def test_default_config(self, full_frame: SyntheticImage):
        result = process(full_frame.data)
        assert result.shape == full_frame.data.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_steps_enabled(self, full_frame: SyntheticImage):
        config = PipelineConfig(denoise=True)
        result = process(full_frame.data, config)
        assert result.shape == full_frame.data.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # A stretched image should have a visible (non-black) background
        assert float(np.median(result)) > 0.01

    def test_all_steps_disabled(self, mono_field: SyntheticImage):
        config = PipelineConfig(autocrop=False, gradient_removal=False, denoise=False)
        result = process(mono_field.data, config)
        assert result.shape == mono_field.data.shape
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_linked_mtf_neutralizes_color_cast(self, full_frame: SyntheticImage):
        """The synthetic RGB image has a strong color cast; linked MTF should
        equalize the per-channel background medians."""
        result = process(full_frame.data)
        interior = result[24:-24, 24:-24, :]
        medians = [float(np.median(interior[:, :, i])) for i in range(3)]
        assert max(medians) - min(medians) < 0.05

    def test_denoise_smooths_output(self, full_frame: SyntheticImage):
        plain = process(full_frame.data, PipelineConfig(denoise=False))
        smoothed = process(full_frame.data, PipelineConfig(denoise=True))
        # Compare local roughness (std of Laplacian-ish residual) in a
        # star-free region; denoised output must be smoother
        region = np.s_[20:80, 20:80, 1]
        rough_plain = float(np.std(np.diff(plain[region], axis=0)))
        rough_smooth = float(np.std(np.diff(smoothed[region], axis=0)))
        assert rough_smooth < rough_plain

    def test_alternate_algorithm(self, mono_field: SyntheticImage):
        config = PipelineConfig(stretch_algorithm=StretchAlgorithm.ARCSINH)
        result = process(mono_field.data, config)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_stretch_kwargs_passthrough(self, mono_field: SyntheticImage):
        bright = process(mono_field.data, PipelineConfig(stretch_kwargs={"bg_percent": 0.4}))
        dim = process(mono_field.data, PipelineConfig(stretch_kwargs={"bg_percent": 0.05}))
        assert float(np.median(bright)) > float(np.median(dim))
