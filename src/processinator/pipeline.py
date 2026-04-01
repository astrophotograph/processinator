"""Processing pipeline for astronomy images.

Composes processing steps (gradient removal, denoising, stretching) into
a configurable pipeline. Each step can be enabled/disabled independently.

Stage order: autocrop detection → normalize → gradient removal →
denoise → renormalize → stretch. All stages after normalization operate
on [0, 1] data, so they compose without rescaling tricks.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from processinator.autocrop import autocrop as _autocrop
from processinator.denoise import denoise
from processinator.gradient import remove_gradient
from processinator.stretching.algorithms import (
    StretchAlgorithm,
    _normalize_to_01_with_stats,
    stretch,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline.

    Attributes:
        autocrop: Detect dark stacking edges and exclude them from all
            statistics (the frame itself is never cropped). Default True.
        gradient_removal: Enable background gradient removal. Default True.
        gradient_order: Polynomial order for gradient model (1-3). Default 2.
        gradient_sigma: Sigma clip for gradient background sampling. Default 2.5.
        denoise: Enable starlet wavelet denoising. Default False.
        denoise_scales: Number of wavelet scales to threshold. Default 4.
        denoise_threshold: Threshold in noise sigmas. Default 3.0.
        denoise_mode: "hard" or "soft" thresholding. Default "hard".
        stretch_algorithm: Stretch algorithm to use. Default MTF.
        stretch_kwargs: Additional kwargs passed to the stretch function.
    """

    autocrop: bool = True
    gradient_removal: bool = True
    gradient_order: int = 2
    gradient_sigma: float = 2.5
    denoise: bool = False
    denoise_scales: int = 4
    denoise_threshold: float = 3.0
    denoise_mode: str = "hard"
    stretch_algorithm: StretchAlgorithm = StretchAlgorithm.MTF
    stretch_kwargs: dict = field(default_factory=dict)


def process(
    data: NDArray[np.floating],
    config: PipelineConfig | None = None,
) -> NDArray[np.floating]:
    """Run the processing pipeline on image data.

    Args:
        data: Raw FITS image data, shape (H, W) or (H, W, 3).
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        Processed image normalized to [0.0, 1.0], same shape as input.
    """
    if config is None:
        config = PipelineConfig()

    result = data.astype(np.float64)
    h, w = result.shape[:2]

    # Detect dark stacking edges once; use the interior for all statistics
    crop = (0, 0, 0, 0)
    if config.autocrop:
        _, crop = _autocrop(result)
        if any(v > 0 for v in crop):
            logger.info(
                "autocrop: excluding edges top=%d bottom=%d left=%d right=%d from stats",
                *crop,
            )

    def interior(arr: NDArray[np.floating]) -> NDArray[np.floating]:
        top, bottom, left, right = crop
        if not any(v > 0 for v in crop):
            return arr
        return arr[top : h - bottom, left : w - right]

    # Normalize to [0, 1]; percentiles come from the interior
    result = _normalize_to_01_with_stats(result, interior(result))

    if config.gradient_removal:
        result = remove_gradient(
            result,
            order=config.gradient_order,
            sigma_clip=config.gradient_sigma,
        )

    if config.denoise:
        result = denoise(
            result,
            n_scales=config.denoise_scales,
            threshold=config.denoise_threshold,
            mode=config.denoise_mode,
        )

    # Gradient removal / denoising shift the data range; renormalize so the
    # stretch sees the full [0, 1] span (keeps bright stars near white)
    if config.gradient_removal or config.denoise:
        result = _normalize_to_01_with_stats(result, interior(result))

    return stretch(
        result,
        algorithm=config.stretch_algorithm,
        pre_normalized=True,
        **config.stretch_kwargs,
    )
