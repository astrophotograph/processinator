"""Processing pipeline for astronomy images.

Composes processing steps (gradient removal, stretching, etc.) into
a configurable pipeline. Each step can be enabled/disabled independently.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from processinator.gradient import remove_gradient
from processinator.stretching.algorithms import StretchAlgorithm, stretch


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline.

    Attributes:
        gradient_removal: Enable background gradient removal. Default True.
        gradient_order: Polynomial order for gradient model (1-3). Default 2.
        gradient_sigma: Sigma clip for gradient background sampling. Default 2.5.
        stretch_algorithm: Stretch algorithm to use. Default MTF.
        stretch_kwargs: Additional kwargs passed to the stretch function.
    """

    gradient_removal: bool = True
    gradient_order: int = 2
    gradient_sigma: float = 2.5
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
        Processed image normalized to [0.0, 1.0].
    """
    if config is None:
        config = PipelineConfig()

    result = data.astype(np.float64)

    # Step 1: Normalize to [0, 1] for gradient removal
    if config.gradient_removal:
        # Quick normalize before gradient removal
        if result.ndim == 3:
            for i in range(result.shape[2]):
                ch = result[:, :, i]
                vmin, vmax = float(np.nanpercentile(ch, 0.1)), float(np.nanpercentile(ch, 99.99))
                if vmax > vmin:
                    result[:, :, i] = np.clip((ch - vmin) / (vmax - vmin), 0, 1)
        else:
            vmin, vmax = float(np.nanpercentile(result, 0.1)), float(np.nanpercentile(result, 99.99))
            if vmax > vmin:
                result = np.clip((result - vmin) / (vmax - vmin), 0, 1)

        result = remove_gradient(
            result,
            order=config.gradient_order,
            sigma_clip=config.gradient_sigma,
        )

    # Step 2: Stretch
    if config.gradient_removal:
        # Data is already in [0, 1] from gradient removal — scale back up
        # so the stretch's internal normalization works correctly
        result = result * 65535.0

    result = stretch(result, algorithm=config.stretch_algorithm, **config.stretch_kwargs)

    return result
