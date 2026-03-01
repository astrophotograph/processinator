"""Image stretching algorithms for astronomy images.

Converts linear FITS data (where most detail is in low pixel values) into
visually useful images by applying nonlinear transfer functions.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class StretchAlgorithm(Enum):
    """Available stretch algorithms."""

    MTF = "mtf"
    """Midtones Transfer Function (GraXpert-style). Default. Good all-around choice."""

    ARCSINH = "arcsinh"
    """Inverse hyperbolic sine. Preserves color ratios well."""

    LOG = "log"
    """Logarithmic stretch. Good for high dynamic range images."""

    LINEAR = "linear"
    """Simple percentile-based linear stretch."""

    STATISTICAL = "statistical"
    """Gamma correction targeting a specific median brightness."""


def stretch(
    data: NDArray[np.floating],
    algorithm: StretchAlgorithm = StretchAlgorithm.MTF,
    **kwargs: float,
) -> NDArray[np.floating]:
    """Stretch image data from linear to nonlinear for display.

    Args:
        data: Image array, shape (H, W) or (H, W, 3). Values should be in
            their original FITS range (not pre-normalized).
        algorithm: Which stretch algorithm to use.
        **kwargs: Algorithm-specific parameters (see individual functions).

    Returns:
        Stretched image normalized to [0.0, 1.0], same shape as input.
    """
    normalized = _normalize_to_01(data)

    match algorithm:
        case StretchAlgorithm.MTF:
            return _stretch_mtf(normalized, **kwargs)
        case StretchAlgorithm.ARCSINH:
            return _stretch_arcsinh(normalized, **kwargs)
        case StretchAlgorithm.LOG:
            return _stretch_log(normalized, **kwargs)
        case StretchAlgorithm.LINEAR:
            return _stretch_linear(normalized, **kwargs)
        case StretchAlgorithm.STATISTICAL:
            return _stretch_statistical(normalized, **kwargs)


def _normalize_to_01(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalize raw FITS data to [0, 1] range."""
    result = data.astype(np.float64)
    vmin = np.nanmin(result)
    vmax = np.nanmax(result)
    if vmax - vmin == 0:
        return np.zeros_like(result)
    return (result - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# MTF (Midtones Transfer Function) - adapted from pyscopinator/GraXpert
# ---------------------------------------------------------------------------


def _mtf(m: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply Midtones Transfer Function.

    MTF(m, x) = (m - 1) * x / ((2m - 1) * x - m)
    """
    numerator = (m - 1.0) * x
    denominator = (2.0 * m - 1.0) * x - m
    # Avoid division by zero
    safe = np.where(np.abs(denominator) < 1e-10, x, numerator / denominator)
    return np.clip(safe, 0.0, 1.0)


def _stretch_mtf(
    data: NDArray[np.floating],
    bg_percent: float = 0.15,
    sigma: float = 3.0,
) -> NDArray[np.floating]:
    """MTF stretch using background/sigma clipping.

    Args:
        data: Normalized [0, 1] image data.
        bg_percent: Target background level (0-1). Default 0.15.
        sigma: Number of sigma above background for shadow clipping. Default 3.0.
    """
    result = data.copy()

    # Process each channel (or the single channel for grayscale)
    if result.ndim == 2:
        channels = [result]
    else:
        channels = [result[:, :, i] for i in range(result.shape[2])]

    processed = []
    for channel in channels:
        flat = channel.ravel()
        valid = flat[(flat > 0.0) & (flat < 1.0)]

        if len(valid) == 0:
            processed.append(channel)
            continue

        median = np.median(valid)
        mad = np.median(np.abs(valid - median))

        shadow_clip = max(0.0, median - sigma * mad * 1.4826)
        highlight_clip = 1.0

        # Normalize between clipping points
        stretched = np.clip(channel, shadow_clip, highlight_clip)
        if highlight_clip - shadow_clip > 0:
            stretched = (stretched - shadow_clip) / (highlight_clip - shadow_clip)

        # Calculate midtone balance for target background
        median_norm = (median - shadow_clip) / (highlight_clip - shadow_clip)
        if 0 < median_norm < 1 and bg_percent > 0:
            midtone = (
                median_norm
                * (bg_percent - 1.0)
                / (2.0 * bg_percent * median_norm - bg_percent - median_norm)
            )
            midtone = np.clip(midtone, 0.01, 0.99)
        else:
            midtone = 0.5

        stretched = _mtf(midtone, stretched)
        processed.append(stretched)

    if result.ndim == 2:
        return processed[0]
    for i, ch in enumerate(processed):
        result[:, :, i] = ch
    return result


# ---------------------------------------------------------------------------
# Arcsinh stretch - adapted from astra
# ---------------------------------------------------------------------------


def _stretch_arcsinh(
    data: NDArray[np.floating],
    factor: float = 0.15,
) -> NDArray[np.floating]:
    """Inverse hyperbolic sine stretch. Preserves color ratios.

    Args:
        data: Normalized [0, 1] image data.
        factor: Controls stretch aggressiveness. Smaller = more aggressive. Default 0.15.
    """
    scale = 1.0 / factor
    result = np.arcsinh(data * scale) / np.arcsinh(scale)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Log stretch - adapted from astra/pyscopinator
# ---------------------------------------------------------------------------


def _stretch_log(
    data: NDArray[np.floating],
    factor: float = 0.15,
) -> NDArray[np.floating]:
    """Logarithmic stretch. Good for high dynamic range.

    Args:
        data: Normalized [0, 1] image data.
        factor: Controls stretch aggressiveness. Smaller = more aggressive. Default 0.15.
    """
    offset = factor * 0.01
    result = np.log1p(data / offset) / np.log1p(1.0 / offset)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Linear stretch
# ---------------------------------------------------------------------------


def _stretch_linear(
    data: NDArray[np.floating],
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> NDArray[np.floating]:
    """Simple percentile-based linear stretch.

    Args:
        data: Normalized [0, 1] image data.
        low_percentile: Lower clipping percentile. Default 1.0.
        high_percentile: Upper clipping percentile. Default 99.0.
    """
    vmin = np.percentile(data, low_percentile)
    vmax = np.percentile(data, high_percentile)
    if vmax - vmin == 0:
        return data
    result = (data - vmin) / (vmax - vmin)
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Statistical stretch (gamma correction) - adapted from astra
# ---------------------------------------------------------------------------


def _stretch_statistical(
    data: NDArray[np.floating],
    target_median: float = 0.15,
    low_percentile: float = 0.5,
    high_percentile: float = 99.9,
) -> NDArray[np.floating]:
    """Stretch using percentile clipping then gamma correction.

    Clips to percentile range, then applies gamma correction to achieve
    a target median brightness.

    Args:
        data: Normalized [0, 1] image data.
        target_median: Desired median value after stretch. Default 0.15.
        low_percentile: Black point percentile. Default 0.5.
        high_percentile: White point percentile. Default 99.9.
    """
    vmin = np.percentile(data, low_percentile)
    vmax = np.percentile(data, high_percentile)
    if vmax - vmin == 0:
        return data

    result = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)

    current_median = np.median(result[result > 0])
    if current_median > 0 and current_median != target_median:
        gamma = np.log(target_median) / np.log(current_median)
        gamma = np.clip(gamma, 0.2, 5.0)
        result = np.power(result, gamma)

    return np.clip(result, 0.0, 1.0)
