"""Image stretching algorithms for astronomy images.

Converts linear FITS data (where most detail is in low pixel values) into
visually useful images by applying nonlinear transfer functions.

When JAX is installed, compute-heavy operations (transcendentals, statistics)
run on GPU automatically. Falls back to numpy otherwise.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from processinator.stretching.backend import from_numpy, to_numpy, xp


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
    """Normalize raw FITS data to [0, 1] range.

    For RGB images, normalizes each channel independently using percentile
    clipping to avoid hot pixels/bright stars from compressing the useful range.
    """
    result = data.astype(np.float64)

    if result.ndim == 3:
        for i in range(result.shape[2]):
            ch = result[:, :, i]
            # nanpercentile stays on numpy (JAX doesn't support it)
            vmin = float(np.nanpercentile(ch, 0.1))
            vmax = float(np.nanpercentile(ch, 99.99))
            if vmax - vmin > 0:
                ch_gpu = from_numpy(ch)
                result[:, :, i] = to_numpy(xp.clip((ch_gpu - vmin) / (vmax - vmin), 0, 1))
            else:
                result[:, :, i] = 0
        return result

    vmin = float(np.nanpercentile(result, 0.1))
    vmax = float(np.nanpercentile(result, 99.99))
    if vmax - vmin == 0:
        return np.zeros_like(result)
    r = from_numpy(result)
    return to_numpy(xp.clip((r - vmin) / (vmax - vmin), 0, 1))


# ---------------------------------------------------------------------------
# MTF (Midtones Transfer Function) - adapted from pyscopinator/GraXpert
# ---------------------------------------------------------------------------


def _mtf(m: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Apply Midtones Transfer Function.

    MTF(m, x) = (m - 1) * x / ((2m - 1) * x - m)
    """
    x_gpu = from_numpy(x)
    numerator = (m - 1.0) * x_gpu
    denominator = (2.0 * m - 1.0) * x_gpu - m
    # Avoid division by zero
    safe = xp.where(xp.abs(denominator) < 1e-10, x_gpu, numerator / denominator)
    return to_numpy(xp.clip(safe, 0.0, 1.0))


def _stretch_mtf(
    data: NDArray[np.floating],
    bg_percent: float = 0.15,
    sigma: float = 3.0,
    linked: bool = True,
) -> NDArray[np.floating]:
    """MTF stretch using background/sigma clipping.

    Args:
        data: Normalized [0, 1] image data.
        bg_percent: Target background level (0-1). Default 0.15.
        sigma: Number of sigma above background for shadow clipping. Default 3.0.
        linked: If True, use the same stretch parameters for all channels
                to preserve color balance. Default True.
    """
    result = data.copy()

    # Process each channel (or the single channel for grayscale)
    if result.ndim == 2:
        channels = [result]
    else:
        channels = [result[:, :, i] for i in range(result.shape[2])]

    # For linked mode: neutralize per-channel backgrounds then apply same stretch.
    # Steps:
    # 1. Compute each channel's background (median) and noise (MAD)
    # 2. Subtract backgrounds to neutralize sky color
    # 3. Rescale so all channels have the same range
    # 4. Apply shared MTF stretch
    if linked and len(channels) > 1:
        # Step 1: Per-channel statistics (numpy — median/MAD need sorting)
        medians = []
        mads = []
        for channel in channels:
            flat = channel.ravel()
            valid = flat[flat > 0.0]
            if len(valid) == 0:
                medians.append(0.0)
                mads.append(0.01)
                continue
            med = float(np.median(valid))
            mad = float(np.median(np.abs(valid - med)))
            medians.append(med)
            mads.append(max(mad, 1e-6))

        # Step 2: Background neutralization
        # Each channel's shadow clip = median - sigma * MAD * 1.4826
        shadows = [max(0.0, med - sigma * mad * 1.4826) for med, mad in zip(medians, mads)]

        # Step 3: Subtract shadows and equalize channel backgrounds.
        # After shadow subtraction, each channel's effective median differs.
        # We rescale so all channels have the same post-subtraction median
        # (the minimum), which neutralizes the sky color cast.
        post_medians = [max(med - shd, 1e-10) for med, shd in zip(medians, shadows)]
        ref_post_median = min(post_medians)

        processed = []
        for i, channel in enumerate(channels):
            ch_gpu = from_numpy(channel)
            stretched = ch_gpu - shadows[i]
            # Equalize: scale this channel so its median matches the dimmest channel's median
            eq_scale = ref_post_median / post_medians[i]
            stretched = stretched * eq_scale
            stretched = xp.clip(stretched, 0, 1)
            processed.append(to_numpy(stretched))

        # Step 4: Compute shared MTF from the reference channel (green)
        ref_idx = min(1, len(processed) - 1)
        ref_flat = processed[ref_idx].ravel()
        ref_valid = ref_flat[ref_flat > 0.0]

        if len(ref_valid) > 0:
            ref_median = float(np.median(ref_valid))
            if 0 < ref_median < 1 and bg_percent > 0:
                midtone = (
                    ref_median
                    * (bg_percent - 1.0)
                    / (2.0 * bg_percent * ref_median - bg_percent - ref_median)
                )
                midtone = float(np.clip(midtone, 0.01, 0.99))
            else:
                midtone = 0.5
        else:
            midtone = 0.5

        for i in range(len(processed)):
            processed[i] = _mtf(midtone, processed[i])
            result[:, :, i] = processed[i]
        return result

    # Unlinked mode (or grayscale): process each channel independently
    processed = []
    for channel in channels:
        flat = channel.ravel()
        valid = flat[(flat > 0.0) & (flat < 1.0)]

        if len(valid) == 0:
            processed.append(channel)
            continue

        # Statistics stay on numpy
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))

        shadow_clip = max(0.0, median - sigma * mad * 1.4826)
        highlight_clip = 1.0

        # Element-wise stretch on GPU
        ch_gpu = from_numpy(channel)
        stretched = xp.clip(ch_gpu, shadow_clip, highlight_clip)
        if highlight_clip - shadow_clip > 0:
            stretched = (stretched - shadow_clip) / (highlight_clip - shadow_clip)
        stretched = to_numpy(stretched)

        # Calculate midtone balance for target background
        median_norm = (median - shadow_clip) / (highlight_clip - shadow_clip)
        if 0 < median_norm < 1 and bg_percent > 0:
            midtone = (
                median_norm
                * (bg_percent - 1.0)
                / (2.0 * bg_percent * median_norm - bg_percent - median_norm)
            )
            midtone = float(np.clip(midtone, 0.01, 0.99))
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
    d = from_numpy(data)
    result = xp.arcsinh(d * scale) / float(xp.arcsinh(scale))
    return to_numpy(xp.clip(result, 0.0, 1.0))


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
    d = from_numpy(data)
    result = xp.log1p(d / offset) / float(xp.log1p(1.0 / offset))
    return to_numpy(xp.clip(result, 0.0, 1.0))


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
    # Percentile stays on numpy (JAX doesn't support all percentile modes)
    vmin = float(np.percentile(data, low_percentile))
    vmax = float(np.percentile(data, high_percentile))
    if vmax - vmin == 0:
        return data
    d = from_numpy(data)
    result = (d - vmin) / (vmax - vmin)
    return to_numpy(xp.clip(result, 0.0, 1.0))


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
    # Percentile/median stay on numpy
    vmin = float(np.percentile(data, low_percentile))
    vmax = float(np.percentile(data, high_percentile))
    if vmax - vmin == 0:
        return data

    d = from_numpy(data)
    result = xp.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    result_np = to_numpy(result)

    current_median = float(np.median(result_np[result_np > 0]))
    if current_median > 0 and current_median != target_median:
        gamma = float(np.log(target_median) / np.log(current_median))
        gamma = float(np.clip(gamma, 0.2, 5.0))
        result = xp.power(from_numpy(result_np), gamma)
        return to_numpy(xp.clip(result, 0.0, 1.0))

    return result_np
