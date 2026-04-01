"""Astronomically-safe denoising via starlet wavelet thresholding.

Uses the isotropic undecimated wavelet transform (the "starlet" / à trous
transform with a B3-spline kernel) that is standard in astronomy image
processing (Starck & Murtagh, "Astronomical Image and Data Analysis").

Why it is safe for astrophotos:
- Stars and real structure produce wavelet coefficients far above the
  noise threshold and pass through (nearly) untouched.
- The coarse residual — faint nebulosity, sky background — is never
  thresholded, so no large-scale signal is destroyed.
- The noise level is estimated from the data itself (MAD of the finest
  wavelet scale), so no manual noise parameter is needed.

Works on linear or stretched data; the estimate is scale-free.
"""

import numpy as np
from numpy.typing import NDArray

from processinator.stretching.backend import from_numpy, to_numpy, xp

# B3-spline scaling kernel used by the à trous transform.
_B3 = (1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0)

# Noise standard deviation of each starlet scale for unit-variance Gaussian
# input (2D, B3 spline). Precomputed constants from Starck & Murtagh.
_NOISE_LEVELS = (0.8907, 0.2007, 0.0855, 0.0412, 0.0203, 0.0102, 0.0051)


def denoise(
    data: NDArray[np.floating],
    n_scales: int = 4,
    threshold: float = 3.0,
    mode: str = "hard",
) -> NDArray[np.floating]:
    """Denoise image data using starlet wavelet thresholding.

    Decomposes each channel into wavelet detail scales plus a smooth
    residual, estimates the noise sigma from the finest scale, thresholds
    the detail coefficients, and reconstructs. Coefficients well above the
    noise (stars, structure) survive; the smooth residual (nebulosity,
    background) is never touched.

    Args:
        data: Image array, shape (H, W) or (H, W, 3). Linear or stretched.
        n_scales: Number of wavelet detail scales to threshold. More scales
            remove lower-frequency noise but risk touching real structure.
            Clamped automatically for small images. Default 4.
        threshold: Threshold in units of the estimated noise sigma. The
            finest scale uses threshold + 1 (standard practice — it is
            almost pure noise). Higher = stronger smoothing. Default 3.0.
        mode: "hard" (keep-or-kill; preserves star photometry exactly, the
            standard choice for astronomy) or "soft" (shrink every
            coefficient; visually smoother background but systematically
            dims point sources — fine for starless/background-dominated
            data). Default "hard".

    Returns:
        Denoised image, same shape, float64 dtype. Values are not
        clipped or rescaled.
    """
    if mode not in ("soft", "hard"):
        raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

    if data.ndim == 2:
        return _denoise_channel(data, n_scales, threshold, mode)

    result = np.empty(data.shape, dtype=np.float64)
    for i in range(data.shape[2]):
        result[:, :, i] = _denoise_channel(data[:, :, i], n_scales, threshold, mode)
    return result


def _denoise_channel(
    channel: NDArray[np.floating],
    n_scales: int,
    threshold: float,
    mode: str,
) -> NDArray[np.floating]:
    h, w = channel.shape

    # The kernel span at scale j is 4 * 2^j + 1 pixels; keep it inside the image
    max_scales = 1
    while 4 * 2**max_scales + 1 < min(h, w) and max_scales < len(_NOISE_LEVELS) - 1:
        max_scales += 1
    n_scales = max(1, min(n_scales, max_scales))

    c = from_numpy(np.ascontiguousarray(channel, dtype=np.float64))

    # À trous decomposition: detail scales + smooth residual
    details = []
    for j in range(n_scales):
        smooth = _b3_smooth(c, 2**j)
        details.append(c - smooth)
        c = smooth

    # Estimate noise sigma from the finest detail scale via MAD.
    # Detail coefficients are zero-mean; stars are sparse enough that the
    # median of |w| is set by the noise.
    finest = to_numpy(details[0])
    sigma = float(np.median(np.abs(finest))) / 0.6745 / _NOISE_LEVELS[0]
    if sigma <= 0.0:
        # No detectable noise — return the input unchanged
        return np.asarray(channel, dtype=np.float64)

    # Threshold detail scales and reconstruct (residual passes untouched)
    result = c
    for j, detail in enumerate(details):
        k = threshold + 1.0 if j == 0 else threshold
        t = k * sigma * _noise_level(j)
        if mode == "soft":
            detail = xp.sign(detail) * xp.maximum(xp.abs(detail) - t, 0.0)
        else:
            detail = xp.where(xp.abs(detail) > t, detail, 0.0)
        result = result + detail

    # JAX computes in float32; deliver the promised float64
    return np.asarray(to_numpy(result), dtype=np.float64)


def _noise_level(scale: int) -> float:
    """Noise std of a starlet scale for unit-variance input noise."""
    if scale < len(_NOISE_LEVELS):
        return _NOISE_LEVELS[scale]
    # Beyond the table each scale roughly halves
    return _NOISE_LEVELS[-1] * 0.5 ** (scale - len(_NOISE_LEVELS) + 1)


def _b3_smooth(img, step: int):
    """Separable B3-spline smoothing with holes ("à trous") of size step.

    Implemented as weighted sums of shifted views so it runs on either
    backend (numpy or JAX) without scipy.
    """
    out = _b3_conv_axis(img, step, axis=0)
    return _b3_conv_axis(out, step, axis=1)


def _b3_conv_axis(img, step: int, axis: int):
    n = img.shape[axis]
    pad = 2 * step
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    padded = xp.pad(img, pad_width, mode="reflect")

    result = None
    for k, weight in enumerate(_B3):
        offset = k * step
        sl: list[slice] = [slice(None), slice(None)]
        sl[axis] = slice(offset, offset + n)
        term = weight * padded[tuple(sl)]
        result = term if result is None else result + term
    return result
