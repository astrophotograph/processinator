"""Background gradient removal for astrophotography images.

Models the sky background as a low-order 2D polynomial surface per channel,
then subtracts it. This removes light pollution gradients and vignetting
while preserving astronomical signal.
"""

import numpy as np
from numpy.typing import NDArray


def remove_gradient(
    data: NDArray[np.floating],
    order: int = 2,
    sigma_clip: float = 2.5,
    sample_grid: int = 32,
) -> NDArray[np.floating]:
    """Remove background gradient from image data.

    Fits a polynomial surface to the background (excluding stars and bright
    objects via sigma clipping) and subtracts it.

    Args:
        data: Image array, shape (H, W) or (H, W, 3). Should be in [0, 1] range.
        order: Polynomial order for the surface fit. 1=linear (tilt),
               2=quadratic (vignetting), 3=cubic. Default 2.
        sigma_clip: Sigma threshold for rejecting bright pixels (stars/objects)
                    from the background model. Default 2.5.
        sample_grid: Number of grid divisions for sampling background points.
                     Higher = more accurate but slower. Default 32.

    Returns:
        Gradient-subtracted image, clipped to [0, 1].
    """
    if data.ndim == 2:
        return _remove_gradient_channel(data, order, sigma_clip, sample_grid)

    result = np.empty_like(data)
    for i in range(data.shape[2]):
        result[:, :, i] = _remove_gradient_channel(
            data[:, :, i], order, sigma_clip, sample_grid
        )
    return result


def _remove_gradient_channel(
    channel: NDArray[np.floating],
    order: int,
    sigma_clip: float,
    sample_grid: int,
) -> NDArray[np.floating]:
    """Remove gradient from a single channel."""
    h, w = channel.shape

    # Sample background on a grid (faster than using every pixel)
    ys = np.linspace(0, h - 1, sample_grid, dtype=int)
    xs = np.linspace(0, w - 1, sample_grid, dtype=int)

    # Extract sample blocks (median of small patches for robustness)
    patch_h = max(1, h // (sample_grid * 2))
    patch_w = max(1, w // (sample_grid * 2))

    sample_y = []
    sample_x = []
    sample_v = []

    for y in ys:
        for x in xs:
            y0 = max(0, y - patch_h)
            y1 = min(h, y + patch_h + 1)
            x0 = max(0, x - patch_w)
            x1 = min(w, x + patch_w + 1)
            patch = channel[y0:y1, x0:x1]
            sample_y.append(float(y))
            sample_x.append(float(x))
            sample_v.append(float(np.median(patch)))

    sample_y = np.array(sample_y)
    sample_x = np.array(sample_x)
    sample_v = np.array(sample_v)

    # Sigma-clip to reject stars and bright objects
    for _ in range(3):
        med = np.median(sample_v)
        mad = np.median(np.abs(sample_v - med))
        std_est = mad * 1.4826
        if std_est < 1e-10:
            break
        mask = np.abs(sample_v - med) < sigma_clip * std_est
        if mask.sum() < 6:
            break
        sample_y = sample_y[mask]
        sample_x = sample_x[mask]
        sample_v = sample_v[mask]

    # Normalize coordinates to [-1, 1] for numerical stability
    yn = (sample_y / max(h - 1, 1)) * 2 - 1
    xn = (sample_x / max(w - 1, 1)) * 2 - 1

    # Build polynomial design matrix
    terms = _poly_terms(xn, yn, order)
    design = np.column_stack(terms)

    # Least-squares fit
    try:
        coeffs, _, _, _ = np.linalg.lstsq(design, sample_v, rcond=None)
    except np.linalg.LinAlgError:
        return channel

    # Evaluate the model over the full image
    full_y, full_x = np.mgrid[0:h, 0:w]
    fyn = (full_y.astype(np.float64) / max(h - 1, 1)) * 2 - 1
    fxn = (full_x.astype(np.float64) / max(w - 1, 1)) * 2 - 1

    full_terms = _poly_terms(fxn.ravel(), fyn.ravel(), order)
    full_design = np.column_stack(full_terms)
    model = (full_design @ coeffs).reshape(h, w)

    # Subtract model, shift so minimum background is near zero
    result = channel - model
    # Shift: the darkest background region should be near zero
    bg_level = np.percentile(result, 1)
    result = result - bg_level

    return np.clip(result, 0, 1)


def _poly_terms(
    x: NDArray[np.floating], y: NDArray[np.floating], order: int
) -> list[NDArray[np.floating]]:
    """Generate polynomial terms up to given order.

    For order=2: [1, x, y, x^2, xy, y^2]
    """
    terms: list[NDArray[np.floating]] = []
    for total in range(order + 1):
        for xpow in range(total, -1, -1):
            ypow = total - xpow
            terms.append((x ** xpow) * (y ** ypow))
    return terms
