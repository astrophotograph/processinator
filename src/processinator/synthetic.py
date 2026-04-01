"""Synthetic astrophotography test images with known ground truth.

Generates realistic linear "FITS-like" data — star field, nebulosity,
sky gradient, read noise, dark stacking edges, hot pixels — so that
stretching, autocrop, gradient removal, and denoising can be tested
against a noise-free reference.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticImage:
    """A generated test image plus the ground truth used to build it.

    Attributes:
        data: Observed image (signal + noise + artifacts), shape (H, W)
            or (H, W, 3). Linear, in raw ADU-like units.
        clean: Noise-free signal (stars + nebula + background + gradient,
            with dark edges applied). Same shape as data.
        stars: (N, 3) array of (x, y, peak_amplitude) per star.
        background: Sky background level in ADU.
        noise_sigma: Gaussian read-noise sigma in ADU.
        dark_edges: (top, bottom, left, right) dark edge widths in pixels.
    """

    data: NDArray[np.floating]
    clean: NDArray[np.floating]
    stars: NDArray[np.floating]
    background: float
    noise_sigma: float
    dark_edges: tuple[int, int, int, int]


def make_test_image(
    shape: tuple[int, int] = (256, 256),
    *,
    rgb: bool = False,
    n_stars: int = 80,
    psf_sigma: float = 1.6,
    star_amplitude: tuple[float, float] = (200.0, 40000.0),
    background: float = 800.0,
    noise_sigma: float = 15.0,
    gradient_amplitude: float = 0.0,
    nebula_amplitude: float = 0.0,
    dark_edges: tuple[int, int, int, int] = (0, 0, 0, 0),
    hot_pixels: int = 0,
    channel_balance: tuple[float, float, float] = (1.0, 0.85, 0.7),
    seed: int = 0,
) -> SyntheticImage:
    """Generate a synthetic linear astrophotograph.

    Args:
        shape: Image dimensions (H, W).
        rgb: If True, produce an (H, W, 3) image with a color cast set by
            channel_balance (exercises linked-channel background
            neutralization).
        n_stars: Number of stars, log-uniform amplitudes.
        psf_sigma: Gaussian PSF sigma in pixels.
        star_amplitude: (min, max) star peak amplitude in ADU.
        background: Sky background level in ADU.
        noise_sigma: Gaussian read-noise sigma in ADU. 0 disables noise.
        gradient_amplitude: Peak amplitude of a smooth linear+bilinear sky
            gradient in ADU. 0 disables.
        nebula_amplitude: Peak amplitude of smooth large-scale nebulosity
            in ADU. 0 disables.
        dark_edges: (top, bottom, left, right) widths of dark stacking
            edges to simulate (signal multiplied by 0.02 there).
        hot_pixels: Number of isolated hot pixels added to the observed
            data (not the clean reference).
        channel_balance: Per-channel signal multipliers for RGB images.
        seed: RNG seed; same seed always produces the same image.

    Returns:
        SyntheticImage with observed data, noise-free reference, and truth.
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # Star field: positions uniform (margin keeps PSF cores in-frame),
    # amplitudes log-uniform so a few stars dominate — like real frames.
    margin = 4.0 * psf_sigma
    xs = rng.uniform(margin, w - margin, n_stars)
    ys = rng.uniform(margin, h - margin, n_stars)
    lo, hi = star_amplitude
    amps = np.exp(rng.uniform(np.log(lo), np.log(hi), n_stars))
    stars = np.column_stack([xs, ys, amps])

    signal = _render_gaussians(shape, stars, psf_sigma)

    if nebula_amplitude > 0:
        signal += _render_nebula(shape, nebula_amplitude, rng)

    if gradient_amplitude > 0:
        signal += _render_gradient(shape, gradient_amplitude)

    signal += background

    # Expand to RGB with a color cast
    if rgb:
        clean = np.stack([signal * b for b in channel_balance], axis=2)
    else:
        clean = signal

    # Dark stacking edges attenuate everything, including noise floor
    top, bottom, left, right = dark_edges
    if any(v > 0 for v in dark_edges):
        mask = np.ones(shape)
        if top:
            mask[:top, :] = 0.02
        if bottom:
            mask[h - bottom :, :] = 0.02
        if left:
            mask[:, :left] = 0.02
        if right:
            mask[:, w - right :] = 0.02
        clean = clean * (mask[:, :, None] if rgb else mask)

    data = clean.copy()
    if noise_sigma > 0:
        data = data + rng.normal(0.0, noise_sigma, data.shape)

    if hot_pixels > 0:
        hy = rng.integers(0, h, hot_pixels)
        hx = rng.integers(0, w, hot_pixels)
        data[hy, hx] = star_amplitude[1] * 1.5

    data = np.maximum(data, 0.0)

    return SyntheticImage(
        data=data,
        clean=clean,
        stars=stars,
        background=background,
        noise_sigma=noise_sigma,
        dark_edges=dark_edges,
    )


def save_fits(image: NDArray[np.floating], path: str | Path) -> Path:
    """Write image data to a FITS file.

    RGB images are stored channels-first (3, H, W), the layout most
    astro software produces and read_fits() expects.
    """
    from astropy.io import fits

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    path = Path(path)
    fits.PrimaryHDU(arr).writeto(path, overwrite=True)
    return path


def _render_gaussians(
    shape: tuple[int, int],
    sources: NDArray[np.floating],
    sigma: float,
) -> NDArray[np.float64]:
    """Render Gaussian point sources onto a zero image.

    Each source is drawn on a small stamp (±4 sigma) for speed.
    """
    h, w = shape
    img = np.zeros(shape, dtype=np.float64)
    r = int(np.ceil(4 * sigma))
    for x, y, amp in sources:
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += amp * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        )
    return img


def _render_nebula(
    shape: tuple[int, int],
    amplitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Smooth large-scale nebulosity: a handful of wide Gaussian blobs."""
    h, w = shape
    n_blobs = 5
    xs = rng.uniform(0.2 * w, 0.8 * w, n_blobs)
    ys = rng.uniform(0.2 * h, 0.8 * h, n_blobs)
    amps = rng.uniform(0.3, 1.0, n_blobs)
    sigmas = rng.uniform(0.08, 0.25, n_blobs) * min(h, w)

    yy, xx = np.mgrid[0:h, 0:w]
    nebula = np.zeros(shape, dtype=np.float64)
    for x, y, a, s in zip(xs, ys, amps, sigmas):
        nebula += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * s**2))
    peak = nebula.max()
    if peak > 0:
        nebula *= amplitude / peak
    return nebula


def _render_gradient(shape: tuple[int, int], amplitude: float) -> NDArray[np.float64]:
    """Smooth sky gradient: linear tilt plus a bilinear term, peak = amplitude."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    xn = xx / max(w - 1, 1)
    yn = yy / max(h - 1, 1)
    surface = 0.5 * xn + 0.3 * yn + 0.2 * xn * yn
    return amplitude * (surface / surface.max())
