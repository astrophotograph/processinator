"""FITS file reading and image output."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from PIL import Image

from processinator.stretching.algorithms import StretchAlgorithm, stretch


def read_fits(file_path: str | Path) -> tuple[NDArray[np.floating], dict]:
    """Read a FITS file and return image data and header metadata.

    Handles common FITS layouts:
    - (H, W) grayscale
    - (3, H, W) RGB channels-first
    - (H, W, 3) RGB channels-last

    Args:
        file_path: Path to the FITS file.

    Returns:
        Tuple of (image_data as float64, header as dict).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the FITS file contains no image data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FITS file not found: {path}")

    with fits.open(path) as hdul:
        # Find the first HDU with image data
        image_data = None
        header = {}

        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                image_data = hdu.data.astype(np.float64)
                header = dict(hdu.header)
                break

        if image_data is None:
            raise ValueError(f"No image data found in FITS file: {path}")

    # Normalize array layout to (H, W) or (H, W, 3)
    if image_data.ndim == 3:
        if image_data.shape[0] == 3:
            # (3, H, W) -> (H, W, 3)
            image_data = np.transpose(image_data, (1, 2, 0))
        elif image_data.shape[2] != 3:
            raise ValueError(
                f"Unexpected FITS shape: {image_data.shape}. "
                "Expected (H, W), (3, H, W), or (H, W, 3)."
            )
    elif image_data.ndim != 2:
        raise ValueError(f"Unexpected FITS dimensions: {image_data.ndim}. Expected 2 or 3.")

    return image_data, header


def fits_to_image(
    fits_path: str | Path,
    output_path: str | Path | None = None,
    algorithm: StretchAlgorithm = StretchAlgorithm.MTF,
    output_format: str = "PNG",
    **stretch_kwargs: float,
) -> Image.Image:
    """Read a FITS file, apply stretching, and produce a displayable image.

    Args:
        fits_path: Path to the input FITS file.
        output_path: If provided, save the image to this path.
        algorithm: Stretch algorithm to use.
        output_format: Image format for saving ("PNG" or "JPEG").
        **stretch_kwargs: Passed through to the stretch algorithm.

    Returns:
        PIL Image object.
    """
    data, _header = read_fits(fits_path)
    stretched = stretch(data, algorithm=algorithm, **stretch_kwargs)

    # Convert to 8-bit
    img_8bit = (stretched * 255.0).clip(0, 255).astype(np.uint8)

    if img_8bit.ndim == 2:
        pil_image = Image.fromarray(img_8bit, mode="L")
    else:
        pil_image = Image.fromarray(img_8bit, mode="RGB")

    if output_path is not None:
        output_path = Path(output_path)
        save_kwargs = {}
        if output_format.upper() == "JPEG":
            save_kwargs["quality"] = 95
            # JPEG doesn't support 'L' with alpha or palette issues, convert if needed
            if pil_image.mode == "L":
                pil_image = pil_image.convert("RGB")
        pil_image.save(output_path, format=output_format, **save_kwargs)

    return pil_image
