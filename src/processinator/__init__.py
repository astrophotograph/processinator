"""Processinator - astronomy image processing library."""

from processinator.stretching import StretchAlgorithm, fits_to_image, read_fits, stretch

__all__ = [
    "StretchAlgorithm",
    "fits_to_image",
    "read_fits",
    "stretch",
]
