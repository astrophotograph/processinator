"""Processinator - astronomy image processing library."""

from processinator.gradient import remove_gradient
from processinator.pipeline import PipelineConfig, process
from processinator.stretching import StretchAlgorithm, fits_to_image, read_fits, stretch
from processinator.stretching.backend import backend_name, using_jax

__all__ = [
    "PipelineConfig",
    "StretchAlgorithm",
    "backend_name",
    "fits_to_image",
    "process",
    "read_fits",
    "remove_gradient",
    "stretch",
    "using_jax",
]
