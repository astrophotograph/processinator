"""Processinator - astronomy image processing library."""

from processinator.autocrop import autocrop
from processinator.denoise import denoise
from processinator.gradient import remove_gradient
from processinator.pipeline import PipelineConfig, process
from processinator.stretching import StretchAlgorithm, fits_to_image, read_fits, stretch
from processinator.stretching.backend import backend_name, using_jax
from processinator.synthetic import SyntheticImage, make_test_image

__all__ = [
    "autocrop",
    "denoise",
    "PipelineConfig",
    "StretchAlgorithm",
    "SyntheticImage",
    "backend_name",
    "fits_to_image",
    "make_test_image",
    "process",
    "read_fits",
    "remove_gradient",
    "stretch",
    "using_jax",
]
