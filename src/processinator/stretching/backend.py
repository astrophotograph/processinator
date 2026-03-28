"""Array backend abstraction: JAX (GPU) with numpy fallback.

Provides a unified `xp` module that mirrors the numpy API. When JAX is
available it is used for GPU-accelerated computation; otherwise plain numpy
is used transparently.

Usage in algorithm code::

    from processinator.stretching.backend import xp, to_numpy

    result = xp.arcsinh(data * scale)  # runs on GPU if JAX is available
    np_array = to_numpy(result)         # always returns a numpy array
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_USE_JAX = False

try:
    import jax
    import jax.numpy as jnp

    # Silence JAX info logs unless user opts in
    jax.config.update("jax_log_compiles", False)

    # Quick smoke test — make sure a device is actually reachable
    _ = jnp.zeros(1)
    _USE_JAX = True

    _backend_name = str(jax.default_backend()).upper()  # "GPU", "CPU", "TPU"
    logger.info("processinator: using JAX backend (%s)", _backend_name)
except Exception:
    logger.debug("processinator: JAX not available, using numpy backend")


def using_jax() -> bool:
    """Return True if JAX is the active backend."""
    return _USE_JAX


def backend_name() -> str:
    """Human-readable backend identifier."""
    if _USE_JAX:
        return f"JAX ({jax.default_backend().upper()})"
    return "numpy"


# ---------------------------------------------------------------------------
# Unified array module
# ---------------------------------------------------------------------------
# `xp` exposes the same API surface as numpy. Algorithm code imports `xp`
# and writes array expressions once.

if _USE_JAX:
    xp = jnp  # type: ignore[assignment]
else:
    xp = np  # type: ignore[assignment]


def to_numpy(arr: "NDArray | jnp.ndarray") -> NDArray:  # type: ignore[name-defined]
    """Convert any array (JAX or numpy) to a plain numpy array."""
    if _USE_JAX:
        return np.asarray(arr)
    return np.asarray(arr)


def from_numpy(arr: NDArray) -> "NDArray | jnp.ndarray":  # type: ignore[name-defined]
    """Convert a numpy array to the active backend's array type."""
    if _USE_JAX:
        return jnp.asarray(arr)
    return arr
