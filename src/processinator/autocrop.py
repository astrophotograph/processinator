"""Auto-crop dark or noisy stacking edges from astrophotography images.

Alt-az mounts and poor guiding produce dark bars or noisy edges after
stacking. This module detects and crops them so they don't skew
stretching and gradient removal.
"""

import numpy as np
from numpy.typing import NDArray


def autocrop(
    data: NDArray[np.floating],
    threshold: float = 0.15,
    min_crop_fraction: float = 0.01,
    max_crop_fraction: float = 0.20,
) -> tuple[NDArray[np.floating], tuple[int, int, int, int]]:
    """Detect and crop dark/noisy stacking edges.

    Examines row and column medians to find edges that are significantly
    darker than the image interior, then crops them.

    Args:
        data: Image array, shape (H, W) or (H, W, 3). Raw or normalized.
        threshold: Edge is considered dark if its median is below this
                   fraction of the interior median. Default 0.15.
        min_crop_fraction: Minimum fraction of dimension to consider
                          cropping (avoids cropping single pixel noise).
                          Default 0.01 (1%).
        max_crop_fraction: Maximum fraction of dimension to crop from
                          each edge. Default 0.20 (20%).

    Returns:
        Tuple of (cropped_data, (top, bottom, left, right)) where the
        crop values are the number of pixels removed from each edge.
        Returns the original data unchanged if no significant edges found.
    """
    # Work with a 2D representation (average channels for RGB)
    if data.ndim == 3:
        mono = np.mean(data, axis=2)
    else:
        mono = data

    h, w = mono.shape

    # Compute row and column medians
    row_medians = np.median(mono, axis=1)
    col_medians = np.median(mono, axis=0)

    # Interior reference: median of the central 60% of the image
    interior_rows = row_medians[int(h * 0.2):int(h * 0.8)]
    interior_cols = col_medians[int(w * 0.2):int(w * 0.8)]

    if len(interior_rows) == 0 or len(interior_cols) == 0:
        return data, (0, 0, 0, 0)

    interior_median = float(np.median(np.concatenate([interior_rows, interior_cols])))

    if interior_median <= 0:
        return data, (0, 0, 0, 0)

    dark_threshold = interior_median * threshold
    min_rows = max(1, int(h * min_crop_fraction))
    min_cols = max(1, int(w * min_crop_fraction))
    max_rows = int(h * max_crop_fraction)
    max_cols = int(w * max_crop_fraction)

    # Find crop bounds from each edge
    top = _find_edge(row_medians, dark_threshold, min_rows, max_rows)
    bottom = _find_edge(row_medians[::-1], dark_threshold, min_rows, max_rows)
    left = _find_edge(col_medians, dark_threshold, min_cols, max_cols)
    right = _find_edge(col_medians[::-1], dark_threshold, min_cols, max_cols)

    # Only crop if we found meaningful edges (at least min_crop on one side)
    total_crop = top + bottom + left + right
    if total_crop == 0:
        return data, (0, 0, 0, 0)

    # Apply crop
    y0 = top
    y1 = h - bottom
    x0 = left
    x1 = w - right

    # Safety: ensure we're not cropping everything
    if y1 - y0 < h * 0.5 or x1 - x0 < w * 0.5:
        return data, (0, 0, 0, 0)

    cropped = data[y0:y1, x0:x1] if data.ndim == 2 else data[y0:y1, x0:x1, :]

    return cropped, (top, bottom, left, right)


def _find_edge(
    medians: NDArray[np.floating],
    dark_threshold: float,
    min_pixels: int,
    max_pixels: int,
) -> int:
    """Find how many pixels from an edge are dark.

    Scans from the edge inward, looking for a contiguous run of dark
    rows/columns. Stops when it finds a non-dark row or hits max_pixels.
    """
    count = 0
    for i in range(min(len(medians), max_pixels)):
        if medians[i] < dark_threshold:
            count = i + 1
        else:
            break

    # Only crop if we found at least min_pixels of dark edge
    return count if count >= min_pixels else 0
