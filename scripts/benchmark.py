#!/usr/bin/env python3
"""Benchmark each step of the processinator pipeline.

Usage:
    uv run python scripts/benchmark.py <fits_file>
    uv run python scripts/benchmark.py <fits_file> --no-gradient
    uv run python scripts/benchmark.py <fits_file> --algorithm arcsinh
"""

import sys
import time

import numpy as np


def benchmark(fits_path: str, algorithm: str = "mtf", gradient: bool = True):
    from processinator import read_fits
    from processinator.autocrop import autocrop
    from processinator.gradient import remove_gradient
    from processinator.stretching.algorithms import (
        StretchAlgorithm,
        _normalize_to_01,
        _normalize_to_01_with_stats,
        _stretch_arcsinh,
        _stretch_linear,
        _stretch_log,
        _stretch_mtf,
        _stretch_statistical,
    )
    from processinator.stretching.backend import backend_name

    print(f"Backend: {backend_name()}")
    print(f"File:    {fits_path}")
    print()

    steps = []

    # 1. Read FITS
    t0 = time.perf_counter()
    data, _header = read_fits(fits_path)
    elapsed = time.perf_counter() - t0
    steps.append(("Read FITS", elapsed))
    size_mb = data.nbytes / 1e6
    print(f"  Read FITS:        {elapsed*1000:7.0f} ms  ({data.shape}, {size_mb:.0f} MB)")

    # 2. Autocrop detection
    t0 = time.perf_counter()
    cropped, crop_info = autocrop(data)
    elapsed = time.perf_counter() - t0
    steps.append(("Autocrop", elapsed))
    crop_str = f"top={crop_info[0]} bot={crop_info[1]} left={crop_info[2]} right={crop_info[3]}"
    print(f"  Autocrop:         {elapsed*1000:7.0f} ms  ({crop_str})")

    # 3. Normalize
    t0 = time.perf_counter()
    if any(v > 0 for v in crop_info):
        normalized = _normalize_to_01_with_stats(data, cropped)
    else:
        normalized = _normalize_to_01(data)
    elapsed = time.perf_counter() - t0
    steps.append(("Normalize", elapsed))
    print(f"  Normalize:        {elapsed*1000:7.0f} ms")

    # 4. Gradient removal (optional)
    if gradient:
        t0 = time.perf_counter()
        work = remove_gradient(normalized)
        elapsed = time.perf_counter() - t0
        steps.append(("Gradient removal", elapsed))
        print(f"  Gradient removal: {elapsed*1000:7.0f} ms")
    else:
        work = normalized
        print(f"  Gradient removal:    skip")

    # 5. Stretch
    stretch_funcs = {
        "mtf": _stretch_mtf,
        "arcsinh": _stretch_arcsinh,
        "log": _stretch_log,
        "linear": _stretch_linear,
        "statistical": _stretch_statistical,
    }
    stretch_fn = stretch_funcs.get(algorithm, _stretch_mtf)

    t0 = time.perf_counter()
    stretched = stretch_fn(work)
    elapsed = time.perf_counter() - t0
    steps.append((f"Stretch ({algorithm})", elapsed))
    print(f"  Stretch ({algorithm:10s}): {elapsed*1000:7.0f} ms")

    # 6. Convert to 8-bit + save JPEG
    t0 = time.perf_counter()
    from PIL import Image

    img_8bit = (stretched * 255.0).clip(0, 255).astype(np.uint8)
    if img_8bit.ndim == 2:
        pil = Image.fromarray(img_8bit, mode="L")
    else:
        pil = Image.fromarray(img_8bit, mode="RGB")
    pil.save("/tmp/benchmark_output.jpg", format="JPEG", quality=95)
    elapsed = time.perf_counter() - t0
    steps.append(("Convert + save", elapsed))
    print(f"  Convert + save:   {elapsed*1000:7.0f} ms")

    total = sum(t for _, t in steps)
    print(f"  {'─' * 35}")
    print(f"  TOTAL:            {total*1000:7.0f} ms")
    print()

    # Show breakdown as percentages
    print("  Breakdown:")
    for name, t in sorted(steps, key=lambda x: -x[1]):
        pct = t / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:20s} {t*1000:6.0f} ms  {pct:4.1f}%  {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: benchmark.py <fits_file> [--no-gradient] [--algorithm name]")
        sys.exit(1)

    fits_path = sys.argv[1]
    algorithm = "mtf"
    gradient = True

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--no-gradient":
            gradient = False
        elif arg == "--algorithm" and i + 1 < len(sys.argv):
            algorithm = sys.argv[i + 1]

    benchmark(fits_path, algorithm=algorithm, gradient=gradient)


if __name__ == "__main__":
    main()
