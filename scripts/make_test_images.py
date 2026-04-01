#!/usr/bin/env python3
"""Generate example FITS test images into examples/.

These are for manual inspection and experimentation — the test suite
generates its own images in memory and does not depend on these files.

Usage:
    uv run python scripts/make_test_images.py [output_dir]
"""

import sys
from pathlib import Path

from processinator.synthetic import make_test_image, save_fits

VARIANTS = {
    "starfield_mono": dict(seed=42),
    "starfield_rgb": dict(rgb=True, seed=42),
    "gradient_nebula": dict(
        gradient_amplitude=400.0, nebula_amplitude=250.0, seed=7
    ),
    "stacking_edges": dict(dark_edges=(12, 8, 20, 0), seed=3),
    "kitchen_sink_rgb": dict(
        rgb=True,
        gradient_amplitude=300.0,
        nebula_amplitude=150.0,
        dark_edges=(10, 0, 14, 0),
        hot_pixels=20,
        seed=13,
    ),
    "noisy": dict(noise_sigma=60.0, nebula_amplitude=300.0, seed=11),
}


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("examples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, kwargs in VARIANTS.items():
        img = make_test_image((512, 512), **kwargs)
        path = save_fits(img.data, out_dir / f"{name}.fits")
        print(f"  wrote {path}  ({img.data.shape})")

    print(f"\n{len(VARIANTS)} example images in {out_dir}/")


if __name__ == "__main__":
    main()
