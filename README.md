# processinator

Astronomy image processing library. Converts linear FITS data into visually useful images using nonlinear stretch algorithms.

## Install

```sh
pip install processinator
```

JAX is included by default for JIT-accelerated stretching on CPU.

### GPU acceleration

For GPU-accelerated processing, install the appropriate extra for your hardware:

**NVIDIA (CUDA 12):**
```sh
pip install processinator[cuda]
```

**AMD (ROCm):**
```sh
pip install processinator[rocm]
# Then install the ROCm JAX wheel:
pip install --upgrade jaxlib-rocm  # see https://jax.readthedocs.io/en/latest/installation.html#amd-gpu-rocm
```

**Apple Silicon (Metal):**
```sh
pip install processinator[metal]
```

Check which backend is active:
```python
import processinator
print(processinator.backend_name())  # "JAX (GPU)", "JAX (CPU)", or "numpy"
```

## Usage

```python
from processinator import stretch, fits_to_image, StretchAlgorithm

# High-level: FITS file → PNG/JPEG
image = fits_to_image("my_image.fits", output_path="stretched.png")

# With gradient removal and denoising
image = fits_to_image(
    "my_image.fits",
    output_path="stretched.png",
    remove_gradient=True,
    denoise=True,
)

# Low-level: numpy array → stretched array
import numpy as np
from astropy.io import fits

data = fits.getdata("my_image.fits").astype(np.float64)
stretched = stretch(data, algorithm=StretchAlgorithm.MTF)

# Full pipeline control
from processinator import process, PipelineConfig

result = process(data, PipelineConfig(
    gradient_removal=True,
    denoise=True,
    denoise_threshold=3.0,
))
```

## Algorithms

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| **MTF** (default) | General use | Midtones Transfer Function with background neutralization |
| **Arcsinh** | Color preservation | Inverse hyperbolic sine, maintains color ratios |
| **Log** | High dynamic range | Logarithmic stretch |
| **Linear** | Quick preview | Simple percentile-based clip and scale |
| **Statistical** | Consistent output | Gamma correction targeting a specific median |

## Denoising

`denoise()` implements starlet (à trous B3-spline) wavelet thresholding —
the standard astronomy approach (Starck & Murtagh). The noise level is
estimated automatically from the finest wavelet scale, so there is no
noise parameter to tune.

```python
from processinator import denoise

quiet = denoise(data)                        # hard thresholding (default)
smooth = denoise(data, mode="soft")          # smoother, for starless data
gentle = denoise(data, threshold=2.0)        # threshold in noise sigmas
```

Why it's astronomically safe:

- Stars and real structure produce wavelet coefficients far above the
  noise threshold and pass through untouched — hard mode preserves
  bright-star photometry exactly.
- The coarse residual (faint nebulosity, sky background) is never
  thresholded, so large-scale signal survives.

## Synthetic test images

`processinator.synthetic` generates linear test frames with known ground
truth (star catalog, noise-free reference) — useful for testing any
processing code, not just this library's:

```python
from processinator import make_test_image
from processinator.synthetic import save_fits

img = make_test_image(
    (512, 512),
    rgb=True,
    gradient_amplitude=300.0,   # sky gradient
    nebula_amplitude=150.0,     # large-scale nebulosity
    dark_edges=(10, 0, 14, 0),  # stacking artifacts
    hot_pixels=20,
    seed=13,
)
img.data     # observed frame (signal + noise)
img.clean    # noise-free truth
img.stars    # (N, 3) star catalog: x, y, peak amplitude
save_fits(img.data, "test.fits")
```

`uv run python scripts/make_test_images.py` writes a set of example FITS
files to `examples/` for manual experimentation.

## License

AGPL-3.0
