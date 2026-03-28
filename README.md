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

# Low-level: numpy array → stretched array
import numpy as np
from astropy.io import fits

data = fits.getdata("my_image.fits").astype(np.float64)
stretched = stretch(data, algorithm=StretchAlgorithm.MTF)
```

## Algorithms

| Algorithm | Best for | Description |
|-----------|----------|-------------|
| **MTF** (default) | General use | Midtones Transfer Function with background neutralization |
| **Arcsinh** | Color preservation | Inverse hyperbolic sine, maintains color ratios |
| **Log** | High dynamic range | Logarithmic stretch |
| **Linear** | Quick preview | Simple percentile-based clip and scale |
| **Statistical** | Consistent output | Gamma correction targeting a specific median |

## License

AGPL-3.0
