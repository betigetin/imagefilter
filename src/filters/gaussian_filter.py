"""
src/filters/gaussian_filter.py
Gaussian Filter
---------------
Weighted averaging using a discretized 2D isotropic Gaussian kernel.
Key properties:
  - Rotationally symmetric (isotropic)
  - Separable: can be applied as two 1D passes (row then column)
  - Frequency response is a Gaussian (no sidelobes, smooth rolloff)
  - Fourier transform of a Gaussian is also a Gaussian
"""
import math
import numpy as np
from src.core.convolution import normalize, denormalize, convolve1d_separable, convolve2d_fast


def build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a (size x size) discretized 2D Gaussian kernel, normalized to sum = 1.

    The continuous 2D Gaussian:
        G(x,y) = (1 / 2*pi*sigma^2) * exp(-(x^2 + y^2) / (2*sigma^2))

    The discrete kernel samples G at integer coordinates and normalizes,
    ensuring the filter preserves mean image brightness (DC-preserving).

    Parameters
    ----------
    size  : int    Kernel dimension (odd). Recommended: 2*ceil(3*sigma)+1
    sigma : float  Standard deviation in pixels. Larger = stronger blur.

    Returns
    -------
    kernel : 2D float32 ndarray of shape (size, size), sum = 1.0
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}.")

    half = size // 2
    ax   = np.arange(-half, half + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)

    # Evaluate unnormalized Gaussian (constant prefactor cancels in normalize step)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # Normalize: ensure all coefficients sum to exactly 1.0
    kernel /= kernel.sum()

    assert np.isclose(kernel.sum(), 1.0, atol=1e-5),         "Gaussian kernel normalization failed."
    return kernel.astype(np.float32)


def build_gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Build a 1D Gaussian kernel for separable convolution.

    Parameters
    ----------
    size  : int    Length of the 1D kernel (odd)
    sigma : float  Standard deviation

    Returns
    -------
    kernel_1d : 1D float32 ndarray of length size, sum = 1.0
    """
    half      = size // 2
    ax        = np.arange(-half, half + 1, dtype=np.float32)
    kernel_1d = np.exp(-(ax**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    return kernel_1d.astype(np.float32)


def auto_kernel_size(sigma: float) -> int:
    """
    Compute recommended kernel size for a given sigma.
    Rule: size = 2 * ceil(3 * sigma) + 1  (covers 3-sigma on each side).
    """
    size = 2 * math.ceil(3 * sigma) + 1
    return size if size % 2 == 1 else size + 1


def apply_gaussian_filter(image: np.ndarray,
                           sigma: float = 1.0,
                           size: int = None) -> np.ndarray:
    """
    Apply Gaussian filter using separable 1D convolution (optimized).

    Separability: G(x,y) = g(x) * g(y) where g is a 1D Gaussian.
    Two 1D convolutions along rows then columns replaces one 2D convolution.
    Complexity: O(2K*H*W) vs O(K^2*H*W) for the naive 2D implementation.
    For a 7x7 kernel this is a 3.5x speedup.

    Parameters
    ----------
    image : 2D uint8 ndarray, shape (H, W)
    sigma : Standard deviation in pixels (controls blur strength)
    size  : Kernel size. If None, auto-computed via auto_kernel_size(sigma).

    Returns
    -------
    filtered : 2D uint8 ndarray, shape (H, W)
    """
    if size is None:
        size = auto_kernel_size(sigma)
    if size % 2 == 0:
        size += 1

    kernel_1d = build_gaussian_kernel_1d(size, sigma)
    img_f     = normalize(image)
    out_f     = convolve1d_separable(img_f, kernel_1d)
    return denormalize(out_f)
