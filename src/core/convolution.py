"""
src/core/convolution.py
Core 2D convolution engine with reflect-padding and normalization helpers.
"""
import numpy as np
from scipy.ndimage import convolve, convolve1d


def normalize(image: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] to float32 [0.0, 1.0]."""
    return image.astype(np.float32) / 255.0


def denormalize(image: np.ndarray) -> np.ndarray:
    """Convert float32 [0.0, 1.0] to uint8 [0, 255] with clipping."""
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def convolve2d_fast(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D discrete spatial convolution using reflect boundary mode.
    Output has identical spatial dimensions to input.

    Parameters
    ----------
    image  : 2D float32 array normalized to [0.0, 1.0]
    kernel : 2D float32 kernel with odd dimensions

    Returns
    -------
    output : 2D float32 array, same shape as image, clipped to [0.0, 1.0]
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel, got shape {kernel.shape}")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel dimensions must be odd (3, 5, 7, ...).")

    result = convolve(image.astype(np.float32),
                      kernel.astype(np.float32),
                      mode='reflect')
    return np.clip(result, 0.0, 1.0)


def convolve1d_separable(image: np.ndarray,
                          kernel_1d: np.ndarray) -> np.ndarray:
    """
    Apply a separable 1D kernel: first along columns (axis=1),
    then along rows (axis=0).

    Complexity: O(2K * H * W) vs O(K^2 * H * W) for full 2D convolution.
    Used by the Gaussian filter for significant speedup on large images.

    Parameters
    ----------
    image     : 2D float32 array, normalized to [0.0, 1.0]
    kernel_1d : 1D float32 array (symmetric Gaussian kernel)

    Returns
    -------
    result : 2D float32 array, same shape, clipped to [0.0, 1.0]
    """
    img = image.astype(np.float32)
    # Horizontal pass
    temp   = convolve1d(img,    kernel_1d, axis=1, mode='reflect')
    # Vertical pass
    result = convolve1d(temp,   kernel_1d, axis=0, mode='reflect')
    return np.clip(result, 0.0, 1.0)
