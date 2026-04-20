"""
src/filters/mean_filter.py
Mean (Box / Averaging) Filter
------------------------------
Replaces each pixel with the arithmetic mean of its local K×K neighborhood.
All kernel coefficients are equal: h[s,t] = 1/(K*K).
Low-pass filter; frequency response is a 2D sinc function.
"""
import numpy as np
from src.core.convolution import normalize, denormalize, convolve2d_fast


def build_mean_kernel(size: int) -> np.ndarray:
    """
    Build a (size x size) mean filter kernel.
    All coefficients equal 1/(size*size) and sum to exactly 1.0.

    Parameters
    ----------
    size : int  Kernel dimension (must be odd: 3, 5, or 7)

    Returns
    -------
    kernel : 2D float32 array of shape (size, size)
    """
    if size % 2 == 0:
        raise ValueError(f"Kernel size must be odd, got {size}.")

    kernel = np.ones((size, size), dtype=np.float32) / float(size * size)

    assert np.isclose(kernel.sum(), 1.0), "Kernel normalization failed."
    return kernel


def apply_mean_filter(image: np.ndarray, size: int = 3) -> np.ndarray:
    """
    Apply the mean (box) filter to a grayscale image.

    Parameters
    ----------
    image : 2D uint8 ndarray, shape (H, W), pixel values in [0, 255]
    size  : kernel size — 3, 5, or 7

    Returns
    -------
    filtered : 2D uint8 ndarray, shape (H, W)

    Notes
    -----
    - Larger kernels produce stronger smoothing but more edge blur.
    - Edge pixels are handled via reflect (mirror) padding.
    - The filter preserves mean image brightness (DC-preserving).
    """
    kernel = build_mean_kernel(size)
    img_f  = normalize(image)
    out_f  = convolve2d_fast(img_f, kernel)
    return denormalize(out_f)
