"""
src/metrics/metrics.py
Image Quality Metrics
---------------------
PSNR   : Peak Signal-to-Noise Ratio [dB]   (higher is better)
MSE    : Mean Squared Error                 (lower is better)
SSIM   : Structural Similarity Index       (higher is better, max 1.0)
Sharpness : Laplacian variance             (higher means sharper)

All metrics computed on float64 images normalized to [0.0, 1.0].
"""
import numpy as np
from scipy.ndimage import convolve


def compute_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Mean Squared Error between original and processed images.

    MSE = (1/M*N) * SUM [f(x,y) - g(x,y)]^2

    Parameters
    ----------
    original, processed : 2D uint8 arrays of same shape

    Returns
    -------
    mse : float in [0.0, 1.0] (normalized to float pixel range)
    """
    orig_f = original.astype(np.float64) / 255.0
    proc_f = processed.astype(np.float64) / 255.0
    return float(np.mean((orig_f - proc_f) ** 2))


def compute_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX_I^2 / MSE)
    where MAX_I = 1.0 for normalized float images.

    A PSNR above 30 dB is generally considered good quality.
    Returns float('inf') if images are identical.
    """
    mse = compute_mse(original, processed)
    if mse < 1e-12:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def compute_sharpness(image: np.ndarray) -> float:
    """
    Compute sharpness as the variance of the Laplacian response.

    Higher values indicate a sharper image with more high-frequency content.
    A blurred image will have low Laplacian variance.

    Parameters
    ----------
    image : 2D uint8 ndarray

    Returns
    -------
    sharpness : float (Laplacian variance)
    """
    img_f  = image.astype(np.float64) / 255.0
    lap_k  = np.array([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], dtype=np.float64)
    lap    = convolve(img_f, lap_k, mode='reflect')
    return float(np.var(lap))


def compute_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    SSIM considers luminance, contrast, and structure simultaneously.
    Range: [-1, 1]; 1.0 means identical images.
    Standard constants: C1 = (0.01*L)^2, C2 = (0.03*L)^2, L = 255.

    This is a global (single-window) SSIM, not the local multi-window version.

    Parameters
    ----------
    original, processed : 2D uint8 arrays of same shape

    Returns
    -------
    ssim : float in approximately [-1, 1]
    """
    x = original.astype(np.float64)
    y = processed.astype(np.float64)
    L  = 255.0
    C1 = (0.01 * L) ** 2   # 6.5025
    C2 = (0.03 * L) ** 2   # 58.5225

    mu_x     = np.mean(x)
    mu_y     = np.mean(y)
    sigma_x  = np.var(x)
    sigma_y  = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator   = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    return float(numerator / (denominator + 1e-12))


def compute_all_metrics(original: np.ndarray,
                         processed: np.ndarray) -> dict:
    """
    Compute all quality metrics and return as a dictionary.

    Returns
    -------
    dict with keys: 'mse', 'psnr', 'sharpness', 'ssim'
    """
    return {
        'mse':       compute_mse(original, processed),
        'psnr':      compute_psnr(original, processed),
        'sharpness': compute_sharpness(processed),
        'ssim':      compute_ssim(original, processed),
    }
