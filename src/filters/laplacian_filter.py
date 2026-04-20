"""
src/filters/laplacian_filter.py
Laplacian Filter and Image Sharpening
--------------------------------------
The Laplacian is a second-order isotropic derivative operator:
    L(x,y) = d^2f/dx^2 + d^2f/dy^2

Discrete kernels:
  4-connectivity:            8-connectivity:
  [ 0   1   0]              [ 1   1   1]
  [ 1  -4   1]              [ 1  -8   1]
  [ 0   1   0]              [ 1   1   1]

Both kernels sum to zero (zero DC response).

Composite sharpening formula:
    g[x,y] = f[x,y] - c * (h_lap * f)[x,y]
where c is the sharpening coefficient in (0, 1.5].
"""
import numpy as np
from scipy.ndimage import convolve
from src.core.convolution import normalize, denormalize


def build_laplacian_kernel(connectivity: int = 4) -> np.ndarray:
    """
    Build the discrete Laplacian kernel.

    Parameters
    ----------
    connectivity : 4 or 8
        4 — cardinal neighbors (N, S, E, W) only
        8 — cardinal + diagonal neighbors

    Returns
    -------
    kernel : 2D float32 ndarray of shape (3, 3), sums to 0
    """
    if connectivity == 4:
        kernel = np.array([[0,  1,  0],
                           [1, -4,  1],
                           [0,  1,  0]], dtype=np.float32)
    elif connectivity == 8:
        kernel = np.array([[1,  1,  1],
                           [1, -8,  1],
                           [1,  1,  1]], dtype=np.float32)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}.")

    assert np.isclose(kernel.sum(), 0.0, atol=1e-6),         "Laplacian kernel must sum to 0."
    return kernel


def _apply_laplacian_raw(img_f: np.ndarray,
                          connectivity: int = 4) -> np.ndarray:
    """
    Compute raw Laplacian response on a float image.
    Output is NOT clipped — may contain negative values.
    Used internally by apply_sharpening.
    """
    kernel = build_laplacian_kernel(connectivity)
    return convolve(img_f.astype(np.float32), kernel, mode='reflect')


def apply_laplacian(image: np.ndarray,
                    connectivity: int = 4) -> np.ndarray:
    """
    Compute and display the normalized Laplacian edge map.
    Output is scaled to [0, 255] for visualization only.

    Parameters
    ----------
    image        : 2D uint8 ndarray, shape (H, W)
    connectivity : 4 or 8

    Returns
    -------
    edge_map : 2D uint8 ndarray, shape (H, W) — normalized edge magnitude
    """
    img_f = normalize(image)
    lap   = _apply_laplacian_raw(img_f, connectivity)

    # Normalize the raw Laplacian to [0, 1] for display
    lap_min, lap_max = lap.min(), lap.max()
    if lap_max - lap_min < 1e-8:
        return np.zeros_like(image)
    lap_norm = (lap - lap_min) / (lap_max - lap_min)
    return denormalize(lap_norm)


def apply_sharpening(image: np.ndarray,
                     c: float = 1.0,
                     connectivity: int = 4,
                     pre_smooth: bool = False,
                     smooth_sigma: float = 0.5) -> np.ndarray:
    """
    Apply Laplacian-based image sharpening.

    Sharpening formula: g[x,y] = f[x,y] - c * L_h[x,y]
    where L_h = (h_lap * f) is the convolved Laplacian response.

    The minus sign is used because the standard 4/8-connectivity Laplacian
    kernel has a negative center coefficient; subtracting it adds the
    edge response back into the original, enhancing contrast at edges.

    Parameters
    ----------
    image        : 2D uint8 ndarray, shape (H, W)
    c            : sharpening coefficient in (0, 1.5]. Higher = sharper.
    connectivity : 4 or 8 (type of Laplacian kernel)
    pre_smooth   : if True, apply mild Gaussian pre-filter before Laplacian.
                   Reduces noise amplification (unsharp-masking style).
    smooth_sigma : sigma for the optional pre-smoothing Gaussian (default 0.5)

    Returns
    -------
    sharpened : 2D uint8 ndarray, shape (H, W)
    """
    if not (0 < c <= 2.0):
        raise ValueError(f"Coefficient c must be in (0, 2.0], got {c}.")

    img_f = normalize(image)

    # Optional pre-smoothing to suppress noise before differentiation
    if pre_smooth:
        from src.filters.gaussian_filter import apply_gaussian_filter
        smoothed   = apply_gaussian_filter(image, sigma=smooth_sigma, size=3)
        lap_source = normalize(smoothed)
    else:
        lap_source = img_f

    # Compute Laplacian of the (optionally pre-smoothed) image
    lap = _apply_laplacian_raw(lap_source, connectivity)

    # Composite sharpening: subtract Laplacian from original
    sharpened = img_f - c * lap

    return denormalize(np.clip(sharpened, 0.0, 1.0))
