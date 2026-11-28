from pathlib import Path

import numpy as np
from scipy.ndimage import convolve1d

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    import cv2
except ImportError:
    cv2 = None


H_B3 = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
NUM_SCALES = 5


def load_hdr_rgb(path):
    """Load an HDR/TIFF image as float64 RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    if iio is not None:
        image = iio.imread(path).astype(np.float64)
    elif cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"OpenCV no pudo leer: {path}")
        image = image.astype(np.float64)
        if image.ndim == 3:
            image = image[..., ::-1]
    else:
        raise RuntimeError("Instala imageio o opencv-python para leer la imagen.")

    if image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]
    return image


def prepare_luminance(rgb_img, log_strength=15.0):
    """Extract luminance and compress it globally to [0, 1]."""
    rgb_pos = np.maximum(rgb_img, 0.0)
    if rgb_pos.ndim == 3:
        luminance = (
            0.2126 * rgb_pos[..., 0]
            + 0.7152 * rgb_pos[..., 1]
            + 0.0722 * rgb_pos[..., 2]
        )
    else:
        luminance = rgb_pos.copy()

    scale_ref = np.percentile(luminance, 99.9)
    if scale_ref <= 0:
        scale_ref = np.max(luminance) if np.max(luminance) > 0 else 1.0
    compressed = np.log1p(log_strength * luminance / scale_ref)
    normalized = (compressed - compressed.min()) / (
        compressed.max() - compressed.min()
    )
    return luminance, normalized


def conv2d_separable(image, kernel_1d, mode="reflect"):
    temp = convolve1d(image, kernel_1d, axis=0, mode=mode)
    return convolve1d(temp, kernel_1d, axis=1, mode=mode)


def laplacian_reduce(image, kernel_1d=H_B3):
    smoothed = conv2d_separable(image, kernel_1d, mode="reflect")
    return smoothed[::2, ::2]


def laplacian_expand(image, target_shape, kernel_1d=H_B3):
    target_height, target_width = target_shape
    upsampled = np.zeros((target_height, target_width), dtype=np.float64)
    upsampled[::2, ::2] = image
    return conv2d_separable(upsampled, 2.0 * kernel_1d, mode="reflect")


def laplacian_pyramid_decompose(image, num_scales=NUM_SCALES, kernel_1d=H_B3):
    current = image.astype(np.float64).copy()
    details = []
    for _ in range(num_scales):
        next_level = laplacian_reduce(current, kernel_1d=kernel_1d)
        expanded = laplacian_expand(
            next_level, target_shape=current.shape, kernel_1d=kernel_1d
        )
        details.append(current - expanded)
        current = next_level
    return details, current


def laplacian_pyramid_reconstruct(details, residual, kernel_1d=H_B3):
    current = residual.astype(np.float64).copy()
    for detail in reversed(details):
        expanded = laplacian_expand(
            current, target_shape=detail.shape, kernel_1d=kernel_1d
        )
        current = detail + expanded
    return current


def build_atrous_kernel(kernel_1d, scale_idx):
    step = 2 ** (scale_idx - 1)
    if step == 1:
        return kernel_1d.copy()
    atrous = np.zeros((len(kernel_1d) - 1) * step + 1, dtype=np.float64)
    atrous[::step] = kernel_1d
    return atrous


def starlet_decompose(image, num_scales=NUM_SCALES, kernel_1d=H_B3):
    current = image.astype(np.float64).copy()
    details = []
    for scale_idx in range(1, num_scales + 1):
        kernel = build_atrous_kernel(kernel_1d, scale_idx)
        next_level = conv2d_separable(current, kernel, mode="reflect")
        details.append(current - next_level)
        current = next_level
    return details, current


def starlet_reconstruct(details, residual):
    reconstruction = residual.astype(np.float64).copy()
    for detail in details:
        reconstruction += detail
    return reconstruction


def hard_threshold_layers(details, alpha):
    thresholded = []
    kept_percentages = []
    for layer in details:
        threshold = alpha * np.max(np.abs(layer))
        mask = np.abs(layer) >= threshold
        thresholded.append(np.where(mask, layer, 0.0))
        kept_percentages.append(100.0 * np.count_nonzero(mask) / layer.size)
    return thresholded, kept_percentages


def apply_scale_gains(details, gains):
    if len(details) != len(gains):
        raise ValueError("Debe haber una ganancia por cada capa de detalle.")
    return [gain * layer for gain, layer in zip(gains, details)]
