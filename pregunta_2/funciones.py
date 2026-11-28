from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift

try:
    import imageio.v3 as iio
except ImportError:
    iio = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.restoration import denoise_bilateral
except ImportError:
    denoise_bilateral = None


EPS = 1e-6


def load_hdr_rgb(path):
    """Load an HDR/TIFF image as nonnegative float64 RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")
    if iio is not None:
        image = iio.imread(path).astype(np.float64)
    elif cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"No se pudo leer: {path}")
        image = image.astype(np.float64)
        if image.ndim == 3:
            image = image[..., ::-1]
    else:
        raise RuntimeError("Se requiere imageio o opencv-python.")
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[..., :3]
    return np.maximum(image, 0.0)


def compute_luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def robust_normalize_01(array, p_low=0.5, p_high=99.5):
    low = np.percentile(array, p_low)
    high = np.percentile(array, p_high)
    if high - low < EPS:
        high, low = array.max(), array.min()
    if high - low < EPS:
        return np.zeros_like(array, dtype=np.float64)
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def linear_to_srgb(rgb_linear):
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    return np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )


def make_display_reference(rgb_hdr, p_white=99.0):
    luminance = compute_luminance(rgb_hdr)
    white = np.percentile(luminance, p_white)
    if white < EPS:
        white = np.max(luminance) + EPS
    rgb_linear = np.clip(rgb_hdr / white, 0.0, 1.0)
    return linear_to_srgb(rgb_linear)


def rgb_to_hsv(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(red, green), blue)
    minc = np.minimum(np.minimum(red, green), blue)
    value = maxc
    delta = maxc - minc
    saturation = np.where(maxc > EPS, delta / (maxc + EPS), 0.0)

    mask = delta > EPS
    rc = np.where(mask, (maxc - red) / (delta + EPS), 0.0)
    gc = np.where(mask, (maxc - green) / (delta + EPS), 0.0)
    bc = np.where(mask, (maxc - blue) / (delta + EPS), 0.0)
    hue = np.zeros_like(maxc)
    hue = np.where(mask & (red == maxc), bc - gc, hue)
    hue = np.where(mask & (green == maxc), 2.0 + rc - bc, hue)
    hue = np.where(mask & (blue == maxc), 4.0 + gc - rc, hue)
    hue = (hue / 6.0) % 1.0
    return np.stack([hue, saturation, value], axis=-1)


def hsv_to_rgb(hsv):
    hue = hsv[..., 0] % 1.0
    saturation = np.clip(hsv[..., 1], 0.0, 1.0)
    value = np.clip(hsv[..., 2], 0.0, 1.0)
    sector = np.floor(hue * 6.0).astype(np.int32)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector %= 6
    red = np.choose(sector, [value, q, p, p, t, value])
    green = np.choose(sector, [t, value, value, q, p, p])
    blue = np.choose(sector, [p, p, t, value, value, q])
    return np.clip(np.stack([red, green, blue], axis=-1), 0.0, 1.0)


def rgb_to_cielab(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    linear = np.where(
        rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
    )
    red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
    x = 0.4124564 * red + 0.3575761 * green + 0.1804375 * blue
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = 0.0193339 * red + 0.1191920 * green + 0.9503041 * blue

    delta = 6.0 / 29.0

    def f_lab(component):
        return np.where(
            component > delta**3,
            np.cbrt(component),
            component / (3.0 * delta**2) + 4.0 / 29.0,
        )

    fx = f_lab(x / 0.95047)
    fy = f_lab(y)
    fz = f_lab(z / 1.08883)
    lightness = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([lightness, a, b], axis=-1)


def compute_color_metrics(rgb_ref, rgb_rec, raw_unclipped=None):
    hsv_ref = rgb_to_hsv(rgb_ref)
    hsv_rec = rgb_to_hsv(rgb_rec)
    lab_rec = rgb_to_cielab(rgb_rec)
    chroma = np.hypot(lab_rec[..., 1], lab_rec[..., 2])

    hue_difference = np.abs(hsv_rec[..., 0] - hsv_ref[..., 0])
    hue_difference = np.minimum(hue_difference, 1.0 - hue_difference) * 360.0
    chroma_mask = (hsv_ref[..., 1] > 0.10) & (hsv_ref[..., 2] > 0.05)
    mean_hue_difference = (
        float(np.mean(hue_difference[chroma_mask])) if np.any(chroma_mask) else 0.0
    )
    if raw_unclipped is not None:
        clip_percentage = float(
            100.0 * np.mean(np.any(raw_unclipped > 1.0, axis=-1))
        )
    else:
        clip_percentage = 0.0

    return {
        "S_mean": float(np.mean(hsv_rec[..., 1])),
        "C_star_mean": float(np.mean(chroma)),
        "Delta_H_deg": mean_hue_difference,
        "Clip_pct": clip_percentage,
    }


def homomorphic_filter_2d(
    channel, gamma_l=0.25, gamma_h=1.65, c=1.0, d0_ratio=0.08
):
    height, width = channel.shape
    scale = np.percentile(channel, 99.9) + EPS
    log_image = np.log(np.maximum(channel / scale, 0.0) + 1e-4)

    pad_h, pad_w = height // 2, width // 2
    padded = np.pad(
        log_image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect"
    )
    padded_h, padded_w = padded.shape
    u = np.arange(padded_h) - padded_h // 2
    v = np.arange(padded_w) - padded_w // 2
    v_grid, u_grid = np.meshgrid(v, u)
    distance_squared = u_grid**2 + v_grid**2
    d0 = d0_ratio * min(padded_h, padded_w)
    filter_response = (gamma_h - gamma_l) * (
        1.0 - np.exp(-c * distance_squared / (d0**2 + EPS))
    ) + gamma_l

    spectrum = fftshift(fft2(padded))
    filtered = np.real(ifft2(ifftshift(spectrum * filter_response)))
    filtered_log = filtered[pad_h : pad_h + height, pad_w : pad_w + width]
    return robust_normalize_01(np.exp(filtered_log), p_low=0.5, p_high=99.5)


def fast_bilateral_2d(image, sigma_spatial, sigma_range):
    if cv2 is not None:
        return cv2.bilateralFilter(
            image.astype(np.float32),
            d=-1,
            sigmaColor=float(sigma_range),
            sigmaSpace=float(sigma_spatial),
        ).astype(np.float64)
    if denoise_bilateral is not None:
        return denoise_bilateral(
            image,
            sigma_color=sigma_range,
            sigma_spatial=sigma_spatial,
            channel_axis=None,
        ).astype(np.float64)
    raise RuntimeError(
        "Se requiere opencv-python o scikit-image para el filtro bilateral."
    )


def durand_tonemap_2d(
    channel, sigma_s_ratio=0.02, sigma_r=0.4, target_contrast=5.0
):
    height, width = channel.shape
    scale = np.percentile(channel, 99.9) + EPS
    log_luminance = np.log10(np.maximum(channel / scale, 0.0) + 1e-5)
    sigma_spatial = max(2.0, sigma_s_ratio * max(height, width))
    base = fast_bilateral_2d(
        log_luminance, sigma_spatial=sigma_spatial, sigma_range=sigma_r
    )
    detail = log_luminance - base
    base_max = np.percentile(base, 99.8)
    base_min = np.percentile(base, 0.2)
    base_range = max(base_max - base_min, EPS)
    compression_factor = np.log10(target_contrast) / base_range
    log_output = compression_factor * (base - base_max) + detail
    return robust_normalize_01(10.0**log_output, p_low=0.5, p_high=99.5)


def apply_hdr_per_channel_rgb(rgb_hdr, method_fn, **kwargs):
    channels = [method_fn(rgb_hdr[..., idx], **kwargs) for idx in range(3)]
    return np.stack(channels, axis=-1)


def reconstruct_color_beta(rgb_hdr, luminance, compressed_luminance, beta=1.0, eps=EPS):
    scale = np.percentile(luminance, 99.0) + eps
    rgb_scaled = np.maximum(rgb_hdr / scale, 0.0)
    luminance_scaled = np.maximum(luminance / scale, 0.0)
    ratio = rgb_scaled / (luminance_scaled[..., None] + eps)
    raw = compressed_luminance[..., None] * np.power(ratio, beta)
    clipped = np.clip(raw, 0.0, 1.0)
    return linear_to_srgb(clipped), raw


def reconstruct_color_hsv_saturation(
    rgb_hdr, luminance, compressed_luminance, sat_gain=1.0, eps=EPS
):
    base_rgb, _ = reconstruct_color_beta(
        rgb_hdr, luminance, compressed_luminance, beta=1.0, eps=eps
    )
    hsv = rgb_to_hsv(base_rgb)
    saturation_raw = hsv[..., 1] * sat_gain
    hsv_modified = hsv.copy()
    hsv_modified[..., 1] = np.clip(saturation_raw, 0.0, 1.0)
    output = hsv_to_rgb(hsv_modified)
    raw_proxy = np.where(saturation_raw[..., None] > 1.0, 1.05, output)
    return output, raw_proxy
