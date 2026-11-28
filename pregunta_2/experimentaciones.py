import numpy as np

from .funciones import (
    apply_hdr_per_channel_rgb,
    compute_color_metrics,
    compute_luminance,
    durand_tonemap_2d,
    homomorphic_filter_2d,
    make_display_reference,
    reconstruct_color_beta,
    reconstruct_color_hsv_saturation,
)


DURAND_PARAMS = dict(sigma_s_ratio=0.02, sigma_r=0.4, target_contrast=5.0)
HOMO_PARAMS = dict(gamma_l=0.25, gamma_h=1.65, c=1.0, d0_ratio=0.08)
BETAS = [0.6, 1.0, 1.4]
SATURATION_GAINS = [0.6, 1.0, 1.4]
METHODS = [
    ("Homomórfico", homomorphic_filter_2d, HOMO_PARAMS),
    ("Durand", durand_tonemap_2d, DURAND_PARAMS),
]


def _print_metrics(label, metrics):
    print(
        f"  {label:<24} | {metrics['S_mean']:7.3f} | "
        f"{metrics['C_star_mean']:8.2f} | {metrics['Delta_H_deg']:7.2f} | "
        f"{metrics['Clip_pct']:7.2f}%"
    )


def run_experiments(images_hdr):
    results = {}
    for image_name, rgb_hdr in images_hdr.items():
        luminance = compute_luminance(rgb_hdr)
        rgb_reference = make_display_reference(rgb_hdr, p_white=99.0)
        reference_metrics = compute_color_metrics(rgb_reference, rgb_reference)
        results[image_name] = {
            "Y_orig": luminance,
            "rgb_ref": rgb_reference,
            "ref_metrics": reference_metrics,
            "methods": {},
        }

        print("=" * 74)
        print(
            f"IMAGEN: {image_name.upper()} | Ref sRGB -> "
            f"S_mean={reference_metrics['S_mean']:.3f}, "
            f"C*_mean={reference_metrics['C_star_mean']:.2f}"
        )
        print("=" * 74)

        for method_name, method_fn, params in METHODS:
            compressed = method_fn(luminance, **params)
            independent = apply_hdr_per_channel_rgb(rgb_hdr, method_fn, **params)
            independent_metrics = compute_color_metrics(
                rgb_reference, independent
            )

            beta_results = {}
            for beta in BETAS:
                reconstructed, raw = reconstruct_color_beta(
                    rgb_hdr, luminance, compressed, beta=beta
                )
                beta_results[beta] = (
                    reconstructed,
                    compute_color_metrics(rgb_reference, reconstructed, raw),
                )

            hsv_results = {}
            for gain in SATURATION_GAINS:
                reconstructed, raw_proxy = reconstruct_color_hsv_saturation(
                    rgb_hdr, luminance, compressed, sat_gain=gain
                )
                hsv_results[gain] = (
                    reconstructed,
                    compute_color_metrics(
                        rgb_reference, reconstructed, raw_proxy
                    ),
                )

            results[image_name]["methods"][method_name] = {
                "Y_comp": compressed,
                "rgb_indep": (independent, independent_metrics),
                "beta": beta_results,
                "hsv": hsv_results,
            }

            print(f"\n--- Método HDR: {method_name} ---")
            print(
                f"  {'Variante':<24} | {'S_mean':>7} | {'C*_mean':>8} | "
                f"{'dH (deg)':>8} | {'Clip (%)':>8}"
            )
            print(f"  {'-' * 64}")
            print(
                f"  {'RGB Independiente':<24} | "
                f"{independent_metrics['S_mean']:7.3f} | "
                f"{independent_metrics['C_star_mean']:8.2f} | "
                f"{independent_metrics['Delta_H_deg']:7.2f} | {'N/A':>8}"
            )
            for beta in BETAS:
                _print_metrics(
                    f"Estrat. 1 (beta = {beta:.1f})", beta_results[beta][1]
                )
            for gain in SATURATION_GAINS:
                suffix = " (id)" if gain == 1.0 else ""
                _print_metrics(
                    f"Estrat. 2 (kS = {gain:.1f}{suffix})",
                    hsv_results[gain][1],
                )

    return results
