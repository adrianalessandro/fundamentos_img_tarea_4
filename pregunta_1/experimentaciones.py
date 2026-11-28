import numpy as np

from .funciones import (
    NUM_SCALES,
    apply_scale_gains,
    hard_threshold_layers,
    laplacian_pyramid_decompose,
    laplacian_pyramid_reconstruct,
    starlet_decompose,
    starlet_reconstruct,
)


ALPHAS = [0.05, 0.15, 0.30]
GAIN_CONFIGS = [
    ("Énfasis finas", [2.5, 2.0, 1.3, 1.0, 1.0]),
    ("Énfasis gruesas", [1.0, 1.0, 1.3, 2.0, 2.5]),
]


def reconstruction_metrics(original, reconstruction):
    error = original - reconstruction
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
    }


def run_experiments(image):
    lap_details, lap_residual = laplacian_pyramid_decompose(
        image, num_scales=NUM_SCALES
    )
    star_details, star_residual = starlet_decompose(image, num_scales=NUM_SCALES)

    lap_reconstruction = laplacian_pyramid_reconstruct(lap_details, lap_residual)
    star_reconstruction = starlet_reconstruct(star_details, star_residual)
    exact_reconstruction = {
        "lap": reconstruction_metrics(image, lap_reconstruction),
        "star": reconstruction_metrics(image, star_reconstruction),
    }

    threshold_results = {"lap": {}, "star": {}}
    print("=== Validación de reconstrucción sin modificar coeficientes ===")
    for method, metrics in exact_reconstruction.items():
        label = "Pirámide Laplaciana" if method == "lap" else "Transformada Starlet"
        print(
            f"{label} -> RMSE: {metrics['rmse']:.4e} | "
            f"Error máx abs: {metrics['max_error']:.4e}"
        )

    original_count = image.size
    lap_count = sum(layer.size for layer in lap_details) + lap_residual.size
    star_count = sum(layer.size for layer in star_details) + star_residual.size
    print("\nDimensiones y cantidad de coeficientes por nivel:")
    for idx, (lap_layer, star_layer) in enumerate(
        zip(lap_details, star_details), start=1
    ):
        print(
            f"  Escala {idx}: Laplaciana d_{idx} = {lap_layer.shape} "
            f"({lap_layer.size:,} coefs) | Starlet w_{idx} = {star_layer.shape} "
            f"({star_layer.size:,} coefs)"
        )
    print(
        f"  Residual: Laplaciana c_5 = {lap_residual.shape} "
        f"({lap_residual.size:,} coefs) | Starlet c_5 = {star_residual.shape} "
        f"({star_residual.size:,} coefs)"
    )
    print(
        f"Total coeficientes -> Original: {original_count:,} | "
        f"Laplaciana: {lap_count:,} ({lap_count / original_count:.2f}x) | "
        f"Starlet: {star_count:,} ({star_count / original_count:.2f}x)"
    )

    print("\n=== Umbral duro: coeficientes retenidos y RMSE ===")
    for alpha in ALPHAS:
        for method, details, residual, reconstruct in (
            ("lap", lap_details, lap_residual, laplacian_pyramid_reconstruct),
            ("star", star_details, star_residual, starlet_reconstruct),
        ):
            thresholded, layer_percentages = hard_threshold_layers(details, alpha)
            raw_reconstruction = reconstruct(thresholded, residual)
            rmse = float(np.sqrt(np.mean((image - raw_reconstruction) ** 2)))
            total_percentage = 100.0 * sum(
                np.count_nonzero(layer) for layer in thresholded
            ) / sum(layer.size for layer in thresholded)
            threshold_results[method][alpha] = {
                "reconstruction": np.clip(raw_reconstruction, 0.0, 1.0),
                "layer_percentages": layer_percentages,
                "total_percentage": total_percentage,
                "rmse": rmse,
            }
            label = "Laplaciana" if method == "lap" else "Starlet"
            print(
                f"alpha={alpha:.2f} | {label}: "
                f"% por capa {[round(float(p), 2) for p in layer_percentages]} | "
                f"total={total_percentage:.2f}% | RMSE={rmse:.4f}"
            )

    gain_results = {}
    for label, gains in GAIN_CONFIGS:
        gain_results[label] = {
            "gains": gains,
            "lap": np.clip(
                laplacian_pyramid_reconstruct(
                    apply_scale_gains(lap_details, gains), lap_residual
                ),
                0.0,
                1.0,
            ),
            "star": np.clip(
                starlet_reconstruct(apply_scale_gains(star_details, gains), star_residual),
                0.0,
                1.0,
            ),
        }

    return {
        "lap_details": lap_details,
        "lap_residual": lap_residual,
        "star_details": star_details,
        "star_residual": star_residual,
        "exact_reconstruction": exact_reconstruction,
        "threshold_results": threshold_results,
        "gain_results": gain_results,
    }
