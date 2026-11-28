from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experimentaciones import ALPHAS, GAIN_CONFIGS


def save_input_figure(luminance, normalized, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(
        np.clip(luminance / np.percentile(luminance, 99), 0, 1), cmap="gray"
    )
    axes[0].set_title("Luminancia lineal (clipeada p99)")
    axes[0].axis("off")
    axes[1].imshow(normalized, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Luminancia comprimida logarítmicamente u ∈ [0, 1]")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "00_luminancia_entrada.png", bbox_inches="tight")
    plt.close(fig)


def save_decomposition_figure(results, output_dir):
    lap_details = results["lap_details"]
    lap_residual = results["lap_residual"]
    star_details = results["star_details"]
    star_residual = results["star_residual"]

    vmax = max(
        np.percentile(np.abs(lap_details[0]), 99.5),
        np.percentile(np.abs(star_details[0]), 99.5),
    )
    fig, axes = plt.subplots(2, len(lap_details) + 1, figsize=(18, 6.5))
    for idx, (lap_layer, star_layer) in enumerate(zip(lap_details, star_details)):
        axes[0, idx].imshow(lap_layer, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0, idx].set_title(
            f"Laplaciana $d_{idx + 1}$\n{lap_layer.shape[0]}x{lap_layer.shape[1]}",
            fontsize=9,
        )
        axes[0, idx].axis("off")
        image = axes[1, idx].imshow(
            star_layer, cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        axes[1, idx].set_title(
            f"Starlet $w_{idx + 1}$\n{star_layer.shape[0]}x{star_layer.shape[1]}",
            fontsize=9,
        )
        axes[1, idx].axis("off")

    axes[0, -1].imshow(lap_residual, cmap="gray", vmin=0, vmax=1)
    axes[0, -1].set_title(
        f"Residual Lap. $c_5$\n{lap_residual.shape[0]}x{lap_residual.shape[1]}",
        fontsize=9,
    )
    axes[0, -1].axis("off")
    axes[1, -1].imshow(star_residual, cmap="gray", vmin=0, vmax=1)
    axes[1, -1].set_title(
        f"Residual Starlet $c_5$\n{star_residual.shape[0]}x{star_residual.shape[1]}",
        fontsize=9,
    )
    axes[1, -1].axis("off")

    colorbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.7])
    fig.colorbar(image, cax=colorbar_ax, label=f"Amplitud detalle (±{vmax:.3f})")
    fig.suptitle(
        "Descomposición Multiescala (5 niveles + residual) — "
        "Escala de visualización coherente",
        fontsize=12,
    )
    fig.subplots_adjust(right=0.90, wspace=0.12, hspace=0.22)
    fig.savefig(output_dir / "01_capas_detalle_y_residual.png", bbox_inches="tight")
    plt.close(fig)


def save_threshold_figures(image, results, output_dir):
    threshold_results = results["threshold_results"]
    height, width = image.shape
    row_start, row_end = int(0.52 * height), int(0.82 * height)
    col_start, col_end = int(0.25 * width), int(0.75 * width)
    crop = image[row_start:row_end, col_start:col_end]

    fig, axes = plt.subplots(2, len(ALPHAS) + 1, figsize=(16, 8))
    for row in range(2):
        axes[row, 0].imshow(crop, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Original u\n(recorte ampliado)", fontsize=9.5)
        axes[row, 0].axis("off")

    for col, alpha in enumerate(ALPHAS, start=1):
        for row, method, label in (
            (0, "lap", "Laplaciana"),
            (1, "star", "Starlet"),
        ):
            result = threshold_results[method][alpha]
            axes[row, col].imshow(
                result["reconstruction"][row_start:row_end, col_start:col_end],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row, col].set_title(
                f"{label} (α={alpha})\n"
                f"Retenido: {result['total_percentage']:.1f}% | "
                f"RMSE: {result['rmse']:.4f}",
                fontsize=9,
            )
            axes[row, col].axis("off")

    fig.suptitle(
        r"Umbral duro relativo $T_j = \alpha \cdot \max|d_j|$ — "
        "Comparación de artefactos en recorte ampliado",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "02_umbral_duro_comparacion.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for col, alpha in enumerate([0.15, 0.30]):
        for row, method, label in (
            (0, "lap", "Laplaciana"),
            (1, "star", "Starlet"),
        ):
            result = threshold_results[method][alpha]
            axes[row, col].imshow(
                result["reconstruction"], cmap="gray", vmin=0, vmax=1
            )
            axes[row, col].set_title(
                f"{label} completa (α={alpha}) | RMSE={result['rmse']:.4f}",
                fontsize=9.5,
            )
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "02b_umbral_duro_completas.png", bbox_inches="tight")
    plt.close(fig)


def save_gain_figure(image, results, output_dir):
    height, width = image.shape
    row_start, row_end = int(0.52 * height), int(0.82 * height)
    col_start, col_end = int(0.25 * width), int(0.75 * width)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))

    for row, (label, _) in enumerate(GAIN_CONFIGS):
        gain_result = results["gain_results"][label]
        axes[row, 0].imshow(gain_result["lap"], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(
            f"Laplaciana — {label} {gain_result['gains']}\n(Completa)",
            fontsize=8.5,
        )
        axes[row, 0].axis("off")
        axes[row, 1].imshow(gain_result["star"], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(
            f"Starlet — {label} {gain_result['gains']}\n(Completa)",
            fontsize=8.5,
        )
        axes[row, 1].axis("off")
        axes[row, 2].imshow(
            gain_result["lap"][row_start:row_end, col_start:col_end],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 2].set_title("Laplaciana (Recorte)", fontsize=8.5)
        axes[row, 2].axis("off")
        axes[row, 3].imshow(
            gain_result["star"][row_start:row_end, col_start:col_end],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 3].set_title("Starlet (Recorte)", fontsize=8.5)
        axes[row, 3].axis("off")

    fig.suptitle(
        "Modificación por ganancias multiplicativas: "
        "Estructuras Finas vs. Estructuras Gruesas",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "03_ganancias_multiescala.png", bbox_inches="tight")
    plt.close(fig)


def save_all_figures(image, luminance, results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_input_figure(luminance, image, output_dir)
    save_decomposition_figure(results, output_dir)
    save_threshold_figures(image, results, output_dir)
    save_gain_figure(image, results, output_dir)
