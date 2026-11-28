from pathlib import Path

import matplotlib.pyplot as plt

from .experimentaciones import BETAS, SATURATION_GAINS
from .funciones import linear_to_srgb


def save_comparison_grids(results, output_dir):
    for image_name in ("memorial", "belgium"):
        rgb_reference = results[image_name]["rgb_ref"]
        for method_name in ("Homomórfico", "Durand"):
            method_data = results[image_name]["methods"][method_name]
            compressed = method_data["Y_comp"]
            rgb_independent, independent_metrics = method_data["rgb_indep"]
            fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5))

            axes[0, 0].imshow(rgb_reference)
            axes[0, 0].set_title("Referencia Original (sRGB)", fontsize=9.5)
            axes[0, 0].axis("off")
            axes[0, 1].imshow(
                linear_to_srgb(compressed), cmap="gray", vmin=0, vmax=1
            )
            axes[0, 1].set_title(
                f"Luminancia comprimida Y'\n({method_name})", fontsize=9.5
            )
            axes[0, 1].axis("off")
            axes[0, 2].imshow(rgb_independent)
            axes[0, 2].set_title(
                f"RGB Independiente\n"
                f"ΔH={independent_metrics['Delta_H_deg']:.1f}° | "
                f"C*={independent_metrics['C_star_mean']:.1f}",
                fontsize=9.5,
            )
            axes[0, 2].axis("off")

            for idx, beta in enumerate(BETAS):
                reconstructed, metrics = method_data["beta"][beta]
                axes[1, idx].imshow(reconstructed)
                axes[1, idx].set_title(
                    f"Estrat. 1: β = {beta}\n"
                    f"ΔH={metrics['Delta_H_deg']:.1f}° | "
                    f"C*={metrics['C_star_mean']:.1f} | "
                    f"Clip={metrics['Clip_pct']:.1f}%",
                    fontsize=9,
                )
                axes[1, idx].axis("off")

            for idx, gain in enumerate(SATURATION_GAINS):
                reconstructed, metrics = method_data["hsv"][gain]
                identity = " (identidad)" if gain == 1.0 else ""
                axes[2, idx].imshow(reconstructed)
                axes[2, idx].set_title(
                    f"Estrat. 2 (HSV): k_S = {gain}{identity}\n"
                    f"ΔH={metrics['Delta_H_deg']:.1f}° | "
                    f"C*={metrics['C_star_mean']:.1f} | "
                    f"Clip={metrics['Clip_pct']:.1f}%",
                    fontsize=9,
                )
                axes[2, idx].axis("off")

            fig.suptitle(
                f"{image_name.capitalize()} — Método {method_name}: "
                "Comparación de Reconstrucción de Color",
                fontsize=12.5,
            )
            fig.tight_layout()
            method_slug = method_name.lower().replace("ó", "o")
            fig.savefig(
                output_dir / f"01_{image_name}_{method_slug}_grilla.png",
                bbox_inches="tight",
            )
            plt.close(fig)


def save_crop_figures(results, output_dir):
    memorial_height, memorial_width = results["memorial"]["rgb_ref"].shape[:2]
    memorial_crop = (
        slice(int(0.58 * memorial_height), int(0.90 * memorial_height)),
        slice(int(0.40 * memorial_width), int(0.95 * memorial_width)),
    )
    belgium_height, belgium_width = results["belgium"]["rgb_ref"].shape[:2]
    belgium_crop = (
        slice(int(0.25 * belgium_height), int(0.68 * belgium_height)),
        slice(int(0.32 * belgium_width), int(0.68 * belgium_width)),
    )
    crops = [
        ("memorial", memorial_crop, "Recorte 1: Colores intensos (vitrales en Memorial)"),
        (
            "belgium",
            belgium_crop,
            "Recorte 2: Transición fuerte de iluminación (interior/exterior en Belgium)",
        ),
    ]

    for image_name, (row_slice, col_slice), title in crops:
        rgb_reference = results[image_name]["rgb_ref"][row_slice, col_slice]
        fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
        for row, method_name in enumerate(("Homomórfico", "Durand")):
            method_data = results[image_name]["methods"][method_name]
            panels = [
                ("Referencia sRGB", rgb_reference),
                (
                    f"{method_name}: RGB Indep.",
                    method_data["rgb_indep"][0][row_slice, col_slice],
                ),
                (
                    f"{method_name}: β = 1.0",
                    method_data["beta"][1.0][0][row_slice, col_slice],
                ),
                (
                    f"{method_name}: β = 1.4",
                    method_data["beta"][1.4][0][row_slice, col_slice],
                ),
                (
                    f"{method_name}: HSV k_S = 1.4",
                    method_data["hsv"][1.4][0][row_slice, col_slice],
                ),
            ]
            for col, (panel_title, panel_image) in enumerate(panels):
                axes[row, col].imshow(panel_image)
                axes[row, col].set_title(panel_title, fontsize=9)
                axes[row, col].axis("off")

        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"02_recorte_{image_name}.png", bbox_inches="tight")
        plt.close(fig)


def save_all_figures(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_grids(results, output_dir)
    save_crop_figures(results, output_dir)
