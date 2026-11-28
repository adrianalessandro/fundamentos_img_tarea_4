from pathlib import Path

import matplotlib.pyplot as plt

from .experimentaciones import run_experiments
from .figuras import save_all_figures
from .funciones import load_hdr_rgb, prepare_luminance


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_PATH = PROJECT_ROOT / "archivos_t4" / "memorial.tif"
OUTPUT_DIR = HERE / "output"


def main():
    plt.rcParams["figure.dpi"] = 120
    rgb_hdr = load_hdr_rgb(INPUT_PATH)
    luminance, normalized = prepare_luminance(rgb_hdr, log_strength=15.0)
    print(f"Imagen cargada: {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Shape RGB: {rgb_hdr.shape} | Dtype original: {rgb_hdr.dtype}")
    print(
        f"Rango luminancia bruta Y: [{luminance.min():.4e}, "
        f"{luminance.max():.4e}]"
    )
    print(
        f"Rango luminancia comprimida u: [{normalized.min():.4f}, "
        f"{normalized.max():.4f}] | Shape: {normalized.shape}"
    )

    results = run_experiments(normalized)
    save_all_figures(normalized, luminance, results, OUTPUT_DIR)
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
