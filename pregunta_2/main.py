from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .experimentaciones import DURAND_PARAMS, HOMO_PARAMS, run_experiments
from .figuras import save_all_figures
from .funciones import compute_luminance, load_hdr_rgb


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INPUT_DIR = PROJECT_ROOT / "archivos_t4"
OUTPUT_DIR = HERE / "output"
IMAGE_FILES = {
    "memorial": "memorial.tif",
    "belgium": "belgium.tif",
}


def main():
    plt.rcParams["figure.dpi"] = 120
    images_hdr = {
        name: load_hdr_rgb(INPUT_DIR / filename)
        for name, filename in IMAGE_FILES.items()
    }
    for name, image in images_hdr.items():
        luminance = compute_luminance(image)
        positive = luminance[luminance > 0]
        dynamic_range = np.percentile(luminance, 99.9) / (
            np.percentile(positive, 0.1) + 1e-6
        )
        print(
            f"Imagen '{name}': shape={image.shape} | "
            f"Rango Y=[{luminance.min():.3e}, {luminance.max():.3e}] | "
            f"DR aprox={dynamic_range:.1f}:1"
        )

    print("Parámetros fijos Homomórfico:", HOMO_PARAMS)
    print("Parámetros fijos Durand:", DURAND_PARAMS)
    results = run_experiments(images_hdr)
    save_all_figures(results, OUTPUT_DIR)
    print(f"\nFiguras guardadas en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
