"""Projections of datasets A, B and C (seed 0), made from the files saved by umap_comparison.py.

  projection_{A,B,C}.png   one projection per dataset
  projections_ABC.png      all three side by side

Usage: python umap_comparison_charts.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from umap_comparison import EMBEDDING_DIR, MIN_DIST, NEIGHBORS, OUTPUT_DIR
from umap_study import SEEDS, embedding_path
from umap_study_charts import draw_projection

CHART_SEED = SEEDS[0]
PANELS = [
    ("A", "A -- pagrindinė aibė"),
    ("B", "B -- be koreliuotų požymių"),
    ("C", "C -- be didžiųjų išskirčių"),
]


def load_coordinates(name):
    path = embedding_path(NEIGHBORS, MIN_DIST, CHART_SEED) if name == "A" else EMBEDDING_DIR / f"{name}_seed{CHART_SEED}.csv"
    return pd.read_csv(path)


def save_projection(name):
    figure, axis = plt.subplots(figsize=(9, 7.5))
    draw_projection(axis, load_coordinates(name))
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(title="Klasė", markerscale=3, frameon=False, loc="best")
    figure.tight_layout()
    path = OUTPUT_DIR / f"projection_{name}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path}")


def main():
    for name, _ in PANELS:
        save_projection(name)

    figure, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS), 5.2))
    for axis, (name, title) in zip(axes, PANELS):
        coordinates = load_coordinates(name)
        draw_projection(axis, coordinates, point_size=2)
        axis.set_title(f"{title} ({len(coordinates)} eil.)", fontsize=11)
        axis.tick_params(labelsize=8)

    handles, names = axes[0].get_legend_handles_labels()
    figure.legend(handles, names, title="Klasė", markerscale=4,
                  frameon=False, loc="lower center", ncol=len(names))
    figure.supxlabel("UMAP 1", y=0.09)
    figure.supylabel("UMAP 2")
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    path = OUTPUT_DIR / "projections_ABC.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
