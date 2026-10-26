"""Charts for the UMAP parameter study, made from the files saved by umap_study.py.

  projection_nn{n}_md{m}.png    one projection per setting (seed 0, or every seed with --all-seeds)
  projections_grid.png          all 20 settings, rows min_dist, columns n_neighbors (seed 0)
  heatmaps/{metric}.png         4 x 5 heatmap per metric, mean over the seeds (std below it)
  projections_n_neighbors.png   projections for every n_neighbors value at the default min_dist
  projections_min_dist.png      projections for every min_dist value at the default n_neighbors
  metrics_vs_n_neighbors.png    metrics against n_neighbors at the default min_dist, mean ± std
  metrics_vs_min_dist.png       metrics against min_dist at the default n_neighbors, mean ± std

Usage: python umap_study_charts.py [--all-seeds]
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from umap_study import (
    DEFAULT_MIN_DIST, DEFAULT_NEIGHBORS, METRIC_COLUMNS, METRIC_K, MIN_DIST_VALUES, NEIGHBOR_VALUES,
    OUTPUT_DIR, RUNS_FILE, SEEDS, SETTINGS, embedding_path, summarize,
)
from umap_visualization import CLASS_COLORS

CHART_DIR = OUTPUT_DIR / "charts"
HEATMAP_DIR = CHART_DIR / "heatmaps"
CHART_SEED = SEEDS[0]
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# one hue from light to dark, darker = larger value
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
METRIC_LABELS = {
    **{f"T_{k}": f"Trustworthiness T({k}) ↑" for k in METRIC_K},
    **{f"C_{k}": f"Continuity C({k}) ↑" for k in METRIC_K},
    **{f"KNN_{k}": f"KNN kaimynų išsaugojimas ({k}) ↑" for k in METRIC_K},
    "spearman": "Spearman ρ ↑",
    "stress": "Normalized Stress ↓",
}


def draw_projection(axis, coordinates, point_size=5):
    for name, color in CLASS_COLORS.items():
        points = coordinates[coordinates["class"] == name]
        axis.scatter(points["umap_1"], points["umap_2"], s=point_size, alpha=0.6, color=color,
                     linewidths=0, label=name)
    # UMAP coordinates have no units, keeping the axes 1:1 avoids stretching the clusters
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)


def save_projection(neighbors, min_dist, seed):
    coordinates = pd.read_csv(embedding_path(neighbors, min_dist, seed))
    figure, axis = plt.subplots(figsize=(9, 7.5))
    draw_projection(axis, coordinates)
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(title="Klasė", markerscale=3, frameon=False, loc="best")
    figure.tight_layout()
    suffix = "" if seed == CHART_SEED else f"_seed{seed}"
    figure.savefig(CHART_DIR / f"projection_nn{neighbors}_md{min_dist}{suffix}.png", dpi=150)
    plt.close(figure)


def save_projection_row(settings, labels, path, columns):
    rows = -(-len(settings) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.2 * rows), squeeze=False)
    for axis, (neighbors, min_dist), label in zip(axes.flat, settings, labels):
        draw_projection(axis, pd.read_csv(embedding_path(neighbors, min_dist, CHART_SEED)), point_size=2)
        axis.set_title(label, fontsize=11)
        axis.tick_params(labelsize=8)
    for axis in axes.flat[len(settings):]:
        axis.set_visible(False)

    handles, names = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, names, title="Klasė", markerscale=4, frameon=False,
                  loc="lower center", ncol=len(names))
    figure.supxlabel("UMAP 1", y=0.06)
    figure.supylabel("UMAP 2")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_projection_grid(path):
    rows, columns = len(MIN_DIST_VALUES), len(NEIGHBOR_VALUES)
    figure, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 3.4 * rows))
    for row, min_dist in enumerate(MIN_DIST_VALUES):
        for column, neighbors in enumerate(NEIGHBOR_VALUES):
            axis = axes[row, column]
            draw_projection(axis, pd.read_csv(embedding_path(neighbors, min_dist, CHART_SEED)), point_size=1.5)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_title(f"n_neighbors = {neighbors}", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"min_dist = {min_dist:g}", fontsize=11)

    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, title="Klasė", markerscale=5, frameon=False,
                  loc="lower center", ncol=len(names))
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_panels(settings, titles, path):
    """Projections side by side with one shared legend below, in the style of the group's other figures."""
    figure, axes = plt.subplots(1, len(settings), figsize=(6 * len(settings), 6.4), squeeze=False)
    for axis, (neighbors, min_dist), title in zip(axes.flat, settings, titles):
        draw_projection(axis, pd.read_csv(embedding_path(neighbors, min_dist, CHART_SEED)), point_size=3)
        axis.set_title(title, fontsize=12)
        # no ticks: UMAP coordinates have no units, only the layout matters
        axis.set_xticks([])
        axis.set_yticks([])

    handles, names = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, names, title="Klasė", markerscale=4, loc="lower center", ncol=len(names))
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_heatmap(summary, metric, path):
    means = summary.pivot(index="min_dist", columns="n_neighbors", values=f"{metric}_mean")
    stds = summary.pivot(index="min_dist", columns="n_neighbors", values=f"{metric}_std")
    low, high = means.min().min(), means.max().max()

    figure, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(means.to_numpy(), cmap=SEQUENTIAL, aspect="auto", vmin=low, vmax=high)
    for row in range(means.shape[0]):
        for column in range(means.shape[1]):
            value = means.iat[row, column]
            # light text on the dark half of the scale so the numbers stay readable
            dark = high > low and (value - low) / (high - low) > 0.55
            axis.text(column, row, f"{value:.3f}\n± {stds.iat[row, column]:.3f}", ha="center",
                      va="center", fontsize=9, color="white" if dark else "#1a1a19")

    axis.set_xticks(range(means.shape[1]), means.columns)
    axis.set_yticks(range(means.shape[0]), [f"{value:g}" for value in means.index])
    axis.set(xlabel="n_neighbors", ylabel="min_dist")
    axis.set_title(METRIC_LABELS[metric], loc="left", fontsize=11)
    axis.tick_params(length=0)
    axis.spines[:].set_visible(False)
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_metric_panel(axis, summary, parameter, series, title):
    for (column, label), color in zip(series, SERIES_COLORS):
        axis.errorbar(summary[parameter], summary[f"{column}_mean"], yerr=summary[f"{column}_std"],
                      color=color, linewidth=2, marker="o", markersize=6, capsize=3, label=label)
    axis.set_title(title, loc="left", fontsize=11)
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=9)


def save_metric_chart(summary, parameter, label, path, log_x):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    panels = [
        ("T", "Trustworthiness T(k) ↑"),
        ("C", "Continuity C(k) ↑"),
        ("KNN", "KNN kaimynų išsaugojimas ↑"),
    ]
    for axis, (metric, title) in zip(axes.flat, panels):
        plot_metric_panel(axis, summary, parameter, [(f"{metric}_{k}", f"k = {k}") for k in METRIC_K], title)
    plot_metric_panel(axes.flat[3], summary, parameter,
                      [("spearman", "Spearman ρ ↑"), ("stress", "Normalized Stress ↓")],
                      "Globali struktūra")

    for axis in axes[1]:
        axis.set_xlabel(label)
        if log_x:
            axis.set_xscale("log")
        values = summary[parameter].tolist()
        axis.set_xticks(values, [f"{value:g}" for value in values])
        axis.minorticks_off()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-seeds", action="store_true", help="save a projection chart for every seed")
    args = parser.parse_args()

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS if args.all_seeds else [CHART_SEED]
    for neighbors, min_dist in SETTINGS:
        for seed in seeds:
            save_projection(neighbors, min_dist, seed)

    neighbor_settings = [(neighbors, DEFAULT_MIN_DIST) for neighbors in NEIGHBOR_VALUES]
    min_dist_settings = [(DEFAULT_NEIGHBORS, min_dist) for min_dist in MIN_DIST_VALUES]
    save_projection_row(neighbor_settings, [f"n_neighbors = {n}" for n in NEIGHBOR_VALUES],
                        CHART_DIR / "projections_n_neighbors.png", columns=3)
    save_projection_row(min_dist_settings, [f"min_dist = {m:g}" for m in MIN_DIST_VALUES],
                        CHART_DIR / "projections_min_dist.png", columns=2)

    save_projection_grid(CHART_DIR / "projections_grid.png")
    pair = [(5, DEFAULT_MIN_DIST), (15, DEFAULT_MIN_DIST)]
    save_panels(pair, [f"n_neighbors = {n}, min_dist = {m:g}" for n, m in pair],
                CHART_DIR / "projections_nn5_vs_nn15.png")

    summary = summarize(pd.read_csv(RUNS_FILE))
    HEATMAP_DIR.mkdir(exist_ok=True)
    for metric in METRIC_COLUMNS:
        save_heatmap(summary, metric, HEATMAP_DIR / f"{metric}.png")

    neighbor_summary = summary[summary["min_dist"].eq(DEFAULT_MIN_DIST)].sort_values("n_neighbors")
    min_dist_summary = summary[summary["n_neighbors"].eq(DEFAULT_NEIGHBORS)].sort_values("min_dist")
    save_metric_chart(neighbor_summary, "n_neighbors", "n_neighbors",
                      CHART_DIR / "metrics_vs_n_neighbors.png", log_x=True)
    save_metric_chart(min_dist_summary, "min_dist", "min_dist",
                      CHART_DIR / "metrics_vs_min_dist.png", log_x=False)

    print(f"Charts saved to: {CHART_DIR}")


if __name__ == "__main__":
    main()
