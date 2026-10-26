"""UMAP parameter study on the robust-scaled A29 dataset (dataset A, all rows and features).

Every n_neighbors value (5, 15, 30, 50, 100) is combined with every min_dist value
(0.0, 0.1, 0.5, 0.99), and every setting is run with 5 seeds. For every run the 2D coordinates are saved and
these projection quality metrics are calculated against the original 48D space:
  T(k), C(k)  trustworthiness and continuity, k = 5, 15, 50
  KNN(k)      share of the k original nearest neighbours that stay among the k nearest in 2D
  spearman    Spearman correlation between original and 2D pairwise distances
  stress      normalized stress (Stress-1) after optimally scaling the 2D distances

Finished runs are skipped, so the script can be stopped and started again.

Usage: python umap_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from umap_visualization import load_data

OUTPUT_DIR = Path(__file__).resolve().parent / "umap_study"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
RUNS_FILE = OUTPUT_DIR / "runs.csv"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

SEEDS = [0, 1, 2, 3, 4]
METRIC_K = [5, 15, 50]
DEFAULT_NEIGHBORS, DEFAULT_MIN_DIST = 15, 0.1

# every n_neighbors value is combined with every min_dist value (4 x 5 grid)
NEIGHBOR_VALUES = [5, 15, 30, 50, 100]
MIN_DIST_VALUES = [0.0, 0.1, 0.5, 0.99]
SETTINGS = [(neighbors, min_dist) for min_dist in MIN_DIST_VALUES for neighbors in NEIGHBOR_VALUES]
TABLE_DIR = OUTPUT_DIR / "tables"

METRIC_COLUMNS = (
    [f"{name}_{k}" for name in ("T", "C", "KNN") for k in METRIC_K] + ["spearman", "stress"]
)


def embedding_path(neighbors, min_dist, seed):
    return EMBEDDING_DIR / f"nn{neighbors}_md{min_dist}_seed{seed}.csv"


def neighbor_ranks(distances):
    """rank[i, j] = position of j in the list of i's neighbours sorted by distance (i itself is 0)."""
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(len(order))[:, None]
    ranks[rows, order] = np.arange(order.shape[1])
    return order, ranks


def rank_error(order_from, ranks_to, k):
    """Trustworthiness when `from` is the 2D space and `to` the original one; continuity when swapped."""
    n = len(order_from)
    neighbors = order_from[:, 1:k + 1]
    rows = np.arange(n)[:, None]
    # neighbours that are not among the k nearest in the other space are penalised by their rank there
    penalty = np.clip(ranks_to[rows, neighbors] - k, 0, None).sum()
    return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * penalty


def knn_preservation(order_original, order_embedded, k):
    original = order_original[:, 1:k + 1]
    embedded = order_embedded[:, 1:k + 1]
    shared = [np.intersect1d(a, b, assume_unique=True).size for a, b in zip(original, embedded)]
    return np.mean(shared) / k


def normalized_stress(original, embedded):
    # UMAP coordinates have no fixed scale, so the 2D distances are first scaled to fit best
    scale = np.dot(original, embedded) / np.dot(embedded, embedded)
    return np.sqrt(np.sum((original - scale * embedded) ** 2) / np.sum(original ** 2))


class OriginalSpace:
    """Everything about the 48D data that the metrics need, calculated once."""

    def __init__(self, features):
        self.pairs = pdist(features.to_numpy())
        self.order, self.ranks = neighbor_ranks(squareform(self.pairs))


def calculate_metrics(original, embedding):
    pairs = pdist(embedding)
    order, ranks = neighbor_ranks(squareform(pairs))
    metrics = {}
    for k in METRIC_K:
        metrics[f"T_{k}"] = rank_error(order, original.ranks, k)
        metrics[f"C_{k}"] = rank_error(original.order, ranks, k)
        metrics[f"KNN_{k}"] = knn_preservation(original.order, order, k)
    metrics["spearman"] = spearmanr(original.pairs, pairs).statistic
    metrics["stress"] = normalized_stress(original.pairs, pairs)
    return metrics


def run_umap(features, neighbors, min_dist, seed):
    reducer = umap.UMAP(n_components=2, n_neighbors=neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(features.to_numpy())


def load_runs():
    return pd.read_csv(RUNS_FILE) if RUNS_FILE.exists() else pd.DataFrame()


def is_done(runs, neighbors, min_dist, seed):
    if runs.empty:
        return False
    match = runs["n_neighbors"].eq(neighbors) & runs["min_dist"].eq(min_dist) & runs["seed"].eq(seed)
    return bool(match.any()) and embedding_path(neighbors, min_dist, seed).exists()


def summarize(runs):
    grouped = runs.groupby(["n_neighbors", "min_dist"])[METRIC_COLUMNS]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    summary.insert(0, "runs", grouped.size())
    # mean and std next to each other for every metric
    ordered = ["runs"] + [f"{metric}_{part}" for metric in METRIC_COLUMNS for part in ("mean", "std")]
    return summary[ordered].reset_index()


def save_grid_tables(summary):
    """One 4 x 5 table per metric: rows min_dist, columns n_neighbors, cells mean ± std."""
    TABLE_DIR.mkdir(exist_ok=True)
    for metric in METRIC_COLUMNS:
        cells = summary.assign(value=summary[f"{metric}_mean"].map("{:.3f}".format)
                               + " ± " + summary[f"{metric}_std"].map("{:.3f}".format))
        table = cells.pivot(index="min_dist", columns="n_neighbors", values="value")
        table.columns = [f"n_neighbors={neighbors}" for neighbors in table.columns]
        # utf-8-sig so Excel shows the ± sign correctly
        table.to_csv(TABLE_DIR / f"{metric}.csv", encoding="utf-8-sig")


def save_averages_table(summary):
    """All 20 settings in one table, one column per metric, mean over the seeds."""
    averages = summary[["min_dist", "n_neighbors"] + [f"{metric}_mean" for metric in METRIC_COLUMNS]]
    averages.columns = ["min_dist", "n_neighbors"] + METRIC_COLUMNS
    averages = averages.sort_values(["min_dist", "n_neighbors"]).round(3)
    averages.to_csv(TABLE_DIR / "averages.csv", index=False)
    return averages


def main():
    features, classes = load_data()
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rows: {len(features)}, features: {features.shape[1]}")
    print("Calculating original distances and neighbour ranks...")
    original = OriginalSpace(features)

    runs = load_runs()
    total = len(SETTINGS) * len(SEEDS)
    for number, (neighbors, min_dist, seed) in enumerate(
        ((n, m, s) for n, m in SETTINGS for s in SEEDS), start=1
    ):
        label = f"[{number}/{total}] n_neighbors={neighbors}, min_dist={min_dist}, seed={seed}"
        if is_done(runs, neighbors, min_dist, seed):
            print(f"{label}: already done")
            continue

        embedding = run_umap(features, neighbors, min_dist, seed)
        coordinates = pd.DataFrame(embedding, columns=["umap_1", "umap_2"])
        coordinates["class"] = classes.to_numpy()
        coordinates.to_csv(embedding_path(neighbors, min_dist, seed), index=False)

        row = {"n_neighbors": neighbors, "min_dist": min_dist, "seed": seed}
        row.update(calculate_metrics(original, embedding))
        runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
        # saved after every run, so nothing is lost if the script is stopped
        runs.to_csv(RUNS_FILE, index=False)
        print(f"{label}: T_15={row['T_15']:.4f}, C_15={row['C_15']:.4f}, "
              f"KNN_15={row['KNN_15']:.4f}, spearman={row['spearman']:.4f}, stress={row['stress']:.4f}")

    summary = summarize(runs)
    summary.to_csv(SUMMARY_FILE, index=False)
    save_grid_tables(summary)
    save_averages_table(summary)
    print(f"\nSaved: {RUNS_FILE}\nSaved: {SUMMARY_FILE}\nSaved 4 x 5 tables and averages.csv to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
