"""Compare UMAP projections of datasets A, B and C with n_neighbors = 15, min_dist = 0.1.

  A  all rows and features (the runs of umap_study.py are reused)
  B  all rows, correlated features removed (|r| > 0.95)
  C  rows with an extreme value (outside Q1 - 3 IQR ... Q3 + 3 IQR) removed

Every dataset is run with the same 5 seeds. Per-run metrics compare each projection with
its own original data: T, C, KNN, spearman and stress as in umap_study.py, plus
  NH(k)  share of the k nearest 2D neighbours that have the same class (neighbourhood hit)

Comparisons (task slides 15 and 16):
  A vs B  same rows: Spearman between the original distances, k-neighbour overlap of the
          original spaces and of the projections made with the same seed
  A vs C  only the rows kept in C: A's metrics recalculated on those rows, Spearman between
          the 2D distances and k-neighbour overlap of the projections made with the same seed
  seed baseline  overlap between projections of A made with different seeds, i.e. how much
          changes from randomness alone

Usage: python umap_comparison.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from umap_study import (
    METRIC_K, SEEDS, OriginalSpace, calculate_metrics, embedding_path, knn_preservation, run_umap,
)
from umap_visualization import load_data, remove_correlated, remove_outliers

NEIGHBORS, MIN_DIST = 15, 0.1
OUTPUT_DIR = Path(__file__).resolve().parent / "umap_comparison"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
# neighbour lists are cut to the 50 nearest, the largest k, to keep memory low
TOP = max(METRIC_K) + 1


def nearest(distances):
    # copy, otherwise the slice keeps the whole n x n array in memory
    return np.argsort(distances, axis=1, kind="stable")[:, :TOP].copy()


def neighbourhood_hit(order, classes, k):
    return float(np.mean(classes[order[:, 1:k + 1]] == classes[:, None]))


def projection(name, features, classes, seed):
    """2D coordinates of one run, loaded if it was already made."""
    path = embedding_path(NEIGHBORS, MIN_DIST, seed) if name == "A" else EMBEDDING_DIR / f"{name}_seed{seed}.csv"
    if path.exists():
        return pd.read_csv(path)[["umap_1", "umap_2"]].to_numpy()
    embedding = run_umap(features, NEIGHBORS, MIN_DIST, seed)
    coordinates = pd.DataFrame(embedding, columns=["umap_1", "umap_2"])
    coordinates["class"] = classes
    # A's runs are stored with umap_study.py's, whose folder may not exist yet
    path.parent.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(path, index=False)
    return embedding


def run_metrics(name, features, classes, embeddings):
    """Per-run metrics, plus the neighbour lists of the original space and of every projection."""
    original = OriginalSpace(features)
    original_nh = {f"NH_original_{k}": neighbourhood_hit(original.order, classes, k) for k in METRIC_K}
    rows, orders = [], {}
    for seed, embedding in embeddings.items():
        orders[seed] = nearest(squareform(pdist(embedding)))
        row = {"dataset": name, "rows": len(features), "features": features.shape[1], "seed": seed}
        row.update(calculate_metrics(original, embedding))
        row.update({f"NH_{k}": neighbourhood_hit(orders[seed], classes, k) for k in METRIC_K})
        row.update(original_nh)
        rows.append(row)
    print(f"{name}: {len(features)} rows x {features.shape[1]} features, metrics done")
    return rows, original.order[:, :TOP].copy(), orders


def mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan}


def compare_projections(comparison, orders_a, orders_b, embeddings_a, embeddings_b):
    """Same-seed overlap and 2D distance correlation, plus the seed baseline of the first dataset."""
    rows = []
    for k in METRIC_K:
        same_seed = [knn_preservation(orders_a[s], orders_b[s], k) for s in SEEDS]
        baseline = [knn_preservation(orders_a[a], orders_a[b], k) for a, b in combinations(SEEDS, 2)]
        rows.append({"comparison": comparison, "measure": "projection_overlap", "k": k, **mean_std(same_seed)})
        rows.append({"comparison": comparison, "measure": "seed_baseline_overlap", "k": k, **mean_std(baseline)})
    distances = [spearmanr(pdist(embeddings_a[s]), pdist(embeddings_b[s])).statistic for s in SEEDS]
    rows.append({"comparison": comparison, "measure": "projection_distance_spearman", "k": np.nan,
                 **mean_std(distances)})
    baseline = [spearmanr(pdist(embeddings_a[a]), pdist(embeddings_a[b])).statistic for a, b in combinations(SEEDS, 2)]
    rows.append({"comparison": comparison, "measure": "seed_baseline_distance_spearman", "k": np.nan,
                 **mean_std(baseline)})
    return rows


def summarize(runs):
    metrics = [column for column in runs.columns if column not in ("dataset", "rows", "features", "seed")]
    grouped = runs.groupby("dataset", sort=False)
    summary = grouped[metrics].mean().add_suffix("_mean").join(grouped[metrics].std().add_suffix("_std"))
    ordered = [f"{metric}_{part}" for metric in metrics for part in ("mean", "std")]
    return grouped[["rows", "features"]].first().join(summary[ordered]).reset_index()


def main():
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    features_a, classes_a = load_data()
    print("Dataset B:")
    features_b = remove_correlated(features_a)
    print("Dataset C:")
    features_c, classes_c = remove_outliers(features_a, classes_a)
    labels_a, labels_c = classes_a.to_numpy(), classes_c.to_numpy()

    datasets = {"A": (features_a, labels_a), "B": (features_b, labels_a), "C": (features_c, labels_c)}
    embeddings = {
        name: {seed: projection(name, features, labels, seed) for seed in SEEDS}
        for name, (features, labels) in datasets.items()
    }

    runs, original_orders, projection_orders = [], {}, {}
    for name, (features, labels) in datasets.items():
        rows, original_orders[name], projection_orders[name] = run_metrics(name, features, labels, embeddings[name])
        runs += rows

    # A on the rows kept in C: same original data as C, A's coordinates of those rows
    common = features_a.index.get_indexer(features_c.index)
    embeddings_a_common = {seed: embedding[common] for seed, embedding in embeddings["A"].items()}
    rows, _, orders_a_common = run_metrics("A (C rows)", features_c, labels_c, embeddings_a_common)
    runs += rows

    comparisons = [{
        "comparison": "A vs B", "measure": "original_distance_spearman", "k": np.nan,
        **mean_std([spearmanr(pdist(features_a.to_numpy()), pdist(features_b.to_numpy())).statistic]),
    }]
    comparisons += [{
        "comparison": "A vs B", "measure": "original_overlap", "k": k,
        **mean_std([knn_preservation(original_orders["A"], original_orders["B"], k)]),
    } for k in METRIC_K]
    comparisons += compare_projections("A vs B", projection_orders["A"], projection_orders["B"],
                                       embeddings["A"], embeddings["B"])
    comparisons += compare_projections("A vs C (C rows)", orders_a_common, projection_orders["C"],
                                       embeddings_a_common, embeddings["C"])

    runs = pd.DataFrame(runs)
    runs.to_csv(OUTPUT_DIR / "runs.csv", index=False)
    summary = summarize(runs)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(OUTPUT_DIR / "comparisons.csv", index=False)

    print(f"\nSaved to: {OUTPUT_DIR}")
    shown = ["dataset", "rows", "features"] + [f"{m}_mean" for m in ("T_15", "C_15", "KNN_15", "NH_15", "spearman", "stress")]
    print(summary[shown].to_string(index=False, float_format="{:.3f}".format))
    print(comparisons.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
