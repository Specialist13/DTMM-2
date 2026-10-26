"""UMAP of the robust-scaled A29 dataset, coloured by class.

Modes:
  1 - all 48 features
  2 - correlated features removed (|Pearson r| > 0.95)
  3 - rows with an extreme value (outside [Q1 - 3 IQR, Q3 + 3 IQR]) removed

Usage: python umap_visualization.py --mode 1   (or 2, 3, all)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import umap

# the data file is in the repository root, two levels above Lab2/code
ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "A29_Fixed_median_by_class_imputed_robust.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "umap_output"
CLASS_COLUMN = "class"
CORRELATION_THRESHOLD = 0.95
# 3 IQR marks extreme outliers; with 1.5 IQR almost 70% of the rows would be removed
IQR_FACTOR = 3.0
RANDOM_STATE = 42

# colour is fixed per class, so the same class has the same colour in every mode
CLASS_COLORS = {"Class_4": "#2a78d6", "Class_5": "#eb6834", "Class_6": "#1baf7a"}

def load_data():
    data = pd.read_csv(DATA_FILE)
    return data.drop(columns=CLASS_COLUMN), data[CLASS_COLUMN]


def remove_correlated(features):
    correlation = features.corr(method="pearson").abs()
    columns = list(features.columns)

    involved = {
        feature
        for i, first in enumerate(columns)
        for second in columns[i + 1:]
        if correlation.loc[first, second] > CORRELATION_THRESHOLD
        for feature in (first, second)
    }

    # a feature is kept unless it is correlated with a feature that is already kept
    keep, omit = [], []
    for feature in columns:
        if any(correlation.loc[feature, kept] > CORRELATION_THRESHOLD for kept in keep):
            omit.append(feature)
        else:
            keep.append(feature)

    print(f"Features with |r| > {CORRELATION_THRESHOLD}: {len(involved)}")
    print(f"Removed ({len(omit)}): {', '.join(omit)}")
    print(f"Kept ({len(keep)}): {', '.join(keep)}")
    return features[keep]


def remove_outliers(features, classes):
    q1, q3 = features.quantile(0.25), features.quantile(0.75)
    iqr = q3 - q1
    outside = features.lt(q1 - IQR_FACTOR * iqr) | features.gt(q3 + IQR_FACTOR * iqr)
    inside = ~outside.any(axis=1)

    print(f"Rows with at least one outlier: {int((~inside).sum())} of {len(features)}")
    print(f"Rows kept: {int(inside.sum())}")
    print("Rows kept per class: " + ", ".join(
        f"{name} = {count}" for name, count in classes[inside].value_counts().sort_index().items()
    ))
    return features[inside], classes[inside]


def prepare(mode, features, classes):
    if mode == 2:
        return remove_correlated(features), classes
    if mode == 3:
        return remove_outliers(features, classes)
    return features, classes


def run_umap(features):
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=RANDOM_STATE)
    return reducer.fit_transform(features.to_numpy())


def plot(embedding, classes, path):
    figure, axis = plt.subplots(figsize=(9, 7.5))
    for name, color in CLASS_COLORS.items():
        mask = (classes == name).to_numpy()
        axis.scatter(embedding[mask, 0], embedding[mask, 1], s=5, alpha=0.6, color=color,
                     linewidths=0, label=name)

    axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    axis.legend(title="Klasė", markerscale=3, frameon=False, loc="best")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_mode(mode, features, classes):
    print(f"\n=== Mode {mode} ===")
    mode_features, mode_classes = prepare(mode, features, classes)
    embedding = run_umap(mode_features)

    OUTPUT_DIR.mkdir(exist_ok=True)
    chart = OUTPUT_DIR / f"A29_umap_mode{mode}.png"
    plot(embedding, mode_classes, chart)

    # coordinates are saved too, so the chart can be redrawn without rerunning UMAP
    coordinates = pd.DataFrame(embedding, columns=["umap_1", "umap_2"], index=mode_features.index)
    coordinates[CLASS_COLUMN] = mode_classes
    coordinates.rename_axis("row").to_csv(OUTPUT_DIR / f"A29_umap_mode{mode}.csv")

    print(f"Saved: {chart}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    features, classes = load_data()
    modes = [1, 2, 3] if args.mode == "all" else [int(args.mode)]
    for mode in modes:
        run_mode(mode, features, classes)


if __name__ == "__main__":
    main()
