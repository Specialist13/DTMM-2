"""Reduce the robust-scaled A29 dataset to 2 dimensions with PCA."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT.parent / "A29_Fixed_median_by_class_imputed_robust.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pca_2d"
N_COMPONENTS = 2
CLASS_COLORS = ("#4a78b5", "#c0392b", "#2e8b57", "#8e44ad", "#d68910")


def load_data(input_path, class_column):
    data = pd.read_csv(input_path, encoding="utf-8-sig")

    if data.empty:
        raise ValueError(f"Input CSV contains no data rows: {input_path}")
    if class_column not in data.columns:
        available = ", ".join(map(str, data.columns))
        raise ValueError(
            f"Class column '{class_column}' was not found. "
            f"Available columns: {available}"
        )

    feature_columns = [column for column in data.columns if column != class_column]
    features = data[feature_columns].apply(pd.to_numeric, errors="coerce")

    if features.isna().any().any():
        invalid_columns = list(features.columns[features.isna().any()])
        raise ValueError(
            f"Found missing or non-numeric values in {invalid_columns}. "
            "Check the scaled input file first."
        )

    return data, feature_columns, features


def run_pca(features, n_components):
    pca = PCA(n_components=n_components)
    component_names = [f"PC{i + 1}" for i in range(n_components)]
    projected = pd.DataFrame(
        pca.fit_transform(features),
        columns=component_names,
        index=features.index,
    )
    return pca, projected


def plot_components(projected, class_values, class_column, explained_variance, output_path, zoom_quantile=None):
    figure, axis = plt.subplots(figsize=(8, 6))
    classes = class_values.unique()

    for class_value, color in zip(classes, CLASS_COLORS):
        mask = class_values == class_value
        axis.scatter(
            projected.loc[mask, "PC1"],
            projected.loc[mask, "PC2"],
            s=14,
            alpha=0.6,
            color=color,
            edgecolors="none",
            label=str(class_value),
        )

    title = "PCA projekcija į 2 dimensijas"
    if zoom_quantile is not None:
        # extreme outliers stretch the axes, clip to the central range so the main cloud stays readable
        for component, set_limits in (("PC1", axis.set_xlim), ("PC2", axis.set_ylim)):
            low, high = projected[component].quantile([zoom_quantile, 1 - zoom_quantile])
            margin = 0.05 * (high - low)
            set_limits(low - margin, high + margin)
        title += f" (be kraštutinių {100 * zoom_quantile:.0f}% outlierių)"

    axis.set(
        xlabel=f"PC1 ({100 * explained_variance[0]:.1f}% dispersijos)",
        ylabel=f"PC2 ({100 * explained_variance[1]:.1f}% dispersijos)",
        title=title,
    )
    axis.grid(alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(title=class_column, frameon=False, markerscale=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)


def run_experiment(input_path, output_dir, class_column):
    data, feature_columns, features = load_data(input_path, class_column)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca, projected = run_pca(features, N_COMPONENTS)
    result = projected.copy()
    result[class_column] = data[class_column].to_numpy()
    result.to_csv(output_dir / f"{input_path.stem}_pca{N_COMPONENTS}.csv", index=False, float_format="%.15g")

    variance = pd.DataFrame({
        "component": projected.columns,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    variance["cumulative_explained_variance_ratio"] = variance["explained_variance_ratio"].cumsum()
    variance.to_csv(output_dir / "explained_variance.csv", index=False)

    chart_path = output_dir / f"{input_path.stem}_pca{N_COMPONENTS}.png"
    plot_components(projected, data[class_column], class_column, pca.explained_variance_ratio_, chart_path)
    zoomed_path = output_dir / f"{input_path.stem}_pca{N_COMPONENTS}_zoomed.png"
    plot_components(
        projected, data[class_column], class_column,
        pca.explained_variance_ratio_, zoomed_path, zoom_quantile=0.01,
    )
    return data, feature_columns, variance, zoomed_path


def main():
    data, feature_columns, variance, chart_path = run_experiment(
        DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, "class"
    )

    print(f"Input: {DEFAULT_INPUT}")
    print(f"Rows: {len(data)}")
    print(f"Features reduced: {len(feature_columns)} -> {N_COMPONENTS}")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Chart: {chart_path}")
    print("\nExplained variance:")
    print(variance.to_string(index=False, float_format="{:.6g}".format))


if __name__ == "__main__":
    main()
