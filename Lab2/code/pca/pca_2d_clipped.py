"""PCA to 2 dimensions on the robust-scaled A29 dataset with per-feature outlier clipping."""

from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from pca_2d import CLASS_COLORS, load_data, plot_components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT.parent / "A29_Fixed_median_by_class_imputed_robust.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pca_2d_clipped"
N_COMPONENTS = 2
CLIP_QUANTILES = (0.01, 0.99)


def clip_features(features, quantiles):
    bounds = features.quantile(quantiles).T
    bounds.columns = ["lower", "upper"]
    clipped = features.clip(bounds["lower"], bounds["upper"], axis="columns")
    clipped_count = features.ne(clipped).sum().rename("clipped_values")
    report = pd.concat([bounds, clipped_count], axis=1).rename_axis("feature").reset_index()
    return clipped, report


def run_experiment(input_path, output_dir, class_column):
    data, feature_columns, features = load_data(input_path, class_column)
    output_dir.mkdir(parents=True, exist_ok=True)

    features, clip_report = clip_features(features, CLIP_QUANTILES)
    clip_report.to_csv(output_dir / "clip_report.csv", index=False)

    pca = PCA(n_components=N_COMPONENTS)
    projected = pd.DataFrame(
        pca.fit_transform(features),
        columns=[f"PC{i + 1}" for i in range(N_COMPONENTS)],
        index=features.index,
    )
    result = projected.copy()
    result[class_column] = data[class_column].to_numpy()
    result.to_csv(output_dir / f"{input_path.stem}_clipped_pca{N_COMPONENTS}.csv", index=False, float_format="%.15g")

    variance = pd.DataFrame({
        "component": projected.columns,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    variance["cumulative_explained_variance_ratio"] = variance["explained_variance_ratio"].cumsum()
    variance.to_csv(output_dir / "explained_variance.csv", index=False)

    chart_path = output_dir / f"{input_path.stem}_clipped_pca{N_COMPONENTS}.png"
    plot_components(projected, data[class_column], class_column, pca.explained_variance_ratio_, chart_path)
    return data, feature_columns, clip_report, variance, chart_path


def main():
    data, feature_columns, clip_report, variance, chart_path = run_experiment(
        DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, "class"
    )

    print(f"Input: {DEFAULT_INPUT}")
    print(f"Rows: {len(data)}")
    print(f"Features reduced: {len(feature_columns)} -> {N_COMPONENTS}")
    print(f"Clip quantiles: {CLIP_QUANTILES}")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print(f"Chart: {chart_path}")
    print(f"\nClipped values total: {clip_report['clipped_values'].sum()}")
    print("\nExplained variance:")
    print(variance.to_string(index=False, float_format="{:.6g}".format))


if __name__ == "__main__":
    main()
