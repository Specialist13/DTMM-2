"""Compare three feature-scaling methods on the imputed A29 dataset."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from analysis_helpers import feature_statistics
from analysis_helpers import compare_correlations


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "A29_Fixed_median_by_class_imputed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scaling_experiment_by_class"

SCALERS = {
    "standardized": StandardScaler(),
    "min_max": MinMaxScaler(),
    "robust": RobustScaler(),
}
METHODS = ("original", *SCALERS)
CORRELATION_TYPES = ("pearson", "spearman")


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
    invalid = features.isna()

    if invalid.any().any():
        invalid_columns = list(invalid.columns[invalid.any()])
        invalid_count = int(invalid.sum().sum())
        raise ValueError(
            f"Found {invalid_count} missing or non-numeric values in "
            f"{invalid_columns}. Check the imputed input file first."
        )

    return data, feature_columns, features


def calculate_parameters(features):
    parameters = features.agg(["mean", "min", "median", "max"]).T
    parameters["population_std"] = features.std(ddof=0)

    quartiles = features.quantile([0.25, 0.75]).T
    quartiles.columns = ["q1", "q3"]
    parameters = parameters.join(quartiles)

    parameters["iqr"] = parameters["q3"] - parameters["q1"]
    parameters["value_range"] = parameters["max"] - parameters["min"]
    parameters["constant_feature"] = parameters["population_std"].eq(0)

    parameters.index.name = "feature"
    return parameters[
        [
            "mean", "population_std", "min", "q1", "median", "q3", "max", "iqr", "value_range",
            "constant_feature",
        ]
    ]


def scale_features(features):
    return {
        method: pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index,
        )
        for method, scaler in SCALERS.items()
    }


def add_class_column(features, class_values, class_column):
    result = features.copy()
    result[class_column] = class_values.to_numpy()
    return result


def calculate_feature_statistics(features, method):
    stats = feature_statistics(features).rename(columns={"std": "sample_std"})
    stats["zero_percent"] = 100 * features.eq(0).mean()
    stats["method"] = method
    return stats.reset_index()


def calculate_all_statistics(transformed_features):
    return pd.concat([
        calculate_feature_statistics(transformed_features[method], method) for method in METHODS
    ], ignore_index=True)


def calculate_method_summary(statistics, transformed_features):
    values = statistics.assign(
        absolute_mean=statistics["mean"].abs(),
        absolute_median=statistics["median"].abs(),
        absolute_skewness=statistics["skewness"].abs(),
        negative=statistics["min"].lt(0),
        above_one=statistics["max"].gt(1),
    )
    return values.groupby("method", sort=False).agg(
        global_min=("min", "min"),
        global_max=("max", "max"),
        mean_absolute_feature_mean=("absolute_mean", "mean"),
        median_absolute_feature_median=("absolute_median", "median"),
        median_feature_sample_std=("sample_std", "median"),
        median_feature_iqr=("iqr", "median"),
        mean_absolute_skewness=("absolute_skewness", "mean"),
        mean_iqr_outlier_percent=("iqr_outlier_percent", "mean"),
        features_with_negative_values=("negative", "sum"),
        features_with_values_above_one=("above_one", "sum"),
    ).reset_index()


def compare_correlation_matrices(original_matrix, scaled_matrix, method, correlation_type, feature_columns):
    return compare_correlations(
        original_matrix, scaled_matrix, method, correlation_type, "scaled_correlation", "1e-10"
    )


def calculate_correlation_results(transformed_features, feature_columns, output_dir):
    comparison_rows = []
    relationship_rows = []

    for correlation_type in CORRELATION_TYPES:
        matrices = {
            method: transformed_features[method].corr(method=correlation_type)
            for method in METHODS
        }

        for method, matrix in matrices.items():
            matrix.to_csv(
                output_dir / f"correlations_{method}_{correlation_type}.csv"
            )

        for method in SCALERS:
            pair_rows, summary = compare_correlation_matrices(
                matrices["original"], matrices[method], method, correlation_type, feature_columns,
            )
            comparison_rows.extend(pair_rows)
            relationship_rows.append(summary)

    comparisons = pd.DataFrame(comparison_rows)
    relationships = pd.DataFrame(relationship_rows)
    comparisons.to_csv(output_dir / "correlation_comparison.csv", index=False)
    relationships.to_csv(output_dir / "relationship_summary.csv", index=False)
    return relationships


def save_class_distribution(data, class_column, output_dir):
    distribution = (
        data[class_column]
        .value_counts(dropna=False)
        .rename_axis(class_column)
        .reset_index(name="count")
    )
    distribution["percent"] = 100 * distribution["count"] / len(data)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)


def run_experiment(input_path, output_dir, class_column):
    data, columns, features = load_data(input_path, class_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    transformed = {"original": features, **scale_features(features)}
    for method in SCALERS:
        add_class_column(transformed[method], data[class_column], class_column).to_csv(
            output_dir / f"{input_path.stem}_{method}.csv", index=False, float_format="%.15g",
        )
    statistics = calculate_all_statistics(transformed)
    summaries = calculate_method_summary(statistics, transformed)
    tables = {
        "scaling_parameters": calculate_parameters(features).reset_index(),
        "feature_statistics": statistics, "method_summary": summaries,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    relationships = calculate_correlation_results(transformed, columns, output_dir)
    save_class_distribution(data, class_column, output_dir)
    return data, columns, summaries, relationships


def main():
    data, feature_columns, summaries, relationships = run_experiment(
        DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, "class"
    )

    print(f"Input: {DEFAULT_INPUT}")
    print(f"Rows: {len(data)}")
    print(f"Features scaled: {len(feature_columns)}")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print("Methods: " + ", ".join(SCALERS))
    print("\nOverall statistics:")
    print(summaries.to_string(index=False, float_format="{:.6g}".format))
    print("\nRelationship changes:")
    print(relationships.to_string(index=False, float_format="{:.6g}".format))


if __name__ == "__main__":
    main()
