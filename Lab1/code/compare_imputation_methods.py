"""Compare class-specific mean and median imputation on A29_Fixed.csv."""

from pathlib import Path

import numpy as np
import pandas as pd

from imputation_by_class import load_data as load_class_data
from analysis_helpers import feature_statistics
from analysis_helpers import compare_correlations as compare_matrices


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "A29_Fixed.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "imputation_comparison_by_class"
IMPUTATION_METHODS = ("median", "mean")
CORRELATION_TYPES = ("pearson", "spearman")


def load_data(input_path, class_column):
    data, feature_columns = load_class_data(input_path, class_column)
    features = data[feature_columns]
    return data, feature_columns, features, features.isna()


def create_imputed_features(features, class_values):
    grouped = features.groupby(class_values, dropna=False)
    replacement_values = {
        method: grouped.transform(method) for method in IMPUTATION_METHODS
    }
    imputed_features = {
        method: features.fillna(values)
        for method, values in replacement_values.items()
    }
    return replacement_values, imputed_features


def add_class_column(features, class_values, class_column):
    result = features.copy()
    result[class_column] = class_values.to_numpy()
    return result


def save_imputed_datasets(
    input_path, output_dir, imputed_features, class_values, class_column
):
    for method, features in imputed_features.items():
        output_path = output_dir / f"{input_path.stem}_{method}_by_class_imputed.csv"
        add_class_column(features, class_values, class_column).to_csv(
            output_path, index=False, float_format="%.15g"
        )


def calculate_feature_statistics(features, method, missing_counts=None):
    stats = feature_statistics(features).reset_index()
    stats["method"] = method
    stats["count"] = stats["count"].astype(int)
    stats["missing_count"] = stats["feature"].map(missing_counts) if missing_counts is not None else 0
    return stats[[
        "feature", "method", "count", "missing_count", "mean", "median", "std",
        "min", "q1", "q3", "max", "iqr", "skewness", "iqr_outlier_percent",
    ]]


def calculate_statistics(features, imputed_features, missing_counts):
    tables = [calculate_feature_statistics(features, "observed", missing_counts)]
    tables += [calculate_feature_statistics(imputed_features[method], method) for method in IMPUTATION_METHODS]
    return pd.concat(tables, ignore_index=True)


def calculate_imputation_values(features, class_values):
    tables = []
    for label, group in features.groupby(class_values, dropna=False):
        counts = group.count()
        if counts.eq(0).any():
            feature = counts.index[counts.eq(0)][0]
            raise ValueError(f"Class {label!r} has no values for feature {feature}.")
        table = pd.DataFrame({
            "missing_count": group.isna().sum(),
            "missing_percent": 100 * group.isna().mean(),
            "mean": group.mean(),
            "median": group.median(),
        })
        table["mean_minus_median"] = table["mean"] - table["median"]
        table["observed_iqr"] = group.quantile(0.75) - group.quantile(0.25)
        table["absolute_gap_over_iqr"] = table["mean_minus_median"].abs() / table["observed_iqr"].replace(0, np.nan)
        table = table.rename_axis("feature").reset_index()
        table.insert(0, "class", label)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def calculate_method_summary(statistics):
    return statistics.assign(absolute_skewness=statistics["skewness"].abs()).groupby(
        "method", sort=False
    ).agg(
        median_feature_std=("std", "median"),
        median_feature_iqr=("iqr", "median"),
        mean_absolute_skewness=("absolute_skewness", "mean"),
        mean_iqr_outlier_percent=("iqr_outlier_percent", "mean"),
    ).reset_index()


def compare_correlations(original, imputed, method, correlation_type, features):
    return compare_matrices(
        original.corr(method=correlation_type), imputed.corr(method=correlation_type),
        method, correlation_type, "imputed_correlation", "1e-4",
    )


def calculate_relationships(features, imputed_features, feature_columns):
    comparison_rows = []
    summary_rows = []
    for correlation_type in CORRELATION_TYPES:
        for method in IMPUTATION_METHODS:
            pair_rows, summary = compare_correlations(
                features, imputed_features[method], method, correlation_type, feature_columns,
            )
            comparison_rows.extend(pair_rows)
            summary_rows.append(summary)
    return pd.DataFrame(comparison_rows), pd.DataFrame(summary_rows)


def run_experiment(input_path, output_dir, class_column):
    data, columns, features, blanks = load_data(input_path, class_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, imputed = create_imputed_features(features, data[class_column])
    save_imputed_datasets(input_path, output_dir, imputed, data[class_column], class_column)
    values = calculate_imputation_values(features, data[class_column])
    statistics = calculate_statistics(features, imputed, blanks.sum())
    summaries = calculate_method_summary(statistics)
    comparisons, relationships = calculate_relationships(features, imputed, columns)
    tables = {
        "imputation_values": values, "feature_statistics": statistics, "method_summary": summaries,
        "correlation_comparison": comparisons, "relationship_summary": relationships,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return data, columns, summaries, relationships


def main():
    data, features, summaries, relationships = run_experiment(
        DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, "class"
    )
    print(f"Input: {DEFAULT_INPUT}")
    print(f"Rows: {len(data)}")
    print(f"Features: {len(features)}")
    print(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    print("\nOverall statistics:")
    print(summaries.to_string(index=False, float_format="{:.6g}".format))
    print("\nRelationship changes:")
    print(relationships.to_string(index=False, float_format="{:.6g}".format))


if __name__ == "__main__":
    main()
