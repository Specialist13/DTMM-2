"""Profile numeric features for variance, skewness, and scale problems.

The parser accepts values with trailing text, such as ``"2.3e-05 a.u."``.
Values that do not begin with a number are treated as missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from analysis_helpers import feature_statistics


NUMBER_RE = (
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
INPUT = Path(__file__).resolve().parent.parent / "A29.csv"
OUTPUT = Path(__file__).resolve().parent.parent / "A29_feature_diagnostics.csv"
CLASS_COLUMN = "class"
LOW_VARIANCE_FRACTION = 0.01
SKEW_THRESHOLD = 2.0
SCALE_RATIO_THRESHOLD = 10.0


def parse_numeric_columns(raw_data, feature_columns):
    raw = raw_data[feature_columns].fillna("").astype(str)
    numeric = raw.apply(lambda column: pd.to_numeric(
        column.str.extract(NUMBER_RE, expand=False), errors="coerce"
    ))
    trailing = raw.apply(lambda column: column.str.replace(NUMBER_RE, "", regex=True).str.strip())
    quality = pd.DataFrame({
        "numeric_values": numeric.count(),
        "missing_or_unparseable": numeric.isna().sum(),
        "numeric_values_with_trailing_text": (numeric.notna() & trailing.ne("")).sum(),
    }).rename_axis("feature").reset_index()
    return numeric, quality

def calculate_statistics(numeric_data):
    stats = feature_statistics(numeric_data).rename(columns={
        "count": "n", "q1": "q25", "q3": "q75", "iqr_outlier_count": "iqr_outliers",
    })
    stats["n"] = stats["n"].astype(int)
    stats["unique"] = numeric_data.nunique()
    stats["variance"] = numeric_data.var()
    stats["q01"], stats["q99"] = numeric_data.quantile(0.01), numeric_data.quantile(0.99)
    stats["central_98_percent_range"] = stats["q99"] - stats["q01"]
    stats["range"] = stats["max"] - stats["min"]
    stats["trimmed_skewness"] = numeric_data.clip(stats["q01"], stats["q99"], axis=1).skew()
    stats["iqr_over_abs_median"] = stats["iqr"] / stats["median"].abs().mask(np.isclose(stats["median"], 0))
    stats = stats[[
        "n", "unique", "mean", "median", "std", "variance", "min", "q01", "q25",
        "q75", "q99", "max", "iqr", "central_98_percent_range", "range", "skewness",
        "trimmed_skewness", "iqr_over_abs_median", "iqr_outliers", "iqr_outlier_percent",
    ]]
    for scale in ("std", "iqr"):
        stats[f"{scale}_scale_ratio"] = stats[scale] / stats[scale].min()
    for scale in ("std", "iqr"):
        ratio = stats[scale] / stats.loc[stats[scale] > 0, scale].median()
        stats[f"{scale}_scale_factor"] = np.maximum(ratio, 1 / ratio)
    stats["constant_or_zero_scale"] = stats[["std", "iqr"]].eq(0).any(axis=1)
    stats["absolute_skewness"] = stats["skewness"].abs()
    return stats.reset_index()

def print_table(title, data, columns):
    print(f"\n{title}")
    if data.empty:
        print("None found.")
        return

    print(data[columns].to_string(index=False, float_format=lambda value: f"{value:.6g}"))


def main():
    csv_path = INPUT
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    raw_data = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")

    if CLASS_COLUMN in raw_data.columns:
        feature_columns = [
            column for column in raw_data.columns if column != CLASS_COLUMN
        ]
        class_message = f"class column excluded: {CLASS_COLUMN}"
    else:
        feature_columns = list(raw_data.columns)
        class_message = "class column not found; all columns treated as features"

    numeric_data, quality = parse_numeric_columns(raw_data, feature_columns)
    statistics = calculate_statistics(numeric_data)

    statistics["low_variance_flag"] = (
        statistics["iqr_over_abs_median"] < LOW_VARIANCE_FRACTION
    )
    statistics["high_skew_flag"] = (
        statistics["absolute_skewness"] > SKEW_THRESHOLD
    )
    statistics["scale_difference_factor"] = statistics[
        ["std_scale_factor", "iqr_scale_factor"]
    ].max(axis=1)
    statistics["scale_difference_flag"] = (
        statistics["constant_or_zero_scale"]
        | (
            statistics["scale_difference_factor"]
            >= SCALE_RATIO_THRESHOLD
        )
    )

    print(f"File: {csv_path}")
    print(f"Rows: {len(raw_data)}")
    print(f"Features: {len(feature_columns)} ({class_message})")

    print_table(
        f"Low-variance candidates (relative IQR < {LOW_VARIANCE_FRACTION:.1%})",
        statistics[statistics["low_variance_flag"]].sort_values(
            "iqr_over_abs_median"
        ),
        [
            "feature", "iqr_over_abs_median", "std", "iqr", "median", "unique",
        ],
    )

    print_table(
        f"Highly asymmetric candidates (|skewness| > {SKEW_THRESHOLD})",
        statistics[statistics["high_skew_flag"]].sort_values(
            "absolute_skewness", ascending=False
        ),
        [
            "feature", "skewness", "trimmed_skewness", "min", "q01", "q99", "max", "iqr_outliers",
            "iqr_outlier_percent",
        ],
    )

    print_table(
        "Features ordered by standard deviation",
        statistics.sort_values("std"),
        ["feature", "std", "std_scale_ratio", "iqr", "iqr_scale_ratio"],
    )

    print_table(
        "Largest maximum-to-minimum differences",
        statistics.sort_values("range", ascending=False),
        [
            "feature", "min", "q01", "q99", "max", "range", "central_98_percent_range",
        ],
    )

    scale_candidates = statistics[
        statistics["scale_difference_flag"]
    ].sort_values("scale_difference_factor", ascending=False)

    print_table(
        (
            "Features with different scales "
            f"(threshold: {SCALE_RATIO_THRESHOLD:g}x)"
        ),
        scale_candidates,
        [
            "feature", "std", "std_scale_factor", "iqr", "iqr_scale_factor",
            "constant_or_zero_scale",
        ],
    )

    if scale_candidates.empty:
        print(
            "\nScaling recommendation: probably not needed; "
            "feature scales are reasonably similar."
        )
    else:
        print(
            f"\nScaling recommendation: YES; {len(scale_candidates)} "
            "feature(s) have substantially different scales."
        )
        if scale_candidates["constant_or_zero_scale"].any():
            print(
                "Note: constant or zero-IQR features should usually be "
                "removed or reviewed; scaling will not add useful variation."
            )

    print_table(
        "Data-quality summary",
        quality[
            (quality["missing_or_unparseable"] > 0)
            | (quality["numeric_values_with_trailing_text"] > 0)
        ],
        [
            "feature", "numeric_values", "missing_or_unparseable",
            "numeric_values_with_trailing_text",
        ],
    )

    statistics.to_csv(OUTPUT, index=False)
    print(f"\nSaved complete statistics to: {OUTPUT}")


if __name__ == "__main__":
    main()
