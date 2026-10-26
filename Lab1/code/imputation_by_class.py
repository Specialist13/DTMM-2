"""Shared pandas implementation for class-specific feature imputation."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "A29_Fixed.csv"


def load_data(input_path, class_column="class"):
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    if data.empty:
        raise ValueError(f"CSV file is empty: {input_path}")
    if class_column not in data.columns:
        raise ValueError(f"Class column '{class_column}' was not found.")

    feature_columns = [column for column in data.columns if column != class_column]
    raw_features = data[feature_columns].replace(r"^\s*$", pd.NA, regex=True)
    features = raw_features.apply(pd.to_numeric, errors="coerce")
    invalid = features.isna() & raw_features.notna()
    if invalid.any().any():
        raise ValueError(f"Non-numeric values found in feature columns: {list(invalid.columns[invalid.any()])}")

    data[feature_columns] = features
    return data, feature_columns


def impute_by_class(data, feature_columns, class_column, method):
    if method not in {"mean", "median"}:
        raise ValueError("method must be 'mean' or 'median'")

    features = data[feature_columns]
    replacements = data.groupby(class_column, dropna=False)[feature_columns].transform(method)
    imputed_features = features.fillna(replacements)

    if imputed_features.isna().any().any():
        missing_columns = list(imputed_features.columns[imputed_features.isna().any()])
        raise ValueError(f"Some class-feature groups have no value for imputation: {missing_columns}")

    result = data.copy()
    result[feature_columns] = imputed_features
    return result, replacements, features.isna()


def default_output_path(input_path, method):
    return input_path.with_name(
        f"{input_path.stem}_{method}_by_class_imputed{input_path.suffix}"
    )


def run_imputation(input_path, output_path, class_column, method):
    data, feature_columns = load_data(input_path, class_column)
    result, _, missing_mask = impute_by_class(data, feature_columns, class_column, method)
    result.to_csv(output_path, index=False, float_format="%.15g")
    return len(data), int(missing_mask.sum().sum())


def main(default_method=None):
    method = default_method or "median"
    output_path = default_output_path(DEFAULT_INPUT, method)
    rows, filled_count = run_imputation(DEFAULT_INPUT, output_path, "class", method)
    print(f"Loaded {rows} rows from {DEFAULT_INPUT}")
    print(f"Filled {filled_count} blank feature values using class-specific {method}s")
    print(f"Saved imputed data to {output_path}")


if __name__ == "__main__":
    main()
