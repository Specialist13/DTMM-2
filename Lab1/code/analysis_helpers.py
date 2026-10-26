import numpy as np
import pandas as pd


def feature_statistics(features):
    stats = features.describe().T.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats["skewness"] = features.skew()
    outliers = features.lt(stats["q1"] - 1.5 * stats["iqr"]) | features.gt(stats["q3"] + 1.5 * stats["iqr"])
    stats["iqr_outlier_count"] = outliers.sum()
    stats["iqr_outlier_percent"] = 100 * outliers.sum() / features.count()
    return stats.rename_axis("feature")


def compare_correlations(original, changed, method, correlation_type, value_name, threshold):
    first, second = np.triu_indices(len(original), k=1)
    before = original.to_numpy()[first, second]
    after = changed.to_numpy()[first, second]
    valid = np.isfinite(before) & np.isfinite(after)
    differences = np.where(valid, np.abs(after - before), np.nan)
    pairs = pd.DataFrame({
        "method": method,
        "correlation_type": correlation_type,
        "feature_1": original.columns.to_numpy()[first],
        "feature_2": original.columns.to_numpy()[second],
        "original_correlation": before,
        value_name: after,
        "absolute_difference": differences,
    })
    summary = {
        "method": method,
        "correlation_type": correlation_type,
        "pair_count": int(valid.sum()),
        "max_absolute_difference": differences[valid].max() if valid.any() else np.nan,
        "mean_absolute_difference": differences[valid].mean() if valid.any() else np.nan,
        f"pairs_changed_above_{threshold}": int((differences > float(threshold)).sum()),
        "correlation_sign_changes": int((np.signbit(before[valid]) != np.signbit(after[valid])).sum()),
    }
    return pairs.to_dict("records"), summary
