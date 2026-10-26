from pathlib import Path

import pandas as pd


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "A29.csv"
OUTPUT = ROOT / "A29_potential_outliers.csv"


def find_outliers(data):
    features = data.drop(columns="class").astype("string").apply(
        lambda column: pd.to_numeric(column.str.extract(f"({NUMBER})", expand=False), errors="coerce")
    ).astype(float)
    q1, q3 = features.quantile(0.25), features.quantile(0.75)
    iqr = q3 - q1
    mask = (features.lt(q1 - 1.5 * iqr) | features.gt(q3 + 1.5 * iqr)) & iqr.ne(0)
    mask = mask.loc[mask.any(axis=1)]
    return pd.DataFrame({
        "row_number": mask.index + 2,
        "class": data.loc[mask.index, "class"],
        "outlier_count": mask.sum(axis=1),
        "outlier_columns": [";".join(row.index[row]) for _, row in mask.iterrows()],
    })


def main():
    outliers = find_outliers(pd.read_csv(INPUT, dtype="string", encoding="utf-8-sig"))
    print(f"Rows with at least one potential outlier value: {len(outliers)}")
    print("\nTop 20 row(s):")
    preview = outliers.sort_values(["outlier_count", "row_number"], ascending=[False, True]).head(20)
    for row in preview.itertuples(index=False, name=None):
        print(*row[:3], row[3].replace(";", ", "))
    outliers.to_csv(OUTPUT, index=False)
    print(f"\nSaved potential outliers to: {OUTPUT.name}")


if __name__ == "__main__":
    main()
