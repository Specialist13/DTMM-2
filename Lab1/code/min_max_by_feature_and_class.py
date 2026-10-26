from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "Sensorless_drive_diagnosis.txt"


def convert_label(label):
    label = str(label).strip().lower()
    if label.startswith("class_"):
        try:
            return int(label[6:])
        except ValueError:
            pass
    raise ValueError(f"Unknown class label: {label}")


def compute_class_ranges(original):
    data = pd.DataFrame(original)
    return {
        int(label): {"min": group.iloc[:, :-1].min().to_numpy(),
                     "max": group.iloc[:, :-1].max().to_numpy()}
        for label, group in data.groupby(data.columns[-1])
    }


def find_out_of_range(rows, class_ranges):
    if not rows:
        return []
    data = pd.DataFrame(rows)
    labels = data.iloc[:, -1].map(convert_label)
    values = data.iloc[:, :-1].replace(r"(?i)\s*a\.\s*u\.?\s*$", "", regex=True)
    values = values.apply(pd.to_numeric, errors="coerce")
    lower = pd.DataFrame({label: limits["min"] for label, limits in class_ranges.items()}).T
    upper = pd.DataFrame({label: limits["max"] for label, limits in class_ranges.items()}).T
    lower = lower.reindex(labels).set_axis(data.index)
    upper = upper.reindex(labels).set_axis(data.index)
    mask = ((values < lower) | (values > upper)) & np.isfinite(values)
    return [
        {"row_index": row, "column_index": column, "class": f"class_{labels.iat[row]}",
         "value": values.iat[row, column], "min": lower.iat[row, column], "max": upper.iat[row, column]}
        for row, column in zip(*np.nonzero(mask.to_numpy()))
    ]


def blank_out_of_range_values(rows, original_path=ORIGINAL):
    records = find_out_of_range(rows, compute_class_ranges(np.loadtxt(original_path)))
    fixed = [list(row) for row in rows]
    for record in records:
        fixed[record["row_index"]][record["column_index"]] = ""
    return fixed, records


def main():
    data = pd.read_csv(ROOT / "A29.csv", dtype=str, keep_default_na=False)
    _, records = blank_out_of_range_values(data.values.tolist())
    details = pd.DataFrame(records, columns=["row_index", "column_index", "class", "value", "min", "max"])
    details = details.rename(columns={"row_index": "line", "column_index": "column"})
    details["line"] += 2
    details["column"] += 1
    for line, column, label, value, lower, upper in details.itertuples(index=False, name=None):
        print(f"Line {line}, column {column}: value {value} is outside "
              f"the range [{lower}, {upper}] for {label}")
    details.to_csv(ROOT / "outliers.csv", index=False)
    print(f"Total outlying values: {len(details)}")
    print("Details written to: outliers.csv")
    if details.empty:
        print("All values fit within their class-specific ranges.")


if __name__ == "__main__":
    main()
