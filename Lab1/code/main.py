from pathlib import Path

import pandas as pd

from check_values import fix_values
from duplicates import remove_duplicate_rows
from impossible_stats import blank_impossible_stat_values
from min_max_by_feature_and_class import blank_out_of_range_values

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "A29.csv"
OUTPUT_CSV = BASE_DIR / "A29_Fixed.csv"
ORIGINAL_DATASET = BASE_DIR / "Sensorless_drive_diagnosis.txt"


def load_rows(path):
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return data.columns.tolist(), data.values.tolist()


def save_rows(path, header, rows):
    pd.DataFrame(rows, columns=header).to_csv(path, index=False, lineterminator="\r\n")


def main():
    header, rows = load_rows(INPUT_CSV)
    print(f"Loaded {len(rows)} data rows from {INPUT_CSV}")

    print("\n[1/4] check_values: strip 'a.u.' suffixes, empty non-numeric strings")
    rows, value_stats = fix_values(rows)
    print(f"  a.u. values stripped:   {value_stats['au_stripped']}")
    print(f"  string values emptied:  {value_stats['strings_emptied']}")

    print("\n[2/4] duplicates: remove every non-first exact duplicate row")
    rows, removed_count = remove_duplicate_rows(rows)
    print(f"  rows removed: {removed_count} (remaining: {len(rows)})")

    print("\n[3/4] impossible_stats: empty negative std / excess kurtosis below -2")
    rows, impossible_records = blank_impossible_stat_values(rows)
    print(f"  values emptied: {len(impossible_records)}")
    records = pd.DataFrame(impossible_records, columns=["row", "column", "value", "reason"])
    records["class"] = records["row"].map(lambda row: rows[row - 1][-1])
    for reason in ("negative standard deviation", "excess kurtosis below -2"):
        matching = records[records["reason"] == reason]
        print(f"    {reason}: {len(matching)}")
        for column, group in matching.groupby("column"):
            print(f"      {header[column]}: {len(group)}")
            for label, count in group.groupby("class").size().items():
                print(f"        {label}: {count}")

    print("\n[4/4] min_max_by_feature_and_class: empty values outside original dataset ranges")
    rows, outlier_records = blank_out_of_range_values(rows, original_path=ORIGINAL_DATASET)
    print(f"  values emptied: {len(outlier_records)}")

    save_rows(OUTPUT_CSV, header, rows)
    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
