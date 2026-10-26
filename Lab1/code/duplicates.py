from pathlib import Path

import pandas as pd


def remove_duplicate_rows(rows):
    data = pd.DataFrame(rows, dtype="string")
    unique = data.drop_duplicates()
    return unique.values.tolist(), len(data) - len(unique)


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "A29.csv"
    data = pd.read_csv(path, dtype="string")
    duplicates = data[data.duplicated(keep="first")]
    counts = duplicates["class"].value_counts().sort_index()

    print(f"Duplicate rows found, excluding first occurrences: {len(duplicates)}")
    for class_name, count in counts.items():
        total = (data["class"] == class_name).sum()
        print(f"{class_name}: {count} ({100 * count / total:.1f}% of class rows)")
