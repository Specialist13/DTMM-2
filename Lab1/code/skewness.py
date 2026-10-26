from pathlib import Path

import pandas as pd

# made by parse_data.py
dataFile = Path(__file__).resolve().parent.parent / "A29_parsed.csv"

data = pd.read_csv(dataFile)

# adjusted Fisher-Pearson skewness, missing values skipped: first for all rows together,
# then for every class separately, so 48 features x (1 + 3) columns
skewness = data.groupby("class").skew().T
skewness.insert(0, "All classes", data.drop(columns="class").skew())
print("Skewness for all classes and by class")
print(skewness.to_string(float_format="{:.4f}".format))

for group, values in skewness.items():
    print(f"\n{group}")
    # the same bands as in the report table
    band = pd.cut(values.abs(), [0, 1, 10, float("inf")],
                  labels=["|s| <= 1", "1 < |s| <= 10", "|s| > 10"], include_lowest=True)
    for label, part in values.groupby(band, observed=False):
        positive, negative = part[part > 0].index, part[part < 0].index
        print(f"  {label:<14} positive {len(positive):>2}: {', '.join(positive)}")
        print(f"  {'':<14} negative {len(negative):>2}: {', '.join(negative)}")
