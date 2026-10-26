from pathlib import Path

import pandas as pd

# made by parse_data.py
root = Path(__file__).resolve().parent.parent
dataFile = root / "A29_parsed.csv"

# features where less than 10% of the values are distinct count as small variance
SMALL_DISTINCT_PERCENT = 10

data = pd.read_csv(dataFile).drop(columns="class")

# variance, std and IQR are in the feature's own units, and dividing them by the median depends
# on where the values sit, the share of distinct values stays the same under any shift or scaling
distinct = (100 * data.nunique() / data.count()).sort_values()
print("Distinct values % for all features, smallest first")
print(distinct.to_string(float_format="{:.2f}".format))

small = distinct[distinct < SMALL_DISTINCT_PERCENT]
print(f"\nSmall variance (distinct values < {SMALL_DISTINCT_PERCENT}%): {len(small)} of {len(distinct)}")
print(", ".join(small.index))
