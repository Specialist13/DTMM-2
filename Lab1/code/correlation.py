from pathlib import Path

import pandas as pd

# made by parse_data.py
dataFile = Path(__file__).resolve().parent.parent / "A29_parsed.csv"
THRESHOLD = 0.95

data = pd.read_csv(dataFile).drop(columns="class")
correlation = data.corr(method="pearson").abs()
features = list(data.columns)

print(f"Correlated pairs (|r| > {THRESHOLD})")
for i, first in enumerate(features):
    for second in features[i + 1:]:
        if correlation.loc[first, second] > THRESHOLD:
            print(f"  {first} - {second}: {correlation.loc[first, second]:.4f}")

# a feature is kept unless it is correlated with a feature that is already kept
keep, omit = [], []
for feature in features:
    if any(correlation.loc[feature, kept] > THRESHOLD for kept in keep):
        omit.append(feature)
    else:
        keep.append(feature)

print(f"\nKeep ({len(keep)}): {', '.join(keep)}")
print(f"Omit ({len(omit)}): {', '.join(omit)}")
