import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles cannot print κ otherwise

# made by parse_data.py
dataFile = Path(__file__).resolve().parent.parent / "A29_parsed.csv"
# Tukey's "far out" fences, only clearly extreme values are thrown away
IQR_MULTIPLIER = 3

data = pd.read_csv(dataFile)


def without_outliers(frame):
    # values outside Q1 - 3 IQR and Q3 + 3 IQR become missing, so kurt() skips them
    q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
    iqr = q3 - q1
    return frame[(frame >= q1 - IQR_MULTIPLIER * iqr) & (frame <= q3 + IQR_MULTIPLIER * iqr)]


# same as kurtosis.py, but outliers are thrown away first, fences are calculated for all
# rows together and then for every class separately, so 48 features x (1 + 3) columns
kurtosis = pd.DataFrame({name: without_outliers(part.drop(columns="class")).kurt()
                         for name, part in data.groupby("class")})
kurtosis.insert(0, "All classes", without_outliers(data.drop(columns="class")).kurt())
print("Kurtosis for all classes and by class (outliers removed)")
print(kurtosis.to_string(float_format="{:.4f}".format))

for group, values in kurtosis.items():
    print(f"\n{group}")
    # κ <= 0: lighter tails than normal, above 0: heavier tails, the higher the heavier
    band = pd.cut(values, [float("-inf"), 0, 1, 3, 10, float("inf")],
                  labels=["κ <= 0", "0 < κ <= 1", "1 < κ <= 3", "3 < κ <= 10", "10 < κ"])
    for label, part in values.groupby(band, observed=False):
        print(f"  {label:<12} {len(part):>2}: {', '.join(part.index)}")
