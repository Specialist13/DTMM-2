import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles cannot print κ otherwise

# made by parse_data.py
dataFile = Path(__file__).resolve().parent.parent / "A29_parsed.csv"

data = pd.read_csv(dataFile)

# excess kurtosis (normal distribution = 0), missing values skipped: first for all rows
# together, then for every class separately, so 48 features x (1 + 3) columns
kurtosis = data.groupby("class").kurt().T
kurtosis.insert(0, "All classes", data.drop(columns="class").kurt())
print("Kurtosis for all classes and by class")
print(kurtosis.to_string(float_format="{:.4f}".format))

for group, values in kurtosis.items():
    print(f"\n{group}")
    # κ <= 0: lighter tails than normal, above 0: heavier tails, the higher the heavier
    band = pd.cut(values, [float("-inf"), 0, 1, 3, 10, float("inf")],
                  labels=["κ <= 0", "0 < κ <= 1", "1 < κ <= 3", "3 < κ <= 10", "10 < κ"])
    for label, part in values.groupby(band, observed=False):
        print(f"  {label:<12} {len(part):>2}: {', '.join(part.index)}")
