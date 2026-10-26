from pathlib import Path

import pandas as pd

# made by parse_data.py
dataFile = Path(__file__).resolve().parent.parent / "A29_parsed.csv"
# Tukey's "far out" fences, only clearly extreme values are thrown away
IQR_MULTIPLIER = 3

data = pd.read_csv(dataFile)


def without_outliers(frame):
    # values outside Q1 - 3 IQR and Q3 + 3 IQR become missing, so skew() skips them
    q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
    iqr = q3 - q1
    return frame[(frame >= q1 - IQR_MULTIPLIER * iqr) & (frame <= q3 + IQR_MULTIPLIER * iqr)]


# same as skewness.py, but outliers are thrown away first, fences are calculated for all
# rows together and then for every class separately, so 48 features x (1 + 3) columns
skewness = pd.DataFrame({name: without_outliers(part.drop(columns="class")).skew()
                         for name, part in data.groupby("class")})
skewness.insert(0, "All classes", without_outliers(data.drop(columns="class")).skew())
print("Skewness for all classes and by class (outliers removed)")
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
