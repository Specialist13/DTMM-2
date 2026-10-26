from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# made by parse_data.py
root = Path(__file__).resolve().parent.parent
dataFile = root / "A29_parsed.csv"

data = pd.read_csv(dataFile).drop(columns="class")

# interquartile range for every feature over all rows, missing values skipped
q1, q3 = data.quantile(0.25), data.quantile(0.75)
iqr = pd.DataFrame({"Q1": q1, "Q3": q3, "IQR": q3 - q1})
print("IQR for all features")
print(iqr.to_string(float_format="{:.4e}".format))

smallest, largest = iqr["IQR"].idxmin(), iqr["IQR"].idxmax()
print(f"\nSmallest IQR: {smallest} = {iqr.loc[smallest, 'IQR']:.4e} "
      f"[{iqr.loc[smallest, 'Q1']:.4e}, {iqr.loc[smallest, 'Q3']:.4e}]")
print(f"Largest IQR:  {largest} = {iqr.loc[largest, 'IQR']:.4e} "
      f"[{iqr.loc[largest, 'Q1']:.4e}, {iqr.loc[largest, 'Q3']:.4e}]")

# how many times every IQR is larger than the smallest one, the ratios go up to ~10^6
# so a log scale is needed, otherwise all but the largest features lie flat on zero
ratio = (iqr["IQR"] / iqr.loc[smallest, "IQR"]).sort_values()
figure, axis = plt.subplots(figsize=(12, 4.5))
axis.plot(ratio.index, ratio, color="#4a78b5", linewidth=2, marker="o", markersize=5)
axis.set_yscale("log")
for feature in (smallest, largest):
    axis.annotate(f"{feature}: {ratio[feature]:,.0f}".replace(",", " "), (feature, ratio[feature]),
                  xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
axis.set(xlabel="Požymis", ylabel=f"IQR / IQR({smallest}) (log skalė)")
axis.tick_params(axis="x", labelrotation=90, labelsize=8)
axis.grid(axis="y", alpha=0.3)
axis.spines[["top", "right"]].set_visible(False)
figure.tight_layout()
(root / "charts").mkdir(exist_ok=True)
figure.savefig(root / "charts" / "A29_iqr_ratio.png", dpi=150)
print("Saved: charts/A29_iqr_ratio.png")
