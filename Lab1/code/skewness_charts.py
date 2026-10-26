from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parent.parent
chartsDirectory = root / "charts"
chartsDirectory.mkdir(exist_ok=True)
# made by parse_data.py
data = pd.read_csv(root / "A29_parsed.csv")

for column in ["V37", "V15"]:
    values = data[column].dropna()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(values, bins=80, color="#4a78b5", edgecolor="white", linewidth=0.5)
    # the long tail holds only a few values, a log scale keeps them visible
    axis.set_yscale("log")
    # with a skewed distribution the mean is pulled towards the long tail
    axis.axvline(values.mean(), color="#c0392b", linewidth=2, label=f"vidurkis = {values.mean():.4g}")
    axis.axvline(values.median(), color="#222222", linewidth=2, linestyle="--",
                 label=f"mediana = {values.median():.4g}")
    axis.set(xlabel=column, ylabel="Dažnis (log skalė)")
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(chartsDirectory / f"A29_skewness_{column}.png", dpi=150)
    plt.close(figure)
    print(f"Saved: charts/A29_skewness_{column}.png")
