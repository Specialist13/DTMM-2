from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# parametrai, kuriu grafikus nupiesti viename paveiksle
params = ["V02", "V36", "V45"]

root = Path(__file__).resolve().parent.parent
data = pd.read_csv(root / "A29_parsed.csv")  # made by parse_data.py

figure, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4.5))
for axis, param in zip(axes, params):
    values = data[param].dropna()
    axis.hist(values, bins=80, color="#4a78b5", edgecolor="white", linewidth=0.5)
    axis.set_yscale("log")  # the long tail holds only a few values, a log scale keeps them visible
    axis.set_ylim(top=axis.get_ylim()[1] * 8)  # room above the bars for the legend
    axis.axvline(values.mean(), color="#c0392b", linewidth=2, label=f"vidurkis = {values.mean():.4g}")
    axis.axvline(values.median(), color="#222222", linewidth=2, linestyle="--", label=f"mediana = {values.median():.4g}")
    axis.set(xlabel=param, ylabel="Dažnis (log skalė)")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))  # tiny values, keep ticks short
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc="best")
figure.tight_layout()
(root / "charts").mkdir(exist_ok=True)
output = root / "charts" / f"A29_distribution_{'_'.join(params)}.png"
figure.savefig(output, dpi=150)
print(f"Saved: charts/{output.name}")
