from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# parametras, kurio grafika nupiesti
param = "V36"

root = Path(__file__).resolve().parent.parent
values = pd.read_csv(root / "A29_parsed.csv")[param].dropna()  # made by parse_data.py

figure, axis = plt.subplots(figsize=(8, 4.5))
axis.hist(values, bins=80, color="#4a78b5", edgecolor="white", linewidth=0.5)
axis.set_yscale("log")  # the long tail holds only a few values, a log scale keeps them visible
axis.axvline(values.mean(), color="#c0392b", linewidth=2, label=f"vidurkis = {values.mean():.4g}")
axis.axvline(values.median(), color="#222222", linewidth=2, linestyle="--", label=f"mediana = {values.median():.4g}")
axis.set(xlabel=param, ylabel="Dažnis (log skalė)")
axis.grid(axis="y", alpha=0.3)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(frameon=False)
figure.tight_layout()
(root / "charts").mkdir(exist_ok=True)
figure.savefig(root / "charts" / f"A29_distribution_{param}.png", dpi=150)
print(f"Saved: charts/A29_distribution_{param}.png")
