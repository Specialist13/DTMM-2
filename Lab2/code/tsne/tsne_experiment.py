from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "tsne"
KS = (5, 15, 50)
PERPLEXITIES = (5, 15, 30, 50, 100)
GRID_PERPLEXITIES = (5, 15, 30, 100)
BEST_PERPLEXITY = 30
CORRELATION_THRESHOLD = 0.95


def correlated_features(df):
    corr = df.corr().abs()
    keep, omit = [], []
    for feature in df.columns:
        (omit if (corr.loc[feature, keep] > CORRELATION_THRESHOLD).any() else keep).append(feature)
    return omit


def neighbours(X, k):
    return NearestNeighbors().fit(X).kneighbors(n_neighbors=k, return_distance=False)


def rho(X, Y):
    return spearmanr(pdist(X), pdist(Y)).statistic


def metrics(X, Y, labels):
    row = {}
    for k in KS:
        nx, ny = neighbours(X, k), neighbours(Y, k)
        row[f"trustworthiness_k{k}"] = trustworthiness(X, Y, n_neighbors=k)
        row[f"continuity_k{k}"] = trustworthiness(Y, X, n_neighbors=k)
        row[f"knn_k{k}"] = (nx[:, :, None] == ny[:, None, :]).any(axis=2).mean()
        row[f"nh_k{k}"] = (labels[ny] == labels[:, None]).mean()
    d, e = pdist(X), pdist(Y)
    scale = d @ e / (e @ e)
    row["normalized_stress"] = np.sqrt(((d - scale * e) ** 2).sum() / (d @ d))
    row["spearman"] = spearmanr(d, e).statistic
    return row


def scatter_grid(panels, shape, path):
    fig, axes = plt.subplots(*shape, figsize=(6 * shape[1], 5.5 * shape[0]), layout="constrained")
    classes = np.unique(panels[0][2])
    for ax, (title, emb, y) in zip(axes.flat, panels):
        for i, label in enumerate(classes):
            ax.scatter(*emb[y == label].T, s=3, alpha=0.6, linewidths=0, color=f"C{i}", label=label)
        ax.set(title=title, xticks=[], yticks=[])
    fig.legend(*axes.flat[0].get_legend_handles_labels(), title="Klasė", loc="outside lower center", ncol=len(classes), markerscale=4)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_csv(ROOT / "A29_Fixed_median_by_class_imputed_robust.csv")
    labels = df.pop("class").to_numpy()
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    iqr = q3 - q1
    keep = ~((df < q1 - 3 * iqr) | (df > q3 + 3 * iqr)).any(axis=1).to_numpy()
    correlated = correlated_features(df)
    print(f"Correlated features ({len(correlated)}): {', '.join(correlated)}")

    data = {
        "A": (df.to_numpy(), labels),
        "B": (df.drop(columns=correlated).to_numpy(), labels),
        "C": (df[keep].to_numpy(), labels[keep]),
    }
    data["A_common"] = (data["C"][0], labels[keep])
    rho_input = {"A_vs_B": rho(data["A"][0], data["B"][0]), "A_vs_C_common": 1.0}

    OUT.mkdir(parents=True, exist_ok=True)
    tables, geometries, embeddings = [], [], {}
    for perplexity in PERPLEXITIES:
        emb = {
            key: TSNE(perplexity=perplexity, init="pca", random_state=42).fit_transform(data[key][0])
            for key in "ABC"
        }
        emb["A_common"] = emb["A"][keep]
        embeddings[perplexity] = emb
        tables.append(pd.DataFrame(
            {"dataset": key, "perplexity": perplexity, "n": len(X), "features": X.shape[1], **metrics(X, emb[key], y)}
            for key, (X, y) in data.items()
        ))
        geometries.append(pd.DataFrame({
            "comparison": list(rho_input),
            "perplexity": perplexity,
            "rho_input": list(rho_input.values()),
            "rho_tsne": [rho(emb["A"], emb["B"]), rho(emb["A_common"], emb["C"])],
        }))
        for key in "ABC":
            y = data[key][1]
            fig, ax = plt.subplots(figsize=(8, 7))
            for label in np.unique(y):
                ax.scatter(*emb[key][y == label].T, s=4, alpha=0.6, linewidths=0, label=label)
            ax.set(xticks=[], yticks=[])
            ax.legend(title="Klasė", markerscale=3)
            fig.savefig(OUT / f"projection_{key}_p{perplexity}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        print(f"perplexity={perplexity} done")

    scatter_grid(
        [(f"Perplexity = {p}", embeddings[p]["A"], labels) for p in GRID_PERPLEXITIES],
        (2, 2), OUT / "projection_A_perplexity_grid.png",
    )
    scatter_grid(
        [(key, embeddings[BEST_PERPLEXITY][key], data[key][1]) for key in "ABC"],
        (1, 3), OUT / f"projection_ABC_p{BEST_PERPLEXITY}.png",
    )

    table = pd.concat(tables, ignore_index=True)
    geometry = pd.concat(geometries, ignore_index=True)
    table.to_csv(OUT / "metrics.csv", index=False)
    geometry.to_csv(OUT / "geometry.csv", index=False)

    for metric in table.columns[4:]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, group in table.groupby("dataset"):
            ax.plot(group["perplexity"], group[metric], marker="o", label=key)
        ax.set(xlabel="Perplexity", ylabel=metric, xticks=PERPLEXITIES)
        ax.legend()
        fig.savefig(OUT / f"perplexity_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, group in geometry.groupby("comparison"):
        ax.plot(group["perplexity"], group["rho_tsne"], marker="o", label=key)
    ax.set(xlabel="Perplexity", ylabel="rho_tsne", xticks=PERPLEXITIES)
    ax.legend()
    fig.savefig(OUT / "perplexity_rho_tsne.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", None):
        print(table.set_index(["perplexity", "dataset"]).round(4).T, geometry.round(4), sep="\n\n")


if __name__ == "__main__":
    main()
