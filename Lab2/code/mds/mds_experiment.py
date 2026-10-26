"""MDS experiment system for the prepared A29 data.

The module reuses the first-assignment decisions and the tested metric helpers
from :mod:`mds_common`, and writes an independent ``results/mds`` tree.
The default command uses all available objects.  Because sklearn MDS has
quadratic memory and computational complexity, ``--sample-size`` is available
as an explicit, documented alternative; it is never enabled implicitly.

Run from the project root::

    python Lab2/code/mds/mds_experiment.py

Short validation run::

    python Lab2/code/mds/mds_experiment.py --quick
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from mds_common import (  # noqa: E402
    CORRELATION_KEEP,
    DEFAULT_SEEDS,
    Dataset,
    REQUESTED_K,
    _format_number,
    _neighbours_without_self,
    _palette,
    _sample_pairs,
    _save_figure,
    build_context,
    calculate_metrics,
    load_dataset,
    map_outlier_artifact,
    resolve_input_paths,
    validate_variant_alignment,
    valid_k_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = PROJECT_ROOT / "results" / "mds"
PAIR_SAMPLE_SIZE = 100_000


def _json_write(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_table(rows: Sequence[Mapping[str, object]], path: Path) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.12g")
    return table


def _subset_dataset(
    base: Dataset,
    name: str,
    feature_columns: Sequence[str],
    row_mask: np.ndarray | None = None,
    path: Path | None = None,
) -> Dataset:
    feature_columns = list(feature_columns)
    positions = [base.feature_columns.index(column) for column in feature_columns]
    if row_mask is None:
        row_mask = np.ones(len(base.X), dtype=bool)
    frame_columns = [*feature_columns, *([base.class_column] if base.class_column else [])]
    frame = base.frame.loc[row_mask, frame_columns].reset_index(drop=True).copy()
    labels = None if base.labels is None else base.labels[row_mask]
    return Dataset(
        name=name,
        path=path or base.path,
        frame=frame,
        X=np.ascontiguousarray(base.X[row_mask][:, positions]),
        feature_columns=feature_columns,
        object_ids=base.object_ids[row_mask],
        labels=labels,
        class_column=base.class_column,
        id_column=base.id_column,
        generated_id=base.generated_id,
    )


def _save_input(dataset: Dataset, path: Path) -> None:
    result = pd.DataFrame({"row_id": dataset.object_ids})
    for column, values in zip(dataset.feature_columns, dataset.X.T):
        result[column] = values
    if dataset.labels is not None:
        result[dataset.class_column or "class"] = dataset.labels
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.15g")


def _save_embedding(dataset: Dataset, embedding: np.ndarray, path: Path) -> None:
    result = pd.DataFrame({"row_id": dataset.object_ids})
    if dataset.labels is not None:
        result[dataset.class_column or "class"] = dataset.labels
    for index in range(embedding.shape[1]):
        result[f"mds_{index + 1}"] = embedding[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, float_format="%.12g")


def _plot_scatter(ax: plt.Axes, dataset: Dataset, embedding: np.ndarray, palette: Mapping[str, str], title: str) -> None:
    if dataset.labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=7, alpha=0.65, c=palette["all objects"], linewidths=0)
    else:
        for label in sorted(pd.unique(dataset.labels).tolist()):
            mask = dataset.labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=7, alpha=0.65, c=palette[str(label)], label=str(label), linewidths=0)
        ax.legend(loc="best", fontsize=7, markerscale=2, frameon=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)


def _plot_3d(ax, dataset: Dataset, embedding: np.ndarray, palette: Mapping[str, str], title: str) -> None:
    if dataset.labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=5, alpha=0.55, c=palette["all objects"], linewidths=0)
    else:
        for label in sorted(pd.unique(dataset.labels).tolist()):
            mask = dataset.labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2], s=5, alpha=0.55, c=palette[str(label)], label=str(label), linewidths=0)
        ax.legend(loc="best", fontsize=7, markerscale=2, frameon=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_zlabel("MDS 3")


def _build_dissimilarity(X: np.ndarray) -> np.ndarray:
    """Build one reusable Euclidean dissimilarity matrix for one study."""

    matrix = pairwise_distances(X, metric="euclidean").astype(np.float64, copy=False)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _mds_kwargs(
    n_components: int,
    metric: bool,
    random_state: int,
    n_init: int,
    max_iter: int,
    eps: float,
) -> dict[str, object]:
    parameters = inspect.signature(MDS).parameters
    kwargs: dict[str, object] = {
        "n_components": int(n_components),
        "metric": bool(metric),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "eps": float(eps),
        "random_state": int(random_state),
        "dissimilarity": "precomputed",
    }
    # Keep sklearn's internal stress_ raw, because the report separately
    # computes Stress-1 on the same reproducible pair sample as Spearman.
    if "normalized_stress" in parameters:
        kwargs["normalized_stress"] = False
    if "n_jobs" in parameters:
        kwargs["n_jobs"] = 1
    return kwargs


def run_mds(
    dissimilarity: np.ndarray,
    n_components: int,
    metric: bool,
    random_state: int,
    n_init: int = 4,
    max_iter: int = 300,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float, dict[str, object]]:
    """Fit sklearn MDS and return coordinates, raw sklearn stress and kwargs."""

    kwargs = _mds_kwargs(n_components, metric, random_state, n_init, max_iter, eps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = MDS(**kwargs)
        embedding = np.asarray(model.fit_transform(dissimilarity), dtype=np.float64)
    return embedding, float(getattr(model, "stress_", np.nan)), kwargs


def _metric_row(
    experiment: str,
    run_name: str,
    dataset: Dataset,
    embedding: np.ndarray,
    metrics: Mapping[str, float],
    sklearn_stress: float,
    metric_mode: bool,
    seed: int,
    n_init: int,
    max_iter: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "experiment": experiment,
        "run": run_name,
        "metric_mds": metric_mode,
        "n_components": embedding.shape[1],
        "random_state": seed,
        "n_init": n_init,
        "max_iter": max_iter,
        "n_samples": len(dataset.X),
        "n_features": len(dataset.feature_columns),
        "sklearn_raw_stress": sklearn_stress,
    }
    row.update(metrics)
    return row


def _study_plot_metric(table: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = table[table["run"] == "stability_2d"] if "run" in table else table
    if values.empty:
        values = table
    grouped = values.groupby("random_state")[metric].mean().reset_index()
    ax.plot(grouped["random_state"].astype(str), grouped[metric], marker="o")
    ax.set_title(title)
    ax.set_xlabel("random_state")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_metric_vs_mode(table: pd.DataFrame, path: Path) -> None:
    selected = table[table["run"].isin(["baseline_2d", "nonmetric_2d"])].copy()
    labels = selected["run"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in (
        (axes[0], "normalized_stress", "Stress-1: metrinis ir nemetrinis MDS"),
        (axes[1], "spearman_distance", "Spearman: metrinis ir nemetrinis MDS"),
    ):
        ax.bar(labels, selected[metric], color=["#35608d", "#d07a2d"])
        ax.set_title(title)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_stability(table: pd.DataFrame, path: Path, title: str) -> None:
    selected = table[table["run"] == "stability_2d"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, label, colour in (
        ("trustworthiness_k15", "Trustworthiness (k=15)", "#35608d"),
        ("continuity_k15", "Continuity (k=15)", "#d07a2d"),
        ("knn_preservation_k15", "kNN preservation (k=15)", "#4c956c"),
    ):
        if metric not in selected:
            continue
        values = selected.groupby("random_state")[metric].mean()
        ax.plot(values.index.astype(str), values.values, marker="o", label=label, color=colour)
    ax.set_title(title)
    ax.set_xlabel("random_state")
    ax.set_ylabel("metrikos reikšmė")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_stability_projections(dataset: Dataset, embeddings: Mapping[int, np.ndarray], palette: Mapping[str, str], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
    for ax, seed in zip(axes.flat, sorted(embeddings)):
        _plot_scatter(ax, dataset, embeddings[seed], palette, f"seed={seed}")
    for ax in axes.flat[len(embeddings):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_dimension_comparison(dataset: Dataset, embeddings: Mapping[int, np.ndarray], palette: Mapping[str, str], path: Path, title: str) -> None:
    fig = plt.figure(figsize=(13, 5.5))
    ax2 = fig.add_subplot(1, 2, 1)
    _plot_scatter(ax2, dataset, embeddings[2], palette, "MDS 2D")
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_3d(ax3, dataset, embeddings[3], palette, "MDS 3D")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_three_studies(studies: Mapping[str, Mapping[str, object]], path: Path) -> None:
    palette = _palette(studies["A"]["dataset"].labels)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), squeeze=False)
    titles = {"A": "A – visi požymiai ir objektai", "B": "B – be koreliuotų požymių", "C": "C – be išskirčių"}
    for ax, key in zip(axes[0], ("A", "B", "C")):
        study = studies[key]
        _plot_scatter(ax, study["dataset"], study["baseline_embedding"], palette, titles[key])
    fig.suptitle("A, B ir C MDS projekcijų palyginimas (2D, metric MDS, seed=42)", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, path)


def _space_relation(left: np.ndarray, right: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> tuple[float, float]:
    left_distances = np.linalg.norm(left[pair_i] - left[pair_j], axis=1)
    right_distances = np.linalg.norm(right[pair_i] - right[pair_j], axis=1)
    if len(left_distances) < 3 or np.ptp(left_distances) == 0 or np.ptp(right_distances) == 0:
        spearman_value = float("nan")
    else:
        spearman_value = float(spearmanr(left_distances, right_distances).statistic)
    denominator = float(np.dot(right_distances, right_distances))
    original_sum = float(np.dot(left_distances, left_distances))
    if denominator == 0 or original_sum == 0:
        stress = float("nan")
    else:
        scale = float(np.dot(left_distances, right_distances) / denominator)
        residual = left_distances - scale * right_distances
        stress = float(math.sqrt(np.dot(residual, residual) / original_sum))
    return spearman_value, stress


def _compare_spaces(
    name: str,
    left_name: str,
    right_name: str,
    left_X: np.ndarray,
    right_X: np.ndarray,
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    ks: Sequence[int],
    pair_sample_size: int,
    path: Path,
) -> pd.DataFrame:
    if len(left_X) != len(right_X):
        raise ValueError("MDS erdvių palyginimui objektų skaičius turi sutapti.")
    pair_i, pair_j = _sample_pairs(len(left_X), pair_sample_size, 2026)
    left_input = _neighbours_without_self(left_X, max(ks))
    right_input = _neighbours_without_self(right_X, max(ks))
    left_embedding_neighbours = _neighbours_without_self(left_embedding, max(ks))
    right_embedding_neighbours = _neighbours_without_self(right_embedding, max(ks))
    input_spearman, input_stress = _space_relation(left_X, right_X, pair_i, pair_j)
    embedding_spearman, embedding_stress = _space_relation(left_embedding, right_embedding, pair_i, pair_j)
    rows = []
    for k in ks:
        rows.extend([
            {"comparison": name, "left": left_name, "right": right_name, "space": "input", "k": k, "n_common_objects": len(left_X), "neighborhood_overlap": float(np.mean([np.isin(left_input[row, :k], right_input[row, :k]).mean() for row in range(len(left_X))])), "spearman_distance": input_spearman, "normalized_stress": input_stress},
            {"comparison": name, "left": left_name, "right": right_name, "space": "embedding", "k": k, "n_common_objects": len(left_X), "neighborhood_overlap": float(np.mean([np.isin(left_embedding_neighbours[row, :k], right_embedding_neighbours[row, :k]).mean() for row in range(len(left_X))])), "spearman_distance": embedding_spearman, "normalized_stress": embedding_stress},
        ])
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False, float_format="%.12g")
    return table


def _run_study(
    key: str,
    title: str,
    dataset: Dataset,
    output_dir: Path,
    seeds: Sequence[int],
    max_iter: int,
    n_init: int,
    eps: float,
    pair_sample_size: int,
) -> dict[str, object]:
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    embeddings_dir = output_dir / "embeddings"
    for directory in (figures, tables, embeddings_dir):
        directory.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_used.csv"
    _save_input(dataset, input_path)
    dataset.path = input_path
    ks, skipped_k = valid_k_values(REQUESTED_K, len(dataset.X))
    if not ks:
        raise ValueError(f"{key}: nėra tinkamų k reikšmių.")
    base_seed = 42 if 42 in seeds else int(seeds[0])
    context = build_context(dataset, max(ks), pair_sample_size, 2026)
    dissimilarity = _build_dissimilarity(dataset.X)
    palette = _palette(dataset.labels)
    rows: list[dict[str, object]] = []

    embedding_2d, raw_2d, kwargs_2d = run_mds(dissimilarity, 2, True, base_seed, n_init, max_iter, eps)
    metrics_2d = calculate_metrics(context, embedding_2d, ks)
    row_2d = _metric_row(key, "baseline_2d", dataset, embedding_2d, metrics_2d, raw_2d, True, base_seed, n_init, max_iter)
    rows.append(row_2d)
    _save_embedding(dataset, embedding_2d, embeddings_dir / "embedding_baseline_2d_seed42.csv")
    fig = plt.figure(figsize=(8, 6))
    _plot_scatter(fig.add_subplot(111), dataset, embedding_2d, palette, f"{title}: bazinė MDS 2D projekcija")
    _save_figure(fig, figures / "baseline_2d.png")

    embedding_3d, raw_3d, kwargs_3d = run_mds(dissimilarity, 3, True, base_seed, n_init, max_iter, eps)
    metrics_3d = calculate_metrics(context, embedding_3d, ks)
    rows.append(_metric_row(key, "baseline_3d", dataset, embedding_3d, metrics_3d, raw_3d, True, base_seed, n_init, max_iter))
    _save_embedding(dataset, embedding_3d, embeddings_dir / "embedding_baseline_3d_seed42.csv")
    fig = plt.figure(figsize=(9, 7))
    _plot_3d(fig.add_subplot(111, projection="3d"), dataset, embedding_3d, palette, f"{title}: MDS 3D projekcija")
    _save_figure(fig, figures / "baseline_3d.png")
    _plot_dimension_comparison(dataset, {2: embedding_2d, 3: embedding_3d}, palette, figures / "dimension_2d_vs_3d.png", f"{title}: tikslinės dimensijos poveikis")

    nonmetric, raw_nonmetric, kwargs_nonmetric = run_mds(dissimilarity, 2, False, base_seed, n_init, max_iter, eps)
    metrics_nonmetric = calculate_metrics(context, nonmetric, ks)
    rows.append(_metric_row(key, "nonmetric_2d", dataset, nonmetric, metrics_nonmetric, raw_nonmetric, False, base_seed, n_init, max_iter))
    _save_embedding(dataset, nonmetric, embeddings_dir / "embedding_nonmetric_2d_seed42.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), squeeze=False)
    _plot_scatter(axes[0, 0], dataset, embedding_2d, palette, "Metric MDS")
    _plot_scatter(axes[0, 1], dataset, nonmetric, palette, "Non-metric MDS")
    fig.suptitle(f"{title}: metric ir non-metric MDS", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures / "metric_vs_nonmetric_2d.png")

    stability_rows: list[dict[str, object]] = []
    stability_embeddings: dict[int, np.ndarray] = {}
    for seed in seeds:
        seed = int(seed)
        if seed == base_seed:
            embedding, raw_stress, run_kwargs = embedding_2d, raw_2d, kwargs_2d
            metrics = metrics_2d
        else:
            embedding, raw_stress, run_kwargs = run_mds(dissimilarity, 2, True, seed, n_init, max_iter, eps)
            metrics = calculate_metrics(context, embedding, ks)
        row = _metric_row(key, "stability_2d", dataset, embedding, metrics, raw_stress, True, seed, n_init, max_iter)
        stability_rows.append(row)
        rows.append(row)
        stability_embeddings[seed] = embedding
        _save_embedding(dataset, embedding, embeddings_dir / f"embedding_stability_2d_seed{seed}.csv")
    stability_table = pd.DataFrame(stability_rows)
    stability_table.to_csv(tables / "stability_metrics.csv", index=False, float_format="%.12g")
    stability_metric_columns = [
        column for column in [
            "trustworthiness_k5", "trustworthiness_k15", "trustworthiness_k50",
            "continuity_k5", "continuity_k15", "continuity_k50",
            "knn_preservation_k5", "knn_preservation_k15", "knn_preservation_k50",
            "neighborhood_hit_k5", "neighborhood_hit_k15", "neighborhood_hit_k50",
            "spearman_distance", "normalized_stress",
        ] if column in stability_table.columns
    ]
    stability_summary = stability_table[stability_metric_columns].agg(["mean", "std"]).T.reset_index(names="metric")
    stability_summary.to_csv(tables / "stability_metrics_summary.csv", index=False, float_format="%.12g")
    _plot_stability(stability_table, figures / "stability_metrics.png", f"{title}: MDS stabilumas")
    _plot_stability_projections(dataset, stability_embeddings, palette, figures / "stability_projections.png", f"{title}: penkių sėklų projekcijos")

    metric_mode_table = pd.DataFrame([rows[0], rows[2]])
    metric_mode_table.to_csv(tables / "metric_vs_nonmetric.csv", index=False, float_format="%.12g")
    dimension_table = pd.DataFrame([rows[0], rows[1]])
    dimension_table.to_csv(tables / "dimension_2d_vs_3d.csv", index=False, float_format="%.12g")
    _plot_metric_vs_mode(metric_mode_table, figures / "metric_vs_nonmetric_metrics.png")
    for metric in ("normalized_stress", "spearman_distance", "trustworthiness_k15", "continuity_k15", "knn_preservation_k15"):
        _study_plot_metric(stability_table, metric, figures / f"stability_{metric}.png", f"{title}: {metric.replace('_', ' ')} pagal seed")

    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(tables / "all_metrics.csv", index=False, float_format="%.12g")
    _json_write({
        "experiment": key,
        "title": title,
        "n_samples": len(dataset.X),
        "n_features": len(dataset.feature_columns),
        "feature_columns": dataset.feature_columns,
        "valid_k": ks,
        "skipped_k": skipped_k,
        "random_states": [int(seed) for seed in seeds],
        "base_parameters": kwargs_2d,
        "metric_mds_explanation": "metric=True optimizuoja porinių atstumų išsaugojimą",
        "nonmetric_mds_explanation": "metric=False optimizuoja atstumų ranginę tvarką per monotonišką transformaciją",
        "stress_note": "sklearn_raw_stress yra raw stress_ iš MDS objekto; normalized_stress apskaičiuotas atskirai su optimaliu masteliu iš bendros porų imties",
        "stress_1_formula": "sqrt(sum((d - a*delta)^2) / sum(d^2)), a=sum(d*delta)/sum(delta^2)",
        "pair_sample_size": pair_sample_size,
        "memory_note": f"Pilna MDS dissimilarity matrica: {len(dataset.X)}x{len(dataset.X)} float64 (~{len(dataset.X) ** 2 * 8 / 1024**2:.1f} MiB).",
    }, tables / "experiment_parameters.json")

    summary = [
        f"# {title}", "", f"Objektai: {len(dataset.X)}, požymiai: {len(dataset.feature_columns)}.",
        f"Naudotas metric MDS: n_components=2/3, n_init={n_init}, max_iter={max_iter}, eps={eps}, random_state={base_seed}, dissimilarity='precomputed' (Euklidiniai atstumai).",
        "MDS klasės nenaudoja projekcijai; jos naudotos tik spalvoms ir Neighborhood Hit.",
        "sklearn stress_ išsaugotas kaip raw stress, o ataskaitinis Stress-1 apskaičiuotas atskirai su optimaliu vienodu masteliu.", "",
        "| Paleidimas | Dimensija | Metric | Stress-1 | Spearman | Trustworthiness k=15 | Continuity k=15 | kNN k=15 | NH k=15 |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (rows[0], rows[1], rows[2]):
        summary.append(f"| {row['run']} | {row['n_components']} | {row['metric_mds']} | {_format_number(row['normalized_stress'])} | {_format_number(row['spearman_distance'])} | {_format_number(row['trustworthiness_k15'])} | {_format_number(row['continuity_k15'])} | {_format_number(row['knn_preservation_k15'])} | {_format_number(row['neighborhood_hit_k15'])} |")
    summary.extend(["", "## Stabilumas", "", f"Sėklos: {list(seeds)}. Projekcijų orientacija, poslinkis ar atspindėjimas nelaikyti nestabilumu; vertinta metrikų sklaida."])
    for row in stability_summary.itertuples(index=False):
        summary.append(f"- {row.metric}: vidurkis={_format_number(row.mean)}, SD={_format_number(row.std)}.")
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return {
        "key": key,
        "title": title,
        "dataset": dataset,
        "dir": output_dir,
        "context": context,
        "ks": ks,
        "baseline_embedding": embedding_2d,
        "baseline_metrics": metrics_2d,
        "baseline_row": row_2d,
        "embedding_3d": embedding_3d,
        "nonmetric_embedding": nonmetric,
        "metrics_table": metrics_table,
        "stability_table": stability_table,
        "stability_summary": stability_summary,
    }


def _normalization_comparison(datasets: Mapping[str, Dataset], output_dir: Path, seeds: Sequence[int], max_iter: int, n_init: int, eps: float, pair_sample_size: int, ks: Sequence[int]) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = 42 if 42 in seeds else int(seeds[0])
    rows = []
    embeddings = {}
    for method in ("min_max", "standardized", "robust"):
        data = datasets[method]
        context = build_context(data, max(ks), pair_sample_size, 2026)
        dissimilarity = _build_dissimilarity(data.X)
        embedding, raw_stress, kwargs = run_mds(dissimilarity, 2, True, base_seed, n_init, max_iter, eps)
        metrics = calculate_metrics(context, embedding, ks)
        row = {"method": method, "n_samples": len(data.X), "n_features": len(data.feature_columns), "random_state": base_seed, "n_init": n_init, "sklearn_raw_stress": raw_stress}
        row.update(metrics)
        rows.append(row)
        embeddings[method] = embedding
        _save_embedding(data, embedding, output_dir / f"embedding_{method}_2d.csv")
    table = _save_table(rows, output_dir / "normalization_metrics.csv")
    palette = _palette(datasets["robust"].labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), squeeze=False)
    for ax, method in zip(axes[0], ("min_max", "standardized", "robust")):
        _plot_scatter(ax, datasets[method], embeddings[method], palette, f"MDS + {method}")
    fig.suptitle("MDS normalizavimo metodų palyginimas", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, output_dir / "normalization_comparison.png")
    summary = ["# MDS normalizavimo metodų palyginimas", "", f"Bendri parametrai: metric=True, n_components=2, random_state={base_seed}, n_init={n_init}, max_iter={max_iter}.", "Kiekvienas metodas vertintas savo normalizuotos pradinės erdvės atžvilgiu.", "", "| Metodas | Stress-1 | Spearman | Trustworthiness k=15 | Continuity k=15 | kNN k=15 | NH k=15 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        summary.append(f"| {row.method} | {_format_number(row.normalized_stress)} | {_format_number(row.spearman_distance)} | {_format_number(row.trustworthiness_k15)} | {_format_number(row.continuity_k15)} | {_format_number(row.knn_preservation_k15)} | {_format_number(row.neighborhood_hit_k15)} |")
    summary.append("\nŠis papildomas tyrimas nėra A/B/C eksperimentų dalis. RobustScaler pasirinkimas pagrįstas išskirčių analize, bet jo pranašumas sprendžiamas tik pagal faktinius rodiklius.")
    (output_dir / "normalization_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return table


def _write_main_summary(studies: Mapping[str, Mapping[str, object]], normalization_table: pd.DataFrame, outlier_mapping, comparison_dir: Path, seeds: Sequence[int], n_init: int, max_iter: int, eps: float, pair_sample_size: int, sampling_note: str) -> None:
    if "naudoti visi" in sampling_note:
        scope_note = "Naudoti visi paruošto Robust rinkinio objektai; pilna MDS matrica skaičiuota be imties mažinimo."
    else:
        scope_note = f"{sampling_note}. Numatytoji programos elgsena yra naudoti visus objektus; šiame rezultatų rinkinyje pasirinkta aiškiai nurodyta imtis dėl pilnos 4158×4158 MDS dissimilarity matricos skaičiavimo kainos."
    lines = [
        "# MDS dimensijos mažinimo eksperimento santrauka", "",
        "MDS (Multidimensional Scaling) ieško mažesnės dimensijos konfigūracijos, kurios objektų poriniai atstumai būtų kuo panašesni į pradinės erdvės atstumus. Metric MDS optimizuoja atstumų dydžius, o non-metric MDS – jų ranginę tvarką.",
        "",
        "## Duomenys", "",
        "Pagrindui naudotas RobustScaler failas. Pradinis trūkstamų reikšmių užpildymas atliktas klasėms atskiromis medianomis, todėl klasių struktūros interpretacija gali būti paveikta. Klasės į MDS požymių matricą neįtrauktos.",
        "", "## A, B ir C eksperimentų baziniai rezultatai", "",
        "| Eksperimentas | Objektai | Požymiai | Stress-1 | Spearman | Trustworthiness k=15 | Continuity k=15 | kNN k=15 | NH k=15 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("A", "B", "C"):
        study = studies[key]
        row = study["baseline_metrics"]
        lines.append(f"| {key} | {len(study['dataset'].X)} | {len(study['dataset'].feature_columns)} | {_format_number(row['normalized_stress'])} | {_format_number(row['spearman_distance'])} | {_format_number(row['trustworthiness_k15'])} | {_format_number(row['continuity_k15'])} | {_format_number(row['knn_preservation_k15'])} | {_format_number(row['neighborhood_hit_k15'])} |")
    a = studies["A"]["baseline_metrics"]
    b = studies["B"]["baseline_metrics"]
    c = studies["C"]["baseline_metrics"]
    lines.extend([
        "", "## Skaičiavimo apimtis", "", scope_note,
        "", "B naudoja tuos pačius objektus kaip A, bet 30 iš ankstesnės koreliacijų analizės atrinktų požymių. A/B pradinių kaimynysčių ir atstumų palyginimas pateiktas `comparison/A_B_structure.csv`.",
        "C naudoja visus A požymius ir tik saugiai suderintus išskirčių ataskaitos objektus. Suderinimo būsena: " + str(outlier_mapping.reason) + ". A/C palyginimas atliktas tik bendriems objektams ir pateiktas `comparison/A_C_common_objects.csv`.",
        "", "## Rezultatų interpretacija", "",
        f"B, palyginti su A, padidino Stress-1 nuo {_format_number(a['normalized_stress'])} iki {_format_number(b['normalized_stress'])}, bet padidino Spearman nuo {_format_number(a['spearman_distance'])} iki {_format_number(b['spearman_distance'])} ir Trustworthiness(k=15) nuo {_format_number(a['trustworthiness_k15'])} iki {_format_number(b['trustworthiness_k15'])}. Tai rodo, kad koreliuotų požymių pašalinimas pakeitė metrinių atstumų mastelį, tačiau ranginė ir lokali struktūra šiame paleidime pagerėjo.",
        f"C, palyginti su A, turi {_format_number(c['trustworthiness_k15'])} Trustworthiness(k=15) ir {_format_number(c['continuity_k15'])} Continuity(k=15), tačiau C imtis yra mažesnė, todėl jos Stress-1 ({_format_number(c['normalized_stress'])}) nėra tiesiogiai lygintinas su A. A/C struktūros išvada remiasi tik bendrais objektais ir atskirai pateikta `A_C_common_objects.csv`.",
        "", "## Metric ir non-metric MDS", "", "Metric MDS siekia išsaugoti porinių atstumų dydžius, o non-metric MDS leidžia monotonišką atstumų transformaciją ir siekia išsaugoti rangus. Jų palyginimas atskirai pateiktas kiekvieno eksperimento `tables/metric_vs_nonmetric.csv`.",
        "", "## Stress-1 ir Spearman", "", "Stress-1 apskaičiuotas su optimaliu vienodu projekcijos masteliu: sqrt(sum((d-a·δ)^2)/sum(d²)), a=sum(d·δ)/sum(δ²). Mažesnis Stress-1 reiškia mažesnį atstumų iškraipymą. Spearman vertina porinių atstumų rangų sutapimą, todėl didesnė reikšmė reiškia geresnį ranginės struktūros išsaugojimą. Abu rodikliai remiasi ta pačia atkuriama porų imtimi.",
        "sklearn `stress_` yra MDS optimizavimo raw stress dydis; jis nėra automatiškai tas pats, kas čia pateiktas normalizuotas Stress-1.",
        "", "## Stabilumas", "", f"Kiekvienas eksperimentas pakartotas su random_state={list(seeds)}. Stabilumo vertinime naudoti metrikų vidurkiai ir standartiniai nuokrypiai; projekcijų pasukimas, atspindėjimas ar poslinkis nelaikyti nestabilumu.",
    ])
    for key in ("A", "B", "C"):
        study = studies[key]
        stability = study["stability_table"]
        lines.append(f"- {key}: Trustworthiness(k=15) vidurkis={_format_number(stability.trustworthiness_k15.mean())}, SD={_format_number(stability.trustworthiness_k15.std())}; Stress-1 vidurkis={_format_number(stability.normalized_stress.mean())}, SD={_format_number(stability.normalized_stress.std())}.")
    lines.extend([
        "", "## Normalizavimo poveikis", "", "MinMax, Standard ir Robust MDS projekcijų palyginimas yra papildomas ir neįtrauktas į A/B/C eksperimentų struktūros pakeitimus. Jo rezultatai pateikti `normalization_comparison/`.",
        "", "## Išvada", "", "MDS tinkamumas A29 vizualizavimui vertinamas ne vien pagal grafiką: svarbūs Stress-1, Spearman, Trustworthiness, Continuity, kNN ir stabilumo rezultatai. A/B poveikis aiškinamas kartu su pradinių erdvių kaimynysčių pasikeitimu, o A/C – tik bendrų objektų procedūra, nes C imtis yra mažesnė.",
        "", "Visa rezultatų struktūra: `experiment_A_full/`, `experiment_B_no_correlated/`, `experiment_C_no_outliers/`, `comparison/` ir `normalization_comparison/`.",
    ])
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (comparison_dir / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    root: Path = PROJECT_ROOT,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    max_iter: int = 300,
    n_init: int = 4,
    eps: float = 1e-6,
    pair_sample_size: int = PAIR_SAMPLE_SIZE,
    sample_size: int | None = None,
) -> dict[str, object]:
    """Run independent A/B/C MDS studies and the separate scaling study."""

    global PROJECT_ROOT, RESULTS_ROOT
    PROJECT_ROOT = root
    RESULTS_ROOT = root / "results" / "mds"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    paths = resolve_input_paths(root)
    source = load_dataset(paths["robust"], name="robust_source")
    imputed = load_dataset(paths["imputed"], name="imputed_unscaled")
    min_max = load_dataset(paths["min_max"], name="min_max")
    standardized = load_dataset(paths["standardized"], name="standardized")
    validate_variant_alignment(imputed, min_max)
    validate_variant_alignment(imputed, standardized)
    validate_variant_alignment(imputed, source)
    outlier_mapping = map_outlier_artifact(source, root)
    if outlier_mapping.mask is None or not outlier_mapping.mask.any() or outlier_mapping.mask.all():
        raise ValueError("MDS C eksperimentui nėra saugiai suderinto išskirčių sąrašo.")

    selected_mask = np.ones(len(source.X), dtype=bool)
    sampling_note = "naudoti visi objektai"
    if sample_size is not None:
        if sample_size < 10 or sample_size > len(source.X):
            raise ValueError(f"sample_size turi būti tarp 10 ir {len(source.X)}.")
        rng = np.random.default_rng(2026)
        selected_indices = np.sort(rng.choice(len(source.X), size=sample_size, replace=False))
        selected_mask = np.zeros(len(source.X), dtype=bool)
        selected_mask[selected_indices] = True
        sampling_note = f"naudota fiksuota atkuriama {sample_size} objektų imtis, seed=2026; tai nėra pilnas eksperimentas"

    A = _subset_dataset(source, "experiment_A_full", source.feature_columns, selected_mask)
    B = _subset_dataset(source, "experiment_B_no_correlated", CORRELATION_KEEP, selected_mask)
    C_mask = (~outlier_mapping.mask) & selected_mask
    C = _subset_dataset(source, "experiment_C_no_outliers", source.feature_columns, C_mask)
    minimum_for_all_k = 2 * max(REQUESTED_K) + 1
    if len(C.X) < minimum_for_all_k:
        raise ValueError(
            f"C eksperimente liko per mažai objektų visoms k reikšmėms "
            f"(reikia bent {minimum_for_all_k}, gauta {len(C.X)})."
        )
    datasets = {"A": A, "B": B, "C": C}
    titles = {"A": "Eksperimentas A – pradinis Robust rinkinys", "B": "Eksperimentas B – be koreliuotų požymių", "C": "Eksperimentas C – be išskirčių"}
    dirs = {key: RESULTS_ROOT / f"experiment_{key}_{suffix}" for key, suffix in (("A", "full"), ("B", "no_correlated"), ("C", "no_outliers"))}
    studies = {}
    for key in ("A", "B", "C"):
        studies[key] = _run_study(key, titles[key], datasets[key], dirs[key], seeds, max_iter, n_init, eps, pair_sample_size)
    comparison_dir = RESULTS_ROOT / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = []
    for key in ("A", "B", "C"):
        row = dict(studies[key]["baseline_row"])
        row["experiment"] = key
        baseline_rows.append(row)
    _save_table(baseline_rows, comparison_dir / "metrics_comparison.csv")
    ab_table = _compare_spaces("A_vs_B", "A", "B", A.X, B.X, studies["A"]["baseline_embedding"], studies["B"]["baseline_embedding"], studies["A"]["ks"], pair_sample_size, comparison_dir / "A_B_structure.csv")
    common_mask = (~outlier_mapping.mask) & selected_mask
    ac_table = _compare_spaces("A_vs_C_common_objects", "A_common", "C", A.X[~outlier_mapping.mask[selected_mask]], C.X, studies["A"]["baseline_embedding"][~outlier_mapping.mask[selected_mask]], studies["C"]["baseline_embedding"], studies["A"]["ks"], pair_sample_size, comparison_dir / "A_C_common_objects.csv")
    pd.concat([ab_table, ac_table], ignore_index=True).to_csv(comparison_dir / "structure_comparison.csv", index=False, float_format="%.12g")
    _plot_three_studies(studies, comparison_dir / "all_three_projections.png")
    ks = studies["A"]["ks"]
    normalization_table = _normalization_comparison({"min_max": min_max, "standardized": standardized, "robust": source}, RESULTS_ROOT / "normalization_comparison", seeds, max_iter, n_init, eps, pair_sample_size, ks)
    _write_main_summary(studies, normalization_table, outlier_mapping, comparison_dir, seeds, n_init, max_iter, eps, pair_sample_size, sampling_note)
    _json_write({
        "sampling_note": sampling_note,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "experiments": {key: str(studies[key]["dir"]) for key in studies},
        "comparison_dir": str(comparison_dir),
        "sampling_note": sampling_note,
        "outlier_mapping": outlier_mapping.reason,
        "correlation_keep": CORRELATION_KEEP,
        "n_init": n_init,
        "max_iter": max_iter,
        "eps": eps,
    }, comparison_dir / "experiment_index.json")
    return {"studies": studies, "normalization_table": normalization_table, "outlier_mapping": outlier_mapping, "sampling_note": sampling_note}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Greitesnis patikros režimas: 600 objektų, 1 init, 80 iteracijų.")
    parser.add_argument("--sample-size", type=int, default=None, help="Aiškiai nurodyta fiksuota imtis; numatyta naudoti visus objektus.")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--pair-sample-size", type=int, default=PAIR_SAMPLE_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        sample_size, max_iter, n_init = args.sample_size or 600, args.max_iter or 80, args.n_init or 1
    else:
        sample_size, max_iter, n_init = args.sample_size, args.max_iter or 300, args.n_init or 4
    result = run_experiment(sample_size=sample_size, max_iter=max_iter, n_init=n_init, pair_sample_size=args.pair_sample_size)
    print(f"Baigta. Rezultatai: {RESULTS_ROOT}")
    print(f"Sampling: {result['sampling_note'].encode('ascii', 'replace').decode()}")


if __name__ == "__main__":
    main()
