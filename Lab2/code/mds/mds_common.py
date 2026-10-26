"""Data loading and evaluation helpers shared by the MDS experiment.

Extracted from the original t-SNE experiment module, which MDS was built on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PERPLEXITIES = (5, 10, 20, 30, 40, 50)
DEFAULT_SEEDS = (0, 21, 42, 123, 2026)
REQUESTED_K = (5, 15, 50)

CORRELATION_KEEP = (
    "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V10", "V13",
    "V15", "V17", "V18", "V19", "V21", "V22", "V25", "V26", "V27",
    "V28", "V29", "V30", "V31", "V34", "V38", "V39", "V40", "V41",
    "V42", "V43", "V46",
)


@dataclass
class Dataset:
    """Numeric features and metadata for one prepared CSV."""

    name: str
    path: Path
    frame: pd.DataFrame
    X: np.ndarray
    feature_columns: list[str]
    object_ids: np.ndarray
    labels: np.ndarray | None
    class_column: str | None
    id_column: str | None
    generated_id: bool


@dataclass
class EvaluationContext:
    """Reusable information tied to one original feature space."""

    dataset: Dataset
    pair_i: np.ndarray
    pair_j: np.ndarray
    original_neighbors: np.ndarray


@dataclass
class OutlierMapping:
    """Result of aligning the first-assignment outlier report."""

    mask: np.ndarray | None
    matched_count: int
    artifact_count: int
    reason: str


def _format_number(value: object, digits: int = 5) -> str:
    if value is None or not np.isfinite(value):
        return "NaN"
    return f"{float(value):.{digits}g}"


def _as_float_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    converted = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any():
        bad = list(converted.columns[converted.isna().any()])
        raise ValueError(f"Ne skaitinės arba trūkstamos reikšmės požymiuose: {bad}")
    values = converted.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Požymiuose yra begalinių reikšmių.")
    return converted


def _find_class_column(columns: Sequence[str]) -> str | None:
    lower = {str(column).lower(): str(column) for column in columns}
    for candidate in ("class", "label", "target", "y"):
        if candidate in lower:
            return lower[candidate]
    return None


def _find_id_column(columns: Sequence[str], class_column: str | None) -> str | None:
    candidates = {"id", "row_id", "object_id", "sample_id", "identifier", "index"}
    for column in columns:
        if column != class_column and str(column).lower() in candidates:
            return str(column)
    return None


def load_dataset(path: Path, name: str | None = None) -> Dataset:
    """Load a prepared CSV and identify class/ID columns without using labels."""

    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"CSV faile nėra eilučių: {path}")

    class_column = _find_class_column(frame.columns)
    id_column = _find_id_column(frame.columns, class_column)
    feature_columns = [
        str(column) for column in frame.columns
        if str(column) not in {class_column, id_column}
    ]
    if not feature_columns:
        raise ValueError(f"Nerasta skaitinių požymių: {path}")

    numeric = _as_float_frame(frame, feature_columns)
    labels = None if class_column is None else frame[class_column].astype(str).to_numpy()
    if id_column is None:
        object_ids = np.arange(1, len(frame) + 1, dtype=np.int64)
        generated_id = True
    else:
        object_ids = frame[id_column].to_numpy()
        generated_id = False
        if pd.Series(object_ids).duplicated().any():
            raise ValueError(f"ID stulpelyje '{id_column}' yra dublikatų: {path}")

    return Dataset(
        name=name or path.stem,
        path=path,
        frame=frame,
        X=np.ascontiguousarray(numeric.to_numpy(dtype=np.float64)),
        feature_columns=feature_columns,
        object_ids=object_ids,
        labels=labels,
        class_column=class_column,
        id_column=id_column,
        generated_id=generated_id,
    )


def resolve_input_paths(root: Path = PROJECT_ROOT) -> dict[str, Path]:
    """Find the prepared imputed and normalised files already in the project."""

    base = root / "A29_Fixed_median_by_class_imputed.csv"
    scaling = root / "scaling_experiment_by_class"

    candidates = {
        "imputed": (base,),
        "min_max": (
            root / "A29_Fixed_median_by_class_imputed_min_max.csv",
            scaling / "A29_Fixed_median_by_class_imputed_min_max.csv",
        ),
        "standardized": (
            root / "A29_Fixed_median_by_class_imputed_standardized.csv",
            scaling / "A29_Fixed_median_by_class_imputed_standardized.csv",
        ),
        "robust": (
            root / "A29_Fixed_median_by_class_imputed_robust.csv",
            scaling / "A29_Fixed_median_by_class_imputed_robust.csv",
        ),
    }
    found: dict[str, Path] = {}
    missing: list[str] = []
    for key, paths in candidates.items():
        match = next((path for path in paths if path.exists()), None)
        if match is None:
            missing.append(f"{key}: " + ", ".join(str(path) for path in paths))
        else:
            found[key] = match
    if missing:
        raise FileNotFoundError(
            "Trūksta paruoštų A29 failų. Reikalingi failai:\n" + "\n".join(missing)
        )
    return found


def validate_variant_alignment(reference: Dataset, variant: Dataset) -> None:
    """Check that normalisation variants have the same objects and feature order."""

    if reference.feature_columns != variant.feature_columns:
        raise ValueError(
            f"{variant.name}: skiriasi požymių rinkinys nuo {reference.name}."
        )
    if len(reference.frame) != len(variant.frame):
        raise ValueError(f"{variant.name}: skiriasi objektų skaičius nuo {reference.name}.")
    if not np.array_equal(reference.object_ids, variant.object_ids):
        raise ValueError(f"{variant.name}: skiriasi objektų tvarka arba ID.")
    if reference.labels is not None and variant.labels is not None:
        if not np.array_equal(reference.labels, variant.labels):
            raise ValueError(f"{variant.name}: skiriasi klasių žymių tvarka.")


def valid_k_values(requested: Iterable[int], n_samples: int) -> tuple[list[int], list[str]]:
    """Keep k values valid for both neighbour metrics and trustworthiness."""

    # trustworthiness requires k < n / 2, while neighbour sets require k < n.
    maximum = min(n_samples - 1, (n_samples - 1) // 2)
    valid: list[int] = []
    skipped: list[str] = []
    for value in requested:
        value = int(value)
        if value < 1 or value > maximum:
            skipped.append(
                f"k={value}: turi tenkinti 1 <= k <= {maximum} (n={n_samples})"
            )
        else:
            valid.append(value)
    return valid, skipped


def _sample_pairs(n_samples: int, sample_size: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fixed reproducible sample of unordered object pairs."""

    total = n_samples * (n_samples - 1) // 2
    target = min(int(sample_size), total)
    if target <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if target == total and total <= 2_000_000:
        rows, cols = np.triu_indices(n_samples, k=1)
        return rows.astype(np.int64), cols.astype(np.int64)

    rng = np.random.default_rng(random_state)
    codes: set[int] = set()
    while len(codes) < target:
        batch = max(10_000, 2 * (target - len(codes)))
        first = rng.integers(0, n_samples, size=batch, dtype=np.int64)
        second = rng.integers(0, n_samples, size=batch, dtype=np.int64)
        keep = first != second
        low = np.minimum(first[keep], second[keep])
        high = np.maximum(first[keep], second[keep])
        codes.update((low * n_samples + high).tolist())
    selected = np.fromiter(list(codes)[:target], dtype=np.int64)
    return selected // n_samples, selected % n_samples


def _neighbours_without_self(X: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Return exactly k neighbours per object, never including the object itself."""

    n_samples = len(X)
    if n_neighbors < 1 or n_neighbors >= n_samples:
        raise ValueError("Kaimynų skaičius turi būti tarp 1 ir n-1.")
    # n_jobs=1 is intentional: on some Windows/Python installations the
    # process/thread backend cannot create the named pipes used by joblib.
    # The expensive part of this experiment is t-SNE itself.
    search = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=1)
    search.fit(X)
    raw = search.kneighbors(return_distance=False)
    result = np.empty((n_samples, n_neighbors), dtype=np.int32)
    for row_number, candidates in enumerate(raw):
        candidates = candidates[candidates != row_number]
        result[row_number] = candidates[:n_neighbors]
    return result


def build_context(dataset: Dataset, max_k: int, pair_sample_size: int, pair_seed: int) -> EvaluationContext:
    pair_i, pair_j = _sample_pairs(len(dataset.X), pair_sample_size, pair_seed)
    original_neighbors = _neighbours_without_self(dataset.X, max_k)
    return EvaluationContext(dataset, pair_i, pair_j, original_neighbors)


def _projected_ranks(embedding: np.ndarray) -> np.ndarray:
    """Build 1-based projected-space ranks for continuity."""

    distances = pairwise_distances(embedding, metric="euclidean").astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int32)
    positions = np.arange(1, len(embedding) + 1, dtype=np.int32)
    ranks[np.arange(len(embedding))[:, None], order] = positions[None, :]
    del distances, order
    return ranks


def _continuity(original_neighbors: np.ndarray, projected_ranks: np.ndarray, k: int) -> float:
    """Continuity using the standard rank-penalty formula."""

    n_samples = len(original_neighbors)
    denominator = n_samples * k * (2 * n_samples - 3 * k - 1)
    if denominator <= 0:
        return float("nan")
    penalty = 0.0
    for row_number in range(n_samples):
        original = original_neighbors[row_number, :k]
        projected = projected_ranks[row_number, original]
        penalty += float(np.maximum(projected - k, 0).sum())
    return float(1.0 - (2.0 * penalty / denominator))


def _knn_preservation(original_neighbors: np.ndarray, projected_neighbors: np.ndarray, k: int) -> float:
    overlaps = np.sum(
        np.array([
            np.isin(original_neighbors[row, :k], projected_neighbors[row, :k])
            for row in range(len(original_neighbors))
        ]),
        dtype=np.float64,
    )
    return float(overlaps / (len(original_neighbors) * k))


def _neighborhood_hit(labels: np.ndarray | None, projected_neighbors: np.ndarray, k: int) -> float:
    if labels is None:
        return float("nan")
    same = np.array([
        labels[projected_neighbors[row, :k]] == labels[row]
        for row in range(len(labels))
    ])
    return float(same.mean())


def _global_pair_metrics(context: EvaluationContext, embedding: np.ndarray) -> tuple[float, float]:
    if len(context.pair_i) < 3:
        return float("nan"), float("nan")
    original_distances = np.linalg.norm(
        context.dataset.X[context.pair_i] - context.dataset.X[context.pair_j], axis=1
    )
    projected_distances = np.linalg.norm(
        embedding[context.pair_i] - embedding[context.pair_j], axis=1
    )
    if np.ptp(original_distances) == 0 or np.ptp(projected_distances) == 0:
        spearman_value = float("nan")
    else:
        spearman_value = float(spearmanr(original_distances, projected_distances).statistic)

    # Stress-1 with an optimal multiplicative scale a:
    #   a = sum(d_ij * delta_ij) / sum(delta_ij^2)
    #   Stress-1 = sqrt(sum((d_ij - a*delta_ij)^2) / sum(d_ij^2))
    denominator = float(np.dot(projected_distances, projected_distances))
    original_sum = float(np.dot(original_distances, original_distances))
    if denominator == 0 or original_sum == 0:
        stress = float("nan")
    else:
        scale = float(np.dot(original_distances, projected_distances) / denominator)
        stress = float(
            math.sqrt(np.dot(original_distances - scale * projected_distances,
                             original_distances - scale * projected_distances) / original_sum)
        )
    return spearman_value, stress


def calculate_metrics(context: EvaluationContext, embedding: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """Calculate local, class-aware and global diagnostics for one embedding."""

    if len(embedding) != len(context.dataset.X):
        raise ValueError("Embedding ir pradiniai duomenys turi turėti tiek pat objektų.")
    max_k = max(ks)
    projected_neighbors = _neighbours_without_self(embedding, max_k)
    projected_ranks = _projected_ranks(embedding)
    metrics: dict[str, float] = {}

    for k in ks:
        prefix = f"k{k}"
        try:
            # The requested metric is deliberately calculated through sklearn's
            # public implementation, as required by the assignment.
            metrics[f"trustworthiness_{prefix}"] = float(
                trustworthiness(context.dataset.X, embedding, n_neighbors=k, metric="euclidean")
            )
        except (ValueError, FloatingPointError):
            metrics[f"trustworthiness_{prefix}"] = float("nan")
        metrics[f"continuity_{prefix}"] = _continuity(
            context.original_neighbors, projected_ranks, k
        )
        metrics[f"knn_preservation_{prefix}"] = _knn_preservation(
            context.original_neighbors, projected_neighbors, k
        )
        metrics[f"neighborhood_hit_{prefix}"] = _neighborhood_hit(
            context.dataset.labels, projected_neighbors, k
        )

    metrics["spearman_distance"] , metrics["normalized_stress"] = _global_pair_metrics(
        context, embedding
    )
    local_values = [
        value
        for key, value in metrics.items()
        if (key.startswith("trustworthiness_")
            or key.startswith("continuity_")
            or key.startswith("knn_preservation_"))
        and np.isfinite(value)
    ]
    metrics["local_structure_score"] = float(np.mean(local_values)) if local_values else float("nan")
    return metrics


def _palette(labels: np.ndarray | None) -> dict[str, str]:
    if labels is None:
        return {"all objects": "#35608d"}
    unique = sorted(pd.unique(labels).tolist())
    colours = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(unique))))
    return {str(label): matplotlib.colors.to_hex(colours[index]) for index, label in enumerate(unique)}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_raw_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    # The first assignment's raw file contains occasional values with an
    # ``a.u.`` suffix.  This parsing is used only to align an existing report;
    # it is not a new imputation/scaling pipeline.
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    values = []
    for column in columns:
        extracted = frame[column].astype(str).str.extract(f"({number})", expand=False)
        values.append(pd.to_numeric(extracted, errors="coerce").to_numpy(dtype=np.float64))
    return np.column_stack(values)


def map_outlier_artifact(dataset: Dataset, root: Path = PROJECT_ROOT) -> OutlierMapping:
    """Align original-row outlier records to the deduplicated prepared rows.

    The first-assignment report stores raw A29 row numbers, while the prepared
    file has 42 duplicates removed and has no explicit ID.  We therefore match
    each prepared row to its raw row using its class and the first three finite
    feature values, followed by an all-finite-value check.  Ambiguous matches
    are not silently guessed.
    """

    artifact_path = root / "A29_potential_outliers.csv"
    raw_path = root / "A29.csv"
    fixed_path = root / "A29_Fixed.csv"
    if not (artifact_path.exists() and raw_path.exists() and fixed_path.exists()):
        return OutlierMapping(None, 0, 0, "neatlikta: trūksta A29 išskirčių suderinimo failų")
    artifact = pd.read_csv(artifact_path)
    if "row_number" not in artifact.columns:
        return OutlierMapping(None, 0, 0, "neatlikta: ataskaitoje nėra row_number")
    artifact_rows = set(pd.to_numeric(artifact["row_number"], errors="coerce").dropna().astype(int) - 2)
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    fixed = pd.read_csv(fixed_path, encoding="utf-8-sig")
    if len(fixed) != len(dataset.frame) or list(fixed.columns) != list(dataset.frame.columns):
        return OutlierMapping(
            None, 0, len(artifact_rows),
            "neatlikta: A29_Fixed eilutės nesutampa su naudojamu paruoštu failu",
        )
    columns = dataset.feature_columns
    raw_classes = raw[dataset.class_column].astype(str).to_numpy() if dataset.class_column else None
    fixed_classes = fixed[dataset.class_column].astype(str).to_numpy() if dataset.class_column else None
    raw_numeric = _parse_raw_numeric(raw, columns)
    fixed_numeric = fixed[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    # Index raw rows by (class, selected finite positions, rounded values).
    indexes: dict[tuple[object, tuple[int, ...], tuple[float, ...]], list[int]] = {}
    for raw_row in range(len(raw)):
        finite_positions = np.flatnonzero(np.isfinite(raw_numeric[raw_row]))[:3]
        if len(finite_positions) < 3:
            continue
        label = None if raw_classes is None else raw_classes[raw_row]
        key = (label, tuple(finite_positions.tolist()), tuple(np.round(raw_numeric[raw_row, finite_positions], 12)))
        indexes.setdefault(key, []).append(raw_row)

    used_raw: set[int] = set()
    prepared_to_raw: dict[int, int] = {}
    ambiguous = 0
    for prepared_row in range(len(fixed_numeric)):
        finite_positions = np.flatnonzero(np.isfinite(fixed_numeric[prepared_row]))[:3]
        if len(finite_positions) < 3:
            continue
        label = None if fixed_classes is None else fixed_classes[prepared_row]
        key = (label, tuple(finite_positions.tolist()), tuple(np.round(fixed_numeric[prepared_row, finite_positions], 12)))
        candidates = indexes.get(key, [])
        matches = []
        for raw_row in candidates:
            if raw_row in used_raw:
                continue
            finite = np.isfinite(fixed_numeric[prepared_row]) & np.isfinite(raw_numeric[raw_row])
            if finite.any() and np.allclose(
                fixed_numeric[prepared_row, finite], raw_numeric[raw_row, finite],
                rtol=1e-8, atol=1e-12,
            ):
                matches.append(raw_row)
        if len(matches) == 1:
            prepared_to_raw[prepared_row] = matches[0]
            used_raw.add(matches[0])
        elif len(matches) > 1:
            ambiguous += 1

    mask = np.zeros(len(dataset.frame), dtype=bool)
    for prepared_row, raw_row in prepared_to_raw.items():
        if raw_row in artifact_rows:
            mask[prepared_row] = True
    if ambiguous or len(prepared_to_raw) < len(dataset.frame):
        reason = (
            f"dalinai suderinta: {int(mask.sum())}/{len(artifact_rows)} išskirčių eilučių, "
            f"{len(prepared_to_raw)}/{len(dataset.frame)} paruoštų eilučių; "
            f"neaiškių sutapimų={ambiguous}"
        )
    else:
        reason = f"suderinta: {int(mask.sum())} objektų pagal pirmos užduoties IQR ataskaitą"
    return OutlierMapping(mask, int(mask.sum()), len(artifact_rows), reason)
