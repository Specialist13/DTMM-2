"""Deterministic tests for the MDS evaluation helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Lab2" / "code" / "mds"))

from mds_common import (  # noqa: E402
    Dataset,
    _continuity,
    _global_pair_metrics,
    _neighbours_without_self,
    build_context,
    calculate_metrics,
    valid_k_values,
)
from mds_experiment import _space_relation  # noqa: E402


def make_dataset() -> Dataset:
    X = np.arange(30, dtype=float).reshape(10, 3)
    frame = pd.DataFrame(X, columns=["V01", "V02", "V03"])
    frame["class"] = np.array(["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"])
    return Dataset(
        name="toy",
        path=Path("toy.csv"),
        frame=frame,
        X=X,
        feature_columns=["V01", "V02", "V03"],
        object_ids=np.arange(1, 11),
        labels=frame["class"].to_numpy(),
        class_column="class",
        id_column=None,
        generated_id=True,
    )


class CommonMetricTests(unittest.TestCase):
    def test_valid_k_values_respect_sample_size(self):
        self.assertEqual(valid_k_values([1, 4, 5], 10)[0], [1, 4])

    def test_neighbours_do_not_include_self(self):
        X = make_dataset().X
        neighbours = _neighbours_without_self(X, 3)
        for row, values in enumerate(neighbours):
            self.assertNotIn(row, values)
            self.assertEqual(len(set(values.tolist())), 3)

    def test_continuity_is_one_for_same_neighbourhoods(self):
        dataset = make_dataset()
        original = _neighbours_without_self(dataset.X, 4)
        # A rank matrix in which the original neighbours have ranks 1..4.
        ranks = np.full((10, 10), 10, dtype=np.int32)
        for row, neighbours in enumerate(original):
            ranks[row, neighbours] = np.arange(1, 5)
        self.assertAlmostEqual(_continuity(original, ranks, 4), 1.0)

    def test_identity_embedding_has_perfect_neighbour_metrics(self):
        dataset = make_dataset()
        context = build_context(dataset, max_k=4, pair_sample_size=1000, pair_seed=123)
        metrics = calculate_metrics(context, dataset.X.copy(), [2, 4])
        for key in (
            "trustworthiness_k2", "trustworthiness_k4", "continuity_k2",
            "continuity_k4", "knn_preservation_k2", "knn_preservation_k4",
        ):
            self.assertAlmostEqual(metrics[key], 1.0, places=12, msg=key)
        self.assertAlmostEqual(metrics["spearman_distance"], 1.0, places=12)
        self.assertAlmostEqual(metrics["normalized_stress"], 0.0, places=12)

    def test_global_metrics_are_undefined_for_constant_distances(self):
        dataset = make_dataset()
        context = build_context(dataset, max_k=2, pair_sample_size=100, pair_seed=1)
        constant = np.zeros((len(dataset.X), 2))
        spearman, stress = _global_pair_metrics(context, constant)
        self.assertTrue(np.isnan(spearman))
        self.assertTrue(np.isnan(stress))


class MDSMetricTests(unittest.TestCase):
    def test_identical_spaces_have_zero_stress_and_unit_spearman(self):
        points = np.arange(24, dtype=float).reshape(8, 3)
        pair_i, pair_j = np.triu_indices(len(points), k=1)
        spearman, stress = _space_relation(points, points.copy(), pair_i, pair_j)
        self.assertAlmostEqual(spearman, 1.0, places=12)
        self.assertAlmostEqual(stress, 0.0, places=12)

    def test_space_relation_is_invariant_to_uniform_scale(self):
        points = np.arange(24, dtype=float).reshape(8, 3)
        pair_i, pair_j = np.triu_indices(len(points), k=1)
        spearman, stress = _space_relation(points, 7.5 * points, pair_i, pair_j)
        self.assertAlmostEqual(spearman, 1.0, places=12)
        self.assertAlmostEqual(stress, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
