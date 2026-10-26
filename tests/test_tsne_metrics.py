"""Small deterministic tests for the t-SNE evaluation helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Lab2" / "code" / "tsne"))

from tsne_experiment import KS, correlated_features, metrics, neighbours, rho  # noqa: E402


def make_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 4)), rng.normal(20, 1, (60, 4))])
    labels = np.array(["a"] * 60 + ["b"] * 60)
    return X, labels


class MetricTests(unittest.TestCase):
    def test_correlated_features_omits_later_duplicate(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"V01": rng.normal(size=50), "V02": rng.normal(size=50)})
        df["V03"] = 2 * df["V01"] + 1
        self.assertEqual(correlated_features(df), ["V03"])

    def test_neighbours_do_not_include_self(self):
        X, _ = make_clusters()
        for row, values in enumerate(neighbours(X, 5)):
            self.assertNotIn(row, values)
            self.assertEqual(len(set(values.tolist())), 5)

    def test_identity_embedding_has_perfect_metrics(self):
        X, labels = make_clusters()
        row = metrics(X, X.copy(), labels)
        for k in KS:
            for key in (f"trustworthiness_k{k}", f"continuity_k{k}", f"knn_k{k}", f"nh_k{k}"):
                self.assertAlmostEqual(row[key], 1.0, places=12, msg=key)
        self.assertAlmostEqual(row["spearman"], 1.0, places=12)
        self.assertAlmostEqual(row["normalized_stress"], 0.0, places=12)

    def test_normalized_stress_is_invariant_to_uniform_scale(self):
        X, labels = make_clusters()
        self.assertAlmostEqual(metrics(X, 7.5 * X, labels)["normalized_stress"], 0.0, places=12)
        self.assertAlmostEqual(rho(X, 7.5 * X), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
