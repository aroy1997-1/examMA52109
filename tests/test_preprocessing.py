import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):

    # -----------------------------------------------------------
    # Test 1: Feature selection should preserve requested order.
    # Why this matters:
    #   A real bug in clustering pipelines is that reordering
    #   columns causes the distance computations to be wrong.
    #   This test ensures select_features NEVER reorders columns.
    # -----------------------------------------------------------
    def test_select_features_preserves_order(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9]
        })

        requested = ["c", "a"]
        X_df = select_features(df, requested)

        self.assertEqual(list(X_df.columns), requested)
        self.assertTrue(np.array_equal(X_df.values[:, 0], df["c"].values))
        self.assertTrue(np.array_equal(X_df.values[:, 1], df["a"].values))


    # -----------------------------------------------------------
    # Test 2: select_features should reject non-numeric columns.
    # Why this matters:
    #   If a user accidentally selects a string column, clustering
    #   will later fail with cryptic NumPy errors. This test checks
    #   select_features raises the correct TypeError early.
    # -----------------------------------------------------------
    def test_select_features_rejects_non_numeric(self):
        df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": ["red", "blue", "green"]   # non-numeric
        })

        with self.assertRaises(TypeError):
            select_features(df, ["x", "y"])


    # -----------------------------------------------------------
    # Test 3: standardisation should return correct shape and
    #         behave as an actual StandardScaler transform.
    # Why this matters:
    #   A common real bug is returning a Python list, or changing
    #   shape, or producing non-zero mean after scaling.
    # -----------------------------------------------------------
    def test_standardise_features_correctness(self):
        X = np.array([[1., 10.],
                      [2., 20.],
                      [3., 30.]])

        X_scaled = standardise_features(X)

        # Same shape must be preserved
        self.assertEqual(X_scaled.shape, X.shape)

        # Column means should be approximately 0
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7))

        # Column stds should be approximately 1
        self.assertTrue(np.allclose(X_scaled.std(axis=0), 1, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
