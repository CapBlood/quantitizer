import unittest
from unittest.mock import patch

import numpy as np

from quantitizer import quantitize


class TestQuantitize(unittest.TestCase):
    @patch('quantitizer.kmeans2')
    def test_quantitize(self, patch_kmeans):
        centroid = [[1.0, 2.0], [3.0, 4.0]]
        label = [5, 6, 7, 8]

        patch_kmeans.return_value = [
            np.array(centroid),
            np.array(label)
        ]

        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        centroids = 256
        sub_dim = 2
        q_a = quantitize(a, sub_dim, n_cluster=centroids)

        labels = [
            [5, 6, 7, 8],
            [5, 6, 7, 8]
        ]

        self.assertEqual(q_a.vectors, len(a))
        self.assertEqual(q_a.centroids, centroids)
        self.assertTrue(
            (q_a.indexes == np.array(labels).T).all())
        self.assertTrue(
            (q_a.codes == np.array([centroid] * sub_dim)).all())
        self.assertEqual(q_a.dim, len(a[0]))
        self.assertEqual(q_a.qdim, len(a[0]) // sub_dim)
