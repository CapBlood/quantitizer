import unittest
from unittest.mock import patch

import numpy as np
from gensim.models.fasttext import FastTextKeyedVectors

from quantitizer.integration.gensim.fasttext import quantitize_ft
from quantitizer._pq_array import PQ


class TestFastText(unittest.TestCase):
    def setUp(self):
        self.ft = FastTextKeyedVectors(
            10, 0, 10, 10)
        self.ft.vectors = np.array([])
        self.ft.vectors_ngrams = np.array([])

    @patch('quantitizer.integration.gensim.fasttext.quantitize')
    def test_quantitize_ft(self, patch_quantitize):
        pq = PQ(
            0, 0, 0, 0, np.array([1, 2]), np.array([1, 2]))
        patch_quantitize.return_value = pq

        q_ft = quantitize_ft(self.ft, 2)

        self.assertTrue((q_ft.vectors == pq).all())
        self.assertTrue((q_ft.vectors_ngrams == pq).all())
