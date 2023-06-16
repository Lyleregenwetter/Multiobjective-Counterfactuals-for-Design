import unittest

import numpy as np

from private.calculate_dtai import calculateDTAI


class CalculateDtaiTest(unittest.TestCase):
    def test_calculate(self):
        self.assertEqual(
            0.8, calculateDTAI(
                np.array([[10, 5]]),
                "maximize",
                np.array([10, 5]),
                np.array([1, 1]),
                np.array([4, 4]))
        )
