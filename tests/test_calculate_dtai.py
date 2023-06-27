import unittest

import numpy as np

from decode_mcd_private.calculate_dtai import calculateDTAI
import numpy.testing as np_test


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

    def test_empty_input(self):
        np_test.assert_equal(calculateDTAI(
            np.array([[], [], []]),
            "maximize",
            np.array([]),
            np.array([]),
            np.array([])), 0)
