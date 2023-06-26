import unittest

import numpy.testing as np_test

from decode_mcd_private.stats_methods import *


class StatsMethodsTest(unittest.TestCase):
    def test_euclidean_distance(self):
        x1 = [[1, 2, 5], [2, 4, 5], [1, 3, 6], [1, 1, 1]]
        reference = [[1, 1, 1]]
        design_distances = euclidean_distance(to_dataframe(np.array(x1)), to_dataframe(np.array(reference)))
        np_test.assert_array_almost_equal(design_distances,
                                          np.array([17 ** 0.5, 26 ** 0.5, 29 ** 0.5, 0]), decimal=5)

    def test_changed_features(self):
        x1 = [[1, 2, 5], [2, 4, 5], [1, 3, 6]]
        x2 = [[1, 3, 6]]
        changes = changed_features_ratio(to_dataframe(np.array(x1)), to_dataframe(np.array(x2)), 3)
        np_test.assert_array_almost_equal(changes, np.array([2 / 3, 1, 0]), decimal=5)

    def test_gower_distance_with_different_dimensions(self):
        x1 = np.array([[5, 10, 3], [5, 10, 3]])
        x2 = np.array([[6, 10, 3]])
        distance = gower_distance(to_dataframe(x1), to_dataframe(x2), np.array([10, 10, 10]))
        np_test.assert_array_almost_equal(distance, np.array([[0.033], [0.033]]), decimal=3)

    def test_to_df(self):
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataframe = to_dataframe(array)
        np_test.assert_equal(dataframe.values, array)
        np_test.assert_equal(dataframe.columns, np.array([0, 1, 2]))

    def test_average_gower_distance(self):
        """Do we really want to the shape to be (4,) instead of (4,1)?"""
        generated_dataset = np.array([
            [1.3, 1.4, 1.6],
            [1, 1, 1],
            [2.1, 2.2, 3],
            [1, 1, 1]
        ])
        original_dataset = np.array([
            [10, 10, 10],
            [0, 0, 0],
            [8, 8, 8],
            [1, 1, 1],
            [2, 2, 2],
        ])
        distance = avg_gower_distance(to_dataframe(generated_dataset),
                                      to_dataframe(original_dataset),
                                      np.array([10, 10, 10]),
                                      {"r": (0, 1, 2)},
                                      3)
        np_test.assert_array_almost_equal(distance, np.array([0.0811, 0.0667, 0.1433, 0.0667]), decimal=4)

    def test_categorical_gower(self):
        x1 = pd.DataFrame.from_records([{
            "x": 5,
            "y": 10,
            "z": 3
        }])
        original = pd.DataFrame.from_records([{
            "x": 12,
            "y": 10,
            "z": 3
        }])
        x1 = pd.concat([x1, x1], axis=0)
        result = categorical_gower(x1, original)
        np_test.assert_array_almost_equal(result, np.array([0.333, 0.333]), decimal=3)

    def test_gower_distance(self):
        x1 = pd.DataFrame.from_records([
            {"x": 5,
             "y": 10,
             "z": 3}
        ])
        x2 = pd.DataFrame.from_records([
            {"x": 6,
             "y": 10,
             "z": 3}
        ])
        self.assertAlmostEqual(0.033,
                               gower_distance(x1, x2, np.array([10, 10, 10]))[0][0],
                               places=3
                               )

    def test_k_edge_case(self):
        """By default, we use the k=3 nearest neighbors in average gower calculations.
        We should not get exceptions when len(dataset) < 3"""
        x1 = np.array([[i + j for i in range(1, 7)] for j in range(1, 6)])
        small_x2 = np.array([[i + j for i in range(1, 7)] for j in range(5, 6)])
        x1 = pd.DataFrame.from_records(x1)
        small_x2 = pd.DataFrame.from_records(small_x2)
        data_types = {"r": (0, 1, 3, 5), "c": (2, 4)}
        results = avg_gower_distance(x1, small_x2, np.array([5, 1, 10, 20]), data_types)
        self.assertIsNotNone(results)

    def test_high_dimensional_mixed_gower(self):
        x1 = np.array([[i + j for i in range(1, 7)] for j in range(1, 6)])
        x2 = np.array([[i + j for i in range(1, 7)] for j in range(5, 8)])
        x1 = pd.DataFrame.from_records(x1)
        x2 = pd.DataFrame.from_records(x2)
        data_types = {"r": (0, 1, 3, 5), "c": (2, 4)}
        results = mixed_gower(x1, x2, np.array([5, 1, 10, 20]), data_types)
        self.assertIsNotNone(results)

    def test_mixed_gower_full(self):
        x1 = pd.DataFrame.from_records(np.array([[15., 0, 20., 500], [15., 1, 25., 500], [100., 2, 50., 501]]))
        x2 = pd.DataFrame.from_records(np.array([[15., 0, 20., 500], [16., 1, 25., 5000]]))
        datatypes = {"r": (0, 2), "c": (1, 3)}
        ranges = np.array([10, 5])
        distance = mixed_gower(x1, x2, ranges, datatypes)
        np_test.assert_equal(distance, np.array([[0, 0.775], [0.5, 0.275], [4.125, 3.85]]))

    def test_mixed_gower_same_as_gower_when_all_real(self):
        features_dataset = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [12, 13, 15]]), columns=["x", "y", "z"])
        ranges = pd.Series({
            "x": 11,
            "y": 11,
            "z": 12})

        features = pd.concat([features_dataset, pd.DataFrame(np.array([[1, 2, 3]]), columns=['x', 'y', 'z'])],
                             axis=0)
        distance = gower_distance(features, features_dataset.iloc[0], ranges.values)
        mixed_distance = mixed_gower(features,
                                     features_dataset.iloc[0:1],
                                     np.array(ranges), {"r": (0, 1, 2)})
        np_test.assert_equal(distance, mixed_distance)
