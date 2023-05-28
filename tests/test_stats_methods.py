import unittest

import numpy.testing as np_test

import pandas_utility as pd_util
from stats_methods import *


class StatsMethodsTest(unittest.TestCase):
    def test_euclidean_distance(self):
        x1 = [[1, 2, 5], [2, 4, 5], [1, 3, 6]]
        reference = [[1, 1, 1]]
        design_distances = np_euclidean_distance(np.array(x1), np.array(reference))
        self.assertAlmostEqual(17 ** 0.5, design_distances[0], places=5)
        self.assertAlmostEqual(29 ** 0.5, design_distances[2], places=5)
        self.assertEqual(3, len(design_distances))

    def test_changed_features(self):
        x1 = [[1, 2, 5], [2, 4, 5], [1, 3, 6]]
        x2 = [[1, 3, 6]]
        changes = np_changed_features_ratio(np.array(x1), np.array(x2), 3)
        self.assertAlmostEqual(2 / 3, changes[0], places=5)
        self.assertEqual(1, changes[1])
        self.assertEqual(0, changes[2])
        self.assertEqual(3, len(changes))

    def test_gower_distance_with_different_dimensions(self):
        x1 = np.array([[5, 10, 3], [5, 10, 3]])
        x2 = np.array([[6, 10, 3]])
        distance = np_gower_distance(x1, x2, np.array([10, 10, 10]))
        self.assertEqual((2, 1), distance.shape)
        self.assertAlmostEqual(0.033, distance[0][0], places=3)
        self.assertAlmostEqual(0.033, distance[1][0], places=3)

    def test_to_df(self):
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dataframe = to_dataframe(array)
        self.assertEqual(1, dataframe.loc[0].loc[0])
        self.assertEqual((3, 3), dataframe.shape)

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
        distance = np_avg_gower_distance(generated_dataset, original_dataset,
                                         np.array([10, 10, 10]), 3)
        self.assertEqual((4,), distance.shape)
        self.assertAlmostEqual(0.0811, distance[0], places=4)
        self.assertAlmostEqual(0.0667, distance[1], places=4)
        self.assertAlmostEqual(0.1433, distance[2], places=4)

    def test_categorical_gower(self):
        x1 = pd_util.get_one_row_dataframe_from_dict({
            "x": 5,
            "y": 10,
            "z": 3
        })
        original = pd_util.get_one_row_dataframe_from_dict({
            "x": 12,
            "y": 10,
            "z": 3
        })
        x1 = pd.concat([x1, x1], axis=0)
        result = categorical_gower(x1, original)
        self.assertAlmostEqual(0.333,
                               result[0],
                               places=3
                               )

    def test_np_gower_distance(self):
        x1 = pd_util.get_one_row_dataframe_from_dict({
            "x": 5,
            "y": 10,
            "z": 3
        })
        x2 = pd_util.get_one_row_dataframe_from_dict({
            "x": 6,
            "y": 10,
            "z": 3
        })
        self.assertAlmostEqual(0.033,
                               np_gower_distance(x1.values, x2.values, np.array([10, 10, 10]))[0][0],
                               places=3
                               )

    def test_gower_distance(self):
        x1 = pd_util.get_one_row_dataframe_from_dict({
            "x": 5,
            "y": 10,
            "z": 3
        })
        x2 = pd_util.get_one_row_dataframe_from_dict({
            "x": 6,
            "y": 10,
            "z": 3
        })
        self.assertAlmostEqual(0.033,
                               gower_distance(x1, x2, np.array([10, 10, 10]))[0][0],
                               places=3
                               )

    def test_k_edge_case(self):
        x1 = np.array([[i + j for i in range(1, 7)] for j in range(1, 6)])
        x2 = np.array([[i + j for i in range(1, 7)] for j in range(5, 6)])
        x1 = pd.DataFrame.from_records(x1)
        x2 = pd.DataFrame.from_records(x2)
        data_types = {"r": (0, 1, 3, 5), "c": (2, 4)}
        results = avg_gower_distance(x1, x2, np.array([5, 1, 10, 20]), data_types)
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
