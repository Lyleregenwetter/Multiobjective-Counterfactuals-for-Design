import unittest

import numpy as np
import pandas as pd
from pymoo.core.variable import Real, Integer, Choice

import pandas_utility as pd_util
from data_package import DataPackage
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator, CFSet
from stat_methods import np_euclidean_distance, np_changed_features_ratio, np_gower_distance, np_avg_gower_distance, \
    gower_distance, to_dataframe, categorical_gower


class FakePredictor:

    def predict(self, data):
        return pd.DataFrame(np.sum(data, axis=1), columns=["performance"])


class MultiObjectiveCFEGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        features = pd.DataFrame(np.random.rand(100, 3), columns=["x", "y", "z"])
        features.loc[0] = pd.Series({
            "x": 10,
            "y": 10,
            "z": 10
        })
        features.loc[1] = pd.Series({
            "x": 0,
            "y": 0,
            "z": 0
        })
        predictor = FakePredictor()
        predictions = predictor.predict(features)
        self.data_package = DataPackage(
            features_dataset=features,
            predictions_dataset=predictions,
            query_x=features[0:1],
            features_to_vary=["x", "y", "z"],
            query_y={"performance": [0.75, 1]},
        )
        self.generator = MultiObjectiveCounterfactualsGenerator(
            data_package=self.data_package,
            predictor=predictor.predict,
            bonus_objs=[],
            constraint_functions=[],
            datatypes=[Real(), Real(), Real()]
        )
        self.static_generator = MultiObjectiveCounterfactualsGenerator

    @unittest.skip
    def test_restrictions_applied_to_dataset_samples(self):
        assert False, "We need to implement a check that samples grabbed from the dataset, " \
                      "when passed through the predictor, meet the query targets"

    def test_generator(self):
        s = CFSet(self.generator, 50)
        s.optimize(1)
        sample = s.sample(num_samples=5, avg_gower_weight=np.array([1]), cfc_weight=np.array([1]),
                          gower_weight=np.array([1]), dtai_beta=np.array([1]), dtai_alpha=np.array([3]),
                          diversity_weight=np.array([1]), dtai_target=np.array([2]))
        performances = FakePredictor().predict(sample.values)
        for performance in performances.values:
            self.assertGreaterEqual(performance, 0.75)
            self.assertLessEqual(performance, 1)

    def test_type_inference(self):
        data = pd.DataFrame([[1, 3, "false"], [45, 23.0, "true"]])
        # noinspection PyTypeChecker
        inferred_types = self.static_generator.infer_if_necessary(None, data)
        self.assertIs(inferred_types[0], Integer)
        self.assertIs(inferred_types[1], Real)
        self.assertIs(inferred_types[2], Choice)

    def test_concat_numpy_arrays(self):
        template_array = np.array([1, 0, 3])
        new_values = np.array([[5, 6, 7, 10]])
        result = self.generator.build_from_template(template_array, new_values, [1])
        self.assertEqual(4, result.shape[0])
        self.assertEqual(3, result.shape[1])
        self.assertEqual(7, result[2][1])

    def test_concat_multi_dimensional_numpy_arrays(self):
        template_array = np.array([1, 0, 3, 0])
        new_values = np.array([[5, 6, 7, 10, 12], [12, 13, 14, 15, 13]])
        result = self.generator.build_from_template(template_array, new_values, [1, 3])
        self.assertEqual(5, result.shape[0])
        self.assertEqual(4, result.shape[1])
        self.assertEqual(12, result[0][3])

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
        distance = np_gower_distance(x1, x2, self.generator.ranges.values)
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
        avg_gower_distance = np_avg_gower_distance(generated_dataset, original_dataset,
                                                   self.generator.ranges.values, 3)
        self.assertEqual((4,), avg_gower_distance.shape)
        self.assertAlmostEqual(0.0811, avg_gower_distance[0], places=4)
        self.assertAlmostEqual(0.0667, avg_gower_distance[1], places=4)
        self.assertAlmostEqual(0.1433, avg_gower_distance[2], places=4)

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
                               np_gower_distance(x1.values, x2.values, self.generator.ranges.values)[0][0],
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
                               gower_distance(x1, x2, self.generator.ranges.values)[0][0],
                               places=3
                               )
