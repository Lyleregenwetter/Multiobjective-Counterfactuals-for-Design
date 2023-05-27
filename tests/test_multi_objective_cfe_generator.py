import unittest

import numpy as np
import pandas as pd
from pymoo.core.variable import Real, Integer, Choice

import pandas_utility as pd_util
from data_package import DataPackage
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator, CFSet
from stats_methods import np_euclidean_distance, np_changed_features_ratio, np_gower_distance, np_avg_gower_distance, \
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
