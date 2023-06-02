import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real, Choice

from data_package import DataPackage
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator as MOCFG


class DummyPredictor:
    def predict(self, x):
        return np.arange(x.shape[0] * 2).reshape(-1, 2)


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
        self.data_package = DataPackage(
            features_dataset=features,
            predictions_dataset=pd.DataFrame(np.random.rand(features.shape[0], 1), columns=["performance"]),
            query_x=features[0:1],
            features_to_vary=["x", "y", "z"],
            query_y={"performance": [0.75, 1]},
            bonus_objectives=[]
        )
        self.generator = MOCFG(
            data_package=self.data_package,
            predictor=lambda x: pd.DataFrame(),
            constraint_functions=[],
            datatypes=[Real(), Real(), Real()]
        )
        self.static_generator = MOCFG

    def test_evaluate_subset(self):
        regressor = self.build_generator(self.build_package(features_to_vary=["x", "y"]))
        out = {}
        regressor._evaluate(
            np.array([[12, 13], [14, 15], [16, 17], [16, 19]]), out, datasetflag=False)
        self.assertTrue("F" in out)
        self.assertTrue("G" in out)

    def test_evaluate_all_features(self):
        regressor = self.build_generator(self.build_package())
        out = {}
        regressor._evaluate(
            np.array([[12, 13, 15], [14, 15, 19], [16, 17, 25], [16, 17, 25]]), out, datasetflag=False
        )
        self.assertTrue("F" in out)
        self.assertTrue("G" in out)

    @unittest.skip
    def test_restrictions_applied_to_dataset_samples(self):
        assert False, "We need to implement a check that samples grabbed from the dataset, " \
                      "when passed through the predictor, meet the query targets"

    def test_get_mixed_constraint_full(self):
        """
        get_mixed_constraint_satisfaction(...) is stateless
        """
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(3)]))
        y = pd.DataFrame.from_records(np.array([
            [1, 200, 3, 500, 0.4, 0.6],
            [3, 250, 10, 550, 0.7, 0.3],
            [5, 300, 15, 500, 0.0, 1.0]
        ]))
        generator = self.build_generator(self.build_package())
        satisfaction = generator._get_mixed_constraint_satisfaction(x_full=x_full,
                                                                    y=y,
                                                                    x_constraint_functions=[],
                                                                    y_regression_constraints={
                                                                        0: (2, 6),
                                                                        2: (10, 16)
                                                                    },
                                                                    y_category_constraints={
                                                                        1: (200, 300),
                                                                        3: (550,)},
                                                                    y_proba_constraints={(4, 5): (5,)})
        np_test.assert_array_equal(satisfaction, np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
        ]))

    def build_generator(self, package):
        return MOCFG(data_package=package, predictor=DummyPredictor().predict, constraint_functions=[],
                     datatypes=[Real() for _ in range(len(package.features_dataset.columns))])

    def test_strict_inequality_of_regression_constraints(self):
        """this is the current behavior, but is it desired?"""
        self.test_get_mixed_constraint_satisfaction()

    def test_get_scores(self):
        """MOCFG()._get_scores() is not stateless """

        features = ["A", "B", "C", "D"]
        features_dataset = pd.DataFrame.from_records(
            np.array([
                [15, 500, 45.5, 1000],
                [34, 600, 23.3, 2000],
                [50, 500, 15.4, 2000],
                [12, 700, 0, 2000],
                [-50, 800, 9.645, 3000],
            ]), columns=features
        )
        predictions_dataset = pd.DataFrame.from_records(
            np.array([
                [15, 500, 50],
                [34, 600, 5000],
                [49, 500, 15.4],
                [12, 700, 17.9],
                [-10, 800, 12.255],
            ]), columns=["O1", "O2", "O3"]
        )
        datatypes = [Real(bounds=(-55, 55)),
                     Choice(options=(500, 600, 700, 800)),
                     Real(bounds=(-5, 50)),
                     Choice(options=(1000, 2000, 3000))]

        data_package = DataPackage(
            features_dataset=features_dataset,
            predictions_dataset=predictions_dataset,
            query_x=pd.DataFrame(np.array([[0, 600, 40, 2000]]), columns=features),
            query_y={"O1": (100, 500)},
            features_to_vary=features,
            bonus_objectives=["O2", "O3"]
        )
        generator = MOCFG(data_package, lambda x: x, [], datatypes)

        scores = generator._get_scores(x=pd.DataFrame(np.array([[25, 500, 45, 2000], [35, 700, 35, 3000]]),
                                                      columns=features),
                                       predictions=pd.DataFrame(np.array([[1, 2.35, 3.33], [1, 3.35, 4.45]]),
                                                                columns=["O1", "O2", "O3"]))
        np_test.assert_array_almost_equal(scores,
                                          np.array([[2.35, 3.33, 0.339, 0.75, 0.298],
                                                    [3.35, 4.45, 0.614, 1, 0.556]]), decimal=3)

    @unittest.skip
    def test_get_mixed_constraint_satisfaction_with_x_constraints(self):
        pass

    def test_get_mixed_constraint_satisfaction(self):
        """get_mixed_constraint_satisfaction() is stateless
        - the generator built and the data package don't matter"""
        y = pd.DataFrame.from_records(np.array([[1, 9], [2, 10],
                                                [3, 12], [3, 8],
                                                [4, 20], [5, 21]]))
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(6)]))
        generator = self.build_generator(self.build_package())
        satisfaction = generator._get_mixed_constraint_satisfaction(x_full=x_full,
                                                                    y=y,
                                                                    x_constraint_functions=[],
                                                                    y_regression_constraints={
                                                                        0: (2, 4),
                                                                        1: (10, 20)},
                                                                    y_category_constraints={},
                                                                    y_proba_constraints={})
        np_test.assert_equal(satisfaction, np.array([[1, 1], [1, 1], [0, 0], [0, 1], [1, 1], [1, 1]]))

    def build_package(self,
                      features_dataset=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [12, 13, 15]]),
                                                    columns=["x", "y", "z"]),
                      predictions_dataset=pd.DataFrame(np.array([[5, 4], [3, 2], [2, 1]]), columns=["A", "B"]),
                      query_x=pd.DataFrame(np.array([[5, 12, 15]]), columns=["x", "y", "z"]),
                      query_y=None,
                      bonus_objectives=None,
                      features_to_vary=None,
                      datatypes=None):
        datatypes = self.get_or_default(datatypes, [])
        features_to_vary = self.get_or_default(features_to_vary, ["x", "y", "z"])
        query_y = self.get_or_default(query_y, {"A": (4, 10)})
        bonus_objectives = self.get_or_default(bonus_objectives, [])
        return DataPackage(features_dataset=features_dataset,
                           predictions_dataset=predictions_dataset,
                           query_x=query_x,
                           features_to_vary=features_to_vary,
                           query_y=query_y,
                           bonus_objectives=bonus_objectives,
                           datatypes=datatypes)

    def get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value
