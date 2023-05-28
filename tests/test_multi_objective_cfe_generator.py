import unittest

import numpy as np
import pandas as pd
from pymoo.core.variable import Real, Integer, Choice
import numpy.testing as np_test

from data_package import DataPackage
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator as MOCFG


class FakePredictor:

    def predict(self, data):
        return pd.DataFrame(np.sum(data, axis=1), columns=["performance"])


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
        predictor = FakePredictor()
        predictions = predictor.predict(features)
        self.data_package = DataPackage(
            features_dataset=features,
            predictions_dataset=predictions,
            query_x=features[0:1],
            features_to_vary=["x", "y", "z"],
            query_y={"performance": [0.75, 1]},
            bonus_objectives=[]
        )
        self.generator = MOCFG(
            data_package=self.data_package,
            predictor=predictor.predict,
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

    @unittest.skip
    def test_type_inference(self):
        data = pd.DataFrame([[1, 3, "false"], [45, 23.0, "true"]])
        # noinspection PyTypeChecker
        inferred_types = MOCFG.infer_if_necessary(None, data)
        self.assertEqual(inferred_types[0], Integer(bounds=(1, 45)))
        self.assertIs(inferred_types[1], Real(bounds=(1, 23)))
        self.assertIs(inferred_types[2], Choice("true", "false"))

    def test_get_mixed_constraint_full(self):
        """

        """
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(3)]))
        y = pd.DataFrame.from_records(np.array([
            [1, 200, 3, 500, 0.4, 0.6],
            [3, 250, 10, 550, 0.7, 0.3],
            [5, 300, 15, 500, 0.0, 1.0]
        ]))
        generator = self.build_generator(self.build_package())
        satisfaction = generator.get_mixed_constraint_satisfaction(x_full=x_full,
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
        np_test.assert_array_almost_equal(satisfaction, np.array([
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

    def test_get_mixed_constraint_satisfaction(self):
        """This does not test the use of constraint functions - hence the dummy x_full"""
        y = pd.DataFrame.from_records(np.array([[1, 9], [2, 10],
                                                [3, 12], [3, 8],
                                                [4, 20], [5, 21]]))
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(6)]))
        satisfaction = self.build_generator(self.build_package()).get_mixed_constraint_satisfaction(x_full=x_full,
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
                      features_to_vary=None,
                      datatypes=None):
        if datatypes is None:
            datatypes = []
        if features_to_vary is None:
            features_to_vary = ["x", "y", "z"]
        if query_y is None:
            query_y = {"A": (4, 10)}
        return DataPackage(features_dataset, predictions_dataset, query_x, features_to_vary, query_y, datatypes)
