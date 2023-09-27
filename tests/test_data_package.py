import unittest

import numpy as np
import pandas as pd
import numpy.testing as np_test
from pymoo.core.variable import Real, Choice, Integer

from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets, ContinuousTarget, ClassificationTarget
from decode_mcd.mcd_exceptions import UserInputException


# noinspection PyTypeChecker
class DataPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_package = self.initialize()

    def test_initialize_valid_package(self):
        self.assertIsNotNone(self.valid_package)

    def test_get_features_to_freeze(self):
        self.assertEqual(["z"], self.valid_package.features_to_freeze)

    def test_invalid_features_dataset(self):
        self._test_invalid(
            {
                lambda: self.initialize(features_dataset=pd.DataFrame()): "features_dataset cannot be empty",
                lambda: self.initialize(features_dataset={}):
                    "features_dataset must either be a pandas dataframe or a numpy ndarray",
                lambda: self.initialize(features_dataset=np.array([])):
                    "features_dataset must be a valid numpy array (non-empty, 2D...)",
            })

    def test_invalid_predictions_dataset(self):
        self._test_invalid({
            lambda: self.initialize(predictions_dataset=pd.DataFrame()): "predictions_dataset cannot be empty",
            lambda: self.initialize(predictions_dataset={}):
                "predictions_dataset must either be a pandas dataframe or a numpy ndarray",
            lambda: self.initialize(predictions_dataset=np.array([])):
                "predictions_dataset must be a valid numpy array (non-empty, 2D...)",
            lambda: self.initialize(predictions_dataset=pd.DataFrame(np.array([[5, 4]]), columns=["A", "B"])):
                "features_dataset and predictions_dataset do not have the same number of rows (2, 1)",
        })

    def test_invalid_design_targets(self):
        self._test_invalid({
            lambda: self.initialize(design_targets=DesignTargets()):
                "Design targets must be provided",
            lambda: self.initialize(design_targets={}): "design_targets must be an instance of DesignTargets",
            lambda: self.initialize(design_targets=DesignTargets([ContinuousTarget("Z", 3, 10)])):
                "Invalid value in design_targets: expected columns ['Z'] "
                "to be in predictions_dataset columns ['A' 'B']",
            lambda: self.initialize(design_targets=DesignTargets([ContinuousTarget("A", 3, 10)],
                                                                 [ClassificationTarget("Z", (1, 2))])):
                "Invalid value in design_targets: expected columns ['Z'] "
                "to be in predictions_dataset columns ['A' 'B']"
        })

    def test_invalid_datatypes(self):
        self.assert_raises_with_message(
            lambda: self.initialize(datatypes=[]),
            "datatypes has length 0, expected length 3 matching features_dataset columns ['x' 'y' 'z']")
        self.assert_raises_with_message(
            lambda: self.initialize(datatypes=[Real(bounds=(5, 10)), None, Real(bounds=(11, 13))]),
            "datatypes must strictly be a sequence of objects belonging to the types [Real, Integer, Choice, Binary]")

    def test_subtle_invalid_datatypes(self):
        self._test_invalid(
            {
                lambda: self.initialize(datatypes=[Real(), Real(), Real()]):
                    "Parameter [datatypes] is invalid: bounds cannot be None for object of type "
                    "pymoo.core.variable.Real",
                lambda: self.initialize(datatypes=[Real(bounds=(1, 3)), Choice(), Real(bounds=(1, 3))]):
                    "Parameter [datatypes] is invalid: options cannot be None for object of type "
                    "pymoo.core.variable.Choice",
                lambda: self.initialize(datatypes=[Real(bounds=(1, 3)), Integer(), Real(bounds=(1, 3))]):
                    "Parameter [datatypes] is invalid: bounds cannot be None for object of type "
                    "pymoo.core.variable.Integer",
                lambda: self.initialize(datatypes=[Real(bounds=(1, 3)), Choice(), Real(bounds=(1, 3))]):
                    "Parameter [datatypes] is invalid: options cannot be None for object of type "
                    "pymoo.core.variable.Choice",
            }
        )

    def test_invalid_bonus_objectives(self):
        self.assert_raises_with_message(
            lambda: self.initialize(bonus_objectives=["NON_EXISTENT"]),
            "Bonus objectives should be a subset of labels!"
        )

    def test_invalid_features_to_vary(self):
        self._test_invalid({
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]])):
                "Invalid value in features_to_vary: expected columns ['x', 'y'] "
                "to be in features_dataset columns [0 1 2]",
            lambda:
            self.initialize(features_to_vary=["N"]):
                "Invalid value in features_to_vary: expected columns ['N'] "
                "to be in features_dataset columns ['x' 'y' 'z']",
            lambda: self.initialize(features_to_vary=[]):
                "features_to_vary cannot be an empty sequence"

        })

    def test_invalid_query_x(self):
        # noinspection PyTypeChecker
        self._test_invalid(
            {
                lambda: self.initialize(query_x=None):
                    "query_x must either be a pandas dataframe or a numpy ndarray",
                lambda: self.initialize(query_x={}):
                    "query_x must either be a pandas dataframe or a numpy ndarray",
                lambda: self.initialize(query_x=pd.DataFrame()):
                    "query_x cannot be empty",
                lambda: self.initialize(query_x=pd.DataFrame(np.array([[1]]), columns=["x"])):
                    "query_x must have 1 row and 3 columns",
                lambda: self.initialize(query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "zz"])):
                    "query_x columns do not match dataset columns!"
            }
        )

    def test_initialize_with_numpy_arrays(self):
        features = np.array([[1, 2, 3], [4, 5, 6]])
        predictions = np.array([[1, 2], [3, 4]])
        data_package = self.initialize(features_dataset=features, features_to_vary=[0, 1],
                                       query_x=np.array([[1, 2, 3]]),
                                       bonus_objectives=[0],
                                       predictions_dataset=predictions,
                                       design_targets=DesignTargets([
                                           ContinuousTarget(0, 5, 10),
                                           ContinuousTarget(1, 10, 15)]))
        np_test.assert_equal(data_package.features_dataset.to_numpy(), features)
        np_test.assert_equal(data_package.predictions_dataset.to_numpy(), predictions)
        self.assertIs(pd.DataFrame, type(data_package.features_dataset))
        self.assertIs(pd.DataFrame, type(data_package.predictions_dataset))
        self.assertEqual({0, 1, 2}, set(data_package.features_dataset.columns))
        self.assertEqual({0, 1}, set(data_package.predictions_dataset.columns))

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(UserInputException) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])

    def initialize(self,
                   features_dataset=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["x", "y", "z"]),
                   predictions_dataset=pd.DataFrame(np.array([[5, 4], [3, 2]]), columns=["A", "B"]),
                   query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "z"]),
                   design_targets=None,
                   features_to_vary=None,
                   bonus_objectives=None,
                   datatypes=None):
        datatypes = self.get_or_default(datatypes, [Real(1, 4), Real(2, 5), Real(3, 6)])
        features_to_vary = self.get_or_default(features_to_vary, ["x", "y"])
        design_targets = self.get_or_default(design_targets, DesignTargets([ContinuousTarget("A", 4, 10)]))
        bonus_objectives = self.get_or_default(bonus_objectives, ["A"])
        return DataPackage(features_dataset=features_dataset,
                           predictions_dataset=predictions_dataset,
                           query_x=query_x,
                           features_to_vary=features_to_vary,
                           design_targets=design_targets,
                           bonus_objectives=bonus_objectives,
                           datatypes=datatypes)

    def _test_invalid(self, invalid_scenarios: dict):
        for factory, exception_message in invalid_scenarios.items():
            with self.subTest(f"Case: {exception_message} ..."):
                self.assert_raises_with_message(factory, exception_message)

    def get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value
