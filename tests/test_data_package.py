import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real, Choice, Integer, Binary

from decode_mcd.data_package import McdDataset
from decode_mcd.design_targets import DesignTargets, ContinuousTarget, CategoricalTarget, MinimizationTarget
from decode_mcd.mcd_exceptions import UserInputException

DEFAULT_FEATURES = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["x", "y", "z"])

# noinspection PyTypeChecker
class DataPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_package = self.initialize()

    def test_initialize_valid_package(self):
        self.assertIsNotNone(self.valid_package)

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
            lambda: self._test_cross_validate(design_targets=DesignTargets()):
                "Design targets must be provided",
            lambda: self._test_cross_validate(design_targets={}): "design_targets must be an instance of DesignTargets",
            lambda: self._test_cross_validate(design_targets=DesignTargets([ContinuousTarget("Z", 3, 10)])):
                "Invalid value in design_targets: expected columns ['Z'] "
                "to be in predictions_dataset columns ['A' 'B']",
            lambda: self._test_cross_validate(design_targets=DesignTargets([ContinuousTarget("A", 3, 10)],
                                                                           [CategoricalTarget("Z", (1, 2))])):
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
            lambda: self._test_cross_validate(design_targets=DesignTargets(
                continuous_targets=[ContinuousTarget("A", 0, 10)],
                minimization_targets=[MinimizationTarget("NON_EXISTENT")]
            )
            ),
            "Minimization targets ['NON_EXISTENT'] do not exist in dataset columns ['A' 'B']"
        )

    def test_invalid_features_to_vary(self):
        self._test_invalid({
            lambda: self._test_cross_validate(features_dataset=np.array([[1, 2, 3], [4, 5, 6]])):
                "Invalid value in features_to_vary: expected columns ['x', 'y'] "
                "to be in features_dataset columns [0 1 2]",
            lambda:
            self._test_cross_validate(features_to_vary=["N"]):
                "Invalid value in features_to_vary: expected columns ['N'] "
                "to be in features_dataset columns ['x' 'y' 'z']",
            lambda: self._test_cross_validate(features_to_vary=[]):
                "features_to_vary cannot be an empty sequence"

        })

    def _test_cross_validate(self, query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "z"]),
                             design_targets=None,
                             features_to_vary=None,
                             features_dataset=None):
        features_dataset = self.get_or_default(features_dataset, DEFAULT_FEATURES)
        features_to_vary = self.get_or_default(features_to_vary, ["x", "y"])
        design_targets = self.get_or_default(design_targets, DesignTargets([ContinuousTarget("A", 4, 10)],
                                                                           minimization_targets=[
                                                                               MinimizationTarget("A")]))
        self.initialize(features_dataset=features_dataset).cross_validate(x_query=query_x, y_targets=design_targets, features_to_vary=features_to_vary)

    def test_invalid_query_x(self):
        # noinspection PyTypeChecker
        self._test_invalid(
            {
                lambda: self._test_cross_validate(query_x=None):
                    "query_x must either be a pandas dataframe or a numpy ndarray",
                lambda: self._test_cross_validate(query_x={}):
                    "query_x must either be a pandas dataframe or a numpy ndarray",
                lambda: self._test_cross_validate(query_x=pd.DataFrame()):
                    "query_x cannot be empty",
                lambda: self._test_cross_validate(query_x=pd.DataFrame(np.array([[1]]), columns=["x"])):
                    "query_x must have 1 row and 3 columns",
                lambda: self._test_cross_validate(
                    query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "zz"])):
                    "query_x columns do not match dataset columns!"
            }
        )

    def test_query_x_outside_of_datatypes_range(self):
        def build_problem_with_query_x_out_of_range():
            self._test_cross_validate(query_x=pd.DataFrame(np.array([[-110, -110, -110]]),
                                                           columns=["x", "y", "z"]
                                                           ))

        def build_problem_with_query_x_integer_out_of_range():
            x = pd.DataFrame(np.array([[-110, -110, -110]]),
                             columns=["x", "y", "z"]
                             )
            design_t = DesignTargets([ContinuousTarget("A", 4, 10)], minimization_targets=[MinimizationTarget("A")])
            self.initialize(datatypes=[Integer(bounds=(0, 5)) for _ in range(3)]).cross_validate(x_query=x,
                                                                                                 y_targets=design_t,
                                                                                                 features_to_vary=["x",
                                                                                                                   "y"])

        self.assert_raises_with_message(build_problem_with_query_x_out_of_range,
                                        "[query_x] parameters fall outside of range specified by datatypes")
        self.assert_raises_with_message(build_problem_with_query_x_integer_out_of_range,
                                        "[query_x] parameters fall outside of range specified by datatypes")

    def test_query_x_with_invalid_choices(self):
        def build_problem_with_invalid_choice_in_query_x():
            dp = self.initialize(
                datatypes=[Real(bounds=(-200, 200)), Choice(options=(-100, -110)), Real(bounds=(-200, 200))])
            dp.cross_validate(x_query=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "z"]),
                              y_targets=DesignTargets([ContinuousTarget("A", 4, 10)],
                                                      minimization_targets=[
                                                          MinimizationTarget("A")]),
                              features_to_vary=["x", "y"])

        self.assert_raises_with_message(build_problem_with_invalid_choice_in_query_x,
                                        "[query_x] has a choice variable that is not permitted by datatypes")

        def build_problem_with_invalid_binary_in_query_x():
            dp = self.initialize(datatypes=[Real(bounds=(-200, 200)), Binary(), Real(bounds=(-200, 200))])
            dp.cross_validate(x_query=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "z"]),
                              y_targets=DesignTargets([ContinuousTarget("A", 4, 10)],
                                                      minimization_targets=[
                                                          MinimizationTarget("A")]),
                              features_to_vary=["x", "y"])

        self.assert_raises_with_message(build_problem_with_invalid_choice_in_query_x,
                                        "[query_x] has a choice variable that is not permitted by datatypes")
        self.assert_raises_with_message(build_problem_with_invalid_binary_in_query_x,
                                        "[query_x] has a variable specified as binary by datatypes"
                                        " whose value is not True, False, 1, or 0")

    def test_initialize_with_numpy_arrays(self):
        features = np.array([[1, 2, 3], [4, 5, 6]])
        predictions = np.array([[1, 2], [3, 4]])
        data_package = self.initialize(features_dataset=features, predictions_dataset=predictions)
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
                   features_dataset=DEFAULT_FEATURES,
                   predictions_dataset=pd.DataFrame(np.array([[5, 4], [3, 2]]), columns=["A", "B"]),
                   datatypes=None):
        datatypes = self.get_or_default(datatypes, [Real(bounds=(1, 4)), Real(bounds=(2, 5)), Real(bounds=(3, 6))])
        return McdDataset(x=features_dataset,
                          y=predictions_dataset,
                          x_datatypes=datatypes)

    def _test_invalid(self, invalid_scenarios: dict):
        for factory, exception_message in invalid_scenarios.items():
            with self.subTest(f"Case: {exception_message} ..."):
                self.assert_raises_with_message(factory, exception_message)

    def get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value
