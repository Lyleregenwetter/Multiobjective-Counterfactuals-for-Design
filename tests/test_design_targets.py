import unittest

import numpy.testing as np_test

from decode_mcd.design_targets import *
from decode_mcd.mcd_exceptions import UserInputException


class DesignTargetsTest(unittest.TestCase):
    def test_count_constrained_labels(self):
        targets = DesignTargets([ContinuousTarget("A", 12, 15), ContinuousTarget("B", 12, 15)],
                                [CategoricalTarget("C", (1, 2)), CategoricalTarget("D", (1, 2))])
        self.assertEqual(4, targets.count_constrained_labels())
        self.assertEqual(("A", "B"), targets.get_continuous_labels())
        self.assertEqual(("C", "D"), targets.get_categorical_labels())
        np_test.assert_equal(np.array([[12, 12], [15, 15]]), targets.get_continuous_boundaries())

    # noinspection PyTypeChecker
    def test_invalid_design_targets(self):
        self._test_invalid({
            lambda: DesignTargets({}): "Continuous targets must be a sequence",
            lambda: DesignTargets(categorical_targets=[None]):
                "Categorical targets must be composed of elements of class CategoricalTarget",
            lambda: DesignTargets(): "Design targets must be provided",
            lambda: DesignTargets(
                continuous_targets=[ContinuousTarget("A", 12, 15),
                                    ContinuousTarget("A", 12, 15)]): "Label was specified twice in targets",
            lambda: DesignTargets(
                continuous_targets=[ContinuousTarget("A", 12, 15)],
                categorical_targets=[CategoricalTarget("A", (1,))]): "Label was specified twice in targets",
        },
        )

    # noinspection PyTypeChecker
    def test_invalid_categorical_target(self):
        self._test_invalid({
            lambda: CategoricalTarget(None, (50,)): "Label must be of type string or an integer index",
            lambda: CategoricalTarget("", (50,)): "Label cannot be an empty string",
            lambda: CategoricalTarget("LABEL", ()): "Desired classes cannot be empty",
            lambda: CategoricalTarget("LABEL", ("A",)): "Desired classes must be an all-integer sequence",
        })

    # noinspection PyTypeChecker
    def test_invalid_minimization_target(self):
        self._test_invalid({
            lambda: MinimizationTarget({}): "Label must be of type string or an integer index",
            lambda: MinimizationTarget(""): "Label cannot be an empty string",
        })

    # noinspection PyTypeChecker
    def test_invalid_continuous_target(self):
        self._test_invalid({
            lambda: ContinuousTarget("LABEL", 50, None): "Upper bound must be a real number",
            lambda: ContinuousTarget("LABEL", None, 500): "Lower bound must be a real number",
            lambda: ContinuousTarget(None, 50, 500): "Label must be of type string or an integer index",
            lambda: ContinuousTarget("LABEL", 500, 50): "Lower bound cannot be greater or equal to upper bound",
            lambda: ContinuousTarget("LABEL", 50, 50): "Lower bound cannot be greater or equal to upper bound",
            lambda: ContinuousTarget("", 50, 500): "Label cannot be an empty string",
        })

    def test_valid_categorical_target(self):
        CategoricalTarget("LABEL", (5,))
        CategoricalTarget(15, (5,))

    def test_valid_continuous_target(self):
        ContinuousTarget("LABEL", 50, 500)
        ContinuousTarget(5, 50, 500)
        ContinuousTarget(5, 50.5, 500.15)

    def _test_invalid(self, map_of_cases):
        for factory, exception_message in map_of_cases.items():
            with self.subTest(f"Case: {exception_message} ..."):
                self.assert_raises_with_message(factory, exception_message)

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(UserInputException) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])
