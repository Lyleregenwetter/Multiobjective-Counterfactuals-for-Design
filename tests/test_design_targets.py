import unittest

import numpy.testing as np_test

from decode_mcd.design_targets import *
from decode_mcd.mcd_exceptions import UserInputException


class DesignTargetsTest(unittest.TestCase):
    def test_count_constrained_labels(self):
        targets = DesignTargets([ContinuousTarget("A", 12, 15), ContinuousTarget("B", 12, 15)],
                                [ClassificationTarget("C", (1, 2)), ClassificationTarget("D", (1, 2))],
                                [ProbabilityTarget(("E", "F"), ("E",)),
                                 ProbabilityTarget(("G", "H", "L"), ("H", "L"))])
        self.assertEqual(9, targets.count_constrained_labels())
        self.assertEqual(("A", "B"), targets.get_continuous_labels())
        self.assertEqual(("C", "D"), targets.get_classification_labels())
        self.assertEqual((("E", "F"), ("G", "H", "L")), targets.get_probability_labels())
        np_test.assert_equal(np.array([[12, 12], [15, 15]]), targets.get_continuous_boundaries())

    # noinspection PyTypeChecker
    def test_invalid_design_targets(self):
        self._test_invalid({
            lambda: DesignTargets({}): "Continuous targets must be a sequence",
            lambda: DesignTargets(classification_targets=[None]):
                "Classification targets must be composed of elements of class ClassificationTarget",
            lambda: DesignTargets(): "Design targets must be provided",
            lambda: DesignTargets(
                continuous_targets=[ContinuousTarget("A", 12, 15),
                                    ContinuousTarget("A", 12, 15)]): "Label was specified twice in targets",
            lambda: DesignTargets(
                continuous_targets=[ContinuousTarget("A", 12, 15)],
                classification_targets=[ClassificationTarget("A", (1,))]): "Label was specified twice in targets",
        },
        )

    # noinspection PyTypeChecker
    def test_invalid_classification_target(self):
        self._test_invalid({
            lambda: ClassificationTarget(None, (50,)): "Label must be of type string or an integer index",
            lambda: ClassificationTarget("", (50,)): "Label cannot be an empty string",
            lambda: ClassificationTarget("LABEL", ()): "Desired classes cannot be empty",
            lambda: ClassificationTarget("LABEL", ("A",)): "Desired classes must be an all-integer sequence",
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

    # noinspection PyTypeChecker
    def test_invalid_probability_target(self):
        self._test_invalid({
            lambda: ProbabilityTarget((1, 2), ()): "Preferred labels cannot be empty",
            lambda: ProbabilityTarget((1, 2), (3,)): "Preferred labels must be a subset of labels",
            lambda: ProbabilityTarget((1,), (1,)): "Labels must have a length greater than 1",
            lambda: ProbabilityTarget((1, 2), ("1",)): "Preferred labels must be a subset of labels",
            lambda: ProbabilityTarget((1, None),
                                      ("1",)): "Expected labels to be an all-integer or all-string sequence",
            lambda: ProbabilityTarget(("A", ""), ("1",)): "Labels cannot contain empty strings",
            lambda: ProbabilityTarget(("A", "B"), ("A", "")): "Preferred labels cannot contain empty strings",
            lambda: ProbabilityTarget((1, 2), (
                "1", None)): "Expected preferred labels to be an all-integer or all-string sequence",
        })

    def test_valid_probability_target(self):
        ProbabilityTarget(labels=("A", "B", "C"), preferred_labels=["A", "B"])
        ProbabilityTarget(labels=(5, 6, 7), preferred_labels=(5,))

    def test_valid_classification_target(self):
        ClassificationTarget("LABEL", (5,))
        ClassificationTarget(15, (5,))

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
