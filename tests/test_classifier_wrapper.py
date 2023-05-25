import unittest

import numpy as np
import numpy.testing as np_test

from classifier_wrapper import ClassifierWrapper


class ClassifierWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = ClassifierWrapper()

    def test_evaluate_simple_percentages(self):
        evaluation = self.classifier.evaluate_proba(np.array([
            [0.4, 0.3, 0.3],
            [0.25, 0.25, 0.5],
            [0.2, 0.8, 0]]),
            (0, 1))
        expected = np.array([[0.1], [-0.25], [0.8]])
        np_test.assert_array_almost_equal(expected, evaluation)

    def test_evaluate_nd_score(self):
        evaluation = self.classifier.evaluate_categorical(
            np.array([['A', 'O'], ['B', 'K'], ['A', 'I'], ['C', 'N'], ['A', 'L']]),
            targets=np.array([['A', 'D', 'F'], ['N', 'I']], dtype='object'))
        np_test.assert_equal(evaluation, np.array([
            [1, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0]]))

    def test_simple_score(self):
        evaluation = self.classifier.evaluate_categorical(np.array([['A'], ['B'], ['D'], ['C'], ['A']]),
                                                          targets=np.array([['A', 'D', 'F']], dtype='object'))
        np_test.assert_equal(evaluation, (np.array([
            [1],
            [0],
            [1],
            [0],
            [1],
        ])))

    def test_dimensional_mismatch(self):
        with self.assertRaises(AssertionError) as context:
            self.classifier.evaluate_categorical(
                np.array([[1, 2, 3], [1, 2, 3]]),
                np.array([[1]])
            )
        self.assertEqual("Dimensional mismatch between actual performances and targets array",
                         context.exception.args[0])
