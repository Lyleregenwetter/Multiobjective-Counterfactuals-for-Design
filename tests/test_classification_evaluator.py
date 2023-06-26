import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from decode_mcd_private.classification_evaluator import ClassificationEvaluator


class ClassifierWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = ClassificationEvaluator()

    def test_evaluate_empty(self):
        dataframe = pd.DataFrame()
        categorical = self.classifier.evaluate_categorical(dataframe, targets=np.array([]))
        proba = self.classifier.evaluate_proba(dataframe, np.array([]))
        np_test.assert_equal(categorical, np.array([]))
        np_test.assert_equal(proba, np.array([]))

    def test_evaluate_simple_percentages(self):
        array = np.array([[0.4, 0.3, 0.3], [0.25, 0.25, 0.5], [0.2, 0.8, 0]])
        evaluation = self.classifier.evaluate_proba(pd.DataFrame.from_records(array),
                                                    (0, 1))
        expected = np.array([[0.1], [-0.25], [0.8]])
        np_test.assert_array_almost_equal(expected, evaluation)

    def test_evaluate_nd_score(self):
        array = np.array([['A', 'O'], ['B', 'K'], ['A', 'I'], ['C', 'N'], ['A', 'L']])
        evaluation = self.classifier.evaluate_categorical(
            pd.DataFrame.from_records(array),
            targets=np.array([['A', 'D', 'F'], ['N', 'I']], dtype='object'))
        np_test.assert_equal(evaluation, np.array([
            [1, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0]]))

    def test_simple_score(self):
        array = np.array([['A'], ['B'], ['D'], ['C'], ['A']])
        evaluation = self.classifier.evaluate_categorical(pd.DataFrame.from_records(array),
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
            array = np.array([[1, 2, 3], [1, 2, 3]])
            self.classifier.evaluate_categorical(
                pd.DataFrame.from_records(array),
                np.array([[1]])
            )
        self.assertEqual("Dimensional mismatch between actual performances and targets array",
                         context.exception.args[0])
