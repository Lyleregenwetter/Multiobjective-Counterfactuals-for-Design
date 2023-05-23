import unittest

import numpy as np
import pandas as pd

from data_package import DataPackage


class DataPackageTest(unittest.TestCase):

    def test_initialize_valid_package(self):
        self.assertIsNotNone(self.initialize())

    def test_get_features_to_freeze(self):
        self.assertEqual(["z"], self.initialize().features_to_freeze)

    def test_invalid_query_x(self):
        # noinspection PyTypeChecker
        self.assert_raises_with_message(
            lambda: self.initialize(query_x=None),
            "Query x is neither a dataframe nor an ndarray!")
        # noinspection PyTypeChecker
        self.assert_raises_with_message(
            lambda: self.initialize(query_x={}),
            "Query x is neither a dataframe nor an ndarray!")
        self.assert_raises_with_message(
            lambda: self.initialize(query_x=pd.DataFrame()),
            "Query x cannot be empty!")
        self.assert_raises_with_message(
            lambda: self.initialize(query_x=pd.DataFrame(np.array([[1]]), columns=["x"])),
            "Dimensional mismatch between query x and dataset!")
        self.assert_raises_with_message(
            lambda: self.initialize(query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "zz"])),
            "Query x columns do not match dataset columns!")

    def test_raises_when_index_out_of_bound(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]]), features_to_vary=[5]),
            "Invalid index provided in list of features to vary")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]]), query_y={5: (3, 10)}),
            "Invalid index provided in query y")

    def test_raises_when_index_index_not_int(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]]), features_to_vary=[5.4]),
            "Invalid index provided in list of features to vary")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]]), query_y={5.4: (3, 10)}),
            "Invalid index provided in query y")

    def test_raises_meaningful_exception_when_inconsistent(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]])),
            "The list of features to vary must be a list of indices when the features dataset is a numpy array")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]])),
            "Query y must contain indices when the predictions dataset is a numpy array")

    # noinspection PyUnresolvedReferences
    def test_initialize_with_numpy_arrays(self):
        features = np.array([[1, 2, 3], [4, 5, 6]])
        predictions = np.array([[1, 2], [3, 4]])
        data_package = self.initialize(features_dataset=features, features_to_vary=[0, 1],
                                       query_x=np.array([[1, 2, 3]]),
                                       predictions_dataset=predictions,
                                       query_y={0: (5, 10), 1: (10, 15)})
        self.assertTrue((data_package.features_dataset.to_numpy() == features).all())
        self.assertTrue((data_package.predictions_dataset.to_numpy() == predictions).all())
        self.assertIs(pd.DataFrame, type(data_package.features_dataset))
        self.assertIs(pd.DataFrame, type(data_package.predictions_dataset))
        self.assertTrue((np.array([0, 1, 2]) == data_package.features_dataset.columns).all())
        self.assertTrue((np.array([0, 1]) == data_package.predictions_dataset.columns).all())

    @unittest.skip
    def test_validate_bonus_objs(self):
        """Bug: query_y is being passed to validate_bonus_obj, and even if not behavior still buggy"""
        pass

    def test_initialize_with_no_features_to_vary(self):
        self.assert_raises_with_message(lambda: self.initialize(features_to_vary=[]),
                                        "User has not provided any features to vary")

    def test_initialize_with_invalid_features_to_vary(self):
        self.assert_raises_with_message(lambda: self.initialize(features_to_vary=["NON_EXISTENT"]),
                                        "Expected label NON_EXISTENT to be in dataset ['x' 'y' 'z']")

    def test_initialize_with_no_targets(self):
        self.assert_raises_with_message(lambda: self.initialize(query_y={}),
                                        "User has not provided any performance targets")

    def test_initialize_with_invalid_targets(self):
        self.assert_raises_with_message(lambda: self.initialize(query_y={"NON_EXISTENT": (4, 10)}),
                                        "Expected label NON_EXISTENT to be in dataset ['A' 'B']")

    def test_data_package_with_mismatch(self):
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=pd.DataFrame(np.array([[5, 4]]), columns=["A", "B"])),
            "Dimensional mismatch between provided datasets")

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(AssertionError) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])

    def initialize(self,
                   features_dataset=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["x", "y", "z"]),
                   predictions_dataset=pd.DataFrame(np.array([[5, 4], [3, 2]]), columns=["A", "B"]),
                   query_x=pd.DataFrame(np.array([[1, 2, 3]]), columns=["x", "y", "z"]),
                   query_y=None,
                   features_to_vary=None,
                   datatypes=None):
        if datatypes is None:
            datatypes = []
        if features_to_vary is None:
            features_to_vary = ["x", "y"]
        if query_y is None:
            query_y = {"A": (4, 10)}
        return DataPackage(features_dataset, predictions_dataset, query_x, features_to_vary, query_y, datatypes)
