import unittest

import numpy as np
import pandas as pd
import numpy.testing as np_test

from data_package import DataPackage
from design_targets import DesignTargets, ContinuousTarget


class DataPackageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_package = self.initialize()

    def test_initialize_valid_package(self):
        self.assertIsNotNone(self.valid_package)

    def test_get_features_to_freeze(self):
        self.assertEqual(["z"], self.valid_package.features_to_freeze)

    @unittest.skip
    def test_validate_labels_in_all_design_targets(self):
        pass

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

    def test_raises_when_index_out_of_bounds(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]]), features_to_vary=[5]),
            """Invalid value in features_to_vary: expected columns [5] to be in features_dataset columns [0 1 2]""")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]]),
                                    design_targets=DesignTargets([ContinuousTarget(5, 3, 10)])),
            """Invalid value in design_targets: expected columns [5] to be in predictions_dataset columns [0 1]""")

    def test_raises_when_index_not_int(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]]), features_to_vary=[5.4]),
            """Invalid value in features_to_vary: expected columns [5.4] to be in features_dataset columns [0 1 2]""")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]]),
                                    design_targets=DesignTargets([ContinuousTarget(5.4, 3, 10)])),
            """Label must be of type string or an integer index""")

    def test_raises_meaningful_exception_when_inconsistent(self):
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=np.array([[1, 2, 3], [4, 5, 6]])),
            """Invalid value in features_to_vary: expected columns ['x', 'y'] to be in features_dataset columns [0 1 2]""")
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=np.array([[1, 2], [4, 5]])),
            """Invalid value in design_targets: expected columns ['A'] to be in predictions_dataset columns [0 1]""")

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

    def test_validate_bonus_objs(self):
        self.assert_raises_with_message(
            lambda: self.initialize(bonus_objectives=["NON_EXISTENT"]),
            "Bonus objectives should be a subset of labels!"
        )

    def test_initialize_with_no_features_to_vary(self):
        self.assert_raises_with_message(lambda: self.initialize(features_to_vary=[]),
                                        "features_to_vary cannot be an empty sequence")

    def test_initialize_with_invalid_features_to_vary(self):
        self.assert_raises_with_message(lambda: self.initialize(features_to_vary=["NON_EXISTENT"]),
                                        """Invalid value in features_to_vary: expected columns ['NON_EXISTENT'] to be in features_dataset columns ['x' 'y' 'z']""")

    def test_initialize_with_no_targets(self):
        self.assert_raises_with_message(lambda: self.initialize(design_targets=DesignTargets()),
                                        "Design targets must be provided")

    def test_initialize_with_invalid_targets(self):
        self.assert_raises_with_message(lambda: self.initialize(
            design_targets=DesignTargets([ContinuousTarget("NON_EXISTENT", 3, 10)])),
                                        """Invalid value in design_targets: expected columns ['NON_EXISTENT'] to be in predictions_dataset columns ['A' 'B']""")

    def test_data_package_with_mismatch(self):
        self.assert_raises_with_message(
            lambda: self.initialize(predictions_dataset=pd.DataFrame(np.array([[5, 4]]), columns=["A", "B"])),
            "Dimensional mismatch between provided datasets")
        self.assert_raises_with_message(
            lambda: self.initialize(features_dataset=pd.DataFrame(np.array([[5, 4, 3]]), columns=["x", "y", "z"])),
            "Dimensional mismatch between provided datasets")

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(ValueError) as context:
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
        datatypes = self.get_or_default(datatypes, [])
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

    def get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value
