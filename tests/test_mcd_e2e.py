import os
import sys
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real, Choice
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

import decode_mcd.mcd_problem as MOP
from decode_mcd import McdGenerator
from decode_mcd.mcd_dataset import McdDataset
from decode_mcd.design_targets import DesignTargets, ContinuousTarget, CategoricalTarget, MinimizationTarget

INFINITY = 1_000_000_000

sys.path.append(os.path.dirname(__file__))


def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


class ToyModel:
    def __init__(self, x, y):
        self.regression_model = LinearRegression()
        self.regression_model.fit(X=x, y=pd.DataFrame(y, columns=["O_R1", "O_P1", "O_P2"]))
        self.classifier_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier_model.fit(X=x, y=pd.DataFrame(y, columns=["O_C1"]))

    def predict(self, x: pd.DataFrame):
        reg = self.regression_model.predict(x)
        classification = self.classifier_model.predict(x).reshape(len(x), 1)
        return pd.DataFrame(np.concatenate([reg, classification], axis=1), columns=["O_R1", "O_P1", "O_P2", "O_C1"])


class McdEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x, self.y = self.load_toy_x_y()
        self.model = ToyModel(self.x, self.y)

    def predict_dummy_multiple_objectives(self, x):
        predictions = self.predict(x)
        predictions["O_R2"] = predictions["O_R1"]
        predictions["O_C2"] = predictions["O_C1"]
        return predictions

    def test_run_with_x_constraints(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        y["PROB"] = y["O_P1"] - y["O_P2"]
        y["X_CONSTRAINT"] = self.x_constraint(x)
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12), ContinuousTarget("O_R2", 0, 6),
             ContinuousTarget("PROB", 0, INFINITY)],
            [CategoricalTarget("O_C1", (1, 2)), CategoricalTarget("O_C2", (1,)),
             CategoricalTarget("X_CONSTRAINT", (0,))],
        )
        dp = McdDataset(x=x, y=y,
                        x_datatypes=datatypes)

        def prediction_function(_x: pd.DataFrame):
            result = self.predict_dummy_multiple_objectives(_x)
            result["X_CONSTRAINT"] = self.x_constraint(_x)
            return result

        problem = MOP.McdProblem(mcd_dataset=dp,
                                 x_query=x.iloc[0:1],
                                 y_targets=targets,
                                 features_to_vary=x.columns,
                                 prediction_function=prediction_function)
        generator = McdGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         max_dpp=10000)

        self.assert_x_constraint_met(cfs)
        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_categorical_target_met(cfs, "O_C1", [1])
        self.assert_greater_than(cfs, "O_P1", "O_P2")
        self.assert_cfs_within_valid_range(cfs)

    def test_non_contiguous_objectives(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12),
             ContinuousTarget("O_R2", 0, 6)],
        )
        dp = McdDataset(x=x, y=y,
                        x_datatypes=datatypes)

        problem = MOP.McdProblem(mcd_dataset=dp,
                                 x_query=x.iloc[1:2],
                                 y_targets=targets,
                                 features_to_vary=["R1", "R2", "R3", "R4", "R5"],
                                 prediction_function=self.predict_dummy_multiple_objectives)
        generator = McdGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         max_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_cfs_within_valid_range(cfs)

    def test_multi_objectives_with_subset_of_features_to_vary(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        y["PROB"] = y["O_P1"] - y["O_P2"]
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12),
             ContinuousTarget("O_R2", 0, 6), ContinuousTarget("PROB", 0, INFINITY)],
            [CategoricalTarget("O_C1", (1, 2)), CategoricalTarget("O_C2", (1,))]
        )
        dp = McdDataset(x=x, y=y,
                        x_datatypes=datatypes)

        problem = MOP.McdProblem(mcd_dataset=dp,
                                 x_query=x.iloc[1:2],
                                 y_targets=targets,
                                 features_to_vary=["R1", "R2", "R3", "R4", "R5"],
                                 prediction_function=self.predict_dummy_multiple_objectives)
        generator = McdGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         max_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_categorical_target_met(cfs, "O_C1", [1])
        self.assert_greater_than(cfs, "O_P1", "O_P2")
        self.assert_cfs_within_valid_range(cfs)

    def test_regression_only_query_y(self):
        x, y = self.x, self.y.drop(columns=self.y.columns.difference(["O_R1"]))
        datatypes = self.build_toy_x_datatypes()
        targets = DesignTargets(
            [ContinuousTarget("O_R1", -5, 5)],
            minimization_targets=[MinimizationTarget("O_R1")]
        )
        dp = McdDataset(x=x, y=y,
                        x_datatypes=datatypes)
        problem = MOP.McdProblem(mcd_dataset=dp,
                                 x_query=x.iloc[0:1],
                                 y_targets=targets,
                                 features_to_vary=x.columns,
                                 prediction_function=lambda any_x: self.predict_subset(["O_R1"],
                                                                                                  any_x))
        generator = McdGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([1]),
                               include_dataset=False)
        self.assert_regression_target_met(cfs, "O_R1", -5, 5)
        self.assert_cfs_within_valid_range(cfs)

    def test_mixed_type_targets(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        y["PROB"] = y["O_P1"] - y["O_P2"]
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12),
             ContinuousTarget("O_R2", 0, 6), ContinuousTarget("PROB", 0, INFINITY)],
            [CategoricalTarget("O_C1", (1, 2)), CategoricalTarget("O_C2", (1,))],
        )
        dp = McdDataset(x=x, y=y,
                        x_datatypes=datatypes)

        problem = MOP.McdProblem(mcd_dataset=dp,
                                 x_query=x.iloc[0:1],
                                 y_targets=targets,
                                 features_to_vary=x.columns,
                                 prediction_function=self.predict_dummy_multiple_objectives)
        generator = McdGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         max_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_categorical_target_met(cfs, "O_C1", [1])
        self.assert_greater_than(cfs, "O_P1", "O_P2")
        self.assert_cfs_within_valid_range(cfs)

    def _build_dummy_multiple_objectives(self):
        x, y = self.x, self.y
        y["O_R2"] = y["O_R1"]
        y["O_C2"] = y["O_C1"]
        return x, y

    def build_toy_x_datatypes(self):
        datatypes = [Real(bounds=(self.x[feature].min(), self.x[feature].max())) for feature
                     in self.x.columns if "R" in feature]
        # noinspection PyTypeChecker
        datatypes.insert(3, Choice(options=tuple(self.x["C1"].unique())))
        return datatypes

    def assert_cfs_within_valid_range(self, cfs):
        regression_cfs = cfs.drop(columns=["C1"]).values
        all_within_range = np.logical_and(np.greater_equal(regression_cfs, self.x.min(numeric_only=True).values),
                                          np.less_equal(regression_cfs, self.x.max(numeric_only=True).values))
        np_test.assert_equal(all_within_range, 1)

    def assert_greater_than(self, cfs, desired_proba: str, other_proba: str):
        proba_results = self.predict_subset([desired_proba, other_proba], cfs)
        satisfaction = np.greater(proba_results[desired_proba].values,
                                  proba_results[other_proba].values)
        np_test.assert_equal(satisfaction, 1)

    def assert_categorical_target_met(self, cfs: pd.DataFrame, label: str, desired_classes: list):
        classification_results = self.predict_subset([label], cfs).values
        satisfaction = np.isin(classification_results, desired_classes)
        np_test.assert_equal(satisfaction, 1)

    def assert_regression_target_met(self, cfs: pd.DataFrame, regression_target: str,
                                     lower_bound: float, upper_bound: float):
        regression_results = self.predict_subset([regression_target], cfs).values
        satisfaction = np.logical_and(np.greater(regression_results, lower_bound),
                                      np.less(regression_results, upper_bound))
        np_test.assert_equal(satisfaction, 1)

    # noinspection PyTypeChecker
    def load_toy_x_y(self):
        y = pd.read_csv(get_path("toy_y.csv"))
        x = pd.read_csv(get_path("toy_x.csv"))
        y["O_C1"] = y["O_C1"].astype("category")
        x["C1"] = x["C1"].astype("category")
        return x, y

    def predict_subset(self, relevant_labels: list, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.predict(x), columns=relevant_labels)

    def predict(self, x):
        return self.model.predict(pd.DataFrame(x, columns=self.x.columns))

    def assert_x_constraint_met(self, cfs: pd.DataFrame):
        np_test.assert_array_equal(
            (cfs['R1'] <= cfs['R2']).astype('int32'),
            1
        )

    @staticmethod
    def x_constraint(x: pd.DataFrame):
        # noinspection PyUnresolvedReferences
        return (x['R1'] > x['R2']).astype('int32')
