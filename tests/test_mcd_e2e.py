import __main__
import os
import sys
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from autogluon.tabular import TabularDataset
from pymoo.core.variable import Real, Choice

import decode_mcd.multi_objective_problem as MOP
from decode_mcd import counterfactuals_generator
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets, ContinuousTarget, ClassificationTarget, ProbabilityTarget
from tests.alt_multi_label_predictor import MultilabelPredictor

sys.path.append(os.path.dirname(__file__))


def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


class McdEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        __main__.alt_multi_label_predictor = MultilabelPredictor
        if not self.grab_trained_model_path():
            self.train_model()
        self.model = MultilabelPredictor.load(self.grab_trained_model_path())
        self.x, self.y = self.load_toy_x_y()

    def predict_dummy_multiple_objectives(self, x):
        predictions = self.predict(x)
        predictions["O_R2"] = predictions["O_R1"]
        predictions["O_C2"] = predictions["O_C1"]
        return predictions

    def test_multi_objectives_with_subset_of_features_to_vary(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12), ContinuousTarget("O_R2", 0, 6)],
            [ClassificationTarget("O_C1", (1, 2)), ClassificationTarget("O_C2", (1,))],
            [ProbabilityTarget(("O_P1", "O_P2"), ("O_P1",))]
        )
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[1:2], features_to_vary=["R1", "R2", "R3", "R4", "R5"],
                         design_targets=targets, datatypes=datatypes)

        problem = MOP.MultiObjectiveProblem(data_package=dp,
                                            prediction_function=self.predict_dummy_multiple_objectives,
                                            constraint_functions=[])
        generator = counterfactuals_generator.CounterfactualsGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         num_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_classification_target_met(cfs, "O_C1", [1])
        self.assert_proba_target_met(cfs, "O_P1")
        self.assert_cfs_within_valid_range(cfs)

    def test_regression_only_query_y(self):
        x, y = self.x, self.y.drop(columns=self.y.columns.difference(["O_R1"]))
        datatypes = self.build_toy_x_datatypes()
        targets = DesignTargets(
            [ContinuousTarget("O_R1", -5, 5)]
        )
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[0:1], features_to_vary=x.columns,
                         design_targets=targets, datatypes=datatypes, bonus_objectives=["O_R1"])
        problem = MOP.MultiObjectiveProblem(data_package=dp,
                                            prediction_function=lambda any_x: self.predict_subset(["O_R1"],
                                                                                                  any_x),
                                            constraint_functions=[])
        generator = counterfactuals_generator.CounterfactualsGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_weights(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([[1]]),
                                            include_dataset=False)
        self.assert_regression_target_met(cfs, "O_R1", -5, 5)
        self.assert_cfs_within_valid_range(cfs)

    def test_mixed_type_targets(self):
        x, y = self._build_dummy_multiple_objectives()
        datatypes = self.build_toy_x_datatypes()
        targets = DesignTargets(
            [ContinuousTarget("O_R1", 0, 12), ContinuousTarget("O_R2", 0, 6)],
            [ClassificationTarget("O_C1", (1, 2)), ClassificationTarget("O_C2", (1,))],
            [ProbabilityTarget(("O_P1", "O_P2"), ("O_P1",))]
        )
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[0:1], features_to_vary=x.columns,
                         design_targets=targets, datatypes=datatypes)

        problem = MOP.MultiObjectiveProblem(data_package=dp,
                                            prediction_function=self.predict_dummy_multiple_objectives,
                                            constraint_functions=[])
        generator = counterfactuals_generator.CounterfactualsGenerator(problem, 500, initialize_from_dataset=False)
        generator.generate(5)
        num_samples = 10
        cfs = generator.sample_with_dtai(num_samples, 0.5, 0.2, 0.5, 0.2, include_dataset=False,
                                         num_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 6)
        self.assert_classification_target_met(cfs, "O_C1", [1])
        self.assert_proba_target_met(cfs, "O_P1")
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

    def assert_proba_target_met(self, cfs, desired_proba):
        proba_results = self.predict_subset(["O_P1", "O_P2"], cfs)
        other_proba = list({"O_P1", "O_P2"}.symmetric_difference({desired_proba}))[0]
        proba_satisfaction = np.greater(proba_results[desired_proba].values,
                                        proba_results[other_proba].values)
        np_test.assert_equal(proba_satisfaction, 1)

    def assert_classification_target_met(self, cfs: pd.DataFrame, label: str, desired_classes: list):
        classification_results = self.predict_subset([label], cfs).values
        satisfaction = np.isin(classification_results, desired_classes)
        np_test.assert_equal(satisfaction, 1)

    def assert_regression_target_met(self, cfs: pd.DataFrame, regression_target: str,
                                     lower_bound: float, upper_bound: float):
        regression_results = self.predict_subset([regression_target], cfs).values
        satisfaction = np.logical_and(np.greater(regression_results, lower_bound),
                                      np.less(regression_results, upper_bound))
        np_test.assert_equal(satisfaction, 1)

    def grab_trained_model_path(self):
        models_path_exists = "AutogluonModels" in os.listdir(os.path.dirname(__file__))
        if models_path_exists:
            return self.find_valid_model()

    def find_valid_model(self):
        trained_models = os.listdir(get_path("AutogluonModels"))
        for trained_model in trained_models:
            model_path = os.path.join(get_path("AutogluonModels"), trained_model)
            valid_model = "multilabel_predictor.pkl" in os.listdir(model_path)
            if valid_model:
                return model_path

    def train_model(self):
        training_predictor = MultilabelPredictor(labels=["O_C1", "O_R1", "O_P1", "O_P2"])
        x, y = self.load_toy_x_y()
        training_predictor.fit(TabularDataset(pd.concat([x, y], axis=1)))

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
