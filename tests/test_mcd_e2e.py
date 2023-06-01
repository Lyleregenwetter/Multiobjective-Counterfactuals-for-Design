import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from autogluon.tabular import TabularDataset
from pymoo.core.variable import Real, Choice

import multi_objective_cfe_generator as MOCG
from alt_multi_label_predictor import MultilabelPredictor
from data_package import DataPackage


class McdEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        if not self.grab_trained_model_path():
            self.train_model()
        self.model = MultilabelPredictor.load(self.grab_trained_model_path())
        self.x, self.y = self.load_toy_x_y()

    def test_regression_only_query_y(self):
        x, y = self.x, self.y.drop(columns=self.y.columns.difference(["O_R1"]))
        datatypes = self.build_toy_x_datatypes()
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[0:1], features_to_vary=x.columns, query_y={"O_R1": (-5, 5)})
        problem = MOCG.MultiObjectiveCounterfactualsGenerator(data_package=dp,
                                                              predictor=lambda any_x: self.predict_subset(["O_R1"],
                                                                                                          any_x),
                                                              constraint_functions=[],
                                                              datatypes=datatypes)
        cf_set = MOCG.CFSet(problem, 500, initialize_from_dataset=False)
        cf_set.optimize(5)
        num_samples = 10
        cfs = cf_set.sample(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([1]), include_dataset=False, num_dpp=10000)
        self.assert_regression_target_met(cfs, "O_R1", -5, 5)
        self.assert_cfs_within_valid_range(cfs)

    def test_mixed_type_targets(self):
        x, y = self.x, self.y
        datatypes = self.build_toy_x_datatypes()
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[0:1], features_to_vary=x.columns, query_y={"O_R1": (0, 12)},
                         y_classification_targets={"O_C1": (1,)}, y_proba_targets={("O_P1", "O_P2"): ("O_P1",)})
        problem = MOCG.MultiObjectiveCounterfactualsGenerator(data_package=dp,
                                                              predictor=self.predict,
                                                              constraint_functions=[],
                                                              datatypes=datatypes)
        cf_set = MOCG.CFSet(problem, 500, initialize_from_dataset=False)
        cf_set.optimize(5)
        num_samples = 10
        cfs = cf_set.sample(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([1]), include_dataset=False, num_dpp=10000)

        self.assert_regression_target_met(cfs, "O_R1", 0, 12)
        self.assert_classification_target_met(cfs, "O_C1", [1])
        self.assert_proba_target_met(cfs, "O_P1")
        self.assert_cfs_within_valid_range(cfs)

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
        models_path_exists = "AutogluonModels" in os.listdir(os.getcwd())
        if models_path_exists:
            return self.find_valid_model()

    def find_valid_model(self):
        trained_models = os.listdir("AutogluonModels")
        for trained_model in trained_models:
            model_path = os.path.join("AutogluonModels", trained_model)
            valid_model = "multilabel_predictor.pkl" in os.listdir(model_path)
            if valid_model:
                return model_path

    def train_model(self):
        training_predictor = MultilabelPredictor(labels=["O_C1", "O_R1", "O_P1", "O_P2"])
        x, y = self.load_toy_x_y()
        training_predictor.fit(TabularDataset(pd.concat([x, y], axis=1)))

    def load_toy_x_y(self):
        y = pd.read_csv("toy_y.csv")
        x = pd.read_csv("toy_x.csv")
        y["O_C1"] = y["O_C1"].astype("category")
        x["C1"] = x["C1"].astype("category")
        return x, y

    def predict_subset(self, relevant_labels: list, x: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(pd.DataFrame(x, columns=self.x.columns)).drop(
            columns=self.y.columns.difference(relevant_labels))

    def predict(self, x):
        return self.model.predict(pd.DataFrame(x, columns=self.x.columns))
