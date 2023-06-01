import __main__
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
        __main__.MultilabelPredictor = MultilabelPredictor
        if not self.grab_trained_model_path():
            self.train_model()
        self.model = MultilabelPredictor.load(self.grab_trained_model_path())
        self.x, self.y = self.load_toy_x_y()

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

    def test_model_example(self):
        x, y = self.x, self.y
        datatypes = [Real(bounds=(x[feature].min(), x[feature].max())) for feature in x.columns if "R" in feature]
        # noinspection PyTypeChecker
        datatypes.insert(3, Choice(options=tuple(x["C1"].unique())))
        dp = DataPackage(features_dataset=x, predictions_dataset=y,
                         query_x=x.iloc[0:1], features_to_vary=x.columns, query_y={"O_R1": (0, 12)},
                         y_classification_targets={"O_C1": (1,)}, y_proba_targets={("O_P1", "O_P2"): ("O_P1",)})
        problem = MOCG.MultiObjectiveCounterfactualsGenerator(data_package=dp,
                                                              predictor=self.call_toy_predictor_,
                                                              constraint_functions=[],
                                                              datatypes=datatypes)
        cf_set = MOCG.CFSet(problem, 500, initialize_from_dataset=False)
        cf_set.optimize(5)
        num_samples = 10
        cfs = cf_set.sample(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([1]), include_dataset=False, num_dpp=10000)
        regression_results = self.call_toy_predictor(["O_R1"], cfs).values
        all_conditions_satisfaction = np.logical_and(np.greater(regression_results, np.array([0])),
                                                     np.less(regression_results, np.array([12])))
        np_test.assert_equal(all_conditions_satisfaction, 1)
        classification_results = self.call_toy_predictor(["O_C1"], cfs).values
        np_test.assert_equal(classification_results, 1)

        proba_results = self.call_toy_predictor(["O_P1", "O_P2"], cfs)
        proba_satisfaction = np.greater(proba_results["O_P1"].values, proba_results["O_P2"].values)
        np_test.assert_equal(proba_satisfaction, 1)

        regression_cfs = cfs.drop(columns=["C1"]).values
        all_within_range = np.logical_and(np.greater_equal(regression_cfs, x.min(numeric_only=True).values),
                                          np.less_equal(regression_cfs, x.max(numeric_only=True).values))
        np_test.assert_equal(all_within_range, 1)

    def call_toy_predictor(self, relevant_labels, x) -> pd.DataFrame:
        return self.model.predict(pd.DataFrame(x, columns=self.x.columns)).drop(
            columns=self.y.columns.difference(relevant_labels))

    def call_toy_predictor_(self, x):
        return self.model.predict(pd.DataFrame(x, columns=self.x.columns))
