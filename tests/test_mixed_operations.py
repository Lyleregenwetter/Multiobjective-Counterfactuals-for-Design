import unittest
from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real

from data_package import DataPackage
from multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator as MOCFG
from stats_methods import mixed_gower


class McdPredictor(metaclass=ABCMeta):
    def __init__(self, data_package: DataPackage, bonus_objs: list, constraint_functions, predictor):
        self.data_package = data_package
        self.bonus_objs = bonus_objs
        self.ranges = self.build_ranges(data_package.features_dataset)
        self.constraint_functions = constraint_functions
        self.query_constraints, self.query_lb, self.query_ub = self.sort_query_y(data_package.query_y)
        self.predictor = predictor

    @staticmethod
    def build_ranges(features_dataset: pd.DataFrame):
        return features_dataset.max() - features_dataset.min()

    def sort_query_y(self, query_y: dict):
        query_constraints = []
        query_lb = []
        query_ub = []
        for key in query_y.keys():
            query_constraints.append(key)
            query_lb.append(query_y[key][0])
            query_ub.append(query_y[key][1])
        return query_constraints, np.array(query_lb), np.array(query_ub)

    @abstractmethod
    def evaluate(self, x, out, *args, **kwargs):
        pass

    def predict(self, x):
        return self.predictor.predict(x)

    def build_full_df(self, x):
        if x.empty:
            return x.values
        n = np.shape(x)[0]
        df = pd.concat([self.data_package.query_x] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self.data_package.features_to_freeze], x], axis=1)
        df = df[self.data_package.features_dataset.columns]
        return df

    def avg_gower_distance(self, dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame,
                           k=3) -> np.array:  # TODO batch this for memory savings
        GD = self.gower_distance(dataframe, reference_dataframe, self.ranges.values)
        bottomk = np.partition(GD, kth=k - 1, axis=1)[:, :k]
        return np.mean(bottomk, axis=1)

    @staticmethod
    def gower_distance(dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame, ranges):
        dists = np.expand_dims(dataframe.values, 1) - np.expand_dims(reference_dataframe.values, 0)
        scaled_dists = np.divide(dists, ranges)
        GD = np.mean(np.abs(scaled_dists), axis=2)
        return GD

    def calculate_scores(self, x, datasetflag):
        x = pd.DataFrame.from_records(x, columns=self.data_package.features_to_vary)
        x_full = self.build_full_df(x)
        if datasetflag:
            prediction = self.data_package.predictions_dataset.copy()
        else:
            prediction = pd.DataFrame(self.predict(x_full), columns=self.data_package.predictions_dataset.columns)
        all_scores = np.zeros((len(x), len(self.bonus_objs) + 3))
        all_scores[:, :-3] = prediction.loc[:, self.bonus_objs]
        # n + 1 is gower distance
        all_scores[:, -3] = mixed_gower(x_full,
                                        self.data_package.query_x,
                                        np.array(self.ranges),
                                        {"r": tuple(
                                            _ for _ in range(len(self.data_package.features_dataset.columns)))}
                                        ).T
        # n + 2 is changed features
        all_scores[:, -2] = self.changed_features(x_full, self.data_package.query_x)
        # all_scores[:, -1] = self.np_euclidean_distance(prediction, self.target_design)
        all_scores[:, -1] = self.avg_gower_distance(x_full, self.data_package.features_dataset)
        return all_scores, self.get_constraint_satisfaction(x_full, prediction)

    def get_constraint_satisfaction(self, x_full, y):
        n_cf = len(self.constraint_functions)
        g = np.zeros((len(x_full), n_cf + len(self.query_constraints)))
        for i in range(n_cf):
            g[:, i] = self.constraint_functions[i](x_full).flatten()
        pred_consts = y.loc[:, self.query_constraints].values
        indiv_satisfaction = np.logical_and(np.less(pred_consts, self.query_ub), np.greater(pred_consts, self.query_lb))
        g[:, n_cf:] = 1 - indiv_satisfaction
        return g

    def changed_features(self, designs_dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame):
        changes = designs_dataframe.apply(
            lambda row: np.count_nonzero(row.values - reference_dataframe.iloc[0].values), axis=1)
        return changes.values / len(self.data_package.features_dataset.columns)


class McdRegressor(McdPredictor):

    def evaluate(self, x: np.ndarray, out, *args, **kwargs):
        # This flag will avoid passing the dataset through the predictor, when the y values are already known
        datasetflag = kwargs.get("datasetflag", False)
        score, validity = self.calculate_scores(x, datasetflag)
        out["F"] = score
        out["G"] = validity


class McdClassifier(McdPredictor):

    def evaluate(self, x, out, *args, **kwargs):
        pass


class DummyPredictor:
    def predict(self, x):
        return np.arange(x.shape[0] * 2).reshape(-1, 2)


class McdPredictorTest(unittest.TestCase):
    @unittest.skip
    def test_k_edge_case(self):
        """If the features dataset is small, the partition method fails with error (K=2) out of bounds"""
        pass

    def test_evaluate_subset(self):
        package = self.build_package(features_to_vary=["x", "y"])
        regressor = self.build_regressor(package)
        out = {}
        regressor.evaluate(
            np.array([[12, 13], [14, 15], [16, 17], [16, 19]]), out, datasetflag=False)
        self.assertTrue("F" in out)
        self.assertTrue("G" in out)

    def test_evaluate_all_features(self):
        package = self.build_package(features_to_vary=["x", "y", "z"])

        regressor = self.build_regressor(package)
        out = {}
        regressor.evaluate(
            np.array([[12, 13, 15], [14, 15, 19], [16, 17, 25], [16, 17, 25]]), out, datasetflag=False
        )
        self.assertTrue("F" in out)
        self.assertTrue("G" in out)

    def build_regressor(self, package):
        return McdRegressor(
            data_package=package,
            bonus_objs=[],
            constraint_functions=[],
            predictor=DummyPredictor()
        )

    def build_package(self,
                      features_dataset=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [12, 13, 15]]),
                                                    columns=["x", "y", "z"]),
                      predictions_dataset=pd.DataFrame(np.array([[5, 4], [3, 2], [2, 1]]), columns=["A", "B"]),
                      query_x=pd.DataFrame(np.array([[5, 12, 15]]), columns=["x", "y", "z"]),
                      query_y=None,
                      features_to_vary=None,
                      datatypes=None):
        if datatypes is None:
            datatypes = []
        if features_to_vary is None:
            features_to_vary = ["x", "y", "z"]
        if query_y is None:
            query_y = {"A": (4, 10)}
        return DataPackage(features_dataset, predictions_dataset, query_x, features_to_vary, query_y, datatypes)
