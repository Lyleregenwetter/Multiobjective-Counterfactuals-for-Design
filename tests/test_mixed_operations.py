import unittest
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from data_package import DataPackage


class McdPredictor(metaclass=ABCMeta):
    def __init__(self, data_package: DataPackage, bonus_objs: list, ranges,
                 constraint_functions, query_constraints, query_ub, query_lb):
        self.data_package = data_package
        self.bonus_objs = bonus_objs
        self.ranges = ranges
        self.constraint_functions = constraint_functions
        self.query_constraints = query_constraints
        self.query_ub = query_ub
        self.query_lb = query_lb

    def predict(self, x):
        pass

    @abstractmethod
    def evaluate(self, x, out, *args, **kwargs):
        pass

    def build_full_df(self, x):
        if x.empty:
            return x.values
        n = np.shape(x)[0]
        df = pd.concat([self.data_package.query_x] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self.data_package.features_to_freeze], x], axis=1)
        df = df[self.data_package.features_dataset.columns]
        return df.values

    def avg_gower_distance(self, dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame,
                           k=3) -> np.array:  # TODO batch this for memory savings
        GD = self.gower_distance(dataframe, reference_dataframe)
        bottomk = np.partition(GD, kth=k - 1, axis=1)[:, :k]
        return np.mean(bottomk, axis=1)

    def gower_distance(self, dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame):
        ranges = self.ranges.values
        dists = np.expand_dims(dataframe.values, 1) - np.expand_dims(reference_dataframe.values, 0)
        scaled_dists = np.divide(dists, ranges)
        GD = np.mean(np.abs(scaled_dists), axis=2)
        return GD

    def calculate_scores(self, x, datasetflag):
        x = pd.DataFrame.from_records(x)
        x_full = self.build_full_df(x)
        if datasetflag:
            prediction = self.data_package.predictions_dataset.copy()
        else:
            prediction = pd.DataFrame(self.predict(x_full), columns=self.data_package.predictions_dataset.columns)
        all_scores = np.zeros((len(x), len(self.bonus_objs) + 3))
        all_scores[:, :-3] = prediction.loc[:, self.bonus_objs]
        # n + 1 is gower distance
        all_scores[:, -3] = self.mixed_gower(x, self.data_package.query_x, [], {}).T
        # n + 2 is changed features
        all_scores[:, -2] = self.changed_features(x, self.data_package.query_x)
        # all_scores[:, -1] = self.np_euclidean_distance(prediction, self.target_design)
        all_scores[:, -1] = self.avg_gower_distance(x, self.data_package.features_dataset)
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

    @staticmethod
    def mixed_gower(x1: pd.DataFrame, original: pd.DataFrame, ranges, datatypes):
        number_of_features = x1.shape[1]
        real_indices = datatypes.get("r", ())
        x1_real = x1.values[:, real_indices]
        original_real = original.values[:, real_indices]
        dists = np.expand_dims(x1_real, 1) - np.expand_dims(original_real, 0)
        scaled_dists = np.divide(dists, ranges)
        scaled_dists: np.ndarray
        scaled_dists = scaled_dists.reshape((x1_real.shape[1], -1))

        categorical_indices = datatypes.get("c", ())
        x1_categorical = x1.values[:, categorical_indices]
        original_categorical = original.values[:, categorical_indices]
        categorical_dists = np.count_nonzero(x1_categorical - original_categorical, axis=1)

        all_dists = np.concatenate([scaled_dists, np.expand_dims(categorical_dists, 1)], axis=1)
        GD = np.divide(np.abs(all_dists), number_of_features)
        GD = np.sum(GD, axis=1)
        return GD

    def changed_features(self, designs_dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame):
        changes = designs_dataframe.apply(
            lambda row: np.count_nonzero(row.values - reference_dataframe.iloc[0].values), axis=1)
        return changes.values / len(self.data_package.features_dataset.columns)


class McdRegressor(McdPredictor):

    def evaluate(self, x, out, *args, **kwargs):
        # This flag will avoid passing the dataset through the predictor, when the y values are already known
        datasetflag = kwargs.get("datasetflag", False)
        score, validity = self.calculate_scores(x, datasetflag)
        out["F"] = score
        out["G"] = validity


class McdClassifier(McdPredictor):

    def evaluate(self, x, out, *args, **kwargs):
        pass


class McdPredictorTest(unittest.TestCase):
    def test_abstract(self):
        pass
