import unittest
from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.testing as np_test
import pandas as pd
from classifier_wrapper import ClassifierWrapper

from data_package import DataPackage


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
        all_scores[:, -3] = self.mixed_gower(x_full,
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

    @staticmethod
    def get_mixed_constraint_satisfaction(x_full: pd.DataFrame,
                                          y: pd.DataFrame,
                                          x_constraint_functions: list,
                                          y_regression_constraints: dict,
                                          y_category_constraints: dict,
                                          y_proba_constraints: dict):
        _, query_lb, query_ub = McdPredictor.sort_regression_constraints(y_regression_constraints)
        n_proba_constrains = len(McdPredictor.flatten_list_of_tuples(list(y_proba_constraints.keys())))

        wrapper = ClassifierWrapper()

        n_cf = len(x_constraint_functions)
        n_total_constraints = n_cf + len(y_regression_constraints) + len(y_category_constraints) + n_proba_constrains
        g = np.zeros((x_full.shape[0], n_total_constraints))
        for i in range(n_cf):
            # TODO: discuss this change with Lyle
            g[:, n_total_constraints - 1 - i] = x_constraint_functions[i](x_full).flatten()

        pred_consts = y.loc[:, y_regression_constraints.keys()].values
        indiv_satisfaction = np.logical_and(np.less(pred_consts, query_ub), np.greater(pred_consts, query_lb))

        category_consts = y.loc[:, y_category_constraints.keys()].values
        category_satisfaction = wrapper.evaluate_categorical(category_consts,
                                                             targets=np.array([[i for i in j] for j in
                                                                               y_category_constraints.values()]))
        for proba_key, proba_targets in y_proba_constraints.items():
            proba_consts = y.loc[:, proba_key]
            proba_satisfaction = wrapper.evaluate_proba(proba_consts, proba_targets)
            g[:, proba_key] = 1 - proba_satisfaction

        g[:, n_cf:] = 1 - indiv_satisfaction
        return g

    @staticmethod
    def sort_regression_constraints(regression_constraints: dict):
        query_constraints = []
        query_lb = []
        query_ub = []
        for key in regression_constraints.keys():
            query_constraints.append(key)
            query_lb.append(regression_constraints[key][0])
            query_ub.append(regression_constraints[key][1])
        return query_constraints, np.array(query_lb), np.array(query_ub)

    @staticmethod
    def flatten_list_of_tuples(list_tuples: list):
        """flattens indices in proba constrains"""
        all_elements = []
        for t in list_tuples:
            for e in t:
                all_elements.append(e)
        return all_elements

    @staticmethod
    def mixed_gower(x1: pd.DataFrame, x2: pd.DataFrame, ranges: np.ndarray, datatypes: dict):

        real_indices = datatypes.get("r", ())
        x1_real = x1.values[:, real_indices]
        x2_real = x2.values[:, real_indices]
        dists = np.expand_dims(x1_real, 1) - np.expand_dims(x2_real, 0)
        # TODO: check whether np.divide will broadcast shapes as desired in all cases
        scaled_dists = np.divide(dists, ranges)

        categorical_indices = datatypes.get("c", ())
        x1_categorical = x1.values[:, categorical_indices]
        x2_categorical = x2.values[:, categorical_indices]
        categorical_dists = np.not_equal(np.expand_dims(x1_categorical, 1), np.expand_dims(x2_categorical, 0))

        all_dists = np.concatenate([scaled_dists, categorical_dists], axis=2)
        total_number_of_features = x1.shape[1]
        GD = np.divide(np.abs(all_dists), total_number_of_features)
        GD = np.sum(GD, axis=2)
        return GD

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

    def test_high_dimensional_mixed_gower(self):
        x1 = np.array([[i + j for i in range(1, 7)] for j in range(1, 6)])
        x2 = np.array([[i + j for i in range(1, 7)] for j in range(5, 8)])
        x1 = pd.DataFrame.from_records(x1)
        x2 = pd.DataFrame.from_records(x2)
        data_types = {"r": (0, 1, 3, 5), "c": (2, 4)}
        mixed_gower = McdPredictor.mixed_gower(x1, x2, np.array([5, 1, 10, 20]), data_types)
        self.assertIsNotNone(mixed_gower)

    def test_get_mixed_constraint_full(self):
        """

        """
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(3)]))
        y = pd.DataFrame.from_records(np.array([
            [1, 200, 3, 500, 0.4, 0.6],
            [3, 250, 10, 550, 0.7, 0.3],
            [5, 300, 15, 500, 0.0, 1.0]
        ]))
        satisfaction = McdPredictor.get_mixed_constraint_satisfaction(x_full=x_full,
                                                                      y=y,
                                                                      x_constraint_functions=[],
                                                                      y_regression_constraints={
                                                                          0: (2, 6),
                                                                          2: (10, 16)
                                                                      },
                                                                      y_category_constraints={
                                                                          1: (200, 300),
                                                                          3: (550,)},
                                                                      y_proba_constraints={(4, 5): (5,)})
        np_test.assert_equal(satisfaction, np.array([
            [1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
        ]))

    def test_strict_inequality_of_regression_constraints(self):
        """this is the current behavior, but is it desired?"""
        self.test_get_mixed_constraint_satisfaction()

    def test_get_mixed_constraint_satisfaction(self):
        """This does not test the use of constraint functions - hence the dummy x_full"""
        y = pd.DataFrame.from_records(np.array([[1, 9], [2, 10],
                                                [3, 12], [3, 8],
                                                [4, 20], [5, 21]]))
        x_full = pd.DataFrame.from_records(np.array([[1] for _ in range(6)]))
        satisfaction = McdPredictor.get_mixed_constraint_satisfaction(x_full=x_full,
                                                                      y=y,
                                                                      x_constraint_functions=[],
                                                                      y_regression_constraints={0: (2, 4),
                                                                                                1: (10, 20)},
                                                                      y_category_constraints={},
                                                                      y_proba_constraints={})
        np_test.assert_equal(satisfaction, np.array([[1, 1], [1, 1], [0, 0], [0, 1], [1, 1], [1, 1]]))

    def test_mixed_gower_full(self):
        x1 = pd.DataFrame.from_records(np.array([[15., 0, 20., 500], [15., 1, 25., 500], [100., 2, 50., 501]]))
        x2 = pd.DataFrame.from_records(np.array([[15., 0, 20., 500], [16., 1, 25., 5000]]))
        datatypes = {"r": (0, 2), "c": (1, 3)}
        ranges = np.array([10, 5])
        gower_distance = McdPredictor.mixed_gower(x1, x2, ranges, datatypes)
        np_test.assert_equal(gower_distance, np.array([[0, 0.775], [0.5, 0.275], [4.125, 3.85]]))

    def test_mixed_gower_same_as_gower_when_all_real(self):
        package = self.build_package()
        regressor = self.build_regressor(package)
        features = pd.concat([package.features_dataset, pd.DataFrame(np.array([[1, 2, 3]]), columns=['x', 'y', 'z'])],
                             axis=0)
        distance = regressor.gower_distance(features, package.features_dataset.iloc[0], regressor.ranges.values)
        mixed_distance = regressor.mixed_gower(features,
                                               package.features_dataset.iloc[0:1],
                                               np.array(regressor.ranges), {"r": (0, 1, 2)})
        np_test.assert_equal(distance, mixed_distance)

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
