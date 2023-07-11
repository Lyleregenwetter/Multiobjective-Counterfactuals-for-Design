from typing import List, Callable, Union

import numpy as np
import pandas as pd
# from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Binary, Choice

from decode_mcd_private.classification_evaluator import ClassificationEvaluator
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets
from decode_mcd_private.stats_methods import mixed_gower, avg_gower_distance, changed_features_ratio

_AVG_GOWER_INDEX = -1

_CHANGED_FEATURE_INDEX = -2

_GOWER_INDEX = -3

_MCD_BASE_OBJECTIVES = 3

# from main.evaluation.Predictor import Predictor

MEANING_OF_LIFE = 42


class MultiObjectiveProblem(Problem):
    def __init__(self,
                 data_package: DataPackage,
                 prediction_function: Callable[[pd.DataFrame], Union[np.ndarray, pd.DataFrame]],
                 constraint_functions: list):
        """A class representing a multiobjective minimization problem"""
        self._validate(isinstance(data_package, DataPackage), "data_package must be an instance of DataPackage")
        self._data_package = data_package
        self._predictor = prediction_function
        self._constraint_functions = constraint_functions
        self._number_of_objectives = _MCD_BASE_OBJECTIVES + len(data_package.bonus_objectives)
        super().__init__(vars=self._build_problem_var_dict(),
                         n_obj=self._number_of_objectives,
                         n_constr=len(constraint_functions) + self._count_y_constraints())
        self._ranges = self._build_ranges(self._data_package.features_dataset)
        self._avg_gower_sample_size = 1000
        self._avg_gower_sample_seed = MEANING_OF_LIFE
        self._set_valid_datasets_subset()  # Remove any invalid designs from the features dataset and predictions
        self._revertible_indexes = self._get_revertible_indexes()
        # dataset

    def set_average_gower_sampling_parameters(self, sample_size: int, sample_seed: int):
        self._validate(sample_size > 0, "Invalid sample size; must be greater than zero")
        self._validate(sample_seed > 0, "Invalid seed; must be greater than zero")
        self._avg_gower_sample_size = sample_size
        self._avg_gower_sample_seed = sample_seed

    def _get_revertible_indexes(self):
        all_candidates = self._data_package.features_to_vary
        var_dict = self._build_problem_var_dict()
        q_x = self._data_package.query_x
        validity = self._get_revert_validity(all_candidates, q_x, var_dict)
        return tuple(list(self._data_package.features_to_vary).index(c) for c in all_candidates if validity[c])

    def _get_revert_validity(self, all_candidates, q_x, var_dict):
        validity = {}
        for candidate in all_candidates:
            value = q_x.iloc[0][candidate]
            if isinstance(var_dict[candidate], (Real, Integer)):
                valid = var_dict[candidate].bounds[0] <= value <= var_dict[candidate].bounds[1]
            elif isinstance(var_dict[candidate], Choice):
                valid = value in var_dict[candidate].options
            else:  # binary
                valid = True
            validity[candidate] = valid
        return validity

    def _count_y_constraints(self):
        return self._data_package.design_targets.count_constrained_labels()

    def _build_problem_var_dict(self):
        variables = {}
        for feature in self._data_package.features_to_vary:
            absolute_feature_index = self._data_package.features_dataset.columns.to_list().index(feature)
            variables[feature] = self._data_package.datatypes[absolute_feature_index]
        return variables

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        # This flag will avoid passing the dataset through the predictor, when the y values are already known
        dataset_flag = kwargs.get("datasetflag", False)
        score, validity = self._calculate_evaluation_metrics(x, dataset_flag)
        out["F"] = score
        out["G"] = validity

    def _calculate_evaluation_metrics(self, x: np.ndarray, dataset_flag: bool):
        x_full = self._build_full_df(x)
        predictions = self._get_predictions(x_full, dataset_flag)

        scores = self._get_scores(x_full, predictions, dataset_flag)
        validity = self._get_mixed_constraint_satisfaction(x_full, predictions, self._constraint_functions,
                                                           self._data_package.design_targets, dataset_flag)
        return scores, validity

    def _get_scores(self, x: pd.DataFrame, predictions: pd.DataFrame, dataset_flag):
        if dataset_flag and (self._data_package.datasets_scores is not None):
            return self._data_package.datasets_scores
        return self._calculate_scores(x, predictions)

    def _calculate_scores(self, x: pd.DataFrame, predictions: pd.DataFrame):
        all_scores = np.zeros((len(x), self._number_of_objectives))
        gower_types = self._build_gower_types()
        all_scores[:, :-_MCD_BASE_OBJECTIVES] = predictions.loc[:, self._data_package.bonus_objectives]
        all_scores[:, _GOWER_INDEX] = mixed_gower(x, self._data_package.query_x, self._ranges.values, gower_types).T
        all_scores[:, _CHANGED_FEATURE_INDEX] = changed_features_ratio(x, self._data_package.query_x,
                                                                       len(self._data_package.features_dataset.columns))
        subset = self._get_features_sample()
        all_scores[:, _AVG_GOWER_INDEX] = avg_gower_distance(x, subset, self._ranges.values, gower_types)
        return all_scores

    def _get_features_sample(self):
        subset_size = min(self._avg_gower_sample_size, len(self._data_package.features_dataset))
        subset = self._data_package.features_dataset.sample(n=subset_size, axis=0,
                                                            random_state=self._avg_gower_sample_seed)
        return subset

    def _get_predictions(self, x_full: pd.DataFrame, dataset_flag: bool):
        if dataset_flag:
            return self._data_package.predictions_dataset.copy()
        return pd.DataFrame(self._predictor(x_full), columns=self._data_package.predictions_dataset.columns)

    def _build_gower_types(self):
        return {
            "r": tuple(self._get_features_by_type([Real, Integer])),
            "c": tuple(self._get_features_by_type([Choice, Binary]))
        }

    def _set_valid_datasets_subset(self):
        # Scans the features_dataset and returns the subset violating the variable categories and ranges,
        # as well as the changed feature specifications
        f_d = self._data_package.features_dataset
        p_d = self._data_package.predictions_dataset
        f_d, p_d = self._get_valid_numeric_entries(f_d, p_d)
        f_d, p_d = self._get_valid_categorical_entries(f_d, p_d)

        f2f = self._data_package.features_to_freeze
        if len(f2f) > 0:
            f_d_view = f_d[f2f]
            query_view = self._data_package.query_x[f2f]
            p_d = p_d[np.equal(f_d_view.values, query_view.values)]
            f_d = f_d[np.equal(f_d_view.values, query_view.values)]
        self._data_package.features_dataset = f_d
        self._data_package.predictions_dataset = p_d

    def _get_valid_categorical_entries(self, f_d, p_d):
        categorical_idx = self._get_features_by_type([Choice])  # pass in the pymoo built in variable types
        for parameter in categorical_idx:  # Filter out any that don't fall into an acceptable category
            f_d, p_d = self._filter_valid_categorical_entries(f_d, p_d, parameter)
        return f_d, p_d

    def _filter_valid_categorical_entries(self, f_d, p_d, parameter):
        # noinspection PyUnresolvedReferences
        features_with_valid_categories = (
            f_d.iloc[:, parameter].isin(self._data_package.datatypes[parameter].options))
        p_d = p_d[features_with_valid_categories]
        f_d = f_d[features_with_valid_categories]
        return f_d, p_d

    def _get_valid_numeric_entries(self, f_d, p_d):
        reals_and_ints_idx = self._get_features_by_type([Real, Integer])  # pass in the pymoo built in variable types
        for index in reals_and_ints_idx:  # Filter out any that don't fall into an acceptable range
            f_d, p_d = self._filter_entries_outside_range(f_d, p_d, index)
        return f_d, p_d

    def _filter_entries_outside_range(self, f_d, p_d, index):
        # noinspection PyUnresolvedReferences
        features_gt_lower_bound = (f_d.iloc[:, index] >= self._data_package.datatypes[index].bounds[0])
        p_d = p_d[features_gt_lower_bound]
        f_d = f_d[features_gt_lower_bound]
        # noinspection PyUnresolvedReferences
        features_lt_upper_bound = (f_d.iloc[:, index] <= self._data_package.datatypes[index].bounds[1])
        p_d = p_d[features_lt_upper_bound]
        f_d = f_d[features_lt_upper_bound]
        return f_d, p_d

    def _get_features_by_type(self, types: list) -> list:
        """Helper function to get a list of parameter indices of a particular datatype"""
        dts = self._data_package.datatypes
        matching_idxs = []
        for i in range(len(dts)):
            if type(dts[i]) in types:
                matching_idxs.append(i)
        return matching_idxs

    def _build_ranges(self, features_dataset: pd.DataFrame):
        # TODO: question this. Do we build ranges based on the
        #  features dataset or based on the limits provided by the user in datatypes?
        indices = self._get_features_by_type([Real, Integer])
        numeric_features = features_dataset.iloc[:, indices]
        return numeric_features.max() - numeric_features.min()

    def _get_mixed_constraint_satisfaction(self,
                                           x_full: pd.DataFrame,
                                           y: pd.DataFrame,
                                           x_constraint_functions: list,
                                           design_targets: DesignTargets,
                                           dataset_flag):
        if dataset_flag and (self._data_package.datasets_validity is not None):
            return self._data_package.datasets_validity
        return self._calculate_mixed_constraint_satisfaction(x_full, y, x_constraint_functions, design_targets)

    def _calculate_mixed_constraint_satisfaction(self,
                                                 x_full: pd.DataFrame,
                                                 y: pd.DataFrame,
                                                 x_constraint_functions: list,
                                                 design_targets: DesignTargets):
        n_total_constraints = design_targets.count_constrained_labels()
        n_rows = x_full.shape[0]
        result = np.zeros(shape=(n_rows, n_total_constraints))

        self._append_x_constraint_satisfaction(result, x_full, x_constraint_functions, n_total_constraints)
        self._append_proba_satisfaction(result, y, design_targets)
        self._append_satisfaction(result, self._evaluate_regression_satisfaction, y, design_targets,
                                  design_targets.get_continuous_labels())
        self._append_satisfaction(result, self._evaluate_categorical_satisfaction, y, design_targets,
                                  design_targets.get_classification_labels())

        return result

    def _append_satisfaction(self, result: np.ndarray, evaluation_function: callable,
                             y: pd.DataFrame, y_constraints: DesignTargets, labels) -> None:
        satisfaction = evaluation_function(y, y_constraints)
        indices = [list(y.columns).index(key) for key in labels]
        result[:, indices] = 1 - satisfaction

    def _append_proba_satisfaction(self, result: np.ndarray, y: pd.DataFrame,
                                   design_targets: DesignTargets) -> None:
        c_evaluator = ClassificationEvaluator()
        for proba_key, proba_targets in zip(design_targets.get_probability_labels(),
                                            design_targets.get_preferred_probability_targets()):
            proba_consts = y.loc[:, proba_key]
            proba_satisfaction = c_evaluator.evaluate_proba(proba_consts, proba_targets)
            indices = [list(y.columns).index(key) for key in proba_key]
            result[:, indices] = 1 - np.greater(proba_satisfaction, 0)

    def _append_x_constraint_satisfaction(self, result: np.ndarray,
                                          x_full: pd.DataFrame,
                                          x_constraint_functions: List[callable],
                                          n_total_constraints: int):
        for i in range(len(x_constraint_functions)):
            # TODO: discuss this change with Lyle
            result[:, n_total_constraints - 1 - i] = x_constraint_functions[i](x_full).flatten()

    def _evaluate_categorical_satisfaction(self, y: pd.DataFrame, y_category_constraints: DesignTargets):
        actual = y.loc[:, y_category_constraints.get_classification_labels()]
        # dtype=object is needed, otherwise the operation is deprecated
        targets = np.array([[i for i in j] for j in y_category_constraints.get_desired_classes()], dtype=object)
        return ClassificationEvaluator().evaluate_categorical(actual, targets=targets)

    def _evaluate_regression_satisfaction(self, y: pd.DataFrame, design_targets: DesignTargets):
        query_lb, query_ub = design_targets.get_continuous_boundaries()
        actual = y.loc[:, design_targets.get_continuous_labels()].values
        satisfaction = np.logical_and(np.less(actual, query_ub), np.greater(actual, query_lb))
        return satisfaction

    def _build_full_df(self, x: np.ndarray):
        x = pd.DataFrame.from_records(x, columns=self._data_package.features_to_vary)
        if x.empty:
            return x
        n = np.shape(x)[0]
        df = pd.concat([self._data_package.query_x] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self._data_package.features_to_freeze], x], axis=1)
        df = df[self._data_package.features_dataset.columns]
        return df

    def _validate(self, mandatory_condition, error_message):
        if not mandatory_condition:
            raise ValueError(error_message)
