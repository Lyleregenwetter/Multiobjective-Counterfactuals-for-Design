from typing import List, Callable, Union, Sequence

import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer, Binary, Choice

from decode_mcd_private.classification_evaluator import ClassificationEvaluator
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets
from decode_mcd_private.stats_methods import mixed_gower, avg_gower_distance, changed_features_ratio
from decode_mcd_private.validation_utils import validate

_MANPROX_INDEX = -1

_SPARSITY_INDEX = -2

_PROXIMITY_INDEX = -3

_MCD_BASE_OBJECTIVES = 3

MEANING_OF_LIFE = 42


class MultiObjectiveProblem(Problem):
    # TODO: allow user to specify score functions
    def __init__(self,
                 data_package: DataPackage,
                 x_query: Union[pd.DataFrame, np.ndarray],
                 y_targets: DesignTargets,
                 prediction_function: Callable[[pd.DataFrame], Union[np.ndarray, pd.DataFrame]],
                 features_to_vary: Union[Sequence[str], Sequence[int]] = None):
        """A class representing a multiobjective minimization problem"""
        self._validate(isinstance(data_package, DataPackage), "data_package must be an instance of DataPackage")
        self._data_package = data_package
        self._x_query = self._to_valid_dataframe(x_query, "x_query")
        self._y_targets = y_targets
        self._features_to_vary: Union[Sequence[str], Sequence[int]] = self._get_or_default(features_to_vary,
                                                                                           list(self._data_package.features_dataset.columns.values))
        self._features_to_freeze = list(set(self._data_package.features_dataset.columns.values) - set(self._features_to_vary))
        self._data_package.cross_validate(self._x_query, self._y_targets, self._features_to_vary)
        self._predictor = prediction_function
        self._bonus_objectives = self._get_or_default(self._grab_minimization_targets(), [])
        self._number_of_objectives = _MCD_BASE_OBJECTIVES + len(self._bonus_objectives)
        super().__init__(vars=self._build_problem_var_dict(),
                         n_obj=self._number_of_objectives,
                         n_constr=self._count_y_constraints())
        self._ranges = self._build_ranges(self._data_package.features_dataset)
        self._avg_gower_sample_size = 1000
        self._avg_gower_sample_seed = MEANING_OF_LIFE
        self._valid_features_dataset, self._predictions_dataset = self._set_valid_datasets_subset()  # Remove any invalid designs from the features dataset and predictions
        self._revertible_indexes = self._get_revertible_indexes()

    def set_average_gower_sampling_parameters(self, sample_size: int, sample_seed: int):
        self._validate(sample_size > 0, "Invalid sample size; must be greater than zero")
        self._validate(sample_seed > 0, "Invalid seed; must be greater than zero")
        self._avg_gower_sample_size = sample_size
        self._avg_gower_sample_seed = sample_seed

    def _get_revertible_indexes(self):
        all_candidates = self._features_to_vary
        var_dict = self._build_problem_var_dict()
        q_x = self._x_query
        validity = self._get_revert_validity(all_candidates, q_x, var_dict)
        return tuple(list(self._features_to_vary).index(c) for c in all_candidates if validity[c])

    @staticmethod
    def _get_revert_validity(all_candidates, q_x, var_dict):
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

    def _validate_bonus_objs(self):
        condition = set(self._bonus_objectives).issubset(set(self._data_package.predictions_dataset.columns))
        validate(condition, "Bonus objectives should be a subset of labels!")

    def _to_valid_dataframe(self, dataset, dataset_name):
        condition = isinstance(dataset, (np.ndarray, pd.DataFrame))
        validate(condition, f"{dataset_name} must either be a pandas dataframe or a numpy ndarray")
        if isinstance(dataset, np.ndarray):
            return self._to_dataframe(dataset, dataset_name)
        mandatory_condition = not dataset.empty
        validate(mandatory_condition, f"{dataset_name} cannot be empty")
        return dataset

    def _to_dataframe(self, numpy_array: np.ndarray, dataset_name: str):
        condition = len(numpy_array.shape) == 2
        validate(condition, f"{dataset_name} must be a valid numpy array (non-empty, 2D...)")
        if dataset_name == "query_x":
            index_based_columns = self._data_package.features_dataset.columns
        else:
            index_based_columns = [_ for _ in range(numpy_array.shape[1])]
        return pd.DataFrame(numpy_array, columns=index_based_columns)


    def _count_y_constraints(self):
        return self._y_targets.count_constrained_labels()

    def _build_problem_var_dict(self):
        variables = {}
        for feature in self._features_to_vary:
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

        scores = self._get_scores(x_full, predictions)
        validity = self._get_mixed_constraint_satisfaction(x_full, predictions,
                                                           self._y_targets)
        return scores, validity

    def _get_scores(self, x: pd.DataFrame, predictions: pd.DataFrame):
        return self._calculate_scores(x, predictions)

    def _calculate_scores(self, x: pd.DataFrame, predictions: pd.DataFrame):
        all_scores = np.zeros((len(x), self._number_of_objectives))
        gower_types = self._build_gower_types()
        all_scores[:, :-_MCD_BASE_OBJECTIVES] = predictions.loc[:, self._bonus_objectives]
        all_scores[:, _PROXIMITY_INDEX] = mixed_gower(x, self._x_query, self._ranges.values, gower_types).T
        all_scores[:, _SPARSITY_INDEX] = changed_features_ratio(x, self._x_query,
                                                                       len(self._valid_features_dataset.columns))
        subset = self._get_features_sample()
        all_scores[:, _MANPROX_INDEX] = avg_gower_distance(x, subset, self._ranges.values, gower_types)
        return all_scores

    def _get_features_sample(self):
        subset_size = min(self._avg_gower_sample_size, len(self._data_package.features_dataset))
        subset = self._data_package.features_dataset.sample(n=subset_size, axis=0,
                                                            random_state=self._avg_gower_sample_seed)
        return subset

    def _get_predictions(self, x_full: pd.DataFrame, dataset_flag: bool):
        if dataset_flag:
            return self._predictions_dataset.copy()
        return pd.DataFrame(self._predictor(x_full), columns=self._predictions_dataset.columns)

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

        f2f = self._features_to_freeze
        if len(f2f) > 0:
            f_d_view = f_d[f2f]
            query_view = self._x_query[f2f]
            # TODO: test this!
            p_d = p_d[np.equal(f_d_view.values, query_view.values).all(axis=1)]
            f_d = f_d[np.equal(f_d_view.values, query_view.values).all(axis=1)]
        return f_d, p_d

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
                                           design_targets: DesignTargets):
        return self._calculate_mixed_constraint_satisfaction(x_full, y, design_targets)

    def _calculate_mixed_constraint_satisfaction(self,
                                                 x_full: pd.DataFrame,
                                                 y: pd.DataFrame,
                                                 design_targets: DesignTargets):
        all_labels = y.columns.tolist()
        initial_num_columns = len(all_labels)
        n_rows = x_full.shape[0]
        result = np.zeros(shape=(n_rows, initial_num_columns))

        self._append_satisfaction(result, self._evaluate_regression_satisfaction, y, design_targets,
                                  design_targets.get_continuous_labels())
        self._append_satisfaction(result, self._evaluate_categorical_satisfaction, y, design_targets,
                                  design_targets.get_categorical_labels())
        result = self.drop_non_constrained_columns(design_targets, result, y)
        return result

    def _grab_minimization_targets(self) -> List[str]:
        return [_target.label for _target in self._y_targets.minimization_targets]


    @staticmethod
    def drop_non_constrained_columns(design_targets, result, y):
        constrained_indices = [list(y.columns).index(key) for key in design_targets.get_all_constrained_labels()]
        constrained_indices.sort()
        return result[:, constrained_indices]

    @staticmethod
    def _append_satisfaction(result: np.ndarray, evaluation_function: callable,
                             y: pd.DataFrame, y_constraints: DesignTargets, labels) -> None:
        satisfaction = evaluation_function(y, y_constraints)
        indices = [list(y.columns).index(key) for key in labels]
        result[:, indices] = satisfaction

    @staticmethod
    def _append_x_constraint_satisfaction(result: np.ndarray,
                                          x_full: pd.DataFrame,
                                          x_constraint_functions: List[callable],
                                          initial_num_columns: int):
        for i in range(len(x_constraint_functions)):
            # TODO: discuss this change with Lyle
            result[:, initial_num_columns - 1 - i] = x_constraint_functions[i](x_full).values.flatten()

    @staticmethod
    def _evaluate_categorical_satisfaction(y: pd.DataFrame, y_category_constraints: DesignTargets):
        actual = y.loc[:, y_category_constraints.get_categorical_labels()]
        # dtype=object is needed, otherwise the operation is deprecated
        targets = np.array([[i for i in j] for j in y_category_constraints.get_desired_classes()], dtype=object)
        return 1 - ClassificationEvaluator().evaluate_categorical(actual, targets=targets)

    @staticmethod
    def _evaluate_regression_satisfaction(y: pd.DataFrame, design_targets: DesignTargets):
        query_lb, query_ub = design_targets.get_continuous_boundaries()
        actual = y.loc[:, design_targets.get_continuous_labels()].values
        satisfaction = np.maximum(actual - query_ub, query_lb - actual)
        return satisfaction

    def _build_full_df(self, x: np.ndarray):
        x = pd.DataFrame.from_records(x, columns=self._features_to_vary)
        if x.empty:
            return x
        n = np.shape(x)[0]
        df = pd.concat([self._x_query] * n, axis=0, )
        df.index = list(range(n))
        df = pd.concat([df.loc[:, self._features_to_freeze], x], axis=1)
        df = df[self._valid_features_dataset.columns]
        return df

    @staticmethod
    def _validate(mandatory_condition, error_message):
        if not mandatory_condition:
            raise ValueError(error_message)

    @staticmethod
    def _get_or_default(_supplied_value, _default_value):
        if _supplied_value is None:
            return _default_value
        return _supplied_value
