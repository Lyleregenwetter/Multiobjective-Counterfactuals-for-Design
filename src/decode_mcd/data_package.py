from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from pymoo.core.variable import Variable, Integer, Binary, Choice, Real

from src.decode_mcd.design_targets import DesignTargets


class DataPackage:
    def __init__(self,
                 features_dataset: Union[pd.DataFrame, np.ndarray],
                 predictions_dataset: Union[pd.DataFrame, np.ndarray],
                 query_x: Union[pd.DataFrame, np.ndarray],
                 design_targets: DesignTargets,
                 datatypes: Sequence[Variable],
                 features_to_vary: Union[Sequence[str], Sequence[int]] = None,
                 bonus_objectives: Union[Sequence[str], Sequence[int]] = None,
                 ):
        """"""
        self.features_dataset = self._to_valid_dataframe(features_dataset, "features_dataset")
        self.predictions_dataset = self._to_valid_dataframe(predictions_dataset, "predictions_dataset")
        self.query_x = self._to_valid_dataframe(query_x, "query_x")
        self.design_targets = design_targets
        self.datatypes = datatypes
        self.features_to_vary = self._get_or_default(features_to_vary, list(self.features_dataset.columns.values))
        self.bonus_objectives = self._get_or_default(bonus_objectives, [])
        self._validate_fields()
        self.features_to_freeze = list(set(self.features_dataset) - set(self.features_to_vary))

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def _to_dataframe(self, numpy_array: np.ndarray, dataset_name: str):
        self._validate(len(numpy_array.shape) == 2, f"{dataset_name} must be a valid numpy array (non-empty, 2D...)")
        index_based_columns = [_ for _ in range(numpy_array.shape[1])]
        return pd.DataFrame(numpy_array, columns=index_based_columns)

    def _to_valid_dataframe(self, dataset, dataset_name):
        self._validate(isinstance(dataset, (np.ndarray, pd.DataFrame)),
                       f"{dataset_name} must either be a pandas dataframe or a numpy ndarray")
        if isinstance(dataset, np.ndarray):
            return self._to_dataframe(dataset, dataset_name)
        self._validate(not dataset.empty, f"{dataset_name} cannot be empty")
        return dataset

    def _validate_fields(self):
        self._cross_validate_datasets()
        self._cross_validate_features_to_vary()
        self._cross_validate_query_x()
        self._validate_design_targets()
        self._validate_bonus_objs()
        self._validate_datatypes()
        # self._validate_bounds(features_to_vary, upper_bounds, lower_bounds)

    def _cross_validate_datasets(self):
        n_f = len(self.features_dataset)
        n_p = len(self.predictions_dataset)
        self._validate(n_f == n_p,
                       f"features_dataset and predictions_dataset do not have the same number of rows ({n_f}, {n_p})")
        nunique = self.features_dataset.nunique()
        uniform_cols = nunique[nunique == 1].index
        self._validate(len(uniform_cols) == 0, f"""Error: The following columns were found to 
            contain completely uniform values: {uniform_cols}. This is not allowed, 
            since it blows proximity values up to infinity!""")

    # def validate_bounds(self, features_to_vary: list, upper_bounds: np.array, lower_bounds: np.array):
    #     valid_length = len(features_to_vary)
    #     assert upper_bounds.shape == (valid_length,)
    #     assert lower_bounds.shape == (valid_length,)

    def _cross_validate_features_to_vary(self):
        self._validate_columns(self.features_dataset, self.features_to_vary,
                               "features_dataset", "features_to_vary")

    def _validate_columns(self,
                          dataset: pd.DataFrame,
                          columns: Sequence,
                          dataset_name,
                          features_name):
        self._validate(len(columns) > 0, f"{features_name} cannot be an empty sequence")
        valid_columns = dataset.columns.values
        invalid_columns = self._get_invalid_columns(columns, valid_columns)
        self._raise_if_invalid_columns(dataset_name, features_name, invalid_columns, valid_columns)

    def _get_invalid_columns(self, columns, valid_columns):
        invalid_columns = set(columns).difference(set(valid_columns))
        invalid_columns = list(invalid_columns)
        invalid_columns.sort()
        return invalid_columns

    def _raise_if_invalid_columns(self, dataset_name, features_name, invalid_columns, valid_columns):
        if len(invalid_columns) != 0:
            self._validate(
                False,
                f"Invalid value in {features_name}: expected columns "
                f"{invalid_columns} to be in {dataset_name} columns {valid_columns}")

    def _validate_bonus_objs(self):
        self._validate(set(self.bonus_objectives).issubset(set(self.predictions_dataset.columns)),
                       "Bonus objectives should be a subset of labels!")

    def _cross_validate_query_x(self):
        self._validate(not self.query_x.empty, "query_x cannot be empty!")
        expected_n_columns = len(self.features_dataset.columns)
        self._validate(self.query_x.values.shape == (1, expected_n_columns),
                       f"query_x must have 1 row and {expected_n_columns} columns")
        self._validate(set(self.query_x.columns) == set(self.features_dataset.columns),
                       "query_x columns do not match dataset columns!")

    def _validate_datatypes(self):
        n_dt = len(self.datatypes)
        f_columns = self.features_dataset.columns.values
        n_f = len(f_columns)
        self._validate(
            n_dt == n_f,
            f"datatypes has length {n_dt}, expected length {n_f} matching features_dataset columns {f_columns}")
        invalid_types = [datatype for datatype in self.datatypes if
                         type(datatype) not in [Real, Choice, Binary, Integer]]
        self._validate(len(invalid_types) == 0,
                       "datatypes must strictly be a sequence of objects belonging to "
                       "the types [Real, Integer, Choice, Binary]")

    def _validate(self, mandatory_condition: bool, exception_message: str):
        if not mandatory_condition:
            raise ValueError(exception_message)

    def _validate_design_targets(self):
        self._validate(isinstance(self.design_targets, DesignTargets),
                       "design_targets must be an instance of DesignTargets")
        invalid_columns = self._get_invalid_columns(self.design_targets.get_all_constrained_labels(),
                                                    self.predictions_dataset.columns)
        self._raise_if_invalid_columns("predictions_dataset", "design_targets",
                                       invalid_columns, self.predictions_dataset.columns.values)