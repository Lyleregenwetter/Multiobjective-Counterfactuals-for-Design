from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
import pymoo.core.variable
from pymoo.core.variable import Variable, Integer, Binary, Choice, Real

from decode_mcd.design_targets import DesignTargets
from decode_mcd.mcd_exceptions import UserInputException
from decode_mcd_private.validation_utils import validate

QUERY_X_INVALID_BINARY_TYPE = "[query_x] has a variable specified as binary by datatypes " \
                              "whose value is not True, False, 1, or 0"

QUERY_X_INVALID_DATATYPES_CHOICE = "[query_x] has a choice variable that is not permitted by datatypes"

QUERY_X_OUTSIDE_TYPES_RANGE = "[query_x] parameters fall outside of range specified by datatypes"


class DataPackage:
    # TODO: move fields from data package to problem
    def __init__(self,
                 x: Union[pd.DataFrame, np.ndarray],
                 y: Union[pd.DataFrame, np.ndarray],
                 x_datatypes: Sequence[Variable]
                 ):
        """
        A data class that encapsulates all design and performance space data.

        @param x: should be 2D and should contain designs that are
        somewhat representative of the desired region of the design space.

        @param y: dataset of the performance metrics of the designs in the features dataset.

        @param x_query: the starting design. Generated designs will generally try to remain similar to this.

        @param y_targets:  describes the desired region of the performance space.

        @param x_datatypes: describes the datatypes of the design features in @features_dataset,
        and must also be supplied with bounds that describe the desired region of the design space.
        Valid: [Real(bounds=(0, 10), Choice(options=(0, 1, 2), ...]
        Invalid: [Real(0, 10), Choice((0, 1, 2))] | [Real(), Choice()]

        @param features_to_vary: the subset of features that are 'actionable' or allowed to vary.

        @param bonus_objectives: the subset of performance metrics that will be minimized in optimization steps.

        @param datasets_scores:

        @param datasets_validity:

        """
        self.features_dataset = self._to_valid_dataframe(x, "features_dataset")
        self.predictions_dataset = self._to_valid_dataframe(y, "predictions_dataset")
        self.datatypes = x_datatypes
        self._validate_fields()

    def cross_validate(self,
                       x_query: Union[pd.DataFrame, np.ndarray],
                       y_targets: DesignTargets,
                       features_to_vary: Union[Sequence[str], Sequence[int]]):
        x_query = self._to_valid_dataframe(x_query, "query_x")
        self._cross_validate_datasets()
        self._cross_validate_features_to_vary(features_to_vary)
        self._cross_validate_query_x(x_query)
        self._validate_design_targets(y_targets)
        self._validate_datatypes()
        self._validate_query_x_against_datatypes(x_query)

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def _to_dataframe(self, numpy_array: np.ndarray, dataset_name: str):
        condition = len(numpy_array.shape) == 2
        validate(condition, f"{dataset_name} must be a valid numpy array (non-empty, 2D...)")
        if dataset_name == "query_x":
            index_based_columns = self.features_dataset.columns
        else:
            index_based_columns = [_ for _ in range(numpy_array.shape[1])]
        return pd.DataFrame(numpy_array, columns=index_based_columns)

    def _to_valid_dataframe(self, dataset, dataset_name):
        condition = isinstance(dataset, (np.ndarray, pd.DataFrame))
        validate(condition, f"{dataset_name} must either be a pandas dataframe or a numpy ndarray")
        if isinstance(dataset, np.ndarray):
            return self._to_dataframe(dataset, dataset_name)
        mandatory_condition = not dataset.empty
        validate(mandatory_condition, f"{dataset_name} cannot be empty")
        return dataset

    def _validate_fields(self):
        self._cross_validate_datasets()
        self._validate_datatypes()
        # self._validate_bounds(features_to_vary, upper_bounds, lower_bounds)

    def _cross_validate_datasets(self):
        n_f = len(self.features_dataset)
        n_p = len(self.predictions_dataset)
        condition = n_f == n_p
        validate(condition,
                 f"features_dataset and predictions_dataset do not have the same number of rows ({n_f}, {n_p})")
        nunique = self.features_dataset.nunique()
        uniform_cols = nunique[nunique == 1].index
        mandatory_condition = len(uniform_cols) == 0
        validate(mandatory_condition, f"""Error: The following columns were found to 
            contain completely uniform values: {uniform_cols}. This is not allowed, 
            since it blows proximity values up to infinity!""")

    # def validate_bounds(self, features_to_vary: list, upper_bounds: np.array, lower_bounds: np.array):
    #     valid_length = len(features_to_vary)
    #     assert upper_bounds.shape == (valid_length,)
    #     assert lower_bounds.shape == (valid_length,)

    def _cross_validate_features_to_vary(self, features_to_vary: Union[Sequence[str], Sequence[int]]):
        self._validate_columns(self.features_dataset, features_to_vary,
                               "features_dataset", "features_to_vary")

    def _validate_columns(self,
                          dataset: pd.DataFrame,
                          columns: Sequence,
                          dataset_name,
                          features_name):
        condition = len(columns) > 0
        validate(condition, f"{features_name} cannot be an empty sequence")
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
            validate(False, f"Invalid value in {features_name}: expected columns "
                            f"{invalid_columns} to be in {dataset_name} columns {valid_columns}")

    def _cross_validate_query_x(self, query_x: pd.DataFrame):
        condition = not query_x.empty
        validate(condition, "query_x cannot be empty!")
        expected_n_columns = len(self.features_dataset.columns)
        mandatory_condition = query_x.values.shape == (1, expected_n_columns)
        validate(mandatory_condition, f"query_x must have 1 row and {expected_n_columns} columns")
        condition1 = set(query_x.columns) == set(self.features_dataset.columns)
        validate(condition1, "query_x columns do not match dataset columns!")

    def _validate_datatypes(self):
        n_dt = len(self.datatypes)
        f_columns = self.features_dataset.columns.values
        n_f = len(f_columns)
        condition = n_dt == n_f
        validate(condition,
                 f"datatypes has length {n_dt}, expected length {n_f} matching features_dataset columns {f_columns}")
        self._validate_class_of_dt_objects()
        self._validate_internals()

    def _validate_class_of_dt_objects(self):
        invalid_types = [datatype for datatype in self.datatypes if
                         type(datatype) not in [Real, Choice, Binary, Integer]]
        mandatory_condition = len(invalid_types) == 0
        validate(mandatory_condition, "datatypes must strictly be a sequence of objects belonging to "
                                      "the types [Real, Integer, Choice, Binary]")

    def _validate_design_targets(self, design_targets: DesignTargets):
        condition = isinstance(design_targets, DesignTargets)
        validate(condition, "design_targets must be an instance of DesignTargets")
        invalid_columns = self._get_invalid_columns(design_targets.get_all_constrained_labels(),
                                                    self.predictions_dataset.columns)
        self._raise_if_invalid_columns("predictions_dataset", "design_targets",
                                       invalid_columns, self.predictions_dataset.columns.values)

        invalid_minimization_targets = self._get_invalid_columns([t.label for t in design_targets.minimization_targets],
                                                                 self.predictions_dataset.columns)
        if len(invalid_minimization_targets) > 0:
            raise UserInputException(
                f"Minimization targets {invalid_minimization_targets} do not exist in dataset columns {self.predictions_dataset.columns.values}")

    def _validate_internals(self):
        for dt in self.datatypes:
            if type(dt) == pymoo.core.variable.Real:
                self._validate_has_field(dt.bounds != (None, None), "bounds", "Real")
            elif type(dt) == pymoo.core.variable.Integer:
                self._validate_has_field(dt.bounds != (None, None), "bounds", "Integer")
            elif type(dt) == pymoo.core.variable.Choice:
                self._validate_has_field(dt.options is not None, "options", "Choice")

    def _validate_has_field(self, condition: bool, field_name: str, class_name: str):
        validate(condition,
                 f"Parameter [datatypes] is invalid: {field_name} cannot be None for object of type "
                 f"pymoo.core.variable.{class_name}")

    def _validate_query_x_against_datatypes(self, query_x: pd.DataFrame):
        for i in range(len(self.datatypes)):
            dt = self.datatypes[i]
            val = query_x.values[0][i]
            if type(dt) in [Real, Integer]:
                self._validate_range(dt, val)
            if type(dt) is Choice:
                validate(val in dt.options,
                         QUERY_X_INVALID_DATATYPES_CHOICE)
            if type(dt) is Binary:
                validate(val in [True, False] or val == 1 or val == 0,
                         QUERY_X_INVALID_BINARY_TYPE)

    def _validate_range(self, dt: Union[Integer, Real], val: Union[float, int]):
        lower_bound = dt.bounds[0]
        upper_bound = dt.bounds[1]
        validate(val >= lower_bound,
                 QUERY_X_OUTSIDE_TYPES_RANGE)
        validate(val <= upper_bound,
                 QUERY_X_OUTSIDE_TYPES_RANGE)
