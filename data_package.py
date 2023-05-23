import numpy as np
import pandas as pd


class DataPackage:
    def __init__(self,
                 features_dataset: pd.DataFrame,
                 predictions_dataset: pd.DataFrame,
                 query_x: pd.DataFrame,
                 features_to_vary: list,
                 query_y: dict,
                 datatypes=None):
        features_dataset = self._features_to_dataframe_if_not(features_dataset, features_to_vary)
        predictions_dataset = self._predictions_to_dataframe_if_not(predictions_dataset, query_y)
        self.features_dataset = features_dataset
        self.predictions_dataset = predictions_dataset
        self.query_x = self._query_x_to_dataframe_if_not(query_x)
        self._validate_parameters(features_dataset, features_to_vary, self.query_x, predictions_dataset, query_y)
        self.features_to_vary = features_to_vary
        self.query_y = query_y
        self.datatypes = datatypes
        self.features_to_freeze = list(set(self.features_dataset) - set(self.features_to_vary))

    def to_dataframe(self, numpy_array: np.ndarray):
        index_based_columns = [_ for _ in range(numpy_array.shape[1])]
        return pd.DataFrame(numpy_array, columns=index_based_columns)

    def _features_to_dataframe_if_not(self, optional_nd_array, features_to_vary):
        if isinstance(optional_nd_array, np.ndarray):
            total_features = optional_nd_array.shape[1]
            self._validate_indices_to_vary(
                features_to_vary,
                total_features,
                "The list of features to vary must be a list of indices when the features dataset is a numpy array",
                "Invalid index provided in list of features to vary")
            return self.to_dataframe(optional_nd_array)
        return optional_nd_array

    def _predictions_to_dataframe_if_not(self, predictions_dataset, query_y):
        if isinstance(predictions_dataset, np.ndarray):
            total_features = predictions_dataset.shape[1]
            self._validate_indices_to_vary(query_y.keys(), total_features,
                                           "Query y must contain indices when the predictions dataset is a numpy array",
                                           "Invalid index provided in query y")
            return self.to_dataframe(predictions_dataset)
        return predictions_dataset

    def _validate_parameters(self, features_dataset, features_to_vary, query_x: pd.DataFrame, predictions_dataset,
                             query_y):
        self._validate_datasets(features_dataset, predictions_dataset)
        self._validate_features_to_vary(features_dataset, features_to_vary)
        self._validate_query_x(features_dataset, query_x)
        self._validate_query_y(predictions_dataset, query_y)
        self._validate_bonus_objs(predictions_dataset, query_y)  # TODO: fix bug here.
        # self._validate_bounds(features_to_vary, upper_bounds, lower_bounds)

    def _validate_datasets(self, features_dataset: pd.DataFrame, predictions_dataset: pd.DataFrame):
        assert len(features_dataset) == len(predictions_dataset), "Dimensional mismatch between provided datasets"
        nunique = features_dataset.nunique()
        uniform_cols = nunique[nunique == 1].index
        assert len(
            uniform_cols) == 0, f"Error: The following columns were found to contain completely uniform values: " \
                                f"{uniform_cols}. This is not allowed, since it blows proximity values up to infinity!"

    # def validate_bounds(self, features_to_vary: list, upper_bounds: np.array, lower_bounds: np.array):
    #     valid_length = len(features_to_vary)
    #     assert upper_bounds.shape == (valid_length,)
    #     assert lower_bounds.shape == (valid_length,)

    def _validate_features_to_vary(self, features_dataset: pd.DataFrame, features_to_vary: list):
        self._validate_labels(features_dataset, features_to_vary, "User has not provided any features to vary")

    def _validate_labels(self, dataset: pd.DataFrame, labels: list,
                         no_labels_message):
        assert len(labels) > 0, no_labels_message
        valid_labels = dataset.columns.values
        for label in labels:
            assert label in valid_labels, f"Expected label {label} to be in dataset {valid_labels}"

    def _validate_query_y(self, predictions_dataset: pd.DataFrame, query_y: dict):
        self._validate_labels(predictions_dataset,
                              list(query_y.keys()),
                              "User has not provided any performance targets")

    def _validate_bonus_objs(self, predictions_dataset: pd.DataFrame, bonus_objs: list):
        self._validate_labels(predictions_dataset,
                              bonus_objs,
                              "User has not provided any performance targets")

    def _validate_indices_to_vary(self, features_to_vary: list, total_features: int, type_error_message: str,
                                  invalid_error_message: str):
        for feature in features_to_vary:
            self._validate_int(feature, total_features, type_error_message,
                               invalid_error_message)

    def _validate_int(self, feature, total_features: int,
                      type_error_message,
                      invalid_error_message):
        try:
            is_int = (float(feature) == int(feature))
        except ValueError:
            raise AssertionError(type_error_message)
        if not (is_int and int(feature) < total_features):
            raise AssertionError(invalid_error_message)

    def _validate_query_x(self, features_dataset: pd.DataFrame, query_x: pd.DataFrame):
        assert query_x is not None, "Query x cannot be none!"
        assert query_x.values.shape == (
            1, len(features_dataset.columns)), "Dimensional mismatch between query x and dataset!"
        assert set(query_x.columns) == set(features_dataset.columns), "Query x columns do not match dataset columns!"

    def _query_x_to_dataframe_if_not(self, query_x):
        if isinstance(query_x, np.ndarray):
            return self.to_dataframe(query_x)
        assert isinstance(query_x, pd.DataFrame), "Query x is neither a dataframe nor an ndarray!"
        return query_x
