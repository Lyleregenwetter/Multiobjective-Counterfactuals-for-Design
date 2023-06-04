import numpy as np
import pandas as pd

from design_targets import DesignTargets


class DataPackage:
    def __init__(self,
                 features_dataset: pd.DataFrame,
                 predictions_dataset: pd.DataFrame,
                 query_x: pd.DataFrame,
                 design_targets: DesignTargets,
                 features_to_vary: list,
                 bonus_objectives: list = None,
                 datatypes=None):
        """"""
        self.features_dataset = self._attempt_features_to_dataframe(features_dataset, features_to_vary)
        self.predictions_dataset = self._attempt_predictions_to_dataframe(predictions_dataset,
                                                                          design_targets.get_continuous_labels())
        self.query_x = self._query_x_to_dataframe_if_not(query_x)
        self.features_to_vary = features_to_vary
        self.design_targets = design_targets
        self.bonus_objectives = self._get_or_default(bonus_objectives, [])
        self.datatypes = datatypes
        self._validate_fields()
        self.features_to_freeze = list(set(self.features_dataset) - set(self.features_to_vary))

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def _to_dataframe(self, numpy_array: np.ndarray):
        index_based_columns = [_ for _ in range(numpy_array.shape[1])]
        return pd.DataFrame(numpy_array, columns=index_based_columns)

    def _attempt_features_to_dataframe(self, features_dataset, features_to_vary):
        return self._attempt_to_dataframe(
            features_dataset, features_to_vary,
            "The list of features to vary must be a list of indices when the features dataset is a numpy array",
            "Invalid index provided in list of features to vary")

    def _attempt_predictions_to_dataframe(self, predictions_dataset, query_y):
        return self._attempt_to_dataframe(
            predictions_dataset, query_y,
            "Query y must contain indices when the predictions dataset is a numpy array",
            "Invalid index provided in query y"
        )

    def _attempt_to_dataframe(self, dataset, provided_features, type_error_message, invalid_error_message):
        if isinstance(dataset, np.ndarray):
            self._validate_indices_to_vary(provided_features,
                                           dataset.shape[1],
                                           type_error_message,
                                           invalid_error_message)
            return self._to_dataframe(dataset)
        return dataset

    def _validate_fields(self):
        self._validate_datasets()
        self._validate_features_to_vary()
        self._validate_query_x()
        self._validate_query_y()
        self._validate_bonus_objs()  # TODO: fix bug here.
        # self._validate_bounds(features_to_vary, upper_bounds, lower_bounds)

    def _validate_datasets(self):
        self._validate(len(self.features_dataset) == len(self.predictions_dataset),
                       "Dimensional mismatch between provided datasets")
        nunique = self.features_dataset.nunique()
        uniform_cols = nunique[nunique == 1].index
        self._validate(len(uniform_cols) == 0, f"""Error: The following columns were found to 
            contain completely uniform values: {uniform_cols}. This is not allowed, 
            since it blows proximity values up to infinity!""")

    # def validate_bounds(self, features_to_vary: list, upper_bounds: np.array, lower_bounds: np.array):
    #     valid_length = len(features_to_vary)
    #     assert upper_bounds.shape == (valid_length,)
    #     assert lower_bounds.shape == (valid_length,)

    def _validate_features_to_vary(self):
        self._validate_labels(self.features_dataset, self.features_to_vary,
                              "User has not provided any features to vary")

    def _validate_labels(self, dataset: pd.DataFrame, labels: list,
                         no_labels_message):
        self._validate(len(labels) > 0, no_labels_message)
        valid_labels = dataset.columns.values
        for label in labels:
            self._validate(label in valid_labels, f"Expected label {label} to be in dataset {valid_labels}")

    def _validate_query_y(self):
        self._validate_labels(self.predictions_dataset,
                              self.design_targets.get_continuous_labels(),
                              "User has not provided any performance targets")

    def _validate_bonus_objs(self):
        self._validate(set(self.bonus_objectives).issubset(set(self.predictions_dataset.columns)),
                       "Bonus objectives should be a subset of labels!")

    def _validate_indices_to_vary(self, features_to_vary: list, number_of_features: int, type_error_message: str,
                                  invalid_error_message: str):
        for feature in features_to_vary:
            self._validate_int(feature, number_of_features, type_error_message, invalid_error_message)

    def _validate_int(self, feature, number_of_features: int,
                      type_error_message,
                      invalid_error_message):
        try:
            is_int = (float(feature) == int(feature))
        except ValueError:
            raise ValueError(type_error_message)
        if not (is_int and int(feature) < number_of_features):
            raise ValueError(invalid_error_message)

    def _validate_query_x(self):
        self._validate(not self.query_x.empty, "Query x cannot be empty!")
        self._validate(self.query_x.values.shape == (1, len(self.features_dataset.columns)),
                       "Dimensional mismatch between query x and dataset!")
        self._validate(set(self.query_x.columns) == set(self.features_dataset.columns),
                       "Query x columns do not match dataset columns!")

    def _query_x_to_dataframe_if_not(self, query_x):
        if isinstance(query_x, np.ndarray):
            return self._to_dataframe(query_x)
        self._validate(isinstance(query_x, pd.DataFrame), "Query x is neither a dataframe nor an ndarray!")
        return query_x

    def _validate(self, mandatory_condition: bool, exception_message: str):
        if not mandatory_condition:
            raise ValueError(exception_message)
