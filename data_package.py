import numpy as np
import pandas as pd


class DataPackage:
    def __init__(self,
                 features_dataset: pd.DataFrame,
                 predictions_dataset: pd.DataFrame,
                 query_x: pd.DataFrame,
                 features_to_vary: list,
                 query_y: dict,
                 y_classification_targets: dict = None,
                 y_proba_targets: dict = None,
                 datatypes=None):
        self.features_dataset = self._attempt_features_to_dataframe(features_dataset, features_to_vary)
        self.predictions_dataset = self._attempt_predictions_to_dataframe(predictions_dataset, query_y.keys())
        self.features_to_vary = features_to_vary
        self.query_x = self._query_x_to_dataframe_if_not(query_x)
        self.query_y = query_y
        self._validate_fields(self.features_dataset, self.features_to_vary,
                              self.query_x, self.predictions_dataset, query_y)
        self.datatypes = datatypes
        self.features_to_freeze = list(set(self.features_dataset) - set(self.features_to_vary))
        self.y_classification_targets = self._get_or_default(y_classification_targets, {})
        self.y_proba_targets = self._get_or_default(y_proba_targets, {})

    def _get_or_default(self, value, default_value):
        if value is None:
            return default_value
        return value

    def sort_query_y(self):
        query_constraints = []
        query_lb = []
        query_ub = []
        for key in self.query_y.keys():
            query_constraints.append(key)
            query_lb.append(self.query_y[key][0])
            query_ub.append(self.query_y[key][1])
        return query_constraints, np.array(query_lb), np.array(query_ub)

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

    def _validate_fields(self,
                         features_dataset,
                         features_to_vary,
                         query_x: pd.DataFrame,
                         predictions_dataset,
                         query_y):
        self._validate_datasets(features_dataset, predictions_dataset)
        self._validate_features_to_vary(features_dataset, features_to_vary)
        self._validate_query_x(features_dataset, query_x)
        self._validate_query_y(predictions_dataset, query_y)
        self._validate_bonus_objs(predictions_dataset, query_y)  # TODO: fix bug here.
        # self._validate_bounds(features_to_vary, upper_bounds, lower_bounds)

    def _validate_datasets(self, features_dataset: pd.DataFrame, predictions_dataset: pd.DataFrame):
        self._validate(len(features_dataset) == len(predictions_dataset),
                       "Dimensional mismatch between provided datasets")
        nunique = features_dataset.nunique()
        uniform_cols = nunique[nunique == 1].index
        self._validate(len(uniform_cols) == 0, f"""Error: The following columns were found to 
            contain completely uniform values: {uniform_cols}. This is not allowed, 
            since it blows proximity values up to infinity!""")

    # def validate_bounds(self, features_to_vary: list, upper_bounds: np.array, lower_bounds: np.array):
    #     valid_length = len(features_to_vary)
    #     assert upper_bounds.shape == (valid_length,)
    #     assert lower_bounds.shape == (valid_length,)

    def _validate_features_to_vary(self, features_dataset: pd.DataFrame, features_to_vary: list):
        self._validate_labels(features_dataset, features_to_vary, "User has not provided any features to vary")

    def _validate_labels(self, dataset: pd.DataFrame, labels: list,
                         no_labels_message):
        self._validate(len(labels) > 0, no_labels_message)
        valid_labels = dataset.columns.values
        for label in labels:
            self._validate(label in valid_labels, f"Expected label {label} to be in dataset {valid_labels}")

    def _validate_query_y(self, predictions_dataset: pd.DataFrame, query_y: dict):
        self._validate_labels(predictions_dataset,
                              list(query_y.keys()),
                              "User has not provided any performance targets")

    def _validate_bonus_objs(self, predictions_dataset: pd.DataFrame, bonus_objs: list):
        self._validate_labels(predictions_dataset,
                              bonus_objs,
                              "User has not provided any performance targets")

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

    def _validate_query_x(self, features_dataset: pd.DataFrame, query_x: pd.DataFrame):
        self._validate(not query_x.empty, "Query x cannot be empty!")
        self._validate(query_x.values.shape == (1, len(features_dataset.columns)),
                       "Dimensional mismatch between query x and dataset!")
        self._validate(set(query_x.columns) == set(features_dataset.columns),
                       "Query x columns do not match dataset columns!")

    def _query_x_to_dataframe_if_not(self, query_x):
        if isinstance(query_x, np.ndarray):
            return self._to_dataframe(query_x)
        self._validate(isinstance(query_x, pd.DataFrame), "Query x is neither a dataframe nor an ndarray!")
        return query_x

    def _validate(self, mandatory_condition: bool, exception_message: str):
        if not mandatory_condition:
            raise ValueError(exception_message)
