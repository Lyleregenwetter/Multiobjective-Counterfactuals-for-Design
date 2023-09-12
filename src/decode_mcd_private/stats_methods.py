import numpy as np
import pandas as pd


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
    return GD.astype(float)


def categorical_gower(first: pd.DataFrame, second: pd.DataFrame):
    return categorical_gower_np(first.values, second.values)


def categorical_gower_np(first: np.ndarray, second: np.ndarray):
    return np.divide(np.count_nonzero(first - second, axis=1), second.shape[1])


def euclidean_distance(dataframe: pd.DataFrame, reference: pd.DataFrame):
    rows = dataframe.values
    reference_row = reference.iloc[0].values
    return np.linalg.norm((rows - reference_row), axis=1)


def avg_gower_distance(dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame,
                       ranges, datatypes, k=3) -> np.array:
    k = min(k, len(reference_dataframe))
    GD = mixed_gower(dataframe, reference_dataframe, ranges, datatypes)
    bottomk = np.partition(GD, kth=k - 1, axis=1)[:, :k]
    return np.mean(bottomk, axis=1)


def gower_distance(dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame, ranges):
    dists = np.expand_dims(dataframe.values, 1) - np.expand_dims(reference_dataframe.values, 0)
    scaled_dists = np.divide(dists, ranges)
    GD = np.mean(np.abs(scaled_dists), axis=2)
    return GD


def changed_features_ratio(designs_dataframe: pd.DataFrame,
                           reference_dataframe: pd.DataFrame,
                           n_features: int):
    designs = designs_dataframe.values
    reference = reference_dataframe.iloc[0].values
    changes = np.count_nonzero(np.not_equal(designs, reference), axis=1)
    return changes / n_features


def to_dataframe(numpy_array: np.ndarray):
    dummy_columns = [_ for _ in range(numpy_array.shape[1])]
    return pd.DataFrame(numpy_array, columns=dummy_columns)
