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
    return GD


def categorical_gower(first: pd.DataFrame, second: pd.DataFrame):
    return categorical_gower_np(first.values, second.values)


def categorical_gower_np(first: np.ndarray, second: np.ndarray):
    return np.divide(np.count_nonzero(first - second, axis=1), second.shape[1])


def euclidean_distance(dataframe: pd.DataFrame, reference: pd.DataFrame):
    reference_row = reference.iloc[0]
    changes = dataframe.apply(lambda row: np.linalg.norm(row - reference_row), axis=1)
    return changes.values


def np_euclidean_distance(designs_matrix: np.array, reference_design: np.array):
    n_columns = reference_design.shape[1]
    return euclidean_distance(alt_to_dataframe(designs_matrix, n_columns),
                              alt_to_dataframe(reference_design, n_columns))


def np_avg_gower_distance(designs_matrix: np.array, reference_designs: np.array, ranges, k=3) -> np.array:
    GD = np_gower_distance(designs_matrix, reference_designs, ranges)
    bottomk = np.partition(GD, kth=k - 1, axis=1)[:, :k]
    return np.mean(bottomk, axis=1)


def avg_gower_distance(dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame,
                       ranges, datatypes, k=3) -> np.array:  # TODO batch this for memory savings
    k = min(k, len(reference_dataframe))
    GD = mixed_gower(dataframe, reference_dataframe, ranges, datatypes)
    bottomk = np.partition(GD, kth=k - 1, axis=1)[:, :k]
    return np.mean(bottomk, axis=1)


def gower_distance(dataframe: pd.DataFrame, reference_dataframe: pd.DataFrame, ranges):
    dists = np.expand_dims(dataframe.values, 1) - np.expand_dims(reference_dataframe.values, 0)
    scaled_dists = np.divide(dists, ranges)
    GD = np.mean(np.abs(scaled_dists), axis=2)
    return GD


def np_changed_features_ratio(designs_matrix: np.array, reference_design: np.array, n_features: int):
    designs_matrix, reference_design = to_dataframe(designs_matrix), to_dataframe(reference_design)
    return changed_features_ratio(designs_matrix, reference_design, n_features)


def changed_features_ratio(designs_dataframe: pd.DataFrame,
                           reference_dataframe: pd.DataFrame,
                           n_features: int):
    changes = designs_dataframe.apply(
        lambda row: np.count_nonzero(row.values - reference_dataframe.iloc[0].values), axis=1)
    return changes.values / n_features


def np_gower_distance(designs_matrix: np.array, reference_design: np.array, ranges):
    return gower_distance(to_dataframe(designs_matrix), to_dataframe(reference_design), ranges)


def to_dataframe(numpy_array: np.ndarray):
    dummy_columns = [_ for _ in range(numpy_array.shape[1])]
    return pd.DataFrame(numpy_array, columns=dummy_columns)


def alt_to_dataframe(matrix: np.array, number_of_columns: int):
    return pd.DataFrame(matrix, columns=[_ for _ in range(number_of_columns)])
