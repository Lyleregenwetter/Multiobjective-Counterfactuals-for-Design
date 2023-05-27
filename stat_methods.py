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
