import numpy as np
import numpy.testing as np_test
import pandas as pd


def generate_regression_column(number_of_rows: int, lower_bound: float, upper_bound: float):
    return (np.random.rand(number_of_rows, 1) * (upper_bound - lower_bound)) + lower_bound


def generate_categorical_column(number_of_rows: int, numer_of_categories: int):
    return np.round(np.random.rand(number_of_rows, 1) * (numer_of_categories - 1)).astype("int32")


if __name__ == "__main__":
    n_rows = 500
    r1 = generate_regression_column(n_rows, 0, 100)
    r2 = generate_regression_column(n_rows, 0, 15)
    c1 = generate_categorical_column(n_rows, 5)
    c2 = generate_categorical_column(n_rows, 3)
    p1 = np.random.rand(n_rows, 1) * 0.5
    p2 = np.random.rand(n_rows, 1) * 0.5
    p3 = 1 - (p1 + p2)

    proba_data = np.concatenate([p1, p2, p3], axis=1)
    np_test.assert_equal(np.sum(proba_data, axis=1), 1)
    c1_valid = np.isin(c1, np.array([0, 1, 2, 3, 4]))
    c2_valid = np.isin(c2, np.array([0, 1, 2]))
    np_test.assert_equal(c1_valid, 1)
    np_test.assert_equal(c2_valid, 1)

    x_r1 = generate_regression_column(500, 50, 500)
    x_r2 = generate_regression_column(500, 30, 50)
    x_c1 = generate_categorical_column(500, 3)
    x_c2 = generate_categorical_column(500, 2)

    x_c1_valid = np.isin(x_c1, np.array([0, 1, 2]))
    x_c2_valid = np.isin(x_c2, np.array([0, 1]))
    np_test.assert_equal(x_c1_valid, 1)
    np_test.assert_equal(x_c2_valid, 1)

    x = np.concatenate([x_r1, x_c1, x_r2, x_c2], axis=1)
    y = np.concatenate([r1, c1, p1, r2, c2, p2, p3], axis=1)
    pd.DataFrame.from_records(x, columns=["R1", "C1", "R2", "C2"]).to_csv("toy_x.csv", index=False)
    pd.DataFrame.from_records(y, columns=["O_R1", "O_C1", "O_P1",
                                          "O_R2", "O_C2", "O_P2", "O_P3"]).to_csv("toy_y.csv", index=False)
