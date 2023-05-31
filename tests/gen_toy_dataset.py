import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler

NUMBER_OF_SAMPLES = 500

MEANING_OF_LIFE = 42


def generate_categorical_column(number_of_rows: int, numer_of_categories: int):
    return np.round(np.random.rand(number_of_rows, 1) * (numer_of_categories - 1)).astype("int32")


def make_proba(number_of_samples):
    x, y = make_regression(n_samples=number_of_samples, n_features=2, n_informative=1, random_state=MEANING_OF_LIFE)
    y = np.expand_dims(y, 1)
    scaled_x = scale(x)
    scaled_y = scale(y)
    scaled_y = scaled_y + abs(scaled_y.min())  # now they're all positive
    scaled_y = scaled_y / scaled_y.max()  # now they're all between 0 and 1
    second_class_proba = 1 - scaled_y
    return scaled_x, np.concatenate([scaled_y, second_class_proba], axis=1)


def scale(x):
    x_scaler = StandardScaler()
    x_scaler.fit(x)
    scaled_x = x_scaler.transform(x)
    return scaled_x


if __name__ == "__main__":
    classification = make_classification(n_samples=NUMBER_OF_SAMPLES, n_features=3, n_repeated=0,
                                         n_redundant=0, n_informative=2, random_state=MEANING_OF_LIFE)
    regression = make_regression(n_samples=NUMBER_OF_SAMPLES, n_features=2,
                                 n_informative=1, random_state=MEANING_OF_LIFE)
    proba = make_proba(NUMBER_OF_SAMPLES)
    uninformative_categorical_column = generate_categorical_column(NUMBER_OF_SAMPLES, 3)
    x_data = np.concatenate([classification[0], uninformative_categorical_column,
                             regression[0]], axis=1)
    y_data = np.concatenate([np.expand_dims(classification[1], 1),
                             np.expand_dims(regression[1], 1),
                             proba[1]], axis=1)
    x_df = pd.DataFrame(x_data, columns=["R1", "R2", "R3", "C1", "R4", "R5"])
    x_df["C1"] = x_df["C1"].astype("category")
    y_df = pd.DataFrame(y_data, columns=["O_C1", "O_R1", "O_P1", "O_P2"])
    x_df.to_csv("toy_x.csv", index=False)
    y_df.to_csv("toy_y.csv", index=False)
