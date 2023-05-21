import pandas as pd


def get_one_row_dataframe_from_dict(model_input_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([list(model_input_dict.values())], columns=list(model_input_dict.keys()))


def get_dict_from_first_row(dataframe: pd.DataFrame) -> dict:
    return dataframe.loc[__first_row_index(dataframe)].to_dict()


def __first_row_index(dataframe) -> int:
    return dataframe.index.values[0]
