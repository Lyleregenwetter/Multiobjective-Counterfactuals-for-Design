import numpy as np
import pandas as pd


class ClassificationEvaluator:
    def evaluate_categorical(self, actual: pd.DataFrame, targets: np.ndarray):
        """Takes an n × m actual_performances array and a targets array consisting of m arrays.
        Checks that for each row in n, the value of each column in m is in the corresponding targets array.
        Examples provided in ClassifierWrapperTest.
        """
        if actual.empty:
            return np.array([])
        # TODO: make this more efficient.
        values = actual.values
        assert values.shape[1] == targets.shape[0], \
            "Dimensional mismatch between actual performances and targets array"
        num_columns = values.shape[1]
        # noinspection PyUnresolvedReferences
        result = np.isin(values[:, 0], targets[0]).astype(int).reshape(values.shape[0], 1)
        for i in range(1, num_columns):
            # noinspection PyUnresolvedReferences
            result = np.concatenate([result, np.isin(values[:, i],
                                                     targets[i])
                                    .astype(int).reshape(values.shape[0], 1)],
                                    axis=1)
        return result

    def evaluate_proba(self, actual: pd.DataFrame, target_classes_indices: tuple):
        if actual.empty:
            return np.array([])
        unwanted_classes = tuple(set(actual.columns.to_list()) - set(target_classes_indices))
        max_desired = np.max(actual.loc[:, target_classes_indices], axis=1)
        max_undesired = np.max(actual.loc[:, unwanted_classes], axis=1)
        return (max_desired - max_undesired).values.reshape(actual.shape[0], 1)
