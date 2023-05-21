import numpy as np


class ClassifierWrapper:
    def evaluate_categorical(self, actual: np.ndarray, targets: np.ndarray):
        """Takes an n × m actual_performances array and a targets array consisting of m arrays.
        Checks that for each row in n, the value of each column in m is in the corresponding targets array.
        Examples provided in ClassifierWrapperTest.
        """
        # TODO: make this more efficient.
        assert actual.shape[1] == targets.shape[0], "Dimensional mismatch between actual performances and targets array"
        num_columns = actual.shape[1]
        # noinspection PyUnresolvedReferences
        result = np.isin(actual[:, 0], targets[0]).astype(int).reshape(actual.shape[0], 1)
        for i in range(1, num_columns):
            # noinspection PyUnresolvedReferences
            result = np.concatenate([result, np.isin(actual[:, i], targets[i]).astype(int).reshape(actual.shape[0], 1)],
                                    axis=1)
        return result

    def evaluate_proba(self, actual: np.ndarray, target_classes_indices: tuple):
        unwanted_classes = tuple(set([_ for _ in range(actual.shape[1])]) - set(target_classes_indices))
        max_desired = np.max(actual[:, target_classes_indices].reshape(actual.shape[0], -1), axis=1)
        max_undesired = np.max(actual[:, unwanted_classes].reshape(actual.shape[0], -1), axis=1)
        # TODO:
        return (max_desired - max_undesired).reshape(actual.shape[0], 1)
