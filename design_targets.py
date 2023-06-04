import numbers
from abc import ABCMeta, abstractmethod
from typing import Union, Tuple


class McdTarget(metaclass=ABCMeta):
    def _validate(self, mandatory_condition: bool, exception_message: str):
        if not mandatory_condition:
            raise ValueError(exception_message)

    @abstractmethod
    def _validate_fields(self):
        pass

    def _validate_label(self, label: str):
        self._validate(isinstance(label, (str, int)),
                       "Label must be of type string or an integer index")
        self._validate(len(str(label)) != 0,
                       "Label cannot be an empty string")


class ContinuousTarget(McdTarget):
    def __init__(self, label: Union[str, int], lower_bound: float, upper_bound: float):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._validate_fields()

    def _validate_fields(self):
        self._validate_label(self.label)
        self._validate_float(self.lower_bound, "Lower bound")
        self._validate_float(self.upper_bound, "Upper bound")
        self._validate(self.upper_bound > self.lower_bound, "Lower bound cannot be greater or equal to upper bound")

    def _validate_float(self, bound, bound_name: str):
        self._validate(isinstance(bound, numbers.Real), f"{bound_name} must be a real number")


class ClassificationTarget(McdTarget):
    def __init__(self, label: Union[str, int], desired_classes: Tuple[Union[str, int]]):
        self.label = label
        self.desired_classes = desired_classes
        self._validate_fields()

    def _validate_fields(self):
        self._validate_label(self.label)
        self._validate(isinstance(self.desired_classes, tuple), "Desired classes must be a tuple")
        # noinspection PyTypeChecker
        self._validate(len(self.desired_classes) > 0, "Desired classes cannot be empty")
        for desired_class in self.desired_classes:
            self._validate(isinstance(desired_class, int), "Desired classes must be an all-integer tuple")


class ProbabilityTarget(McdTarget):
    def __init__(self, labels: Union[Tuple[str, ...], Tuple[int, ...]],
                 preferred_labels: Union[Tuple[str, ...], Tuple[int, ...]]):
        self.labels = labels
        self.preferred_labels = preferred_labels
        self._validate_fields()

    def _validate_fields(self):
        self._validate_tuple(self.labels, "Labels")
        self._validate_tuple(self.preferred_labels, "Preferred labels")
        self._validate(len(self.labels) > 1, "Labels must have a length greater than 1")
        self._validate(len(self.preferred_labels) > 0, "Preferred labels cannot be empty")
        self._validate_type_consistency(self.labels, "labels")
        self._validate_type_consistency(self.preferred_labels, "preferred labels")
        self._validate_no_empty_strings(self.labels, "Labels")
        self._validate_no_empty_strings(self.preferred_labels, "Preferred labels")
        self._validate(set(self.preferred_labels).issubset(self.labels),
                       "Preferred labels must be a subset of labels")

    def _validate_tuple(self, _tuple, tuple_name):
        self._validate(isinstance(_tuple, tuple), f"{tuple_name} must be a tuple")

    def _validate_type_consistency(self, _tuple, tuple_name):
        list_of_types = [type(element) for element in _tuple]
        count_strings = list_of_types.count(str)
        count_integers = list_of_types.count(int)
        exception_message = f"Expected {tuple_name} to be an all-integer or all-string tuple"
        self._validate((count_integers == 0) or (count_strings == 0), exception_message)
        self._validate((count_integers + count_strings) == len(list_of_types), exception_message)

    def _validate_no_empty_strings(self, _tuple, tuple_name):
        lengths = [len(str(element)) for element in _tuple]
        self._validate(0 not in lengths, f"{tuple_name} cannot contain empty strings")
