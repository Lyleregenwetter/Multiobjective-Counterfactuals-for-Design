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


class ContinuousTarget(McdTarget):
    def __init__(self, label: Union[str, int], lower_bound: float, upper_bound: float):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._validate_fields()

    def _validate_fields(self):
        self._validate(isinstance(self.label, (str, int)), "Label must be of type string or an integer index")
        self._validate(len(str(self.label)) != 0, "Label cannot be an empty string")
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
        self._validate(isinstance(self.label, (int, str)), "Label must be of type string or an integer index")
        self._validate(len(str(self.label)) != 0, "Label cannot be an empty string")
        self._validate(isinstance(self.desired_classes, tuple), "Desired classes must be a tuple")
        # noinspection PyTypeChecker
        self._validate(len(self.desired_classes) > 0, "Desired classes cannot be empty")
        for desired_class in self.desired_classes:
            self._validate(isinstance(desired_class, int), "Desired classes must be an all-integer tuple")


class ProbabilityTarget:
    pass
