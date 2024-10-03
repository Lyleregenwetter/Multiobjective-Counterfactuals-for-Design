import numbers
from abc import ABCMeta, abstractmethod
from typing import Union, Sequence, Tuple, List

import numpy as np

from decode_mcd.mcd_exceptions import UserInputException
from decode_mcd_private.validation_utils import validate


class McdTarget(metaclass=ABCMeta):

    @abstractmethod
    def _validate_fields(self):
        pass

    def _validate_label(self, label: Union[str, int]):
        validate(isinstance(label, (str, int)),
                 "Label must be of type string or an integer index")
        validate(len(str(label)) != 0,
                 "Label cannot be an empty string")


class MinimizationTarget(McdTarget):
    def __init__(self, label: Union[str, int]):
        self.label = label
        self._validate_fields()

    def _validate_fields(self):
        self._validate_label(self.label)


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
        validate(self.upper_bound > self.lower_bound, "Lower bound cannot be greater or equal to upper bound")

    def _validate_float(self, bound, bound_name: str):
        validate(isinstance(bound, numbers.Real), f"{bound_name} must be a real number")


class CategoricalTarget(McdTarget):
    def __init__(self, label: Union[str, int], desired_classes: Union[Sequence[str], Sequence[int]]):
        self.label = label
        self.desired_classes = desired_classes
        self._validate_fields()

    def _validate_fields(self):
        self._validate_label(self.label)
        validate(isinstance(self.desired_classes, Sequence), "Desired classes must be a sequence")
        # noinspection PyTypeChecker
        validate(len(self.desired_classes) > 0, "Desired classes cannot be empty")
        for desired_class in self.desired_classes:
            validate(isinstance(desired_class, int), "Desired classes must be an all-integer sequence")


class DesignTargets:
    def __init__(self, continuous_targets: Sequence[ContinuousTarget] = None,
                 categorical_targets: Sequence[CategoricalTarget] = None,
                 # TODO: remove probability targets
                 minimization_targets: Sequence[MinimizationTarget] = None):
        self.continuous_targets = self._get_or_default(continuous_targets, ())
        self.categorical_targets = self._get_or_default(categorical_targets, ())
        self.minimization_targets = self._get_or_default(minimization_targets, ())
        self._validate_fields()

    def get_all_constrained_labels(self):
        return self.get_continuous_labels() + self.get_categorical_labels()

    def count_constrained_labels(self):
        return len(self.get_all_constrained_labels())

    def _get_or_default(self, value, default_value):
        if value is not None:
            return value
        return default_value

    def _validate_fields(self):
        self._validate_sequence(self.continuous_targets, "Continuous targets", ContinuousTarget)
        self._validate_sequence(self.categorical_targets, "Categorical targets", CategoricalTarget)
        self._validate_no_crossover()
        validate(self.count_constrained_labels() > 0, "Design targets must be provided")

    def _validate(self, mandatory_condition: bool, exception_message: str):
        if not mandatory_condition:
            raise UserInputException(exception_message)

    def _validate_sequence(self, _sequence, _sequence_name, element_type):
        validate(isinstance(_sequence, Sequence), f"{_sequence_name} must be a sequence")
        for element in _sequence:
            validate(isinstance(element, element_type),
                     f"{_sequence_name} must be composed of elements of class {element_type.__name__}")

    def _validate_no_crossover(self):
        continuous = set(self.get_continuous_labels())
        categorical = set(self.get_categorical_labels())
        len_union = len((continuous | categorical))
        validate(self.count_constrained_labels() == len_union,
                 "Label was specified twice in targets")

    def get_continuous_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        lower_bounds = [target.lower_bound for target in self.continuous_targets]
        upper_bounds = [target.upper_bound for target in self.continuous_targets]
        return np.array(lower_bounds), np.array(upper_bounds)

    def get_desired_classes(self) -> List[Union[Sequence[str], Sequence[int]]]:
        return [target.desired_classes for target in self.categorical_targets]

    def get_continuous_labels(self) -> Tuple[str, ...]:
        return tuple(target.label for target in self.continuous_targets)

    def get_categorical_labels(self) -> Tuple[str, ...]:
        return tuple(target.label for target in self.categorical_targets)
