import itertools
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

    def _validate_label(self, label: str):
        validate(isinstance(label, (str, int)),
                 "Label must be of type string or an integer index")
        validate(len(str(label)) != 0,
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
        validate(self.upper_bound > self.lower_bound, "Lower bound cannot be greater or equal to upper bound")

    def _validate_float(self, bound, bound_name: str):
        validate(isinstance(bound, numbers.Real), f"{bound_name} must be a real number")


class ClassificationTarget(McdTarget):
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


class ProbabilityTarget(McdTarget):
    def __init__(self, labels: Union[Sequence[str], Sequence[int]],
                 preferred_labels: Union[Sequence[str], Sequence[int]]):
        self.labels = labels
        self.preferred_labels = preferred_labels
        self._validate_fields()

    def _validate_fields(self):
        self._validate_sequence(self.labels, "Labels")
        self._validate_sequence(self.preferred_labels, "Preferred labels")
        validate(len(self.labels) > 1, "Labels must have a length greater than 1")
        validate(len(self.preferred_labels) > 0, "Preferred labels cannot be empty")
        self._validate_type_consistency(self.labels, "labels")
        self._validate_type_consistency(self.preferred_labels, "preferred labels")
        self._validate_no_empty_strings(self.labels, "Labels")
        self._validate_no_empty_strings(self.preferred_labels, "Preferred labels")
        validate(set(self.preferred_labels).issubset(self.labels),
                 "Preferred labels must be a subset of labels")

    def _validate_sequence(self, _object, object_name):
        validate(isinstance(_object, Sequence), f"{object_name} must be a sequence")

    def _validate_type_consistency(self, _sequence, sequence_name):
        list_of_types = [type(element) for element in _sequence]
        count_strings = list_of_types.count(str)
        count_integers = list_of_types.count(int)
        exception_message = f"Expected {sequence_name} to be an all-integer or all-string sequence"
        validate((count_integers == 0) or (count_strings == 0), exception_message)
        validate((count_integers + count_strings) == len(list_of_types), exception_message)

    def _validate_no_empty_strings(self, _sequence, sequence_name):
        lengths = [len(str(element)) for element in _sequence]
        validate(0 not in lengths, f"{sequence_name} cannot contain empty strings")


class DesignTargets:
    def __init__(self, continuous_targets: Sequence[ContinuousTarget] = None,
                 classification_targets: Sequence[ClassificationTarget] = None,
                 probability_targets: Sequence[ProbabilityTarget] = None):
        self.continuous_targets = self._get_or_default(continuous_targets, ())
        self.classification_targets = self._get_or_default(classification_targets, ())
        self.probability_targets = self._get_or_default(probability_targets, ())
        self._validate_fields()

    def get_all_constrained_labels(self):
        return self._get_probability_labels() + self.get_continuous_labels() + self.get_classification_labels()

    def count_constrained_labels(self):
        return len(self.get_all_constrained_labels())

    def _get_or_default(self, value, default_value):
        if value is not None:
            return value
        return default_value

    def _get_probability_labels(self) -> Tuple[str, ...]:
        return tuple(itertools.chain.from_iterable([probability_target.labels for probability_target
                                                    in self.probability_targets]))

    def _validate_fields(self):
        self._validate_sequence(self.continuous_targets, "Continuous targets", ContinuousTarget)
        self._validate_sequence(self.classification_targets, "Classification targets", ClassificationTarget)
        self._validate_sequence(self.probability_targets, "Probability targets", ProbabilityTarget)
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
        classification = set(self.get_classification_labels())
        probability = set(self._get_probability_labels())
        len_union = len((continuous | classification) | probability)
        validate(self.count_constrained_labels() == len_union,
                 "Label was specified twice in targets")

    def get_continuous_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        lower_bounds = [target.lower_bound for target in self.continuous_targets]
        upper_bounds = [target.upper_bound for target in self.continuous_targets]
        return np.array(lower_bounds), np.array(upper_bounds)

    def get_desired_classes(self) -> List[Union[Sequence[str], Sequence[int]]]:
        return [target.desired_classes for target in self.classification_targets]

    def get_preferred_probability_targets(self) -> List[Union[Sequence[str], Sequence[int]]]:
        return [target.preferred_labels for target in self.probability_targets]

    def get_continuous_labels(self) -> Tuple[str, ...]:
        return tuple(target.label for target in self.continuous_targets)

    def get_classification_labels(self) -> Tuple[str, ...]:
        return tuple(target.label for target in self.classification_targets)

    def get_probability_labels(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(target.labels for target in self.probability_targets)
