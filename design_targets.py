import numbers


class ContinuousTarget:
    def __init__(self, label, lower_bound, upper_bound):
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

    def _validate_float(self, bound, bound_name):
        self._validate(isinstance(bound, numbers.Real), f"{bound_name} must be a real number")

    def _validate(self, mandatory_condition, exception_message):
        if not mandatory_condition:
            raise ValueError(exception_message)


class ClassificationTarget:
    pass


class ProbabilityTarget:
    pass
