from pymoo.core.mixed import MixedVariableDuplicateElimination


class EfficientMixedVariableDuplicateElimination(MixedVariableDuplicateElimination):

    def is_equal(self, a, b):
        return a.X == b.X
