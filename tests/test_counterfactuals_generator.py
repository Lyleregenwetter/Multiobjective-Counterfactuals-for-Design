import unittest

import numpy as np
import pandas as pd
from pymoo.core.variable import Real
import numpy.testing as np_test

from decode_mcd.counterfactuals_generator import RevertToQueryRepair
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets, ContinuousTarget
from decode_mcd.multi_objective_problem import MultiObjectiveProblem


class CounterfactualsGeneratorTest(unittest.TestCase):
    def test_query_repair_does_not_revert_to_x_values_outside_range(self):
        repair = RevertToQueryRepair()
        package = self.build_package(np.array([[11, 12, 13]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real((0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        np_test.assert_equal(repaired_array, pd.DataFrame.from_records(z).values)

    def test_percentage_reverted(self):
        repair = RevertToQueryRepair()
        package = self.build_package(np.array([[11, 12, 13]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real((0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        differences = repaired_array - pd.DataFrame.from_records(z).values
        changed = np.count_nonzero(differences)
        percentage_reverted = changed / 300_000
        self.assertGreaterEqual(percentage_reverted, 0.05)
        self.assertLessEqual(percentage_reverted, 0.07)

    def build_package(self, query_x, datatypes):
        _array = np.array([[5, 10, 15], [12, 15, 123], [13, 145, 13]])
        package = DataPackage(features_dataset=_array,
                              predictions_dataset=_array,
                              query_x=query_x,
                              design_targets=DesignTargets([ContinuousTarget(0, 10, 15)]),
                              datatypes=datatypes)
        return package
