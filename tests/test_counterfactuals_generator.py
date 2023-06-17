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
    def test_revert_none_when_all_outside_valid_range(self):
        repair = RevertToQueryRepair()
        package = self.build_package(np.array([[11, 12, 13]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        revertible = problem._get_revertible_indexes()
        self.assertEqual((), revertible)
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        initial_array = pd.DataFrame.from_records(z).values
        np_test.assert_equal(repaired_array, initial_array)

    def test_revert_some(self):
        repair = RevertToQueryRepair()
        package = self.build_package(np.array([[11, 8, 13]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        initial_array = pd.DataFrame.from_records(z).values

        differences = repaired_array[:, (1,)] - initial_array[:, (1,)]
        changed = np.count_nonzero(differences)
        percentage_changed = changed / 100_000
        self.assertGreaterEqual(percentage_changed, 0.05)
        self.assertLessEqual(percentage_changed, 0.07)
        np_test.assert_equal(repaired_array[:, (0, 2)], initial_array[:, (0, 2)])

    def test_revert_all(self):
        repair = RevertToQueryRepair()
        package = self.build_package(np.array([[1, 2, 3]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
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
