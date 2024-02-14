import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real

from decode_mcd.counterfactuals_generator import _RevertToQueryRepair, CounterfactualsGenerator
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import DesignTargets, ContinuousTarget
from decode_mcd.mcd_exceptions import UserInputException
from decode_mcd.multi_objective_problem import MultiObjectiveProblem


class RevertToQueryRepairTest(unittest.TestCase):
    def test_revert_none_when_all_outside_valid_range(self):
        repair = _RevertToQueryRepair()
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
        repair = _RevertToQueryRepair()
        package = self.build_package(np.array([[11, 8, 13]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        initial_array = pd.DataFrame.from_records(z).values

        np_test.assert_equal(repaired_array[:, (0, 2)], initial_array[:, (0, 2)])
        self.assertGreaterEqual(np.average(repaired_array[:, (1,)]), 4)
        self.assertLessEqual(np.average(repaired_array[:, (1,)]), 8)

    def test_revert_all(self):
        repair = _RevertToQueryRepair()
        package = self.build_package(np.array([[1, 2, 3]]),
                                     [Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = MultiObjectiveProblem(data_package=package, prediction_function=lambda x: x, constraint_functions=[])
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values

        self.assertTrue(1 <= np.average(repaired_array[:, 0]) <= 3)
        self.assertTrue(2 <= np.average(repaired_array[:, 1]) <= 4)
        self.assertTrue(3 <= np.average(repaired_array[:, 2]) <= 5)

    def build_package(self, query_x, datatypes):
        _array = np.array([[5, 10, 15], [12, 15, 123], [13, 145, 13]])
        package = DataPackage(features_dataset=_array,
                              predictions_dataset=_array,
                              query_x=query_x,
                              design_targets=DesignTargets([ContinuousTarget(0, 10, 15)]),
                              datatypes=datatypes)
        return package


# noinspection PyTypeChecker
class CounterfactualsGeneratorTest(unittest.TestCase):
    def test_validates_constructor_parameters(self):
        self.assert_raises_with_message(lambda:
                                        CounterfactualsGenerator(problem=None, pop_size=500),
                                        "problem must be an instance of decode_mcd.MultiObjectiveProblem")
        problem = self.build_valid_problem()
        self.assert_raises_with_message(lambda: CounterfactualsGenerator(problem, 1000.15),
                                        "pop_size must be an integer")
        self.assert_raises_with_message(lambda: CounterfactualsGenerator(problem, -100),
                                        "pop_size must be a positive integer")

    def test_validates_generation_parameters(self):
        generator = CounterfactualsGenerator(self.build_valid_problem(), 500)
        self.assert_raises_with_message(lambda: generator.generate(50.5),
                                        "n_generations must be an integer")
        self.assert_raises_with_message(lambda: generator.generate(-50),
                                        "n_generations must be a positive integer")
        self.assert_raises_with_message(lambda: generator.generate(5, 50.5),
                                        "seed must be an integer")
        self.assert_raises_with_message(lambda: generator.generate(5, -50),
                                        "seed must be a positive integer")

    def test_invalid_sampling_parameters(self):
        generator = CounterfactualsGenerator(self.build_valid_problem(), 500)
        generator.generate(1)
        self.assert_raises_with_message(
            lambda: generator.sample_with_weights(num_samples=5.5, avg_gower_weight=1, cfc_weight=1, gower_weight=1,
                                                  diversity_weight=1),
            "num_samples must be an integer")
        self.assert_raises_with_message(
            lambda: generator.sample_with_weights(num_samples=-5, avg_gower_weight=1, cfc_weight=1, gower_weight=1,
                                                  diversity_weight=1),
            "num_samples must be a positive integer")

    def build_valid_problem(self):
        return MultiObjectiveProblem(
            data_package=DataPackage(
                features_dataset=np.array([[1, 2, 3], [4, 5, 6]]),
                predictions_dataset=np.array([[1, 2, 3], [4, 5, 6]]),
                query_x=np.array([[5, 3, 1]]),
                design_targets=DesignTargets([ContinuousTarget(0, 0, 10)]),
                datatypes=[Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))]
            ),
            prediction_function=lambda x: x,
            constraint_functions=[]
        )

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(UserInputException) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])
