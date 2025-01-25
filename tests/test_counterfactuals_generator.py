import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pymoo.core.variable import Real

from decode_mcd.mcd_generator import _RevertToQueryRepair, McdGenerator
from decode_mcd.mcd_dataset import McdDataset
from decode_mcd.design_targets import DesignTargets, ContinuousTarget
from decode_mcd.mcd_exceptions import UserInputException
from decode_mcd.mcd_problem import McdProblem


class RevertToQueryRepairTest(unittest.TestCase):
    @unittest.skip
    def test_records_with_incorrect_column_order(self):
        pass

    @unittest.skip
    def test_revert_some(self):
        """We no longer allow query_x to be outside valid ranges,
        so this check is needless. I'm keeping the test so we know
        to remove the redundant check in the code or otherwise replace it with
        a more meaningful check."""
        repair = _RevertToQueryRepair()
        query_x = np.array([[11, 8, 13]])
        design_targets = DesignTargets([ContinuousTarget(0, 10, 15)])
        package = self.build_package([Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = McdProblem(mcd_dataset=package,
                             x_query=query_x,
                             y_targets=design_targets,
                             prediction_function=lambda x: x)
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values
        initial_array = pd.DataFrame.from_records(z).values

        np_test.assert_equal(repaired_array[:, (0, 2)], initial_array[:, (0, 2)])
        self.assertGreaterEqual(np.average(repaired_array[:, (1,)]), 4)
        self.assertLessEqual(np.average(repaired_array[:, (1,)]), 8)

    def test_revert_all(self):
        repair = _RevertToQueryRepair()
        package = self.build_package([Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        problem = McdProblem(mcd_dataset=package,
                             x_query=np.array([[1, 2, 3]]),
                             y_targets=DesignTargets([ContinuousTarget(0, 10, 15)]),
                             prediction_function=lambda x: x)
        z = [{0: 3, 1: 4, 2: 5} for _ in range(100_000)]
        repaired = repair._do(problem, z)
        repaired_array = pd.DataFrame.from_records(repaired).values

        self.assertTrue(1 <= np.average(repaired_array[:, 0]) <= 3)
        self.assertTrue(2 <= np.average(repaired_array[:, 1]) <= 4)
        self.assertTrue(3 <= np.average(repaired_array[:, 2]) <= 5)

    def build_package(self, datatypes):
        _array = np.array([[5, 10, 15], [12, 15, 123], [13, 145, 13]])
        package = McdDataset(x=_array,
                             y=_array,
                             x_datatypes=datatypes)
        return package


# noinspection PyTypeChecker
class CounterfactualsGeneratorTest(unittest.TestCase):
    def test_validates_constructor_parameters(self):
        self.assert_raises_with_message(lambda:
                                        McdGenerator(mcd_problem=None, pop_size=500),
                                        "problem must be an instance of decode_mcd.MultiObjectiveProblem")
        problem = self.build_valid_problem()
        self.assert_raises_with_message(lambda: McdGenerator(problem, 1000.15),
                                        "pop_size must be an integer")
        self.assert_raises_with_message(lambda: McdGenerator(problem, -100),
                                        "pop_size must be a positive integer")

    def test_validates_generation_parameters(self):
        generator = McdGenerator(self.build_valid_problem(), 500)
        self.assert_raises_with_message(lambda: generator.generate(50.5),
                                        "n_generations must be an integer")
        self.assert_raises_with_message(lambda: generator.generate(-50),
                                        "n_generations must be a positive integer")
        self.assert_raises_with_message(lambda: generator.generate(5, 50.5),
                                        "seed must be an integer")
        self.assert_raises_with_message(lambda: generator.generate(5, -50),
                                        "seed must be a positive integer")

    def test_invalid_sampling_parameters(self):
        generator = McdGenerator(self.build_valid_problem(), 500)
        generator.generate(1)
        self.assert_raises_with_message(
            lambda: generator.sample(num_samples=5.5, manifold_proximity_weight=1, sparsity_weight=1, proximity_weight=1,
                                     diversity_weight=1),
            "num_samples must be an integer")
        self.assert_raises_with_message(
            lambda: generator.sample(num_samples=-5, manifold_proximity_weight=1, sparsity_weight=1, proximity_weight=1,
                                     diversity_weight=1),
            "num_samples must be a positive integer")

    def build_valid_problem(self):
        _x_query = np.array([[5, 3, 1]])
        targets = DesignTargets([ContinuousTarget(0, 0, 10)])
        _data_package = McdDataset(x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[1, 2, 3], [4, 5, 6]]),
                                   x_datatypes=[Real(bounds=(0, 10)), Real(bounds=(0, 10)), Real(bounds=(0, 10))])
        return McdProblem(
            mcd_dataset=_data_package,
            x_query=_x_query,
            y_targets=targets,
            prediction_function=lambda x: x
        )

    def assert_raises_with_message(self, faulty_call: callable, expected_message: str):
        with self.assertRaises(UserInputException) as context:
            faulty_call()
        self.assertEqual(expected_message, context.exception.args[0])
