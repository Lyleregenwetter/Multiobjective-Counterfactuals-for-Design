import random
import unittest

from pymoo.core.variable import Real

import numpy as np
from decode_mcd import DesignTargets, DataPackage, \
    MultiObjectiveProblem, CounterfactualsGenerator, ContinuousTarget


class QuickStartTest(unittest.TestCase):
    def test_quick_start(self):
        x = np.random.random(100)
        x = x.reshape(100, 1)
        y = x * 100

        def predict(_x):
            return _x * 100 + random.random()

        data_package = DataPackage(features_dataset=x,
                                   predictions_dataset=y,
                                   query_x=x[0].reshape(1, 1),
                                   design_targets=DesignTargets([ContinuousTarget(label=0,
                                                                                  lower_bound=25,
                                                                                  upper_bound=75)]),
                                   datatypes=[Real(bounds=(0, 1))])

        problem = MultiObjectiveProblem(data_package=data_package,
                                        prediction_function=lambda design: predict(design),
                                        constraint_functions=[])

        generator = CounterfactualsGenerator(problem=problem,
                                             pop_size=10,
                                             initialize_from_dataset=False)

        generator.generate(n_generations=10)
        counterfactuals = generator.sample_with_dtai(num_samples=10, gower_weight=1,
                                                     avg_gower_weight=1, cfc_weight=1,
                                                     diversity_weight=50)
        print(counterfactuals)
