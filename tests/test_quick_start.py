import random
import unittest

from pymoo.core.variable import Real

import numpy as np
from decode_mcd import DesignTargets, McdDataset, \
    McdProblem, McdGenerator, ContinuousTarget


class QuickStartTest(unittest.TestCase):
    def test_quick_start(self):
        x = np.random.random(100)
        x = x.reshape(100, 1)
        y = x * 100

        def predict(_x):
            return _x * 100 + random.random()

        data_package = McdDataset(x=x,
                                  y=y,
                                  x_datatypes=[Real(bounds=(0, 1))])

        problem = McdProblem(mcd_dataset=data_package,
                             prediction_function=lambda design: predict(design),
                             x_query=x[0].reshape(1, 1),
                             y_targets=DesignTargets([ContinuousTarget(label=0,
                                                                                  lower_bound=25,
                                                                                  upper_bound=75)]))

        generator = McdGenerator(mcd_problem=problem,
                                 pop_size=10,
                                 initialize_from_dataset=False)

        generator.generate(n_generations=10)
        counterfactuals = generator.sample_with_dtai(num_samples=10, proximity_weight=1,
                                                     manifold_proximity_weight=1, sparsity_weight=1,
                                                     diversity_weight=50)
        print(counterfactuals)
