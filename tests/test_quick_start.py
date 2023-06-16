import unittest


import random
from pymoo.core.variable import Real
from decode_mcd.data_package import DataPackage
from decode_mcd.design_targets import *
from decode_mcd.multi_objective_cfe_generator import MultiObjectiveCounterfactualsGenerator, CFSet


class QuickStartTest(unittest.TestCase):
    def test_quick_start(self):
        x = np.random.random(100)
        x = x.reshape(100, 1)
        y = x * 100 + random.random()

        def predict(_x):
            return _x * 100

        data_package = DataPackage(x, y, x[0].reshape(1, 1),
                                   design_targets=DesignTargets([ContinuousTarget(label=0,
                                                                                  lower_bound=25,
                                                                                  upper_bound=75)]),
                                   datatypes=[Real(bounds=(0, 1))])
        gen = MultiObjectiveCounterfactualsGenerator(data_package, lambda design: predict(design), [])
        cf_set = CFSet(gen, 10, False)
        cf_set.optimize(10)
        counterfactuals = cf_set.sample(10, 1, 1, 1, 1, 50)
        print(counterfactuals)
