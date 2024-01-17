import unittest

import numpy as np
import pandas as pd

import decode_mcd.multi_objective_problem as MOP
from decode_mcd import counterfactuals_generator
from decode_mcd import data_package
from decode_mcd import design_targets
from pymoo.core.variable import Real


def validity(_x):  # Validity function for the 2D case
    a = _x["X"]  # Separate the two dimensions for clarity
    b = _x["Y"]
    fc = np.less(np.sqrt(np.power((a - 0.3), 2) + np.power((b - 0.3), 2)), 0.1)  # Circle
    sc = np.less(np.power(np.power(np.power((a - b), 6) - 1, 2) + np.power(np.power((a + b), 6) - 1, 2), 2),
                 0.99)  # Arcs
    res = np.logical_or(fc, sc)  # If points are in circle or arcs they are valid
    res = np.tile(res, (4, 1)).T  # Tile the single objective into four identical ones for testing
    return res


class Mcd2dExampleTest(unittest.TestCase):
    def test_case_runs(self):
        all_datapoints = np.random.rand(10000, 2)  # Sample 10000 2D points
        all_datapoints = all_datapoints * 2.2 - 1.1  # Scale from -1.1 to 1.1
        x_df = pd.DataFrame(all_datapoints, columns=["X", "Y"])
        validity_mask = validity(x_df)

        y_df = pd.DataFrame(validity_mask, columns=["O1", "O2", "O3", "O4"])
        all_df = pd.concat([x_df, y_df], axis=1)
        print(all_df)
        v = 100 * np.mean(all_df["O1"])
        print(f"{v}% of the points are valid")
        print(x_df)

        query_x = pd.DataFrame([[0, 0]], columns=["X", "Y"])

        data = data_package.DataPackage(features_dataset=x_df,
                                        predictions_dataset=y_df,
                                        query_x=query_x,
                                        design_targets=design_targets.DesignTargets(
                                            [design_targets.ContinuousTarget(label="O1", lower_bound=0.9,
                                                                             upper_bound=1.1),
                                             design_targets.ContinuousTarget(label="O3", lower_bound=0.9,
                                                                             upper_bound=1.1),
                                             # design_targets.ContinuousTarget(label="O4", lower_bound=0.9,
                                             #                                 upper_bound=1.1)
                                             ]),
                                        datatypes=[Real(bounds=(-1.1, 1.1)), Real(bounds=(-1.1, 1.1))])
        problem = MOP.MultiObjectiveProblem(data_package=data,
                                            prediction_function=validity,
                                            constraint_functions=[])

        generator = counterfactuals_generator.CounterfactualsGenerator(problem=problem,
                                                                       pop_size=100,
                                                                       initialize_from_dataset=True)
        generator.generate(n_generations=100)
