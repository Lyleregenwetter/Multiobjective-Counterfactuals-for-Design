import __main__
import os
import unittest

import numpy as np
import pandas as pd
from pymoo.core.variable import Real
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import multi_objective_cfe_generator as MOCG
from alt_multi_label_predictor import MultilabelPredictor
from data_package import DataPackage
from load_data import load_scaled_framed_dataset

DEFAULT_MODEL_PATH = "/home/yazan/Repositories/Personal/BikeCAD-integration/service_resources/generated/models" \
                     "/Trained Models/AutogluonModels/ag-20230311_154304"


class McdEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        __main__.MultilabelPredictor = MultilabelPredictor
        self.MODEL_PATH = os.getenv("MODEL_FULL_PATH",
                                    DEFAULT_MODEL_PATH)
        try:
            self.predictor = MultilabelPredictor.load(self.MODEL_PATH)
        except Exception as e:
            print(e)
            raise EnvironmentError("""Could not load model. 
            You might need to override the MODEL_FULL_PATH environment variable""")
        self.x, self.y = self._load_data()

    def _test_model_loaded(self):
        predictor = MultilabelPredictor.load(self.MODEL_PATH)
        x, y = self._load_data()
        predictions = predictor.predict(x)
        self.assertGreater(r2_score(y, predictions), 0.72)

    def test_framed_example(self):
        x, y = self.x, self.y
        lbs = np.quantile(x.values, 0.01, axis=0)
        ubs = np.quantile(x.values, 0.99, axis=0)
        datatypes = []
        for i in range(len(x.columns)):
            datatypes.append(Real(bounds=(lbs[i], ubs[i])))
        dp = DataPackage(x, y, x.iloc[0:1], x.columns, {"Model Mass Magnitude": (2, 4)}, None)
        problem = MOCG.MultiObjectiveCounterfactualsGenerator(dp, self.call_predictor, [], [], datatypes)
        cf_set = MOCG.CFSet(problem, 500, initialize_from_dataset=False)
        cf_set.optimize(5)
        num_samples = 10
        cfs = cf_set.sample(num_samples, 0.5, 0.2, 0.5, 0.2, np.array([1]), include_dataset=False, num_dpp=10000)
        print(cfs)

    def call_predictor(self, x):

        return self.predictor.predict(pd.DataFrame(x, columns=self.x.columns)).drop(
            columns=self.y.columns.difference(["Model Mass Magnitude"]))

    def _load_data(self):
        x_scaled, y_scaled, _, _ = load_scaled_framed_dataset()
        x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                            y_scaled,
                                                            test_size=0.2,
                                                            random_state=1950)
        return x_test, y_test