import __main__
import os
import unittest

import data_package
import multi_objective_cfe_generator as MOCG
from sklearn.model_selection import train_test_split
from alt_multi_label_predictor import MultilabelPredictor
from load_data import load_scaled_framed_dataset
from sklearn.metrics import r2_score

DEFAULT_MODEL_PATH = "/home/yazan/Repositories/Personal/BikeCAD-integration/service_resources/generated/models" \
                     "/Trained Models/AutogluonModels/ag-20230311_154304"


class McdEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        __main__.MultilabelPredictor = MultilabelPredictor
        self.MODEL_PATH = os.getenv("MODEL_FULL_PATH",
                                    DEFAULT_MODEL_PATH)

    def test_model_loaded(self):
        predictor = MultilabelPredictor.load(self.MODEL_PATH)
        x, y = self._load_data()
        predictions = predictor.predict(x)
        self.assertGreater(r2_score(y, predictions), 0.72)

    def test_framed_example(self):
        pass

    def _load_data(self):
        x_scaled, y_scaled, _, _ = load_scaled_framed_dataset()
        x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                                            y_scaled,
                                                            test_size=0.2,
                                                            random_state=1950)
        return x_test, y_test
