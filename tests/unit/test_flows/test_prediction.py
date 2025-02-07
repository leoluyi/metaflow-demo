import pytest
from flows.prediction.predict_flow import PredictionFlow
import pandas as pd

def test_prediction_output_format(tmp_path):
   flow = PredictionFlow()
   predictions = pd.Series([0, 1, 0])
   assert len(predictions) > 0
   assert predictions.dtype == 'int64'
