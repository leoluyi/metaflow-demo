import pytest
from metaflow import Flow
import pandas as pd

def test_end_to_end_pipeline():
   # Train
   train_run = Flow('TrainingFlow').latest_successful_run
   assert train_run is not None
   
   # Predict
   predict_run = Flow('PredictionFlow').latest_successful_run
   assert predict_run is not None
   
   # Evaluate
   eval_run = Flow('EvaluationFlow').latest_successful_run
   assert eval_run is not None
   assert 'test_accuracy' in eval_run.data.metrics
