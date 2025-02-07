import pytest
from flows.training.train_flow import TrainingFlow
from metaflow.exception import MetaflowException

def test_training_flow_initialization():
   flow = TrainingFlow()
   assert flow.env == 'dev'

def test_invalid_env():
   with pytest.raises(MetaflowException):
       flow = TrainingFlow(env='invalid')
