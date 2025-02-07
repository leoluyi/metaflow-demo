import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
   return pd.DataFrame({
       'feature1': np.random.normal(0, 1, 100),
       'feature2': np.random.normal(0, 1, 100),
       'target': np.random.randint(0, 2, 100)
   })

@pytest.fixture
def config():
   from config.config_parser import Config
   return Config('dev')
