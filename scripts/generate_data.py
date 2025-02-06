from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/predict").mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n_samples = 1000

# Generate features
feature1 = np.random.normal(0, 1, n_samples)
feature2 = np.random.normal(0, 1, n_samples)
feature3 = np.sin(feature1) + np.cos(feature2)
feature4 = feature1 * feature2

# Generate target
target = (
    feature1 + feature2 + feature3 + feature4 + np.random.normal(0, 0.1, n_samples)
) > 0
target = target.astype(int)

# Create DataFrames
train_test_data = pd.DataFrame(
    {
        "feature1": feature1,
        "feature2": feature2,
        "feature3": feature3,
        "feature4": feature4,
        "target": target,
    }
)

# Generate prediction data
n_predict = 100
predict_data = pd.DataFrame(
    {
        "feature1": np.random.normal(0, 1, n_predict),
        "feature2": np.random.normal(0, 1, n_predict),
        "feature3": np.sin(feature1[:n_predict]) + np.cos(feature2[:n_predict]),
        "feature4": feature1[:n_predict] * feature2[:n_predict],
    }
)

# Split and save
train_df, test_df = train_test_split(train_test_data, test_size=0.2, random_state=42)
train_df.to_csv("data/raw/train.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)
predict_data.to_csv("data/predict/new_samples.csv", index=False)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Prediction samples: {len(predict_data)}")
