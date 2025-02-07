import numpy as np
import pandas as pd


def generate_predict_data(output_path: str, size: int = 1000):
    df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, size),
            "feature2": np.random.normal(0, 1, size),
        }
    )

    # Print dataset information
    print("\nGenerated Prediction Dataset Summary:")
    print("-" * 40)
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nFeature Statistics:")
    print("-" * 40)
    print("\nSaving data to:", output_path)

    df.to_csv(output_path, index=False)
    print("Data generation completed!")
