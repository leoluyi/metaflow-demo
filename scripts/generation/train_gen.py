import numpy as np
import pandas as pd


def generate_train_data(output_path: str, size: int = 10000):
    df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, size),
            "feature2": np.random.normal(0, 1, size),
            "target": np.random.randint(0, 2, size),
        }
    )

    # Print dataset information
    print("\nGenerated Training Dataset Summary:")
    print("-" * 40)
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")  # excluding target
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True).round(3)}")
    print("\nFeature Statistics:")
    print("-" * 40)
    print("\nSaving data to:", output_path)

    df.to_csv(output_path, index=False)
    print("Training data generation completed!")
