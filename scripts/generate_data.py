# scripts/generate_data.py
from pathlib import Path

from generation.predict_gen import generate_predict_data
from generation.test_gen import generate_test_data
from generation.train_gen import generate_train_data


def main():
    data_dir = Path("data")
    paths = {
        "train": data_dir / "training/train.csv",
        "test": data_dir / "training/test.csv",
        "new": data_dir / "prediction/new.csv",
    }

    # Create directories
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    # Generate data
    generate_train_data(paths["train"])
    generate_test_data(paths["test"])
    generate_predict_data(paths["new"])


if __name__ == "__main__":
    main()
