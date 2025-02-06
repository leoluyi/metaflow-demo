.PHONY: setup data train test clean

setup:
	uv venv
	uv pip install -e ".[dev]"

data:
	uv pip install pandas scikit-learn numpy
	uv run python scripts/generate_data.py

train: data
	uv run python -m scripts.train

predict: train
	uv run python -m scripts.predict

test:
	uv run pytest tests/

clean:
	rm -rf logs/* models/* mlruns/* data/raw/* data/predict/*
	rm -rf .venv
