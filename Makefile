# Makefile
.PHONY: setup data train predict test clean mlflow

setup:
	uv venv
	uv pip install -e ".[dev]"
	mkdir -p data/{raw,predict} models mlruns

data:
	uv run python scripts/generate_data.py

train:
	uv run python -m flows.train_flow run --experiment="experiment-$(shell date +%Y%m%d)"

predict:
	uv run python -m flows.predict_flow run

mlflow:
	uv run mlflow ui --backend-store-uri mlruns

test:
	uv run pytest tests/

clean:
	rm -rf data/{raw,predict}/* models/* .metaflow/ mlruns/
	rm -rf .venv
