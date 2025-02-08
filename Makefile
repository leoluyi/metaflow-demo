.PHONY: setup data ingest train predict test clean mlflow

LOG_DIR := logs
DATE := $(shell date +%Y%m%d_%H%M%S)
ENV ?= dev
export PIPE_ENV = $(ENV)
export PYTHONPATH := $(PWD):$(PYTHONPATH)

setup:
	uv venv
	# uv pip install -e ".[dev]"
	uv pip compile pyproject.toml -o requirements.lock
	uv pip sync requirements.lock
	mkdir -p data/training data/prediction
	mkdir -p models/{dev,staging,prod}
	mkdir -p logs

data:
	uv run python scripts/generate_data.py

train:
	uv run python -m ml_pipeline.flows.training.train_flow run --env $(PIPE_ENV) | tee -a $(LOG_DIR)/train_flow_$(DATE).log

predict:
	uv run python -m ml_pipeline.flows.prediction.predict_flow run --env $(PIPE_ENV) | tee -a $(LOG_DIR)/predict_flow_$(DATE).log

evaluate:
	uv run python -m ml_pipeline.flows.evaluation.eval_flow run --env $(PIPE_ENV) | tee -a $(LOG_DIR)/evaluate_flow_$(DATE).log

train-all: train-dev train-staging train-prod

train-dev:
	$(MAKE) train PIPE_ENV=dev

train-staging:
	$(MAKE) train PIPE_ENV=staging

train-prod:
	$(MAKE) train PIPE_ENV=prod

mlflow:
	uv run mlflow ui -p 5999

test:
	uv run pytest tests/

clean:
	rm -rf data/* models/* logs/* mlruns/*

clean-all:
	rm -rf data/* models/* logs/* mlruns/*
	rm -rf .venv/
	rm -f uv.lock requirements.lock
