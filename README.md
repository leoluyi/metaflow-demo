# Metaflow Demo

## Setup

1. Create required directories:
```bash
mkdir -p data/raw logs models/trained mlruns
```

2. Install dependencies:
```bash
make setup
```

3. Generate sample data:
```bash
make data
```

4. Train model:
```bash
make train
```

## Development

Run tests:
```bash
make test
```

Clean all generated files:
```bash
make clean
```

## Project Structure

```
medium_ml_project/
├── configs/                   # Configuration files
│   ├── base/                  # Base configurations
│   │   ├── data.yaml
│   │   ├── model.yaml
│   │   └── train.yaml
│   └── experiments/           # Experiment configurations
│       ├── exp001.yaml
│       └── exp002.yaml
│
├── data/                      # Data directory
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── features/              # Feature files
│
├── logs/                      # Log files
├── mlruns/                    # MLflow tracking
├── models/                    # Model files
│   ├── trained/              # Training outputs
│   └── deployed/             # Production models
│
├── notebooks/                 # Jupyter notebooks
│   ├── exploration/
│   └── experiments/
│
├── src/                       # Source code
│   ├── data/                  # Data operations
│   │   ├── __init__.py
│   │   ├── loader.py         # Data loading
│   │   └── processor.py      # Data processing
│   │
│   ├── features/             # Feature engineering
│   │   ├── __init__.py
│   │   ├── creator.py
│   │   └── selector.py
│   │
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── trainer.py
│   │
│   ├── utils/                # Utilities
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   └── pipelines/            # Training/inference
│       ├── __init__.py
│       ├── train.py
│       └── predict.py
│
├── tests/                     # Test files
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
│
├── scripts/                   # Utility scripts
│   ├── train.py
│   └── predict.py
│
├── .env                       # Environment variables
├── .gitignore                # Git ignore rules
├── pyproject.toml            # Project metadata
├── README.md                 # Project documentation
└── Makefile                  # Automation commands
```
