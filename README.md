# ML Pipeline Project

Machine learning pipeline using Metaflow and MLflow.

## Project Features

**Environment**

- Self-contained

**Params and Configs**

- Environment variable for DEV, STAGING, PROD stages
- CLI Params

**Data:**

- Versioning (including transformation, features)

**ML Model:**

- Tracking
- Versioning

**Debug:**

- Logging

## Setup

```bash
# Create environment and install dependencies
make setup

# Generate training/test/prediction data
make data

# Train model
make train ENV=[dev|staging|prod]

# Run predictions
make predict ENV=[dev|staging|prod]
```

## Project project

```
ml_pipeline/
├── src/ml_pipeline/             # Main package
│   ├── flows/                   # Metaflow pipelines
│   │   ├── training/           # Training pipeline
│   │   └── prediction/         # Prediction pipeline
│   ├── models/                 # Model definitions
│   │   ├── architectures/      # Model architectures
│   │   └── components/         # Reusable components
│   ├── config/                 # Configuration
│   ├── utils/                  # Utilities
│   │   ├── logger.py
│   │   └── errors.py
│   └── monitoring/             # Monitoring components
│
├── scripts/                    # Data generation
├── notebooks/                  # Experiments
├── tests/                      # Testing
│   ├── unit/
│   └── integration/
├── Makefile                    # Build automation
└── pyproject.toml             # Project metadata
```

## Development

```bash
# Run tests
make test

# Launch MLflow UI
make mlflow ENV=[dev|staging|prod]

# Clean environment
make clean ENV=[dev|staging|prod]

# Show current settings
make status ENV=[dev|staging|prod]
```

## Environment Management

- dev: Local development
- staging: Testing environment
- prod: Production environment

## Dependencies

- Python 3.10+
- uv package manager
- MLflow
- Metaflow
