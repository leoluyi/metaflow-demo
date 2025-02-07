# ML Pipeline Project

ML pipeline using Metaflow for workflow management and MLflow for experiment tracking.

## Requirements
- Python 3.10+
- uv

## Quick Start

1. Setup
```bash
make setup
```

2. Generate Data
```bash
make data
```

3. Train
```bash
make train
```

4. Predict
```bash
make predict
```

## Project Structure
```
/data        - Data directory
/flows       - Metaflow workflows
/models      - Model architectures
/config      - Configurations
/scripts     - Utility scripts
/tests       - Tests
/monitoring  - Metrics & alerts
/docs        - Documentation
```

## Development
- Tests: `make test`
- MLflow UI: `make mlflow`
- Cleanup: `make clean`

## Monitoring
Monitor metrics and alerts in `/monitoring`

## Documentation
Detailed docs in `/docs`
