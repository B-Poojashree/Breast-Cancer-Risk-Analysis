# CardioRiskOps: End-to-End MLOps Pipeline & Breast Cancer Risk Analysis
Bridging Machine Learning experimentation and production with MLflow, Docker, Streamlit, and Railway.

## Executive Summary
CardioRiskOps is a production-style MLOps project that performs real-time breast cancer risk classification from clinical features.

The project demonstrates the complete lifecycle:
- experiment tracking with MLflow
- model training and evaluation
- model registration
- deployment-ready Streamlit dashboard
- containerized execution with Docker

## Key Features
- Automated Experiment Tracking: Logs hyperparameters, metrics, artifacts, and model versions in MLflow.
- Containerized Architecture: Ships with Docker to run consistently across local and cloud environments.
- Interactive Analytics Dashboard: Streamlit UI for user inputs and distribution visualization.
- Smart Data Preprocessing: Inputs are scaled automatically through a training pipeline (`StandardScaler` + calibrated `RandomForest`).
- Cloud Deployment Ready: Includes Railway-compatible process and deployment configuration.

## Performance Metrics
The model logs and tracks:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Best model selection is based on ROC-AUC and is automatically registered into MLflow Model Registry.

## System Architecture
1. Training Phase (`train.py`)
- Loads the Scikit-Learn Breast Cancer dataset
- Trains multiple calibrated Random Forest configurations
- Logs every run to MLflow
- Registers the best model

2. Preprocessing Phase
- Accepts user-friendly raw feature values
- Applies standardization in the pipeline before inference

3. Deployment Phase (`app.py` + Docker)
- Serves predictions through Streamlit
- Loads the latest registered model (or best run fallback)

## Project Structure
```
.
|-- app.py
|-- train.py
|-- requirements.txt
|-- Dockerfile
|-- Procfile
|-- railway.json
|-- .gitignore
`-- README.md
```

## Installation & Setup
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and log experiments
```bash
python train.py
```

### 3. Launch dashboard
```bash
streamlit run app.py
```

## Docker Usage
### Build image
```bash
docker build -t cardioriskops .
```

### Run container
```bash
docker run -p 8501:8501 cardioriskops
```

## Railway Deployment
1. Push this repository to GitHub.
2. Create a Railway project and connect the repo.
3. Railway automatically detects `Procfile` / `railway.json` and deploys.
4. Ensure env var `MLFLOW_TRACKING_URI` is set if using a remote MLflow server.

## Optional Environment Variables
- `MLFLOW_TRACKING_URI`: Tracking backend URI.
  - Default: local `mlruns` folder.
- `PORT`: Runtime port for Streamlit in cloud containers.

## Notes
- If Model Registry is unavailable in your tracking backend, `app.py` first tries the top run by ROC-AUC and then falls back to `artifacts/best_model.joblib`.
- For production hardening, add authentication, input validation, and CI checks.
