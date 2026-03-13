import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EXPERIMENT_NAME = "healthpredict_breast_cancer"
REGISTERED_MODEL_NAME = "HealthPredictBreastCancerClassifier"
RANDOM_STATE = 42
FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
]


def get_tracking_uri() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        return tracking_uri
    local_store = Path("mlruns").absolute().as_uri()
    return local_store


def load_data():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data[FEATURES]
    y = dataset.target
    return X, y


def build_pipeline(n_estimators: int, max_depth: int | None, min_samples_split: int) -> Pipeline:
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    calibrated_model = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=3)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", calibrated_model),
        ]
    )


def evaluate_model(model: Pipeline, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def main():
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    search_space = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
        {"n_estimators": 300, "max_depth": 12, "min_samples_split": 4},
    ]

    best_run_id = None
    best_pipeline = None
    best_model_uri = None
    best_score = -np.inf

    for idx, params in enumerate(search_space, start=1):
        with mlflow.start_run(run_name=f"rf_calibrated_run_{idx}") as run:
            pipeline = build_pipeline(**params)
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            signature = infer_signature(X_train, pipeline.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:3],
            )

            print(f"Run ID: {run.info.run_id}")
            print(f"Params: {params}")
            print(f"Metrics: {metrics}")

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_run_id = run.info.run_id
                best_pipeline = pipeline
                best_model_uri = model_info.model_uri

    if best_run_id is None or best_pipeline is None or best_model_uri is None:
        raise RuntimeError("No successful MLflow runs were created.")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifacts_dir / "best_model.joblib")

    model_version = mlflow.register_model(model_uri=best_model_uri, name=REGISTERED_MODEL_NAME)

    print("\nBest run registered successfully.")
    print(f"Best run ID: {best_run_id}")
    print(f"Best ROC-AUC: {best_score:.4f}")
    print(f"Registered model: {REGISTERED_MODEL_NAME} v{model_version.version}")


if __name__ == "__main__":
    main()
