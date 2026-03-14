import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAME = "HealthPredictBreastCancerClassifier"
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
    return Path("mlruns").absolute().as_uri()


def build_fallback_model() -> Pipeline:
    data = load_breast_cancer(as_frame=True)
    X = data.data[FEATURES]
    y = data.target

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_split=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


@st.cache_resource
def load_model():
    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = f"models:/{MODEL_NAME}/latest"

    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        pass

    # Prefer a bundled artifact if available in the runtime image.
    local_model = Path("artifacts") / "best_model.joblib"
    if local_model.exists():
        return joblib.load(local_model)

    # Fallback to latest local run if registry is unavailable.
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("healthpredict_breast_cancer")
        if experiment is not None:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.roc_auc DESC"],
                max_results=1,
            )
            if runs:
                run_id = runs[0].info.run_id
                try:
                    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                except Exception:
                    pass
    except Exception:
        pass

    # Last-resort fallback keeps the app usable on first deploys.
    return build_fallback_model()


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame[FEATURES + ["target"]].copy()


def prediction_label(probability: float) -> str:
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.70:
        return "Moderate Risk"
    return "High Risk"


def main():
    st.set_page_config(page_title="HealthPredict - Breast Cancer Risk", layout="wide")

    st.title("HealthPredict: Breast Cancer Risk Analysis")
    st.caption("MLOps-ready Streamlit app with MLflow model loading and calibrated confidence output.")

    data = load_reference_data()
    model = load_model()

    st.subheader("Clinical Inputs")
    c1, c2, c3 = st.columns(3)

    feature_values = {}
    for idx, feature in enumerate(FEATURES):
        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        default_val = float(data[feature].median())

        container = [c1, c2, c3][idx % 3]
        with container:
            feature_values[feature] = st.slider(
                feature.title(),
                min_value=min_val,
                max_value=max_val,
                value=default_val,
            )

    input_df = pd.DataFrame([feature_values])

    if st.button("Analyze Risk", type="primary"):
        probability = float(model.predict_proba(input_df)[0][1])

        label = prediction_label(probability)

        st.markdown("### Risk Assessment")
        st.metric("Predicted Risk", label)
        st.progress(probability)
        st.write(f"Model confidence score: **{probability:.2%}**")

    st.divider()
    st.subheader("Feature Distribution Explorer")
    feature_choice = st.selectbox("Choose a feature", FEATURES)

    fig = px.histogram(
        data,
        x=feature_choice,
        color="target",
        barmode="overlay",
        nbins=30,
        title=f"Distribution of {feature_choice.title()} by Class",
        opacity=0.75,
    )
    fig.update_layout(legend_title_text="Target (1=Benign, 0=Malignant)")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How preprocessing works"):
        st.write(
            "Inputs are accepted in standard clinical scale and then normalized inside "
            "the model pipeline using StandardScaler before RandomForest inference."
        )


if __name__ == "__main__":
    main()
