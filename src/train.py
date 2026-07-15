"""
train.py
--------
Trains two models (Logistic Regression and Random Forest) on the
electrical fault detection dataset, tracks every run with MLflow,
compares the models, and saves the best one to disk so that
predict.py / app.py can load it later.

Why this file exists:
This is the "brain" of the MLOps pipeline. It is responsible for
turning raw data into a trained, evaluated, and saved model, while
logging everything (parameters, metrics, and the model itself) to
MLflow so the whole experiment is reproducible and comparable.

Run it with:
    python src/train.py
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from utils import load_clean_data, FEATURE_COLUMNS, TARGET_COLUMN

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "detect_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# All MLflow experiment runs will be grouped under this experiment name.
# Using a fixed name means every time you re-run train.py, the new run
# is added to the same experiment instead of creating a new one.
EXPERIMENT_NAME = "Smart_Electrical_Fault_Detection"

RANDOM_STATE = 42


def evaluate_model(model, X_test, y_test):
    """
    Compute the standard classification metrics for a fitted model.

    Returns a dictionary of metrics plus the confusion matrix, so the
    caller can both log the numeric metrics to MLflow and print the
    confusion matrix for a human to read.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def train_and_log(model_name, model, params, X_train, X_test, y_train, y_test):
    """
    Train a single model inside its own MLflow run, log its
    hyperparameters + metrics + the model artifact itself, and
    return the fitted model along with its metrics.
    """
    with mlflow.start_run(run_name=model_name):
        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Evaluate on the held-out test set
        metrics, cm = evaluate_model(model, X_test, y_test)

        # --- MLflow logging ---
        # Log hyperparameters so we know exactly how this run was configured
        mlflow.log_params(params)

        # Log metrics so runs can be compared in the MLflow UI
        mlflow.log_metrics(metrics)

        # Log the trained model itself as an MLflow artifact.
        # This also registers the model's "flavor" (sklearn) so it can
        # later be loaded back with mlflow.sklearn.load_model(...).
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n--- {model_name} ---")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")
        print("Confusion Matrix:")
        print(cm)

    return model, metrics


def main():
    print("Step 1/6: Loading and cleaning data...")
    df = load_clean_data(DATA_PATH)
    print(f"Clean dataset shape: {df.shape}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print("\nStep 2/6: Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Feature scaling: Logistic Regression is sensitive to feature scale,
    # so we standardize (mean=0, std=1) the current/voltage readings.
    # Random Forest does not need this, but scaling does not hurt it either,
    # so we use the same scaled data for both models to keep the pipeline simple.
    print("\nStep 3/6: Scaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Point MLflow at a local folder so no external server is required.
    # This creates an "mlruns" folder in the project root.
    mlflow.set_tracking_uri("file:" + os.path.join(os.path.dirname(__file__), "..", "mlruns"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nStep 4/6: Training Logistic Regression...")
    lr_params = {"max_iter": 1000, "random_state": RANDOM_STATE}
    lr_model = LogisticRegression(**lr_params)
    lr_model, lr_metrics = train_and_log(
        "LogisticRegression", lr_model, lr_params,
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\nStep 5/6: Training Random Forest...")
    rf_params = {"n_estimators": 200, "max_depth": 10, "random_state": RANDOM_STATE}
    rf_model = RandomForestClassifier(**rf_params)
    rf_model, rf_metrics = train_and_log(
        "RandomForest", rf_model, rf_params,
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print("\nStep 6/6: Selecting and saving the best model...")
    # We select the "best" model using F1 score because the dataset is
    # only mildly imbalanced (about 54% / 46%) and F1 balances precision
    # and recall, which matters for a fault-detection system: missing a
    # real fault (low recall) and raising false alarms (low precision)
    # are both costly.
    if rf_metrics["f1_score"] >= lr_metrics["f1_score"]:
        best_name, best_model, best_metrics = "RandomForest", rf_model, rf_metrics
    else:
        best_name, best_model, best_metrics = "LogisticRegression", lr_model, lr_metrics

    print(f"Best model: {best_name} (F1 = {best_metrics['f1_score']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved scaler to:     {SCALER_PATH}")

    # Also register the best model in the MLflow Model Registry so it has
    # a stable, named reference ("FaultDetectionModel") that other tools
    # (like a deployment script) could look up later, instead of relying
    # on a specific run ID.
    with mlflow.start_run(run_name=f"BestModel_{best_name}"):
        mlflow.log_params({"selected_model": best_name})
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="FaultDetectionModel",
        )

    print("\nTraining complete. Run 'mlflow ui' to view all experiments.")


if __name__ == "__main__":
    main()
