"""
predict.py
----------
Loads the trained model + scaler from disk and exposes a single
function, `predict_fault`, that turns raw sensor readings into a
"Normal" or "Fault" label.

Why this file exists:
We want the exact same prediction logic to be usable in two places:
1. From the command line (for quick manual testing, see the __main__
   block below).
2. From the FastAPI app (app.py), which imports `predict_fault`
   directly instead of duplicating this logic.

Run it directly for a quick manual test:
    python src/predict.py
"""

import os
import joblib
import pandas as pd

from utils import FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Load the model and scaler once, when this module is imported, rather
# than on every single prediction. Loading from disk is relatively slow,
# so doing it once and reusing the objects keeps predictions fast.
_model = None
_scaler = None


def _load_artifacts():
    """Lazily load the trained model and scaler from disk (only once)."""
    global _model, _scaler
    if _model is None or _scaler is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                "Trained model not found. Please run 'python src/train.py' first."
            )
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def predict_fault(Ia: float, Ib: float, Ic: float, Va: float, Vb: float, Vc: float) -> str:
    """
    Predict whether the electrical system is in a Normal or Fault state.

    Parameters
    ----------
    Ia, Ib, Ic : float
        Line currents for phases A, B, C.
    Va, Vb, Vc : float
        Line voltages for phases A, B, C.

    Returns
    -------
    str
        "Normal" if the model predicts class 0, otherwise "Fault".
    """
    model, scaler = _load_artifacts()

    # Build a single-row DataFrame in the exact same column order used
    # during training (see utils.FEATURE_COLUMNS). Using a DataFrame
    # (instead of a raw array) keeps the feature names attached, which
    # avoids sklearn's "X does not have valid feature names" warning.
    raw_features = pd.DataFrame([[Ia, Ib, Ic, Va, Vb, Vc]], columns=FEATURE_COLUMNS)

    # Apply the same scaling that was fit during training
    scaled_features = scaler.transform(raw_features)

    prediction = model.predict(scaled_features)[0]

    return "Fault" if prediction == 1 else "Normal"


if __name__ == "__main__":
    # Quick manual test using one real "fault" row and one real
    # "normal" row taken from the dataset, so you can sanity-check
    # that the saved model actually works before wiring up the API.
    fault_example = predict_fault(
        Ia=-170.4721962, Ib=9.219613499, Ic=161.2525827,
        Va=0.054490004, Vb=-0.659920931, Vc=0.605430928,
    )
    print(f"Example 1 prediction: {fault_example}")
