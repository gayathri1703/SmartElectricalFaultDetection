"""
app.py
------
FastAPI application that exposes the trained fault-detection model
as a web API.

Why this file exists:
A trained model sitting in a .pkl file is not useful to other
programs (a dashboard, a monitoring system, etc.) unless it is
served over a standard interface. FastAPI lets us wrap the model
in a simple HTTP endpoint: send in sensor readings as JSON, get
back a "Normal"/"Fault" prediction as JSON.

Run it with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for interactive API docs.
"""

import sys
import os

# Make sure Python can find the src/ package when this file is run
# directly (e.g. `uvicorn app:app`) rather than as an installed package.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_fault

app = FastAPI(
    title="Smart Electrical Fault Detection API",
    description="Predicts whether an electrical system is Normal or in a Fault state.",
    version="1.0.0",
)


class SensorReading(BaseModel):
    """
    Request body schema for the /predict endpoint.

    These six fields mirror the feature columns used during training:
    three line currents (Ia, Ib, Ic) and three line voltages (Va, Vb, Vc).
    """

    Ia: float = Field(..., description="Line current, phase A")
    Ib: float = Field(..., description="Line current, phase B")
    Ic: float = Field(..., description="Line current, phase C")
    Va: float = Field(..., description="Line voltage, phase A (per-unit)")
    Vb: float = Field(..., description="Line voltage, phase B (per-unit)")
    Vc: float = Field(..., description="Line voltage, phase C (per-unit)")

    class Config:
        json_schema_extra = {
            "example": {
                "Ia": -170.4721962,
                "Ib": 9.219613499,
                "Ic": 161.2525827,
                "Va": 0.054490004,
                "Vb": -0.659920931,
                "Vc": 0.605430928,
            }
        }


class PredictionResponse(BaseModel):
    """Response body schema for the /predict endpoint."""
    status: str  # "Normal" or "Fault"


@app.get("/")
def read_root():
    """Simple health-check endpoint."""
    return {"message": "Smart Electrical Fault Detection API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    """
    Predict whether the electrical system is Normal or in a Fault state,
    given the six current/voltage sensor readings.
    """
    try:
        result = predict_fault(
            Ia=reading.Ia,
            Ib=reading.Ib,
            Ic=reading.Ic,
            Va=reading.Va,
            Vb=reading.Vb,
            Vc=reading.Vc,
        )
        return PredictionResponse(status=result)
    except FileNotFoundError as e:
        # This happens if someone starts the API before running train.py
        raise HTTPException(status_code=500, detail=str(e))
