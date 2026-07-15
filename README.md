# Smart Electrical Fault Detection System using MLOps and MLflow

A beginner-friendly, end-to-end MLOps project that predicts whether an
electrical system is in a **Normal** or **Fault** state, based on
three-phase current and voltage readings. Built with scikit-learn,
tracked with MLflow, and served with FastAPI.

## Project Overview

- **Problem type:** Binary classification (Normal vs Fault)
- **Input features:** `Ia, Ib, Ic` (line currents), `Va, Vb, Vc` (line voltages)
- **Models trained:** Logistic Regression and Random Forest (best one is auto-selected by F1 score)
- **Experiment tracking:** MLflow (local, file-based — no external server needed)
- **Serving:** FastAPI, with a single `POST /predict` endpoint
- **Containerization:** A single, simple Dockerfile (no Kubernetes/orchestration)

## Folder Structure

```
SmartElectricalFaultDetection/
│
├── data/
│   └── detect_dataset.csv        # Raw dataset
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis
│
├── src/
│   ├── utils.py                  # Shared data-cleaning logic
│   ├── train.py                  # Trains models + logs to MLflow
│   └── predict.py                # Loads model, makes predictions
│
├── models/                       # Created after training (best_model.pkl, scaler.pkl)
├── mlruns/                       # Created after training (MLflow tracking data)
│
├── app.py                        # FastAPI application
├── main.py                       # Runs the full pipeline end-to-end
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

1. Make sure you have **Python 3.11** installed.
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

The fastest way to run everything (train the models, save the best one,
and verify it with a sample prediction) is:

```bash
python main.py
```

## Training

To train the models directly:

```bash
python src/train.py
```

This will:
1. Load and clean `data/detect_dataset.csv`
2. Split the data into train/test sets (80/20)
3. Scale the features
4. Train Logistic Regression and Random Forest
5. Log parameters, metrics, and models to MLflow
6. Select the best model by F1 score and save it to `models/best_model.pkl`
   (along with `models/scaler.pkl`)
7. Register the best model in the MLflow Model Registry as `FaultDetectionModel`

## Running MLflow

After training at least once, launch the MLflow UI to explore your experiments:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open **http://127.0.0.1:5000** in your browser. You'll be able to
compare the Logistic Regression and Random Forest runs side by side,
view logged parameters/metrics, and inspect the registered model.

## Prediction

**Option A — quick script test:**
```bash
python src/predict.py
```

**Option B — via the API** (see below).

## Running the API

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs,
or send a request directly:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "Ia": -170.4721962,
        "Ib": 9.219613499,
        "Ic": 161.2525827,
        "Va": 0.054490004,
        "Vb": -0.659920931,
        "Vc": 0.605430928
      }'
```

Response:
```json
{"status": "Normal"}
```

> **Note:** You must run `python src/train.py` (or `python main.py`) at
> least once before starting the API, since it loads `models/best_model.pkl`.

## Docker Commands

Build the image (make sure you've trained a model first, so `models/`
exists and gets copied into the image):

```bash
docker build -t fault-detection-api .
```

Run the container:

```bash
docker run -p 8000:8000 fault-detection-api
```

The API will then be available at **http://127.0.0.1:8000**.

## Why Each Part of This Project Exists

- **`src/utils.py`** — centralizes data cleaning so training and prediction
  never disagree about how the raw data should be interpreted.
- **`src/train.py`** — the core ML pipeline: trains, evaluates, compares,
  and saves the best model. This is where MLflow tracking happens.
- **`src/predict.py`** — a thin, reusable prediction function so the same
  logic can be called from the command line or from the API.
- **`app.py`** — turns the saved model into a real, usable web service.
- **`main.py`** — a single convenience command that runs the whole
  pipeline for you.
- **MLflow** — without it, every training run's parameters and metrics
  would only exist in your terminal history. MLflow gives you a
  searchable, comparable record of every experiment, plus a registry
  for your best model — the core idea behind reproducible MLOps.
- **FastAPI** — turns a static `.pkl` file into something other
  programs (a dashboard, a monitoring tool, etc.) can actually call
  over HTTP.
- **Docker** — packages the API and its dependencies into one portable
  unit that runs the same way on any machine.
