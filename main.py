"""
main.py
-------
Single entry point that runs the whole pipeline end to end:
    1. Train the models (logs everything to MLflow, saves the best model)
    2. Run one example prediction to prove the saved model works

Why this file exists:
For a beginner project, it's convenient to have one command that
does "everything" without needing to remember which script to run
in which order. This is NOT a replacement for train.py/predict.py —
it simply calls them in sequence.

Run it with:
    python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from train import main as run_training
from predict import predict_fault


def main():
    print("=" * 60)
    print("SMART ELECTRICAL FAULT DETECTION - FULL PIPELINE")
    print("=" * 60)

    print("\n[1/2] Running training pipeline...\n")
    run_training()

    print("\n[2/2] Running a sample prediction to verify the saved model...\n")
    result = predict_fault(
        Ia=-170.4721962, Ib=9.219613499, Ic=161.2525827,
        Va=0.054490004, Vb=-0.659920931, Vc=0.605430928,
    )
    print(f"Sample prediction result: {result}")

    print("\nPipeline finished successfully.")
    print("Next steps:")
    print("  - Run 'mlflow ui' to explore experiment tracking")
    print("  - Run 'uvicorn app:app --reload' to start the prediction API")


if __name__ == "__main__":
    main()
