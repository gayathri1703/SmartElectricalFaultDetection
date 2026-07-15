# Dockerfile
# ----------
# Builds a container that serves the trained fault-detection model
# via the FastAPI app. Kept intentionally simple: one stage, one
# base image, no orchestration tools.

FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first (this layer is cached unless
# requirements.txt changes, which speeds up rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# The FastAPI app listens on port 8000
EXPOSE 8000

# NOTE: the model must already be trained (models/best_model.pkl and
# models/scaler.pkl must exist) before building this image, since the
# container only serves predictions — it does not train on startup.
# Run "python src/train.py" locally first, or add a training step to
# your own build process if you want training to happen at build time.

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
