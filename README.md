# Time-Series Forecasting API ⏳

A robust time-series forecasting API built with **FastAPI** and visualized using **Streamlit**, deployed on **GCP Cloud Run**. This project predicts weekly sales for the M4 dataset using an XGBoost model, with interpretability via SHAP plots, input validation, and an OpenAPI schema.

---

## 🌟 Project Overview

This project delivers a scalable forecasting solution for weekly sales data from the M4 dataset. The backend, powered by FastAPI, handles predictions using a pre-trained XGBoost model, while the Streamlit frontend provides an interactive interface for users to input sales data and visualize results, including a SHAP plot for model interpretability.

### Key Features

- **SHAP Plot (Feature 1)**: Visualize the impact of features like `lag_1` to `lag_4`, `rolling_mean_4`, `rolling_std_4`, and `trend` on predictions.
- **Input Validation (Feature 2)**: Ensures robust inputs (e.g., rejects negative values, requires at least 5 values).
- **OpenAPI Schema (Feature 3)**: Fully documented API endpoints for seamless integration.

---

## 🚀 Deployment

The app is live on **GCP Cloud Run** for real-time forecasting.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python 3.10) with XGBoost for forecasting.
- **Frontend**: Streamlit for interactive visualization.
- **Containerization**: Docker and Docker Compose for local development.
- **Deployment**: Google Cloud Platform (Cloud Run, Artifact Registry).
- **Dataset**: M4 Weekly Dataset for training and testing.

---

## 📂 File Structure

- `main.py`: FastAPI backend for prediction endpoints.
- `streamlit.py`: Streamlit frontend for user interaction and SHAP visualization.
- `docker-compose.yml`: Orchestrates FastAPI and Streamlit containers.
- `Dockerfile`: Builds the FastAPI container.
- `Dockerfile.streamlit`: Builds the Streamlit container.
- `requirements.txt`: Dependencies for FastAPI.
- `requirements-streamlit.txt`: Dependencies for Streamlit.
- `forecasting_model.pkl`: Pre-trained XGBoost model (included in the repository)

---

## 🔧 Local Setup

### Prerequisites

- Docker Desktop
- Python 3.10 (optional, if running without Docker)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/gurisapuriya/time-series-forecasting.git
   cd time-series-forecasting
   ```
2. Build and run the containers:

   ```bash
   docker-compose up --build
   ```
3. Access the apps:
   - FastAPI (Local): `http://localhost:8080/docs`
   - Streamlit (Local): `http://localhost:8501`
   - **FastAPI (Deployed)**: `https://fastapi-service-995431799163.us-central1.run.app`
   - **Streamlit (Deployed)**: `https://streamlit-service-995431799163.us-central1.run.app`

*Note: Replace the deployed URLs with your actual Cloud Run service URLs before submission.*

---

## 📊 Usage

### FastAPI

- **Endpoint**: `/predict`
- **Example Request**:

  ```json
  {
    "series_id": "T100000",
    "values": [1111, 1234, 2345, 1237, 1234.6]
  }
  ```
- **Response**: Predicted next week's sales (e.g., `~1300`).

### Streamlit

- Enter a time series ID (e.g., `T100000`) and sales values (e.g., `1111, 1234, 2345, 1237, 1234.6`).
- Click **Predict** to see the forecast and a SHAP plot.