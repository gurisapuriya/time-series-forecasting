from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import time
import psutil

# Load the trained model
model = joblib.load("forecasting_model.pkl")

app = FastAPI()

# Define Input Data Contract
class ForecastRequest(BaseModel):
    series_id: str
    values: list[float]  # Raw time series values (at least 4 points for features)

# Feature engineering function (matches training)
def create_features_from_series(values, n_lags=3, window=4):
    if len(values) < max(n_lags + 1, window):
        raise ValueError(f"Series must have at least {max(n_lags + 1, window)} values for feature engineering.")
    
    df = pd.DataFrame({'value': values})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df['lag_4'] = df['value'].shift(4)  # Replace lag_52 with lag_4
    df['rolling_mean_4'] = df['value'].rolling(window=window, min_periods=1).mean()
    df['rolling_std_4'] = df['value'].rolling(window=window, min_periods=1).std()
    df['trend'] = np.arange(len(df))

    # Define feature columns to match training (assumed)
    feature_cols = [f'lag_{i}' for i in range(1, n_lags + 1)] + ['lag_4', 'rolling_mean_4', 'rolling_std_4', 'trend']
    features = df[feature_cols].dropna().iloc[-1:].values
    if features.size == 0:
        raise ValueError("No valid features generated; check input length or NaN values.")
    return features

@app.get("/")
def read_root():
    return {"message": "Welcome to the Forecasting API! Use /health to check status or /predict for forecasts."}

@app.post("/predict")
def predict_forecast(request: ForecastRequest):
    start_time = time.time()
    memory_before = psutil.virtual_memory().used / (1024 * 1024)  # MB

    try:
        # Create features from input series
        input_features = create_features_from_series(request.values)
        input_df = pd.DataFrame(input_features, columns=['lag_1', 'lag_2', 'lag_3', 'lag_4', 'rolling_mean_4', 'rolling_std_4', 'trend'])
        
        # Predict
        predictions = model.predict(input_df)

        memory_after = psutil.virtual_memory().used / (1024 * 1024)
        end_time = time.time()
        
        response_time = end_time - start_time
        memory_used = memory_after - memory_before

        return {
            "series_id": request.series_id,
            "forecast": predictions.tolist(),
            "performance": {
                "response_time_seconds": response_time,
                "memory_usage_mb": memory_used
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)