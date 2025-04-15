import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import joblib
import os

st.title("Weekly Sales Forecaster")
st.write("Enter a time series ID and past sales to predict next week's sales.")

# Use environment variable for FastAPI URL
# FASTAPI_URL = "http://localhost:8080"  # Temporary for local testing
# st.write(f"Debug: Using FASTAPI_URL = {FASTAPI_URL}")

#for docker build testing
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8080")

#for deploying on gcp
# FASTAPI_URL = os.getenv("FASTAPI_URL", "https://fastapi-service-<hash>.uc.r.appspot.com")  # Replace with your FastAPI service URL

# Input form
with st.form("sales_form"):
    series_id = st.text_input("Time Series ID (e.g., T100000)", value="T100000")
    sales_input = st.text_area("Enter sales (comma-separated, e.g., 1111, 1234, 2345, 1237, 1234.6):")
    submit = st.form_submit_button("Predict")

if submit and sales_input and series_id:
    try:
        # Parse sales
        sales = [float(x.strip()) for x in sales_input.split(",")]
        if len(sales) < 5:  # Need at least 5 values to get 1 row after dropna
            st.error("Please provide at least 5 sales values to generate features and SHAP plot.")
            st.stop()

        # Call FastAPI
        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json={"series_id": series_id, "values": sales}
            )
            response.raise_for_status()  # Raise an error for bad status codes
            result = response.json()
            prediction = result["forecast"][0]
            st.success(f"Predicted next week's sales for {series_id}: {prediction:.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {str(e)}")
            st.stop()

        # Feature engineering (match main.py)
        df = pd.DataFrame({"value": sales})
        for lag in range(1, 5):
            df[f"lag_{lag}"] = df["value"].shift(lag)
        df["rolling_mean_4"] = df["value"].rolling(window=4, min_periods=1).mean()
        df["rolling_std_4"] = df["value"].rolling(window=4, min_periods=1).std()
        df["trend"] = np.arange(len(df))
        feature_cols = ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_mean_4", "rolling_std_4", "trend"]
        X = df[feature_cols].dropna()

        if X.empty:
            st.error("Not enough data to generate features after processing. Please provide more sales values.")
            st.stop()

        # Load actual model
        model = joblib.load("forecasting_model.pkl")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Beeswarm plot
        shap_data = pd.DataFrame({
            "Feature": feature_cols * len(shap_values),
            "SHAP Value": shap_values.flatten(),
            "Feature Value": X.values.flatten()
        })
        fig = px.scatter(
            shap_data,
            x="SHAP Value",
            y="Feature",
            color="Feature Value",
            title=f"SHAP Values for {series_id} Prediction",
            color_continuous_scale="Viridis"
        )
        fig.update_traces(marker=dict(size=15))
        fig.update_layout(yaxis_title="Features", xaxis_title="SHAP Value (Impact on Prediction)")
        st.plotly_chart(fig)

    except ValueError:
        st.error("Invalid input. Use numbers separated by commas.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")