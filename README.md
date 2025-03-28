# Forecasting API

This project deploys a time series forecasting model as a FastAPI service on Google Cloud Run for Assignment 2.

## Overview
- **Task**: Weekly sales forecasting using the `m4_weekly` dataset.
- **Model**: XGBoost (`XGBRegressor`).
- **Deployment**: Docker + Google Cloud Run.

## Users
- **Target Users**: Data analysts/business intelligence teams.
- **Daily Requests**: ~100 requests/day.
- **Requirements**: Real-time responses (<1s).

## Setup Instructions
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/gurisapuriya/forecasting-api.git
   cd forecasting-api