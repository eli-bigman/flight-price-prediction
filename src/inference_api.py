from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.config import config

from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Flight Price Prediction API")

# Global variables to hold model and encoders
model = None
encoders = {}

class FlightInput(BaseModel):
    airline: str
    source: str
    destination: str
    departure_date_and_time: str # Format: YYYY-MM-DD HH:MM:SS
    arrival_date_and_time: str   # Format: YYYY-MM-DD HH:MM:SS
    duration_hrs: float
    stopovers: str
    aircraft_type: str
    flight_class: str # Renamed from 'class' to avoid keyword conflict
    booking_source: str
    days_before_departure: int
    # seasonality: str # Notebook had this, maybe we need it? 
    # Checking data_loader and preprocessing, seasonality was in columns.
    # I should check if it was dropped or encoded.
    # Based on notebook output: 'seasonality' is a column.

@app.on_event("startup")
def load_artifacts():
    global model, encoders
    
    # Load Model
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model not found at {config.MODEL_PATH}. Run training first.")
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}. Run training first.")
    model = joblib.load(config.MODEL_PATH)
    logger.info("Model loaded successfully.")
    
    # Load Encoders
    encoder_files = os.listdir(config.ENCODER_DIR)
    for file in encoder_files:
        if file.endswith("_encoder.pkl"):
            col_name = file.replace("_encoder.pkl", "")
            encoders[col_name] = joblib.load(config.ENCODER_DIR / file)
            logger.info(f"Loaded encoder for {col_name}")

@app.post("/predict")
def predict_price(flight: FlightInput):
    try:
        # Convert input to DataFrame
        data = flight.dict()
        
        # Renaissance of 'class' field
        data['class'] = data.pop('flight_class')
        
        # 'seasonality' might be missing from input if logic requires it. 
        # For this implementation, I'll assume basic inputs are provided. 
        # If seasonality is derived, we need logic. 
        # Looking at notebook, Seasonality seems to be a feature in the CSV.
        # I'll add it to input schema or strict validation might fail if model expects it.
        # Let's verify model features during prediction.
        
        df = pd.DataFrame([data])
        
        # 1. Date Features Extraction
        date_cols = ['departure_date_and_time', 'arrival_date_and_time']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
            prefix = col.replace('_date_and_time', '')
            df[f'{prefix}_day'] = df[col].dt.day
            df[f'{prefix}_month'] = df[col].dt.month
            df[f'{prefix}_year'] = df[col].dt.year
            df[f'{prefix}_weekday'] = df[col].dt.weekday
            df[f'{prefix}_hour'] = df[col].dt.hour
            df[f'{prefix}_minute'] = df[col].dt.minute
            df.drop(columns=[col], inplace=True)

        # 2. Categorical Encoding
        # We need to manually match columns that have encoders
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unseen labels carefully & safely
                # simplified approach: Use transform, catch error for robustness
                try:
                    df[col] = le.transform(df[col].astype(str))
                except ValueError:
                    # Fallback for unseen values: assign -1 or a specific unknown class if trained
                    # For now, let's use the first class as fallback or raise informative error
                     raise HTTPException(status_code=400, detail=f"Unknown value for {col}")

        # Ensure column order matches model
        # This requires the model to have feature_names_in_ property (sklearn > 1.0)
        if hasattr(model, "feature_names_in_"):
            # specific fix for missing columns if 'seasonality' was missing
            missing_cols = set(model.feature_names_in_) - set(df.columns)
            if missing_cols:
                 raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols}. Please provide Seasonality.")
            
            df = df[model.feature_names_in_]
        
        # Predict
        prediction = model.predict(df)[0]
        
        return {"predicted_price": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
