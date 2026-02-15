import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

import os

logger = get_logger(__name__)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Flight Price Predictor", layout="wide")

st.title("✈️ Flight Price Prediction App")
st.markdown("Enter flight details to estimated the ticket price!")

# --- Sidebar Inputs ---
st.sidebar.header("Flight Details")
airline = st.sidebar.selectbox("Airline", ['Biman Bangladesh Airlines', 'US-Bangla Airlines', 'Novoair', 'Air Astra'])
travel_class = st.sidebar.selectbox("Class", ['Economy', 'Business', 'First Class'])
aircraft_type = st.sidebar.selectbox("Aircraft Type", ['Boeing 737', 'Dash 8-Q400', 'ATR 72-600'])

# --- Main Layout ---
col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Source", ['DAC', 'CGP', 'ZYL', 'CXB', 'RJH', 'SPD', 'BZL', 'JSR'])
    destination = st.selectbox("Destination", ['DAC', 'CGP', 'ZYL', 'CXB', 'RJH', 'SPD', 'BZL', 'JSR'])
    
    dept_date = st.date_input("Departure Date", min_value=datetime.today())
    dept_time = st.time_input("Departure Time")
    
    arrival_date = st.date_input("Arrival Date", min_value=dept_date)
    arrival_time = st.time_input("Arrival Time")

with col2:
    total_stops = st.number_input("Total Stops", min_value=0, max_value=5, step=1)
    duration_hrs = st.number_input("Duration (Hours)", min_value=0.5, step=0.5)
    seasonality = st.selectbox("Seasonality", ['Regular', 'Eid', 'Winter Holidays', 'Hajj'])

# --- Calculated Logic ---
days_before_departure = (pd.to_datetime(dept_date) - pd.Timestamp.today()).days
if days_before_departure < 0: days_before_departure = 0

dept_datetime = f"{dept_date} {dept_time}"
arrival_datetime = f"{arrival_date} {arrival_time}"

if st.button("Predict Price"):
    payload = {
        "airline": airline,
        "source": source,
        "destination": destination,
        "departure_time": dept_datetime,
        "arrival_time": arrival_datetime,
        "duration_hrs": duration_hrs,
        "total_stops": total_stops,
        "seasonality": seasonality,
        "days_before_departure": days_before_departure,
        "aircraft_type": aircraft_type,
        "travel_class": travel_class
    }
    
    try:
        # Option 1: Call API (Production Way)
        logger.info(f"Sending request to API: {payload}")
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            price = response.json().get("predicted_price")
            st.success(f"💰 Estimated Price: BDT {price:,.2f}")
        else:
            st.error(f"Error: {response.text}")
            
    except Exception as e:
        logger.error(f"Streamlit Error: {e}")
        st.error(f"Connection Error. Is the API running? {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by Refined Random Forest Model & FastAPI")
