import streamlit as st
import requests
import joblib
import os
import datetime
import pandas as pd
from src.config import config

from src.logger import get_logger

logger = get_logger(__name__)

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
logger.info(f"Streamlit app started. API_URL: {API_URL}")

st.set_page_config(page_title="Flight Price Predictor", layout="wide")

st.title("✈️ Flight Price Prediction")
st.markdown("Enter flight details to get an estimated fare.")

# Load Encoders for Dropdown Options
@st.cache_data
def load_options():
    options = {}
    if os.path.exists(config.ENCODER_DIR):
        files = os.listdir(config.ENCODER_DIR)
        for file in files:
            if file.endswith("_encoder.pkl"):
                col = file.replace("_encoder.pkl", "")
                le = joblib.load(config.ENCODER_DIR / file)
                options[col] = list(le.classes_)
    return options

options = load_options()

# --- INPUT FORM ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Airline
        airline_opts = options.get('airline', [])
        airline = st.selectbox("Airline", airline_opts)
        
        # Source
        source_opts = options.get('source', [])
        source = st.selectbox("Source City", source_opts)
        
        # Destination
        dest_opts = options.get('destination', [])
        destination = st.selectbox("Destination City", dest_opts)
        
        # Departure Date
        dep_date = st.date_input("Departure Date", datetime.date.today())
        dep_time = st.time_input("Departure Time", datetime.time(10, 0))
        
    with col2:
        # Class
        class_opts = options.get('class', ['Economy', 'Business']) # Fallback if not found
        flight_class = st.selectbox("Class", class_opts)
        
        # Stops
        stops_opts = options.get('stopovers', ['0', '1', '2+'])
        stopovers = st.selectbox("Stopovers", stops_opts)
        
        # Arrival (Simplified: User might not know arrival, maybe estimate duration?)
        # For model input, we need arrival time or duration. 
        # Let's ask for duration and calculate arrival, or ask for arrival.
        # Notebook used duration. Let's ask for duration.
        duration = st.number_input("Duration (Hours)", min_value=0.5, value=2.0)
        
        # Days before departure (calculated)
        days_before = (dep_date - datetime.date.today()).days
        st.write(f"Days before departure: {days_before}")

    # Hidden/Default Fields
    # The model expects these but we might not want to ask user everything
    # Aircraft Type, Booking Source, Source Name, Destination Name
    # We can use mode/defaults or selectboxes if we have options
    
    with st.expander("Advanced Options"):
        aircraft_opts = options.get('aircraft_type', [])
        aircraft_type = st.selectbox("Aircraft Type", aircraft_opts) if aircraft_opts else st.text_input("Aircraft Type", "Airbus A320")
        
        booking_opts = options.get('booking_source', [])
        booking_source = st.selectbox("Booking Source", booking_opts) if booking_opts else "Website"

    submit = st.form_submit_button("Predict Price")

if submit:
    # Construct Datetime Strings
    dep_dt = datetime.datetime.combine(dep_date, dep_time)
    arr_dt = dep_dt + datetime.timedelta(hours=duration)
    
    # Payload
    payload = {
        "airline": airline,
        "source": source,
        "destination": destination,
        "departure_date_and_time": dep_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "arrival_date_and_time": arr_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_hrs": duration,
        "stopovers": stopovers,
        "aircraft_type": aircraft_type,
        "flight_class": flight_class,
        "booking_source": booking_source,
        "days_before_departure": days_before
    }
    
    with st.spinner("Calculating..."):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            price = result.get("predicted_price", 0)
            
            st.success(f"Estimated Ticket Price: BDT {price:,.2f}")
        except Exception as e:
            logger.error(f"Error predicting price: {e}")
            st.error(f"Error predicting price: {e}")
            st.error("Make sure the Inference API is running!")
