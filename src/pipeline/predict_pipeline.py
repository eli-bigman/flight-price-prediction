import sys
import os
import pandas as pd
import numpy as np
import joblib
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.config.configuration import config

logger = get_logger(__name__)

class PredictPipeline:
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.preprocessor_path = config.PROCESSED_DATA_DIR / "preprocessor.pkl"

    def predict(self, features):
        try:
            # Load Artifcats
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)

            logger.info("Artifacts loaded for prediction")
            
            # Use the 'transform' method, not fit_transform
            data_scaled = preprocessor.transform(features)
            
            # Predict (Log Scale)
            log_preds = model.predict(data_scaled)
            
            # Inverse Transform
            preds = np.expm1(log_preds)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 airline: str,
                 source: str,
                 destination: str,
                 departure_time: str,
                 arrival_time: str,
                 duration_hrs: float,
                 total_stops: int,
                 seasonality: str,
                 days_before_departure: int,
                 aircraft_type: str,
                 travel_class: str):
        
        self.airline = airline
        self.source = source
        self.destination = destination
        self.departure_time = departure_time
        self.arrival_time = arrival_time
        self.duration_hrs = duration_hrs
        self.total_stops = total_stops
        self.seasonality = seasonality
        self.days_before_departure = days_before_departure
        self.aircraft_type = aircraft_type
        self.travel_class = travel_class # Mapped to 'class' in dataframe

    def get_data_as_data_frame(self):
        try:
            # Match the input names expected by FeatureEngineering
            # Note: FeatureEngineering expects raw column names initially, then cleans them.
            # We will provide names that match the cleaned version or raw version?
            # FeatureEngineering cleans: .strip().lower().replace(' ', '_')...
            # So 'Departure Date and Time' -> 'departure_date_and_time'
            
            custom_data_input_dict = {
                "airline": [self.airline],
                "source": [self.source],
                "destination": [self.destination],
                "departure_date_and_time": [self.departure_time],
                "arrival_date_and_time": [self.arrival_time],
                "duration_hrs": [self.duration_hrs],
                "total_stops": [self.total_stops],
                "seasonality": [self.seasonality],
                "days_before_departure": [self.days_before_departure],
                "aircraft_type": [self.aircraft_type],
                "class": [self.travel_class] # Original csv had 'Class' or 'class'? Input csv usually has 'class' based on previous checks.
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
