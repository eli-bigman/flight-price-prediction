from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Flight Price Prediction API", version="1.0.0")

class FlightInput(BaseModel):
    airline: str
    source: str
    destination: str
    departure_time: str
    arrival_time: str
    duration_hrs: float
    total_stops: int
    seasonality: str
    days_before_departure: int
    aircraft_type: str
    travel_class: str

@app.get("/")
def home():
    return {"message": "Flight Price Prediction API is Running!"}

@app.post("/predict")
def predict_flight_price(flight: FlightInput):
    logger.info(f"Received prediction request: {flight}")
    try:
        # 1. Convert Input to DataFrame
        data = CustomData(
            airline=flight.airline,
            source=flight.source,
            destination=flight.destination,
            departure_time=flight.departure_time,
            arrival_time=flight.arrival_time,
            duration_hrs=flight.duration_hrs,
            total_stops=flight.total_stops,
            seasonality=flight.seasonality,
            days_before_departure=flight.days_before_departure,
            aircraft_type=flight.aircraft_type,
            travel_class=flight.travel_class
        )
        final_df = data.get_data_as_data_frame()
        
        # 2. Get Prediction
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_df)
        
        result = round(float(pred[0]), 2)
        logger.info(f"Prediction success: {result}")
        
        return {"predicted_price": result}

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
