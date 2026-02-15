import os
from pathlib import Path

class Config:
    # Project root directory
    result = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = Path(result).parent

    # Data directories
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_DIR / "Flight_Price_Dataset_of_Bangladesh.csv"
    
    # Processed data directory (create if not exists)
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FINAL_DATA_DIR = DATA_DIR / "final"
    
    # Model directory (create if not exists)
    MODEL_DIR = PROJECT_ROOT / "models"
    ENCODER_DIR = MODEL_DIR / "encoders"
    MODEL_PATH = MODEL_DIR / "flight_price_model.pkl"

    # Constants
    RANDOM_STATE = 42
    TARGET_COL = 'total_fare_bdt'
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.FINAL_DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.ENCODER_DIR, exist_ok=True)

config = Config()
