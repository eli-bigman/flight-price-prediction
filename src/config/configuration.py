import os
from pathlib import Path

class Config:
    def __init__(self):
        self.PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_PATH = self.DATA_DIR / "Flight_Price_Dataset_of_Bangladesh.csv"
        
        # Artifacts
        self.ARTIFACTS_DIR = self.PROJECT_ROOT / "artifacts"
        self.PROCESSED_DATA_DIR = self.ARTIFACTS_DIR / "data"
        self.MODEL_DIR = self.ARTIFACTS_DIR / "models"
        self.ENCODER_DIR = self.MODEL_DIR / "encoders"
        
        # File Paths
        self.TRAIN_DATA_PATH = self.PROCESSED_DATA_DIR / "train.csv"
        self.TEST_DATA_PATH = self.PROCESSED_DATA_DIR / "test.csv"
        self.MODEL_PATH = self.MODEL_DIR / "refined_model.pkl"
        
        # Create directories
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.ENCODER_DIR, exist_ok=True)

        # Modeling Constants
        self.RANDOM_STATE = 111 # As per notebook refined_rf
        self.TARGET_COL = 'total_fare_bdt'

config = Config()
