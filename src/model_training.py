import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.config import config

from src.logger import get_logger

logger = get_logger(__name__)

def train_model():
    """
    Loads features, trains a Random Forest model, and saves it.
    """
    logger.info("Starting model training...")
    
    # Load features
    input_path = config.FINAL_DATA_DIR / "features.csv"
    if not os.path.exists(input_path):
        logger.error(f"Features data not found at {input_path}. Run feature engineering first.")
        raise FileNotFoundError(f"Features data not found at {input_path}. Run feature engineering first.")
    
    df = pd.read_csv(input_path)
    
    # Separate Features and Target
    if config.TARGET_COL not in df.columns:
        logger.error(f"Target column {config.TARGET_COL} not found in dataset.")
        raise ValueError(f"Target column {config.TARGET_COL} not found in dataset.")
        
    X = df.drop(columns=[config.TARGET_COL])
    y = df[config.TARGET_COL]
    
    # Train/Test Split
    logger.info(f"Splitting data with random state {config.RANDOM_STATE}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    
    # Initialize and Train Model
    logger.info("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Save Model
    logger.info(f"Saving model to {config.MODEL_PATH}...")
    joblib.dump(model, config.MODEL_PATH)
    
    logger.info("Model training complete.")
    return model

if __name__ == "__main__":
    train_model()
