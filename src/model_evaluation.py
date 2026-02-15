import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import config

from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model():
    """
    Loads features and trained model, evaluates on test set, and logs metrics.
    """
    logger.info("Starting model evaluation...")
    
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
    
    # Train/Test Split (Must match training split)
    logger.info(f"Splitting data with random state {config.RANDOM_STATE}...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    
    # Load Model
    if not os.path.exists(config.MODEL_PATH):
        logger.error(f"Model not found at {config.MODEL_PATH}. Run training first.")
        raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}. Run training first.")
        
    logger.info(f"Loading model from {config.MODEL_PATH}...")
    model = joblib.load(config.MODEL_PATH)
    
    # Predict
    logger.info("Predicting on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Log Metrics
    logger.info("Model Evaluation Metrics:")
    logger.info(f"MAE:  {mae:.4f}")
    logger.info(f"MSE:  {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2:   {r2:.4f}")
    
    # Save metrics to file (optional, good for pipeline artifacts)
    metrics_path = config.MODEL_DIR / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    evaluate_model()
