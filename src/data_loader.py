import pandas as pd
from src.config import config
import os

from src.logger import get_logger

logger = get_logger(__name__)

def load_raw_data():
    """
    Loads the raw flight price dataset.
    
    Returns:
        pd.DataFrame: The raw dataframe.
    """
    if not os.path.exists(config.RAW_DATA_PATH):
        logger.error(f"Raw data file not found at {config.RAW_DATA_PATH}")
        raise FileNotFoundError(f"Raw data file not found at {config.RAW_DATA_PATH}")
    
    df = pd.read_csv(config.RAW_DATA_PATH)
    logger.info(f"Data loaded successfully from {config.RAW_DATA_PATH}")
    logger.info(f"Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test loading
    try:
        df = load_raw_data()
        logger.info(df.head())
    except Exception as e:
        logger.error(f"Error loading data: {e}")
