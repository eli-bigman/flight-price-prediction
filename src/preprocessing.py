import pandas as pd
import os
from src.config import config
from src.data_loader import load_raw_data

from src.logger import get_logger

logger = get_logger(__name__)

def preprocess_data():
    """
    Loads raw data, performs cleaning and preprocessing, and saves to processed data directory.
    """
    logger.info("Starting data preprocessing...")
    
    # Load Data
    df = load_raw_data()
    
    # 1. Standardize column names
    logger.info("Standardizing column names...")
    df.columns = [col.strip().lower().replace(' ', '_').replace('&', 'and').replace('(', '').replace(')', '') for col in df.columns]
    
    # 2. Handle Duplicates
    initial_len = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_len - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows.")
    
    # 3. Handle Missing Values (Imputation or Drop)
    # Based on notebook, we check but might not have missing values. We'll drop for safety or impute if needed.
    # For now, dropping any rows with nulls in critical columns
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        logger.info(f"Found {missing_count} missing values. Dropping rows...")
        df.dropna(inplace=True)
    
    # 4. Convert Date Columns to DateTime
    date_cols = ['departure_date_and_time', 'arrival_date_and_time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            logger.info(f"Converted {col} to datetime.")

    # Save cleaned data
    save_path = config.PROCESSED_DATA_DIR / "cleaned_data.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Cleaned data saved to {save_path}")
    
    return df

if __name__ == "__main__":
    preprocess_data()
