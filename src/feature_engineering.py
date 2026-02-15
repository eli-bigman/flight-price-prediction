import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from src.config import config

from src.logger import get_logger

logger = get_logger(__name__)

def run_feature_engineering():
    """
    Loads cleaned data, performs feature engineering (date extraction, encoding),
    saves encoders, and saves final features.
    """
    logger.info("Starting feature engineering...")
    
    # Load cleaned data
    input_path = config.PROCESSED_DATA_DIR / "cleaned_data.csv"
    if not os.path.exists(input_path):
        logger.error(f"Cleaned data not found at {input_path}. Run preprocessing first.")
        raise FileNotFoundError(f"Cleaned data not found at {input_path}. Run preprocessing first.")
    
    df = pd.read_csv(input_path)
    
    # Ensure date columns are datetime objects (read_csv might load as string)
    date_cols = ['departure_date_and_time', 'arrival_date_and_time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # 1. Extract Date Features
    logger.info("Extracting date features...")
    for col in date_cols:
        if col in df.columns:
            prefix = col.replace('_date_and_time', '')
            df[f'{prefix}_day'] = df[col].dt.day
            df[f'{prefix}_month'] = df[col].dt.month
            df[f'{prefix}_year'] = df[col].dt.year
            df[f'{prefix}_weekday'] = df[col].dt.weekday
            df[f'{prefix}_hour'] = df[col].dt.hour
            df[f'{prefix}_minute'] = df[col].dt.minute
            
            # Drop original date column as models can't ingest datetime objects directly
            df.drop(columns=[col], inplace=True)

    # 1.5 Drop leakage columns (price components)
    leakage_cols = ['base_fare_bdt', 'tax_and_surcharge_bdt']
    for col in leakage_cols:
        if col in df.columns:
            logger.info(f"Dropping leakage column: {col}")
            df.drop(columns=[col], inplace=True)

    # 2. Categorical Encoding
    # Identify categorical columns (object type)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Categorical columns to encode: {cat_cols}")
    
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
        # Save the encoder for inference
        encoder_path = config.ENCODER_DIR / f"{col}_encoder.pkl"
        joblib.dump(le, encoder_path)
        logger.info(f"Saved encoder for {col} at {encoder_path}")

    # 3. Save Final Features
    save_path = config.FINAL_DATA_DIR / "features.csv"
    df.to_csv(save_path, index=False)
    logger.info(f"Feature engineering complete. Data saved to {save_path}")
    logger.info(f"Data shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    run_feature_engineering()
