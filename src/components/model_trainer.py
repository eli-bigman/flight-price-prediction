import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.config.configuration import config

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        self.model_path = config.MODEL_PATH

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Split training and test input data")
            
            # Assuming last column is target, as concatenated in DataTransformation
            X_train = train_array.iloc[:, :-1]
            y_train = train_array.iloc[:, -1]
            X_test = test_array.iloc[:, :-1]
            y_test = test_array.iloc[:, -1]

            logger.info("Applying Log Transformation to Target (refined_rf logic)")
            y_train_log = np.log1p(y_train)

            logger.info("Initializing Refined Random Forest Model")
            # Logic from Notebook 'refined_rf'
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=25,
                max_features=None, # As explicitly requested
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )

            logger.info("Training Model...")
            model.fit(X_train, y_train_log)

            logger.info("Model training complete. Evaluating on Test data...")
            
            # Predict
            log_preds = model.predict(X_test)
            
            # Inverse Transform predictions
            final_preds = np.expm1(log_preds)
            
            r2 = r2_score(y_test, final_preds)
            logger.info(f"Model R2 Score: {r2}")

            # Save Model
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved at {self.model_path}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
