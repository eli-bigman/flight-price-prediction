import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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

            # 1. Target Transformation (The 'Senior' move)
            # np.log1p is used to handle zero values safely, though fares are > 0
            y_train_log = np.log1p(y_train)

            # 2. Initialize the Refined Random Forest
            # We use the deep-tree architecture that performed best in our initial run
            refined_rf = RandomForestRegressor(
                n_estimators=500, 
                max_depth=25,       # Increased depth for finer detail
                max_features=None,   # Using all features based on our heatmap analysis 
                n_jobs=-1
            )

            # 3. Train on Log-Transformed Data
            logger.info("Training Refined Random Forest on Log-Transformed Data...")
            refined_rf.fit(X_train, y_train_log)

            # 4. Predict and Inverse Transform
            # We must use np.expm1 to bring the log-prices back to BDT
            log_preds = refined_rf.predict(X_test)
            final_preds = np.expm1(log_preds)

            # 5. Final Metrics
            refined_r2 = r2_score(y_test, final_preds)
            refined_mae = mean_absolute_error(y_test, final_preds)
            refined_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
            
            logger.info(f"Model R2 Score: {refined_r2}")
            logger.info(f"Model MAE: {refined_mae}")
            logger.info(f"Model RMSE: {refined_rmse}")
            
            model = refined_rf # Assign to model for saving
            r2 = refined_r2

            # Save Model
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved at {self.model_path}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
