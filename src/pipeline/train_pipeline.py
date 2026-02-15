from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import get_logger
import sys

logger = get_logger(__name__)

def run_training_pipeline():
    try:
        logger.info(">>>>> Starting Training Pipeline <<<<<")
        
        # 1. Ingestion
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()
        
        # 2. Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
        
        # 3. Training
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logger.info(f"Training Pipeline Completed. Final R2 Score: {r2_score}")
        print(f"Training Completed. R2 Score: {r2_score}")
        
    except Exception as e:
        logger.error(f"Training Pipeline Failed: {e}")
        # We don't raise here to allow standard exit but we log the error
        sys.exit(1)

if __name__ == "__main__":
    run_training_pipeline()
