import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.config.configuration import config

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.ingestion_config.RAW_DATA_PATH)
            logger.info('Read the dataset as dataframe')

            logger.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=self.ingestion_config.RANDOM_STATE)

            train_set.to_csv(self.ingestion_config.TRAIN_DATA_PATH, index=False, header=True)
            test_set.to_csv(self.ingestion_config.TEST_DATA_PATH, index=False, header=True)

            logger.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.TRAIN_DATA_PATH,
                self.ingestion_config.TEST_DATA_PATH
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
