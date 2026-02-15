from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'flight_price_granular_pipeline',
    default_args=default_args,
    description='A granular production pipeline for Flight Price Prediction',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False
)

def ingest_data(**kwargs):
    logger.info("Starting Data Ingestion Task")
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    # Push paths to XCom for next task
    kwargs['ti'].xcom_push(key='train_path', value=train_path)
    kwargs['ti'].xcom_push(key='test_path', value=test_path)
    return "Data Ingestion Completed"

def transform_data(**kwargs):
    logger.info("Starting Data Transformation Task")
    ti = kwargs['ti']
    train_path = ti.xcom_pull(key='train_path', task_ids='data_ingestion')
    test_path = ti.xcom_pull(key='test_path', task_ids='data_ingestion')
    
    # Path objects from pathlib might need conversion to string if passed loosely,
    # but XCom handles basic types well. Our code uses Path objects.
    # Ideally, we serialize/deserialize cleanly, but for local airflow, paths as strings work if consistent.
    # DataTransformation expects paths (str or Path).
    
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)
    
    # We can't easily pass big arrays via XCom (DB limits). 
    # Best practice: Save intermediate artifacts to shared storage (S3/GCS or Shared Volume).
    # Since our DataTransformation returns arrays in memory but DOES NOT save the transformed array by default 
    # (wait, looking at code: it returns arrays but only saves preprocessor.pkl).
    
    # PRODUCTION GAPS REVEALED:
    # The current `initiate_data_transformation` returns in-memory arrays.
    # In a real Airflow setup, we MUST save these to disk so the next task (Training) can read them.
    # For now, we will MODIFY this task to save them explicitly to a temp location or `artifacts/transformed`.
    
    # Let's save them here manually to bridge the gap without rewriting the component logic yet.
    import numpy as np
    import joblib
    from src.config.configuration import config
    
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    train_arr_path = config.PROCESSED_DATA_DIR / "train_arr.npy"
    test_arr_path = config.PROCESSED_DATA_DIR / "test_arr.npy"
    
    # DataTransformation returns DataFrames/Arrays.
    # Code review: `train_arr` is a DataFrame (pd.concat).
    import pandas as pd
    train_arr.to_csv(train_arr_path, index=False)
    test_arr.to_csv(test_arr_path, index=False)
    
    ti.xcom_push(key='train_arr_path', value=str(train_arr_path))
    ti.xcom_push(key='test_arr_path', value=str(test_arr_path))
    return "Data Transformation Completed"

def train_model(**kwargs):
    # This task will fail if we don't handle the loading of data we just saved.
    # ModelTrainer expects `train_array` and `test_array`.
    logger.info("Starting Model Training Task")
    ti = kwargs['ti']
    train_arr_path = ti.xcom_pull(key='train_arr_path', task_ids='data_transformation')
    test_arr_path = ti.xcom_pull(key='test_arr_path', task_ids='data_transformation')
    
    import pandas as pd
    train_arr = pd.read_csv(train_arr_path)
    test_arr = pd.read_csv(test_arr_path)
    
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    logger.info(f"Model Training Finished. R2: {r2_score}")
    return r2_score

t1 = PythonOperator(
    task_id='data_ingestion',
    python_callable=ingest_data,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='data_transformation',
    python_callable=transform_data,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='model_trainer',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

t1 >> t2 >> t3
