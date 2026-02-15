from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to sys.path to allow importing from src
# Assuming the dags folder is at project_root/dags
# We need to add project_root to path
dag_folder = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(dag_folder)
sys.path.append(project_root)

from src.preprocessing import preprocess_data
from src.feature_engineering import run_feature_engineering
from src.model_training import train_model
from src.model_evaluation import evaluate_model

# Default Arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
with DAG(
    'flight_price_prediction_pipeline',
    default_args=default_args,
    description='End-to-end flight price prediction pipeline',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Task 1: Preprocessing
    # This also handles data loading internally
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # Task 2: Feature Engineering
    feature_eng_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_feature_engineering,
    )

    # Task 3: Model Training
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 4: Model Evaluation
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    # Task Dependencies
    preprocess_task >> feature_eng_task >> train_task >> evaluate_task
