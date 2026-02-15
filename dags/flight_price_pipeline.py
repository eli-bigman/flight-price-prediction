from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Ensure src is in pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.train_pipeline import run_training_pipeline

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'flight_price_training_pipeline',
    default_args=default_args,
    description='A training pipeline for Flight Price Prediction',
    schedule_interval=timedelta(days=1),
)

def run_pipeline():
    run_training_pipeline()

t1 = PythonOperator(
    task_id='run_training_pipeline',
    python_callable=run_pipeline,
    dag=dag,
)
