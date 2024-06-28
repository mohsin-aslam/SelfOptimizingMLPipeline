from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from ml_project.model_training import execute_ml_pipeline, execute_mlflow_pipeline
from airflow.sensors.external_task_sensor import ExternalTaskSensor
import sys
import os


default_args = {
    'owner': 'Mohsin',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='A simple ML pipeline DAG',
    schedule_interval='@weekly',
    catchup=False,
)

wait_for_feature_engineering = ExternalTaskSensor(
task_id='wait_for_feature_engineering',
external_dag_id='feature_engineering_dag',
external_task_id='prepare_data_for_modeling',
dag=dag)

execute_ml_pipeline = PythonOperator(
    task_id='execute_ml_pipeline',
    python_callable=execute_ml_pipeline,
    dag=dag,
)

execute_mlflow_pipeline = PythonOperator(
    task_id='execute_mlflow_pipeline',
    python_callable=execute_mlflow_pipeline,
    op_kwargs={
        'tracking_uri': 'http://172.27.0.5:5000',
        'experiment_name': 'Fare_Prediction'},
    dag=dag,
)


wait_for_feature_engineering >> execute_ml_pipeline >> execute_mlflow_pipeline
