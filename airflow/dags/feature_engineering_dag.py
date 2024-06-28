from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from ml_project.feature_engineering import execute_feature_engineering_pipeline, apply_cyclic_encoding, typecasting_variables, prepare_data_for_modeling, feature_selection,scale_features
import sys
import os


default_args = {
    'owner': 'Mohsin',
    'start_date': datetime(2022, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('feature_engineering_dag',
         default_args=default_args,
         schedule_interval='@weekly',
         catchup=False) as dag:


    wait_for_preprocessing = ExternalTaskSensor(
    task_id='wait_for_preprocessing',
    external_dag_id='data_cleaning_dag',
    external_task_id='remove_outliers',
    dag=dag)


    feature_engineering_task = PythonOperator(
        task_id='execute_feature_engineering_pipeline',
        python_callable=execute_feature_engineering_pipeline,
        op_kwargs={'chunk_size': 50000},
        dag=dag
    )

    apply_cyclic_encoding_task = PythonOperator(
        task_id='apply_cyclic_encoding',
        python_callable=apply_cyclic_encoding,
        op_kwargs={'column_name': 'pickup_day_of_week'},
        provide_context=True
    )

    typecasting_variables_task = PythonOperator(
        task_id='typecasting_variables',
        python_callable=typecasting_variables,
        provide_context=True
    )

    feature_selection_task = PythonOperator(
        task_id='feature_selection',
        python_callable=feature_selection,
        op_kwargs={
            'fare_prediction_columns': ['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
                                        'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'trip_duration',
                                        'congestion_surcharge', 'pickup_hour', 'day_sin', 'day_cos',
                                        'pickup_month', 'fare_amount'
                                    ]},
        provide_context=True
    )

    scale_features_task = PythonOperator(
        task_id='scale_features',
        python_callable=scale_features,
        op_kwargs={'columns_to_scale': ['trip_distance', 'pickup_hour', 'pickup_month', 'PULocationID', 'DOLocationID', 'trip_duration']},
        dag=dag
    )

    prepare_data_for_modeling_task = PythonOperator(
    task_id='prepare_data_for_modeling',
    python_callable=prepare_data_for_modeling,
    op_kwargs={
        'base_features': ['trip_distance', 'pickup_hour', 'pickup_month', 'PULocationID', 'DOLocationID', 'trip_duration'],
        'categorical_vars': ['VendorID', 'RatecodeID', 'store_and_fwd_flag', ],
        'target_variable': 'fare_amount'},
    provide_context=True
    )   

    wait_for_preprocessing >> feature_engineering_task >> apply_cyclic_encoding_task >> typecasting_variables_task >> feature_selection_task >> scale_features_task >> prepare_data_for_modeling_task
