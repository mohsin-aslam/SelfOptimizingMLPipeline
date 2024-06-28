from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from ml_project.data_preprocessing import sanity_checks, handle_missing_values, correct_data_types, remove_duplicates
from ml_project.removing_outliers import remove_outliers



default_args = {
    'owner': 'Mohsin',
    'start_date': datetime(2022, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('data_cleaning_dag',
         default_args=default_args,
         schedule_interval='@weekly',
         catchup=False) as dag:

    
    handle_missing_values_task = PythonOperator(
        task_id='handle_missing_values',
        python_callable=handle_missing_values,
        op_kwargs={'strategy': 'drop', 'columns': None},
        provide_context=True
    )
    
    remove_duplicates_task = PythonOperator(
        task_id='remove_duplicates',
        python_callable=remove_duplicates,
        provide_context=True
    )
    
    correct_data_types_task = PythonOperator(
        task_id='correct_data_types',
        python_callable=correct_data_types,
        provide_context=True
    )
    
    sanity_checks_task = PythonOperator(
        task_id='sanity_checks',
        python_callable=sanity_checks,
        provide_context=True
    ),

    remove_outliers_task = PythonOperator(
        task_id='remove_outliers',
        python_callable=remove_outliers,
        provide_context=True
    )

    # Setting up the task sequence
    handle_missing_values_task >> remove_duplicates_task >> correct_data_types_task >> sanity_checks_task >> remove_outliers_task

