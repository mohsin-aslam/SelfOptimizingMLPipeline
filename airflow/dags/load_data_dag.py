from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
import os

def load_data():
    file_path = os.getcwd() + '/dags/ms_iba/resources/2020_Yellow_Taxi_Trip_Data.csv'
    print(file_path)
    try:
        df = pd.read_csv(file_path)        
        print("Data loading complete.")
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except pd.errors.EmptyDataError:
        print("Error: No data found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def save_to_sql():
    # Database connection parameters
    db_connection_string = 'postgresql+psycopg2://root:root@172.27.0.7:5432/airflow'
    engine = create_engine(db_connection_string)
    file_path = os.getcwd() + '/dags/ms_iba/resources/2020_Yellow_Taxi_Trip_Data.csv'
    df = pd.read_csv(file_path)

    try:
        # Save the DataFrame to SQL
        df.to_sql('taxi_service_data', con=engine, if_exists='replace', index=False)
        print("Data successfully saved to SQL.")
    except Exception as e:
        print(f"An error occurred while saving to SQL: {e}")

default_args = {
    'owner': 'Mohsin',
    'start_date': datetime(2022, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('load_data_dag',
          description='DAG for loading data',
          default_args=default_args,
          schedule_interval='@weekly',
          start_date=datetime(2021, 1, 1),
          catchup=False)

load_data_task = PythonOperator(task_id='load_data_task',
                                python_callable=load_data,
                                dag=dag)

save_to_sql_task = PythonOperator(
    task_id='save_to_sql_task',
    python_callable=save_to_sql,  
    dag=dag)

load_data_task >> save_to_sql_task  # Set task dependency