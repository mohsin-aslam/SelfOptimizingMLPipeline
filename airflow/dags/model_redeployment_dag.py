from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

db_connection_string = 'postgresql+psycopg2://root:root@172.27.0.7:5432/airflow'
engine = create_engine(db_connection_string)

def check_condition(**kwargs):
    query = """
    SELECT 
        CASE 
            WHEN COUNT(*) = 0 THEN true  -- No rows for today, so return true
            WHEN COUNT(*) = SUM(
                CASE 
                    WHEN e.evaluation_rmse > (e.model_rmse + 3) AND e.evaluation_r2 < (e.model_r2 - 0.3)
                    THEN 1 
                    ELSE 0 
                END
            ) THEN true
            ELSE false
        END as all_conditions_met
    FROM 
        ml_project.evaluations e
    WHERE 
        DATE(e.execution_date) = CURRENT_DATE;
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            all_conditions_met = result.scalar()  # Get the single boolean result
            kwargs['ti'].xcom_push(key='condition_met', value=all_conditions_met)
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")
        kwargs['ti'].xcom_push(key='condition_met', value=False)

def trigger_conditionally(**kwargs):
    ti = kwargs['ti']
    condition_met = ti.xcom_pull(task_ids='check_condition_task', key='condition_met')
    if condition_met:
        return 'trigger_other_dag_task'
    else:
        return 'no_op_task'

default_args = {
    'owner': 'Mohsin',
    'start_date': days_ago(1),
}

with DAG(
    'automated_redeployment_dag',
    default_args=default_args,
    description='Main DAG to decide triggering another DAG',
    schedule_interval='0 0 * * *',
) as dag:
    check_condition_task = PythonOperator(
        task_id='check_condition_task',
        python_callable=check_condition,
        provide_context=True,
    )

    trigger_conditionally_task = PythonOperator(
        task_id='trigger_conditionally_task',
        python_callable=trigger_conditionally,
        provide_context=True,
    )

    trigger_other_dag_task = TriggerDagRunOperator(
        task_id='trigger_other_dag_task',
        trigger_dag_id='load_data_dag',  # DAG id of the DAG to trigger
    )

    no_op_task = PythonOperator(
        task_id='no_op_task',
        python_callable=lambda: print("Condition not met, no operation performed."),
    )

    check_condition_task >> trigger_conditionally_task
    trigger_conditionally_task >> [trigger_other_dag_task, no_op_task]
