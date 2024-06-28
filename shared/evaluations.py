from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db_engine(connection_string):
    try:
        engine = create_engine(connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        return engine, session
    except SQLAlchemyError as e:
        print(f"Error creating engine: {e}")
        return None, None

def define_table(engine, table_name, schema_name):
    if engine is None:
        print("Engine is not initialized.")
        return None

    try:
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine, schema=schema_name)
        return table
    except SQLAlchemyError as e:
        print(f"Error defining table: {e}")
        return None

def convert_data_types(data_dict, table):
    for column in table.columns:
        column_name = column.name
        if column_name in data_dict:
            try:
                if column.type.python_type == float:
                    data_dict[column_name] = float(data_dict[column_name])
                elif column.type.python_type == int:
                    data_dict[column_name] = int(data_dict[column_name])
                # Add more type conversions as needed
            except ValueError:
                logger.error(f"Invalid value for column '{column_name}': {data_dict[column_name]}")
                data_dict[column_name] = None
    return data_dict

def insert_data(session, table, data_dict):
    if session is None or table is None:
        print("Session or table is not initialized.")
        return

    try:
        data_dict = convert_data_types(data_dict, table)
        stmt = insert(table).values(data_dict)
        session.execute(stmt)
        session.commit()
        print("Row inserted successfully.")
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Error inserting data: {e}")


def insert_evaluations(evaluation_data):
    # Define the database connection string
    db_connection_string = 'postgresql+psycopg2://root:root@172.27.0.7:5432/airflow'
    
    
    # Create the database engine and session
    engine, session = create_db_engine(db_connection_string)
    if engine is None or session is None:
        print("Failed to create database engine or session.")
        return
    
    # Define the table schema
    schema_name = 'ml_project'
    table_name = 'evaluations'  # Replace with your table name
    table = define_table(engine, table_name, schema_name)
    if table is None:
        print("Failed to define table schema.")
        return

    
    # Insert the data
    insert_data(session, table, evaluation_data)

