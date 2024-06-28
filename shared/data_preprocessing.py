import pandas as pd
import os

def load_data(file_path):
    # path = kwargs.get('path', 'path_to_your_data.csv')
    try:
        print(os.getcwd())
        # Get the absolute path to the current file
        current_file_path = os.path.abspath(__file__)
        print("Current File Path:", current_file_path)

        # Get the directory containing the current file
        current_dir_path = os.path.dirname(current_file_path)
        print("Directory of Current File:", current_dir_path)
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")

def handle_missing_values(strategy='drop', columns=None):
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/2020_Yellow_Taxi_Trip_Data.csv')
        print('load complete')
        if strategy == 'drop':
            print('handle_missing_values complete')
            print(len(df))
            save_data(df.dropna(), os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
            print('save complete')
            return True
        elif strategy == 'fill' and columns is not None:
            for column in columns:
                if df[column].dtype in ['float64', 'int64']:
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            print('handle_missing_values complete')
            save_data(df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
            print('save complete')
            return True
    except Exception as e:
        raise ValueError(f"Error in handling missing values: {str(e)}")

def correct_data_types():
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        datetime_format = '%m/%d/%Y %I:%M:%S %p'
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format=datetime_format, errors='coerce')
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=datetime_format, errors='coerce')
        save_data(df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
    except Exception as e:
        raise ValueError(f"Error in correcting data types: {str(e)}")

def remove_duplicates():
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        save_data(df.drop_duplicates(), os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
    except Exception as e:
        raise ValueError(f"Error in removing duplicates: {str(e)}")

def sanity_checks():
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        conditions = (
            (df['passenger_count'] == 0) |
            (df['trip_distance'] == 0) |
            (df['fare_amount'] < 0) |
            (df['total_amount'] < 0)
        )
        df_cleaned = df[~conditions]
        print(len(df))
        save_data(df_cleaned, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        save_data(df_cleaned, os.path.dirname(os.path.abspath(__file__)) + '/resources/data_processing.csv')
        return True
    except Exception as e:
        raise ValueError(f"Error in performing sanity checks: {str(e)}")

def save_data(df, file_path):
    # path = kwargs.get('path', 'path_to_your_data.csv')
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise ValueError(f"Failed to save data to {file_path}: {str(e)}")
