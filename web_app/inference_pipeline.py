import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
print(os.getcwd())

def handle_missing_values(df, strategy='drop', columns=None):
    try:
        print('load complete')
        if strategy == 'drop':
            print('handle_missing_values complete')
            return df
        elif strategy == 'fill' and columns is not None:
            for column in columns:
                if df[column].dtype in ['float64', 'int64']:
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            print('handle_missing_values complete')
            return df
    except Exception as e:
        raise ValueError(f"Error in handling missing values: {str(e)}")

def correct_data_types(df):
    try:
        print(df.head(10))
        print(type(df['tpep_pickup_datetime'][0]))
        datetime_format = '%m/%d/%Y %I:%M:%S %p'
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format=datetime_format, errors='coerce')
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=datetime_format, errors='coerce')
        print('checking',df.head(10))
        return df
    except Exception as e:
        raise ValueError(f"Error in correcting data types: {str(e)}")

# def correct_data_types(df):
#     try:
#         print(df.columns)
#         datetime_format = '%m/%d/%Y %I:%M:%S %p'
#         if 'tpep_pickup_datetime' not in df.columns or 'tpep_dropoff_datetime' not in df.columns:
#             raise KeyError("DataFrame does not contain required datetime columns.")

#         print("Before conversion:", df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head())
        
#         df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format=datetime_format, errors='coerce')
#         df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format=datetime_format, errors='coerce')
        
#         print("After conversion:", df[['tpep_pickup_datetime', 'tpep_dropoff_datetime']].head())

#     except Exception as e:
#         raise ValueError(f"Error in correcting data types: {str(e)}")

def remove_duplicates(df):
    try:
        return df.drop_duplicates()
    except Exception as e:
        raise ValueError(f"Error in removing duplicates: {str(e)}")

def sanity_checks(df):
    try:
        conditions = (
            (df['passenger_count'] == 0) |
            (df['trip_distance'] == 0) |
            # (df['fare_amount'] < 0) |
            (df['total_amount'] < 0)
        )
        df_cleaned = df[~conditions]
        return df_cleaned
    except Exception as e:
        raise ValueError(f"Error in performing sanity checks: {str(e)}")
    
def extract_time_features(df):
    try:
        print(df.head(10))
        print(df['tpep_pickup_datetime'].dtype)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        print(df['tpep_pickup_datetime'].dtype)
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
        print(df.head(10))
        return df
    except Exception as e:
        raise Exception(f"Error in extract_time_features: {str(e)}")

def calculate_distance_rate(df):
    try:
        df['fare_per_mile'] = df.apply(lambda x: x['fare_amount'] / x['trip_distance'] if x['trip_distance'] != 0 else 0, axis=1)
        return df
    except Exception as e:
        raise Exception(f"Error in calculate_distance_rate: {str(e)}")

def feature_engineering_pipeline(df):
    """
    A pipeline function that applies all feature engineering functions.
    :param df: DataFrame to be processed.
    :returns: DataFrame with all features engineered.
    """
    try:
        df = extract_time_features(df)
    except Exception as e:
        raise Exception(f"Error in extracting time features: {str(e)}")
    
    # try:
    #     df = calculate_distance_rate(df)
    # except Exception as e:
    #     raise Exception(f"Error in calculating distance rate: {str(e)}")
    
    # Example of how you might include encoding if uncommented
    # try:
    #     df = encode_categorical_features(df, ['pickup_day_of_week', 'payment_type'])
    # except Exception as e:
    #     raise Exception(f"Error in encoding categorical features: {str(e)}")
    
    return df


def execute_feature_engineering_pipeline(df, chunk_size=50000):
    """
    Process the DataFrame in chunks and apply feature engineering.
    :param df: DataFrame to be processed.
    :param chunk_size: Number of rows per chunk.
    :returns: DataFrame with all features engineered concatenated from all chunks.
    """
    try:
        # Calculate the number of chunks
        num_chunks = int(np.ceil(len(df) / chunk_size))
        
        # Split the DataFrame into chunks
        chunks = np.array_split(df, num_chunks)
        
        # List to store each chunk after processing
        chunk_list = []
        
        # Process each chunk
        for chunk in chunks:
            try:
                chunk_processed = feature_engineering_pipeline(chunk)
                chunk_list.append(chunk_processed)
            except Exception as e:
                raise Exception(f"Error processing chunk: {str(e)}")
        
        # Concatenate all processed chunks into a single DataFrame
        engineered_df = pd.concat(chunk_list, ignore_index=True)
        return engineered_df
    except Exception as e:
        raise Exception(f"Error in processing chunks: {str(e)}")

def apply_cyclic_encoding(df, column_name):
    try:
        day_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['day_number'] = df[column_name].map(day_to_num)
        if df['day_number'].isna().any():
            missing_values = df[df['day_number'].isna()][column_name].unique()
            raise ValueError(f"Mapping not found for: {missing_values}")
        df['day_sin'] = np.sin(2 * np.pi * df['day_number'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_number'] / 7)
        df.drop('day_number', axis=1, inplace=True)
        
        return df
    except Exception as e:
        raise Exception(f"Error in apply_cyclic_encoding: {str(e)}")

def typecasting_variables(df):
    try:
        df.dropna(inplace=True)
        df['VendorID'] = df['VendorID'].astype(int).astype(str)
        
        return df
    except Exception as e:
        raise Exception(f"Error in typecasting_variables: {str(e)}")

def feature_selection(df, fare_prediction_columns):
    try:
        fare_prediction_df = df[fare_prediction_columns]
        return fare_prediction_df
    except Exception as e:
        raise Exception(f"Error in feature_selection: {str(e)}")
    
def scale_features(df, columns_to_scale):
    """
    Scale selected features in a DataFrame using StandardScaler.

    Args:
    df (DataFrame): The DataFrame containing the data.
    columns_to_scale (list): A list of column names in the DataFrame to be scaled.

    Returns:
    DataFrame: The DataFrame with the specified columns scaled.
    """
    try:
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Check if all columns exist in the Daalign_featurestaFrame
        missing_columns = [col for col in columns_to_scale if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

        # Fit and transform the selected columns
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        
        return df
    except Exception as e:
        raise Exception(f"Error scaling features: {str(e)}")

def encode_categorical_features(df, base_features, categorical_vars):
    """
    Prepare the DataFrame for modeling by converting specified categorical variables
    to dummy variables and compiling a complete list of features.
    
    Args:
    df (DataFrame): The input data frame.
    base_features (list): List of base feature names to be included.
    categorical_vars (list): List of categorical variable names to be one-hot encoded.

    Returns:
    DataFrame, list: The modified DataFrame and the complete list of feature names.
    """
    try:
        # Check if specified categorical variables exist in the DataFrame
        missing_vars = [var for var in categorical_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"Missing columns in DataFrame that are specified for encoding: {missing_vars}")

        # Convert specified categorical variables to category data type
        for var in categorical_vars:
            df[var] = df[var].astype('category')

        # Apply one-hot encoding to the specified categorical variables
        df = pd.get_dummies(df, columns=categorical_vars)

        # Initialize the feature list with the base features
        features = base_features[:]

        # Dynamically add features for each categorical variable
        for var in categorical_vars:
            features.extend([col for col in df.columns if col.startswith(var + '_')])

        return df, features
    except Exception as e:
        raise Exception(f"Failed to encode categorical features: {str(e)}")
    

def prepare_data_for_modeling(df, base_features, categorical_vars):
    """
    Prepare the DataFrame for modeling by encoding categorical variables,
    extracting features, and separating the target variable.

    Args:
    df (DataFrame): The input DataFrame.
    base_features (list): List of base feature names to be included.
    categorical_vars (list): List of categorical variable names to be one-hot encoded.
    target_variable (str): Name of the column to use as the target variable.

    Returns:
    DataFrame, Series, list: Feature DataFrame (X), target variable Series (y), and list of feature names.
    """
    try:
        # Encode categorical features and update feature list
        encoded_df, features = encode_categorical_features(df, base_features, categorical_vars)
        # Extract the features for the model
        X = encoded_df[features]
        print(X.head(10))
        # Extract the target variable
        return X
    except Exception as e:
        raise Exception(f"Error preparing data for modeling: {str(e)}")

def align_features(base_df, target_df):
    # Add missing columns to the target_df with all zeros
    missing_cols = set(base_df.columns) - set(target_df.columns)
    for c in missing_cols:
        target_df[c] = 0
    # Ensure the order of columns in the target_df matches that of base_df
    target_df = target_df[base_df.columns]
    return target_df
    

def execute_inference_pipeline(df):
    try:
        base_df = pd.read_csv(os.getcwd() + '/airflow_data/dags/ml_project/resources/training_features_data1.csv')

        # Handle missing values
        df = handle_missing_values(df, strategy='drop', columns=None)
        print("Missing values handled.")
        print(df.head(10))

        # Correct data types
        df = correct_data_types(df)
        print("Data types corrected.")

        # Remove duplicates
        df = remove_duplicates(df)
        print("Duplicates removed.")

        # Perform sanity checks
        # df = sanity_checks(df)
        # print("Sanity checks completed.")

        # Apply cyclic encoding
        df= execute_feature_engineering_pipeline(df, chunk_size=50000)

        # Apply cyclic encoding
        df = apply_cyclic_encoding(df, 'pickup_day_of_week')
        print("Cyclic encoding applied.")

        df = typecasting_variables(df)
        fare_prediction_columns = ['VendorID', 'passenger_count', 'trip_distance', 'RatecodeID',
                                        'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'trip_duration',
                                        'congestion_surcharge', 'pickup_hour', 'day_sin', 'day_cos',
                                        'pickup_month'
                                    ]
        df = feature_selection(df, fare_prediction_columns)

        # Scale features
        columns_to_scale = ['trip_distance', 'pickup_hour', 'pickup_month', 'PULocationID', 'DOLocationID', 'trip_duration']
        df = scale_features(df, columns_to_scale)
        print("Features scaled.")

        # Encode categorical features
        base_features = ['trip_distance', 'pickup_hour', 'pickup_month', 'PULocationID', 'DOLocationID', 'trip_duration']
        categorical_vars = ['VendorID', 'RatecodeID', 'store_and_fwd_flag' ]
        # Prepare data for modeling
        X = prepare_data_for_modeling(df, base_features, categorical_vars)
        print("Categorical features encoded.")
        print("Data prepared for modeling.")

        processed_data = align_features(base_df, X)
        print("All processing steps completed successfully.")
        return processed_data, X
    except Exception as e:
        print(f"An error occurred: {str(e)}")

