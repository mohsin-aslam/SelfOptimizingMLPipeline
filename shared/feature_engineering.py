import pandas as pd
import numpy as np
from ml_project.data_preprocessing import save_data, load_data
import os
from sklearn.preprocessing import StandardScaler


def extract_time_features(df):
    try:
        print(df['tpep_pickup_datetime'].dtype)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        print(df['tpep_pickup_datetime'].dtype)
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['tpep_pickup_datetime'].dt.day_name()
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60.0
        print(df.head(10))
        print(len(df))
        return df
    except Exception as e:
        raise Exception(f"Error in extract_time_features: {str(e)}")

def calculate_distance_rate(df):
    try:
        df['fare_per_mile'] = df.apply(lambda x: x['fare_amount'] / x['trip_distance'] if x['trip_distance'] != 0 else 0, axis=1)
        print(len(df))
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
    
    try:
        df = calculate_distance_rate(df)
    except Exception as e:
        raise Exception(f"Error in calculating distance rate: {str(e)}")
    
    # Example of how you might include encoding if uncommented
    # try:
    #     df = encode_categorical_features(df, ['pickup_day_of_week', 'payment_type'])
    # except Exception as e:
    #     raise Exception(f"Error in encoding categorical features: {str(e)}")
    print(len(df))
    return df


def execute_feature_engineering_pipeline(chunk_size=50000):
    """
    Process the DataFrame in chunks and apply feature engineering.
    :param df: DataFrame to be processed.
    :param chunk_size: Number of rows per chunk.
    :returns: DataFrame with all features engineered concatenated from all chunks.
    """
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/data_processing.csv')
        print(len(df))
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
        print(len(engineered_df))
        save_data(engineered_df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
    except Exception as e:
        raise Exception(f"Error in processing chunks: {str(e)}")

def apply_cyclic_encoding(column_name):
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        day_to_num = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        df['day_number'] = df[column_name].map(day_to_num)
        if df['day_number'].isna().any():
            missing_values = df[df['day_number'].isna()][column_name].unique()
            raise ValueError(f"Mapping not found for: {missing_values}")
        df['day_sin'] = np.sin(2 * np.pi * df['day_number'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_number'] / 7)
        df.drop('day_number', axis=1, inplace=True)
        print(len(df))
        save_data(df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        
        return True
    except Exception as e:
        raise Exception(f"Error in apply_cyclic_encoding: {str(e)}")

def typecasting_variables():
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        df['VendorID'] = df['VendorID'].astype(int).astype(str)
        save_data(df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
    except Exception as e:
        raise Exception(f"Error in typecasting_variables: {str(e)}")

def feature_selection(fare_prediction_columns):
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        fare_prediction_df = df[fare_prediction_columns]
        save_data(fare_prediction_df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
    except Exception as e:
        raise Exception(f"Error in feature_selection: {str(e)}")
    
def scale_features(columns_to_scale):
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
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        scaler = StandardScaler()

        # Check if all columns exist in the DataFrame
        missing_columns = [col for col in columns_to_scale if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

        # Fit and transform the selected columns
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        save_data(df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        return True
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
    

def prepare_data_for_modeling(base_features, categorical_vars, target_variable):
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
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        print(len(df))
        # Encode categorical features and update feature list
        encoded_df, features = encode_categorical_features(df, base_features, categorical_vars)
        save_data(encoded_df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        # Extract the features for the model
        X = encoded_df[features]
        
        # Extract the target variable
        y = encoded_df[target_variable]
        print(len(X))
        save_data(X, os.path.dirname(os.path.abspath(__file__)) + '/resources/training_features.csv')
        save_data(y, os.path.dirname(os.path.abspath(__file__)) + '/resources/target_variable.csv')
        return True
    except Exception as e:
        raise Exception(f"Error preparing data for modeling: {str(e)}")

