import pandas as pd
from ml_project.data_preprocessing import save_data, load_data
import os


def remove_outliers():
    try:
        df = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        outlier_columns = []  # List to store columns which contain outliers
        clean_df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data
        print(len(df))
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:  # Ensure the column has numeric data
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Check for outliers
                if ((df[column] < lower_bound) | (df[column] > upper_bound)).any():
                    outlier_columns.append(column)

                # Condition to identify rows without outliers
                condition = (df[column] >= lower_bound) & (df[column] <= upper_bound)
                clean_df = clean_df[condition]

        print(len(clean_df))
        save_data(clean_df, os.path.dirname(os.path.abspath(__file__)) + '/resources/intermediate_data.csv')
        return True
    except Exception as e:
        raise Exception(f"Failed to remove outliers: {str(e)}")
