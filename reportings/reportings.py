import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(os.getcwd() + '/airflow_data/dags/ml_project/resources/2020_Yellow_Taxi_Trip_Data.csv')
    return df

df = load_data()

# Streamlit app
st.title("NYC Taxi Trip Data Analysis")

# Display the dataset
st.header("Dataset")
st.dataframe(df.head())

# Basic statistics
st.header("Basic Statistics")
st.write(df.describe())

st.header("Removing Outliers and reruning Statistics")

def remove_outliers(df):
    outlier_columns = []  # List to store columns which contain outliers
    clean_df = df.copy()  # Create a copy of the DataFrame to avoid modifying the original data
    
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:  # Ensure the column has numeric data
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Before removing, check if there are any outliers
            if ((df[column] < lower_bound) | (df[column] > upper_bound)).any():
                outlier_columns.append(column)
            
            # Condition to identify rows with outliers
            condition = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            clean_df = clean_df[condition]

    return clean_df, outlier_columns

df, columns_with_outliers = remove_outliers(df)
st.write(df.describe())

df = df[:100000]
df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
df.dropna(inplace=True)
# Filter data by date range
st.header("Filter by Date Range")
start_date = st.date_input("Start Date", pd.to_datetime(df['tpep_pickup_datetime']).min())
end_date = st.date_input("End Date", pd.to_datetime(df['tpep_pickup_datetime']).max())
filtered_df = df[(pd.to_datetime(df['tpep_pickup_datetime']) >= pd.to_datetime(start_date)) & 
                 (pd.to_datetime(df['tpep_pickup_datetime']) <= pd.to_datetime(end_date))]
st.write(f"Filtered data from {start_date} to {end_date}")
st.dataframe(filtered_df.head())

# Visualize distributions of numerical features
st.header("Distributions of Numerical Features")
numerical_features = ['passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
                      'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']

for feature in numerical_features:
    st.subheader(f"{feature.capitalize()} Distribution")
    fig, ax = plt.subplots()
    filtered_df[feature].dropna().hist(bins=30, ax=ax)
    ax.set_title(f"{feature.capitalize()} Distribution")
    st.pyplot(fig)

# Correlation heatmap
st.header("Correlation Heatmap")
corr = filtered_df[numerical_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# Pairplot for numerical features
st.header("Pairplot of Numerical Features")
if st.checkbox("Show Pairplot"):
    pd.plotting.scatter_matrix(filtered_df[numerical_features].dropna(), figsize=(15, 15))
    st.pyplot()

# Summarize total trips, total passengers, and total revenue
st.header("Summary")
total_trips = filtered_df.shape[0]
total_passengers = filtered_df['passenger_count'].sum()
total_revenue = filtered_df['total_amount'].sum()
st.write(f"Total Trips: {total_trips}")
st.write(f"Total Passengers: {total_passengers}")
st.write(f"Total Revenue: ${total_revenue:,.2f}")

# Show congestion surcharge
st.header("Congestion Surcharge Analysis")
congestion_surcharge_total = filtered_df['congestion_surcharge'].sum()
st.write(f"Total Congestion Surcharge: ${congestion_surcharge_total:,.2f}")

if st.checkbox("Show detailed congestion surcharge distribution"):
    fig, ax = plt.subplots()
    filtered_df['congestion_surcharge'].dropna().hist(bins=30, ax=ax)
    ax.set_title("Congestion Surcharge Distribution")
    st.pyplot(fig)

# Additional report: Payment type distribution
st.header("Payment Type Distribution")
payment_counts = filtered_df['payment_type'].value_counts()
fig, ax = plt.subplots()
payment_counts.plot(kind='bar', ax=ax)
ax.set_title("Payment Type Distribution")
ax.set_xlabel("Payment Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# Peak hours analysis
st.header("Peak Hours Analysis")
filtered_df['pickup_hour'] = pd.to_datetime(filtered_df['tpep_pickup_datetime']).dt.hour
peak_hours = filtered_df['pickup_hour'].value_counts().sort_index()
fig, ax = plt.subplots()
peak_hours.plot(kind='bar', ax=ax)
ax.set_title("Peak Hours Distribution")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Trips")
st.pyplot(fig)

# Trip duration analysis
st.header("Trip Duration Analysis")
filtered_df['tpep_pickup_datetime'] = pd.to_datetime(filtered_df['tpep_pickup_datetime'])
filtered_df['tpep_dropoff_datetime'] = pd.to_datetime(filtered_df['tpep_dropoff_datetime'])
filtered_df['trip_duration'] = (filtered_df['tpep_dropoff_datetime'] - filtered_df['tpep_pickup_datetime']).dt.total_seconds() / 60
fig, ax = plt.subplots()
filtered_df['trip_duration'].dropna().hist(bins=30, ax=ax)
ax.set_title("Trip Duration Distribution")
ax.set_xlabel("Duration (minutes)")
st.pyplot(fig)

# Average fare per mile
st.header("Average Fare per Mile")
filtered_df['fare_per_mile'] = filtered_df.apply(lambda x: x['fare_amount'] / x['trip_distance'] if x['trip_distance'] != 0 else 0, axis=1)
# filtered_df['fare_per_mile'] = filtered_df['fare_amount'] / filtered_df['trip_distance']
fig, ax = plt.subplots()
filtered_df['fare_per_mile'].dropna().hist(bins=30, ax=ax)
ax.set_title("Fare per Mile Distribution")
ax.set_xlabel("Fare per Mile")
st.pyplot(fig)

# Trip distribution by pickup and dropoff locations
st.header("Trip Distribution by Pickup and Dropoff Locations")
pickup_counts = filtered_df['PULocationID'].value_counts().head(10)
dropoff_counts = filtered_df['DOLocationID'].value_counts().head(10)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
pickup_counts.plot(kind='bar', ax=ax[0])
ax[0].set_title("Top 10 Pickup Locations")
ax[0].set_xlabel("Location ID")
ax[0].set_ylabel("Number of Trips")

dropoff_counts.plot(kind='bar', ax=ax[1])
ax[1].set_title("Top 10 Dropoff Locations")
ax[1].set_xlabel("Location ID")
ax[1].set_ylabel("Number of Trips")

st.pyplot(fig)

st.write("This app provides comprehensive insights into the NYC Taxi Trip dataset. Use the filters and visualizations to explore the data further.")



# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load data
# @st.cache
# def load_data():
#     # base_df = pd.read_csv(os.getcwd() + '/airflow_data/dags/ml_project/resources/training_features_data.csv')
#     # url = "path_to_your_dataset.csv"  # Replace with your dataset path
#     df = pd.read_csv(os.getcwd() + '/airflow_data/dags/ml_project/resources/2020_Yellow_Taxi_Trip_Data.csv')
#     return df

# df = load_data()

# # Streamlit app
# st.title("NYC Taxi Trip Data Analysis")

# # Display the dataset
# st.header("Dataset")
# st.dataframe(df.head())

# # Basic statistics
# st.header("Basic Statistics")
# st.write(df.describe())

# # Filter data by date range
# st.header("Filter by Date Range")
# start_date = st.date_input("Start Date", pd.to_datetime(df['tpep_pickup_datetime']).min())
# end_date = st.date_input("End Date", pd.to_datetime(df['tpep_pickup_datetime']).max())
# filtered_df = df[(pd.to_datetime(df['tpep_pickup_datetime']) >= pd.to_datetime(start_date)) & 
#                  (pd.to_datetime(df['tpep_pickup_datetime']) <= pd.to_datetime(end_date))]
# st.write(f"Filtered data from {start_date} to {end_date}")
# st.dataframe(filtered_df.head())

# # Visualize distributions of numerical features
# st.header("Distributions of Numerical Features")
# numerical_features = ['passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 
#                       'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge']

# for feature in numerical_features:
#     st.subheader(f"{feature.capitalize()} Distribution")
#     fig, ax = plt.subplots()
#     sns.histplot(filtered_df[feature].dropna(), bins=30, kde=True, ax=ax)
#     ax.set_title(f"{feature.capitalize()} Distribution")
#     st.pyplot(fig)

# # Correlation heatmap
# st.header("Correlation Heatmap")
# corr = filtered_df[numerical_features].corr()
# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
# ax.set_title("Correlation Heatmap")
# st.pyplot(fig)

# # Pairplot for numerical features
# st.header("Pairplot of Numerical Features")
# if st.checkbox("Show Pairplot"):
#     sns.pairplot(filtered_df[numerical_features].dropna())
#     st.pyplot()

# # Summarize total trips, total passengers, and total revenue
# st.header("Summary")
# total_trips = filtered_df.shape[0]
# total_passengers = filtered_df['passenger_count'].sum()
# total_revenue = filtered_df['total_amount'].sum()
# st.write(f"Total Trips: {total_trips}")
# st.write(f"Total Passengers: {total_passengers}")
# st.write(f"Total Revenue: ${total_revenue:,.2f}")

# # Show congestion surcharge
# st.header("Congestion Surcharge Analysis")
# congestion_surcharge_total = filtered_df['congestion_surcharge'].sum()
# st.write(f"Total Congestion Surcharge: ${congestion_surcharge_total:,.2f}")

# if st.checkbox("Show detailed congestion surcharge distribution"):
#     fig, ax = plt.subplots()
#     sns.histplot(filtered_df['congestion_surcharge'].dropna(), bins=30, kde=True, ax=ax)
#     ax.set_title("Congestion Surcharge Distribution")
#     st.pyplot(fig)

# # Additional report: Payment type distribution
# st.header("Payment Type Distribution")
# payment_counts = filtered_df['payment_type'].value_counts()
# st.bar_chart(payment_counts)

# # Peak hours analysis
# st.header("Peak Hours Analysis")
# filtered_df['pickup_hour'] = pd.to_datetime(filtered_df['tpep_pickup_datetime']).dt.hour
# peak_hours = filtered_df['pickup_hour'].value_counts().sort_index()
# fig, ax = plt.subplots()
# peak_hours.plot(kind='bar', ax=ax)
# ax.set_title("Peak Hours Distribution")
# ax.set_xlabel("Hour of Day")
# ax.set_ylabel("Number of Trips")
# st.pyplot(fig)

# # Trip duration analysis
# st.header("Trip Duration Analysis")
# filtered_df['tpep_pickup_datetime'] = pd.to_datetime(filtered_df['tpep_pickup_datetime'])
# filtered_df['tpep_dropoff_datetime'] = pd.to_datetime(filtered_df['tpep_dropoff_datetime'])
# filtered_df['trip_duration'] = (filtered_df['tpep_dropoff_datetime'] - filtered_df['tpep_pickup_datetime']).dt.total_seconds() / 60
# fig, ax = plt.subplots()
# sns.histplot(filtered_df['trip_duration'].dropna(), bins=30, kde=True, ax=ax)
# ax.set_title("Trip Duration Distribution")
# ax.set_xlabel("Duration (minutes)")
# st.pyplot(fig)

# # Average fare per mile
# st.header("Average Fare per Mile")
# filtered_df['fare_per_mile'] = filtered_df['fare_amount'] / filtered_df['trip_distance']
# fig, ax = plt.subplots()
# sns.histplot(filtered_df['fare_per_mile'].dropna(), bins=30, kde=True, ax=ax)
# ax.set_title("Fare per Mile Distribution")
# ax.set_xlabel("Fare per Mile")
# st.pyplot(fig)

# # Trip distribution by pickup and dropoff locations
# st.header("Trip Distribution by Pickup and Dropoff Locations")
# pickup_counts = filtered_df['PULocationID'].value_counts().head(10)
# dropoff_counts = filtered_df['DOLocationID'].value_counts().head(10)

# fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# pickup_counts.plot(kind='bar', ax=ax[0])
# ax[0].set_title("Top 10 Pickup Locations")
# ax[0].set_xlabel("Location ID")
# ax[0].set_ylabel("Number of Trips")

# dropoff_counts.plot(kind='bar', ax=ax[1])
# ax[1].set_title("Top 10 Dropoff Locations")
# ax[1].set_xlabel("Location ID")
# ax[1].set_ylabel("Number of Trips")

# st.pyplot(fig)

# st.write("This app provides comprehensive insights into the NYC Taxi Trip dataset. Use the filters and visualizations to explore the data further.")
