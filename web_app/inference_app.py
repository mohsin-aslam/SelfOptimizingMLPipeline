from flask import Flask, request, render_template, session, redirect, url_for
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import traceback
from inference_pipeline import execute_inference_pipeline
from predictions import predict_data
from evaluations import insert_evaluations
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


app = Flask(__name__)

# Configuration for MLflow
MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://minio:9000"

def find_experiment_by_name(exp_name):
    try:
        experiments = client.search_experiments()  # List all experiments
        for experiment in experiments:
            if experiment.name == exp_name:
                return experiment.experiment_id
        raise ValueError(f"No experiment found with name {exp_name}")
    except Exception as e:
        print(f"Failed to find experiment by name: {e}")
        return None

def find_best_model(experiment_id):
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["metrics.rmse ASC"],
            max_results=1
        )
        if not runs:
            raise ValueError("No runs found for this experiment.")
        best_run = runs[0]
        print(best_run.info.run_id, best_run.data.tags.get('model_type'))
        return {'model_run_id': best_run.info.run_id, 
                'model_type': best_run.data.tags.get('model_type'), 
                'model_rmse':best_run.data.metrics['rmse'], 
                'model_r2': best_run.data.metrics['r2']}
    except Exception as e:
        print(f"Failed to find best model: {e}")
        traceback.print_exc()
        return None, None

# def load_model(run_id, model_type):
#     try:
#         model_uri = f"runs:/{run_id}/{model_type}"
#         print(model_uri)
#         model = mlflow.pyfunc.load_model(model_uri)
#         return model
#     except Exception as e:
#         print(f"Failed to load model: {e}")
#         return None

def load_model(run_id, model_type):
    try:
        model_uri = f"runs:/{run_id}/{model_type}"
        print(f"Attempting to load model from URI: {model_uri}")
        # Check if the model can be loaded without actually loading it
        # model_path = mlflow.artifacts.download_artifacts(model_uri)
        # print(f"Model path resolved to: {model_path}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Failed to load model due to: {e}")
        return None

def highlight_last_column(s):
    # Apply 'last-column' class to the last column
    return [''] * (len(s) - 1) + ['last-column']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'csvfile' not in request.files:
        return render_template('index.html', error="No file part")
    
    file = request.files['csvfile']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if file and file.filename.endswith('.csv'):
        # Read the file directly from the stream
        try:
            df = pd.read_csv(file)
            processed_data, process_data = execute_inference_pipeline(df)
            print(process_data.columns)
            print(processed_data.columns)
            experiment_id = find_experiment_by_name("Fare_Prediction")
            print(experiment_id)
            best_model = find_best_model(experiment_id)
            best_run_id = best_model.get('model_run_id')
            model_type = best_model.get('model_type')
            print(best_run_id, model_type)
            # best_model = load_model(best_run_id, model_type)
            # prediction = best_model.predict(processed_data)
            print('predict_data')
            try:
                processed_data = processed_data.drop(columns=['Unnamed: 0'])
                prediction =predict_data(processed_data, best_run_id, model_type)
            except Exception as e:
                print(str(e))
            df['predicted_fare_amount'] = pd.DataFrame(prediction, columns=['predicted_fare_amount'])
            df['predicted_fare_amount'] = df['predicted_fare_amount'].abs()
            print(df.head(10))
            fare_prediction_columns = ['VendorID','tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 
                                        'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'predicted_fare_amount', 'fare_amount' ]
            predicted_df = df[fare_prediction_columns]
            predicted_df.to_csv('predictions.csv', index=False)
            fare_prediction_columns = ['VendorID','tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 
                                        'store_and_fwd_flag', 'PULocationID', 'DOLocationID', 'predicted_fare_amount']
            # df_styled = df[fare_prediction_columns].style.apply(highlight_last_column, axis=1, subset=pd.IndexSlice[:, [-1]])
            table_html = df[fare_prediction_columns].to_html(classes='table table-bordered table-hover', index=False)
            # table_html = df_styled.to_html(classes='table table-bordered table-hover', index=False)
            # table_html = df_styled.render()
            return render_template('index.html', table=table_html)
        except Exception as e:
            print(str(e))
            return render_template('index.html', error=f"Error reading the file: {str(e)}")
    else:
        return render_template('index.html', error="Unsupported file type")


@app.route('/evaluate')
def evaluate():
    try:
        df = pd.read_csv('predictions.csv')
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        max_inference_date = df['tpep_pickup_datetime'].max().date()
        df = df[['fare_amount','predicted_fare_amount']]
        # df.fillna(df.mean(), inplace=True)
        df.dropna(inplace=True)
        y_true = df['fare_amount']
        y_pred = df['predicted_fare_amount']

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(rmse,r2)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(rmse,r2)

        experiment_id = find_experiment_by_name("Fare_Prediction")
        print(experiment_id)
        best_model = find_best_model(experiment_id)
        best_run_id = best_model.get('model_run_id')
        model_type = best_model.get('model_type')
        print(best_run_id, model_type)

        evaluation_data = {'experiment_id': experiment_id,
                'model_run_id': best_model.get('model_run_id'),
                'model_type': best_model.get('model_type'),
                'model_rmse': best_model.get('model_rmse'),
                'model_r2': best_model.get('model_r2'),
                'evaluation_rmse': rmse,
                'evaluation_r2': r2,
                'max_inference_date':max_inference_date, 
                'execution_date': datetime.datetime.now()
        }
        try:
            insert_evaluations(evaluation_data)
        except Exception as e:
            print('Error in inserting evaluation Data', e)
        # Create a dataframe for the results
        results = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
        results['Residuals'] = results['Actual'] - results['Predicted']

        # Generate plots
        img = io.BytesIO()

        # Residual Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results['Predicted'], results['Residuals'])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.savefig(img, format='png')
        img.seek(0)
        residual_plot_url = base64.b64encode(img.getvalue()).decode()

        # Predicted vs Actual Plot
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.scatter(results['Actual'], results['Predicted'])
        plt.plot([results['Actual'].min(), results['Actual'].max()], [results['Actual'].min(), results['Actual'].max()], 'r--')
        plt.title('Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.savefig(img, format='png')
        img.seek(0)
        pred_vs_actual_plot_url = base64.b64encode(img.getvalue()).decode()

        # Distribution of Residuals
        img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.hist(results['Residuals'], bins=20, edgecolor='k')
        plt.title('Distribution of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.savefig(img, format='png')
        img.seek(0)
        residuals_dist_plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('evaluation.html', mae=mae, mse=mse, rmse=rmse, r2=r2,
                               residual_plot_url=residual_plot_url,
                               pred_vs_actual_plot_url=pred_vs_actual_plot_url,
                               residuals_dist_plot_url=residuals_dist_plot_url)
    except Exception as e:
        print(str(e))
        return render_template('index.html', error=f"Error evaluating the predictions: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5001)