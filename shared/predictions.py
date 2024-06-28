import boto3
from botocore.client import Config
from io import BytesIO
import joblib
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError


def create_s3_client():
    s3_client = boto3.client('s3',
                             endpoint_url='http://172.27.0.8:9000',
                             aws_access_key_id='minioadmin',
                             aws_secret_access_key='minioadmin',
                             config=Config(signature_version='s3v4'),
                             region_name='us-east-1')
    return s3_client


def list_model_files(s3_client, bucket_name, best_run_id, model_type):
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    model_files = [obj['Key'] for obj in response.get('Contents', []) if 'pkl' in obj['Key'] and  best_run_id in obj['Key'] and model_type in obj['Key']]
    return model_files


# def download_and_load_model(s3_client, bucket_name, object_key):
#     model_buffer = BytesIO()
#     print('download_fileobj')
#     s3_client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=model_buffer)
#     print('download done')
#     model_buffer.seek(0)  # Move to the start of the file before loading
#     model = joblib.load(model_buffer)
#     print('model loaded')
    
#     return model

def download_and_load_model(s3_client, bucket_name, object_key):
    try:
        model_buffer = BytesIO()
        print('Attempting to download the model...')

        # Try to download the model file from the specified S3 bucket
        s3_client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=model_buffer)
        print('Model download completed successfully.')

        # Reset the buffer pointer to the beginning of the stream before loading it
        model_buffer.seek(0)

        # Load the model from the buffer
        model = joblib.load(model_buffer)
        print('Model loaded successfully.')

        return model
    except NoCredentialsError:
        print("Error: No valid credentials provided for AWS S3. Check your credentials.")
    except PartialCredentialsError:
        print("Error: Incomplete credentials provided. Please provide full access credentials.")
    except ClientError as e:
        print(f"ClientError: {e.response['Error']['Message']}")
        if e.response['Error']['Code'] == '404':
            print("Error: The object does not exist at the provided key in the bucket.")
    except joblib.externals.loky.process_executor.TerminatedWorkerError:
        print("Error: The worker processing the joblib file was unexpectedly terminated.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    return None

def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions



def predict_data(df, best_run_id, model_type):
    # Define your bucket name and specific object key
    bucket_name = 'mlflow'
    s3_client = create_s3_client()
    # List all .pkl files in the bucket
    model_files = list_model_files(s3_client, bucket_name, best_run_id, model_type)
    print("Available model files:", model_files)

    # Assume you choose a specific model file to load
    if model_files:
        print('model_files found')
        selected_model_file = model_files[0]  # Select the first model file
        print(selected_model_file)
        model = download_and_load_model(s3_client, bucket_name, selected_model_file)

        # Assume X_test_subset is your test dataset loaded elsewhere
        predictions = make_predictions(model, df)
        print(type(predictions))
        print("Predictions:", predictions)
        return predictions
    else:
        print("No model files found.")
        return None