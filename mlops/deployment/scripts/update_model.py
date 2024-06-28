from mlflow.tracking import MlflowClient
import mlflow.tensorflow
import tensorflow as tf
import argparse
import os


def save_model_for_tf_serving(model_name, tag_key, tag_value, save_directory):
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        if version.tags.get(tag_key) == tag_value:
            # Load the TensorFlow model from MLflow
            model_uri = f"models:/{model_name}/{version.version}"
            model = mlflow.tensorflow.load_model(model_uri)

            # Include model_name in the directory path where the model for this version will be saved
            model_version_save_path = os.path.join(save_directory, model_name, str(version.version))
            os.makedirs(model_version_save_path, exist_ok=True)
            
            # Save the model in SavedModel format
            tf.saved_model.save(model, model_version_save_path)
            print(f"Model version {version.version} saved for TensorFlow Serving at {model_version_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Save models for TensorFlow Serving")
    parser.add_argument('model_name', type=str, help='The name of the model to save')
    parser.add_argument('--tag_key', type=str, default="stage", help='The tag key to filter models')
    parser.add_argument('--tag_value', type=str, default="Production", help='The tag value to filter models')
    parser.add_argument('--save_directory', type=str, default='', help='The directory where models will be saved')

    args = parser.parse_args()

    save_model_for_tf_serving(args.model_name, args.tag_key, args.tag_value, args.save_directory)


if __name__ == "__main__":
    main()
