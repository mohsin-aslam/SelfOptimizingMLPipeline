# import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from ml_project.data_preprocessing import save_data, load_data
import mlflow
from mlflow import sklearn as mlflow_sklearn


def split_data(X, y, test_size=0.3, random_state=2242):
    """
    Split the data into training and testing sets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Failed to split data: {e}")

def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    """
    try:
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using RMSE and R^2 score.
    """
    try:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Root Mean Squared Error: {rmse}")
        print(f"R^2 Score: {r2}")
        return rmse, r2
    except Exception as e:
        raise ValueError(f"Failed to evaluate model: {e}")
    
def execute_ml_pipeline():
    X = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/training_features.csv')
    y = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/target_variable.csv')
    model = LinearRegression()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(model, X_train, y_train)
    rmse, r2 = evaluate_model(model, X_test, y_test)
    # Print results
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R^2 Score: {r2}")
    return rmse, r2


def train_and_predict_with_mlflow(model, X_train, y_train, X_test, y_test, model_name, model_tags=None, register_model=False):
    """
    Train the model, predict, evaluate, and log the experiment to MLflow with optional tagging and model registration.

    Args:
        model (estimator): The model to train.
        X_train, y_train (DataFrame, Series): Training data.
        X_test, y_test (DataFrame, Series): Testing data.
        model_name (str): Name of the model for tracking.
        model_tags (dict, optional): Tags to add to the MLflow run.
        register_model (bool, optional): Whether to register the model in MLflow Model Registry.
    """
    with mlflow.start_run():
        # Fit the model
        model.fit(X_train, y_train)
        
        # Predict on the testing set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters, metrics, and model
        params = model.get_params()
        mlflow.log_params(params)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        
        # Log the model
        mlflow_sklearn.log_model(model, model_name)
        
        # Add tags if provided
        if model_tags:
            mlflow.set_tags(model_tags)
        
        print(f'Logged {model_name}: RMSE={rmse}, R2={r2}')
        
        # Register the model if flagged to do so
        if register_model:
            mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
        
        return y_pred


def execute_mlflow_pipeline(tracking_uri, experiment_name):

    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://172.27.0.8:9000"
    mlflow.set_tracking_uri(tracking_uri)  
    mlflow.set_experiment(experiment_name)
    X = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/training_features.csv')
    y = load_data(os.path.dirname(os.path.abspath(__file__)) + '/resources/target_variable.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    tree_model = DecisionTreeRegressor(max_depth=50,min_samples_split=5)
    model_name = "DecisionTreeRegressorModel"
    model_tags = {
        "model_type": "DecisionTreeRegressorModel","max_depth": "50","min_samples_split": "5"
        
    }

    predictions = train_and_predict_with_mlflow(tree_model, X_train, y_train, X_test, y_test, model_name, model_tags=model_tags, register_model=True)
    print(predictions)
# def perform_grid_search_and_log_all_models(model, param_grid, X_train, y_train, X_test, y_test):
#     """
#     Performs a grid search to find the best hyperparameters for the given model,
#     evaluates the best model on the test set, and logs all models and results to MLflow.

#     Args:
#     model (estimator): The model for which the hyperparameters are tuned.
#     param_grid (dict): The hyperparameter grid to explore.
#     X_train, y_train (arrays): Training data.
#     X_test, y_test (arrays): Testing data.
#     """
#     # Initialize MLflow run
#     # Set the tracking URI to a local folder or a remote server
#     mlflow.set_tracking_uri('http://mlflow_server:5000')  # Replace with your path or URI
#     # Set the experiment name
#     mlflow.set_experiment('Fare_Prediction')
#     with mlflow.start_run():
#         try:
#             # Set up grid search
#             grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
#             grid_search.fit(X_train, y_train)

#             # Log the best model
#             best_model = grid_search.best_estimator_
#             y_pred = best_model.predict(X_test)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             r2 = r2_score(y_test, y_pred)

#             mlflow.log_params(grid_search.best_params_)
#             mlflow.log_metric('rmse', rmse)
#             mlflow.log_metric('r2', r2)
#             mlflow.sklearn.log_model(best_model, "best_model")

#             # Log all other models
#             for i in range(len(grid_search.cv_results_['params'])):
#                 params = grid_search.cv_results_['params'][i]
#                 mean_test_score = -grid_search.cv_results_['mean_test_score'][i]  # Convert back from negative MSE to positive
#                 std_test_score = grid_search.cv_results_['std_test_score'][i]

#                 with mlflow.start_run(nested=True):  # Create a nested run for each model
#                     mlflow.log_params(params)
#                     mlflow.log_metric('mean_test_score', mean_test_score)
#                     mlflow.log_metric('std_test_score', std_test_score)

#         except ValueError as ve:
#             logging.error(f"Value error occurred: {ve}")
#             mlflow.log_param('error', str(ve))
#         except Exception as e:
#             logging.error(f"An unexpected error occurred: {e}")
#             mlflow.log_param('error', str(e))
#         finally:
#             mlflow.end_run()

# def train_and_tune_with_mlflow(model, param_grid, X_train, y_train, X_test, y_test, model_name):
#     """
#     Train the model with hyperparameter tuning, predict, and log the experiment to MLflow.

#     Args:
#     model (estimator): The base model to train.
#     param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
#     X_train, y_train: Training data.
#     X_test, y_test: Testing data.
#     model_name (str): Name of the model for tracking.
#     """
#     try:
#         # Set the tracking URI to a local folder or a remote server
#         mlflow.set_tracking_uri('http://mlflow_server:5000')  # Replace with your path or URI
#         # Set the experiment name
#         mlflow.set_experiment('Fare_Prediction')
#         with mlflow.start_run():
#             grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
#             grid_search.fit(X_train, y_train)
            
#             # Best model
#             best_model = grid_search.best_estimator_
            
#             # Predict on the testing set
#             y_pred = best_model.predict(X_test)
            
#             # Calculate metrics
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             r2 = r2_score(y_test, y_pred)
            
#             # Log parameters and metrics
#             mlflow.log_params(grid_search.best_params_)
#             mlflow.log_metric('rmse', rmse)
#             mlflow.log_metric('r2', r2)
            
#             # Log the model
#             mlflow.sklearn.log_model(best_model, model_name)

#             print(f'Logged {model_name}: Best Params={grid_search.best_params_}, RMSE={rmse}, R2={r2}')
            
#             return best_model, grid_search.best_params_
#     except Exception as e:
#         mlflow.end_run()
#         raise e


# def train_and_predict_with_mlflow(model, X_train, y_train, X_test, y_test, model_name):
#     """
#     Train the model, predict, and log the experiment to MLflow, with added exception handling.

#     Args:
#     model (estimator): The model to train.
#     X_train, y_train: Training data.
#     X_test, y_test: Testing data.
#     model_name (str): Name of the model for tracking.
#     """
#     mlflow.set_tracking_uri('http://mlflow_server:5000')  # Replace with your path or URI
#     # Set the experiment name
#     mlflow.set_experiment('Fare_Prediction')
#     with mlflow.start_run():
#         try:
#             # Fit the model
#             model.fit(X_train, y_train)
            
#             # Predict on the testing set
#             y_pred = model.predict(X_test)
            
#             # Calculate metrics
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             r2 = r2_score(y_test, y_pred)
            
#             # Log parameters (if any)
#             params = model.get_params()
#             mlflow.log_params(params)
            
#             # Log metrics
#             mlflow.log_metric('rmse', rmse)
#             mlflow.log_metric('r2', r2)
            
#             # Log the model
#             mlflow.sklearn.log_model(model, model_name)

#             print(f'Logged {model_name}: RMSE={rmse}, R2={r2}')

#             return y_pred

#         except Exception as e:
#             # Log the exception details in MLflow for troubleshooting
#             mlflow.log_param('exception', str(e))
#             print(f"Error occurred during MLflow logging for {model_name}: {str(e)}")
#             raise e  # Optionally re-raise the exception to ensure that it's caught by higher-level error management
#         finally:
#             mlflow.end_run()  # Ensure that MLflow run is closed properly in case of an exception


# def tune_linear_regression(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(LinearRegression(), param_grid, X_train, y_train, X_test, y_test, 'Linear_Regression')

# def tune_ridge_regression(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(Ridge(), param_grid, X_train, y_train, X_test, y_test, 'Ridge_Regression')

# def tune_gradient_boosting(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(GradientBoostingRegressor(random_state=6542), param_grid, X_train, y_train, X_test, y_test, 'GradientBoostingRegressor')

# def tune_decision_tree(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(DecisionTreeRegressor(random_state=4312), param_grid, X_train, y_train, X_test, y_test, 'DecisionTreeRegressor')

# def tune_mlp(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(MLPRegressor(random_state=4122), param_grid, X_train, y_train, X_test, y_test, 'MLPRegressor')

# def tune_random_forest(X_train, y_train, X_test, y_test, param_grid):
#     return train_and_tune_with_mlflow(RandomForestRegressor(random_state=456), param_grid, X_train, y_train, X_test, y_test, 'RandomForestRegressor')


# def setup_and_train_ridge(X, y, test_size=0.2, random_state=42, alpha=1.0):
#     """
#     Splits the data, trains a Ridge regression model, and logs the process to MLflow.

#     Args:
#     X (array-like): Feature matrix.
#     y (array-like): Target vector.
#     test_size (float): Fraction of the dataset to be used as test set.
#     random_state (int): Seed used by the random number generator.
#     alpha (float): Regularization strength of the Ridge regressor.

#     Returns:
#     None
#     """
#     try:
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
#         # Initialize and train the Ridge model
#         ridge_model = Ridge(alpha=alpha)
#         train_and_predict_with_mlflow(ridge_model, X_train, y_train, X_test, y_test, 'Ridge_Regression')
    
#     except ValueError as e:
#         print(f"ValueError during model training: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred during the training process: {e}")


# def setup_and_train_gradient_boosting(X, y, test_size=0.2, random_state=42):
#     """
#     Splits the data, trains a Gradient Boosting Regressor, and logs the process to MLflow.
#     """
#     try:
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
#         # Initialize and train the model
#         gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#         train_and_predict_with_mlflow(gbm_model, X_train, y_train, X_test, y_test, 'GradientBoostingRegressor')
    
#     except Exception as e:
#         print(f"An error occurred with Gradient Boosting Regressor: {e}")


# def setup_and_train_mlp(X, y, test_size=0.2, random_state=42):
#     """
#     Splits the data, trains an MLP Regressor, and logs the process to MLflow.
#     """
#     try:
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
#         # Initialize and train the model
#         nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
#         train_and_predict_with_mlflow(nn_model, X_train, y_train, X_test, y_test, 'MLPRegressor')
    
#     except Exception as e:
#         print(f"An error occurred with MLP Regressor: {e}")


# def setup_and_train_decision_tree(X, y, test_size=0.2, random_state=42):
#     """
#     Splits the data, trains a Decision Tree Regressor, and logs the process to MLflow.
#     """
#     try:
#         # Split the data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
#         # Initialize and train the model
#         tree_model = DecisionTreeRegressor(max_depth=5)
#         train_and_predict_with_mlflow(tree_model, X_train, y_train, X_test, y_test, 'DecisionTreeRegressor')
    
#     except Exception as e:
#         print(f"An error occurred with Decision Tree Regressor: {e}")