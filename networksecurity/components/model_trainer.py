import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get MLflow tracking details from environment variables
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Set MLflow environment variables
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_tracking_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_tracking_password

# Alternatively, you can use dagshub.init to set up MLflow
# dagshub.init(repo_owner='bilalafzalaf', repo_name='networksecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            
            # Log MLflow configuration
            logging.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
            logging.info(f"MLflow Username: {mlflow_tracking_username}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric, run_name=None):
        # Set MLflow registry URI
        mlflow.set_registry_uri(mlflow_tracking_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Get metrics
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score

            # Log model parameters
            for param_name, param_value in best_model.get_params().items():
                mlflow.log_param(param_name, param_value)
            
            # Log model type
            mlflow.log_param("model_type", type(best_model).__name__)
                
            # Log metrics
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(
                    best_model, 
                    "model", 
                    registered_model_name=type(best_model).__name__
                )
            else:
                mlflow.sklearn.log_model(best_model, "model")
        
    def train_model(self, X_train, y_train, x_test, y_test):
        try:
            logging.info("Starting model training process")
            
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss': ['log_loss', 'exponential'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    'learning_rate': [.1, .01, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }
            
            logging.info("Evaluating model performance")
            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=x_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")
            best_model = models[best_model_name]
            
            # Get predictions and metrics for training data
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            # Track the training metrics with mlflow
            logging.info("Tracking training metrics with MLflow")
            self.track_mlflow(best_model, classification_train_metric, run_name=f"{best_model_name}_train")

            # Get predictions and metrics for test data
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Track the test metrics with mlflow
            logging.info("Tracking test metrics with MLflow")
            self.track_mlflow(best_model, classification_test_metric, run_name=f"{best_model_name}_test")

            # Load preprocessor
            logging.info("Loading preprocessor object")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                
            # Create model directory if it doesn't exist
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Create and save NetworkModel
            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            logging.info(f"Saving model at: {self.model_trainer_config.trained_model_file_path}")
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)
            
            # Save final model
            final_model_dir = "final_model"
            os.makedirs(final_model_dir, exist_ok=True)
            final_model_path = os.path.join(final_model_dir, "model.pkl")
            logging.info(f"Saving final model at: {final_model_path}")
            save_object(final_model_path, obj=best_model)
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise NetworkSecurityException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading training data from: {train_file_path}")
            logging.info(f"Loading test data from: {test_file_path}")
            
            # Loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info(f"Training data shape: X={x_train.shape}, y={y_train.shape}")
            logging.info(f"Test data shape: X={x_test.shape}, y={y_test.shape}")

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact
            
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {str(e)}")
            raise NetworkSecurityException(e, sys)