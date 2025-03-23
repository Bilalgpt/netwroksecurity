# This should be in networksecurity/pipeline/training_pipeline.py
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

from networksecurity.cloud.s3_syncer import S3Sync
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get S3 bucket name from environment variables
TRAINING_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME", "netwroksec")

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        
        # Configure AWS credentials from environment variables
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        
        # Initialize S3 sync utility
        self.s3_sync = S3Sync()
        
        logging.info(f"Training pipeline initialized with artifact directory: {self.training_pipeline_config.artifact_dir}")
        logging.info(f"S3 bucket for artifacts: {TRAINING_BUCKET_NAME}")
        

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            logging.info("Starting data validation")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            
            logging.info("Starting data transformation")
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            logging.info("Starting model training")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed and artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise NetworkSecurityException(e, sys)

    # Sync local artifact directory to S3 bucket    
    def sync_artifact_dir_to_s3(self):
        try:
            logging.info(f"Syncing artifacts to S3 bucket: {TRAINING_BUCKET_NAME}")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            
            logging.info(f"Local artifact directory: {self.training_pipeline_config.artifact_dir}")
            logging.info(f"S3 destination: {aws_bucket_url}")
            
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.artifact_dir,
                aws_bucket_url=aws_bucket_url
            )
            
            logging.info("Artifact sync to S3 completed successfully")
        except Exception as e:
            logging.error(f"Error syncing artifacts to S3: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    # Sync local final model to S3 bucket     
    def sync_saved_model_dir_to_s3(self):
        try:
            logging.info(f"Syncing final model to S3 bucket: {TRAINING_BUCKET_NAME}")
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            
            logging.info(f"Local model directory: {self.training_pipeline_config.model_dir}")
            logging.info(f"S3 destination: {aws_bucket_url}")
            
            self.s3_sync.sync_folder_to_s3(
                folder=self.training_pipeline_config.model_dir,
                aws_bucket_url=aws_bucket_url
            )
            
            logging.info("Model sync to S3 completed successfully")
        except Exception as e:
            logging.error(f"Error syncing model to S3: {str(e)}")
            raise NetworkSecurityException(e, sys)
        
    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")
            
            # Step 1: Data Ingestion
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Step 2: Data Validation
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            
            # Step 3: Data Transformation
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact
            )
            
            # Step 4: Model Training
            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact=data_transformation_artifact
            )
            
            # Step 5: Sync artifacts and model to S3
            self.sync_artifact_dir_to_s3()
            self.sync_saved_model_dir_to_s3()
            
            logging.info("Training pipeline completed successfully")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in training pipeline: {str(e)}")
            raise NetworkSecurityException(e, sys)