import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

# Updated environment variable names to match your .env file
mongo_db_url = os.getenv("MONGO_DB_URL")
mongo_db_database = os.getenv("MONGO_DB_DATABASE", "Bilalnetwork")
mongo_db_collection = os.getenv("MONGO_DB_COLLECTION", "NetworkData")

print(f"MongoDB URL: {mongo_db_url}")
print(f"MongoDB Database: {mongo_db_database}")
print(f"MongoDB Collection: {mongo_db_collection}")

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# Connect to MongoDB with proper SSL certificate
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

# You can use your existing constant values or override them with environment variables
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

# Use environment variables if available, otherwise use constants
database_name = mongo_db_database or DATA_INGESTION_DATABASE_NAME
collection_name = mongo_db_collection or DATA_INGESTION_COLLECTION_NAME

database = client[database_name]
collection = database[collection_name]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        logging.info("Starting training pipeline from API endpoint")
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        logging.info(f"Received prediction request with file: {file.filename}")
        
        # Create prediction_output directory if it doesn't exist
        os.makedirs("prediction_output", exist_ok=True)
        
        # Read uploaded file
        df = pd.read_csv(file.file)
        logging.info(f"Uploaded file contains {len(df)} rows and {len(df.columns)} columns")
        
        # Load preprocessor and model
        processor_path = "final_model/preprocessor.pkl"
        model_path = "final_model/model.pkl"
        
        logging.info(f"Loading preprocessor from {processor_path}")
        preprocessor = load_object(processor_path)
        
        logging.info(f"Loading model from {model_path}")
        final_model = load_object(model_path)
        
        # Create NetworkModel instance
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        
        # Make predictions
        logging.info("Making predictions")
        y_pred = network_model.predict(df)
        
        # Add predictions to dataframe
        df['predicted_column'] = y_pred
        logging.info(f"Predictions completed. Prediction distribution: {df['predicted_column'].value_counts().to_dict()}")
        
        # Save predictions to file
        output_path = 'prediction_output/output.csv'
        df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")
        
        # Generate HTML table
        table_html = df.to_html(classes='table table-striped')
        
        # Return HTML response
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise NetworkSecurityException(e, sys)

    
if __name__ == "__main__":
    # Add more logging for startup
    logging.info("Starting Network Security API")
    app_run(app, host="0.0.0.0", port=8005)