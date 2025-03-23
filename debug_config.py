import os
import sys
from dotenv import load_dotenv
import pandas as pd
import pymongo
import certifi

# Load environment variables
load_dotenv()

# Get MongoDB connection details
mongo_db_url = os.getenv("MONGO_DB_URL")

# Connect to MongoDB
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=certifi.where())

# Debug information
print("Available databases:")
print(client.list_database_names())

# Check Bilalnetwork database
if "Bilalnetwork" in client.list_database_names():
    db = client["Bilalnetwork"]
    print("\nCollections in Bilalnetwork:")
    print(db.list_collection_names())
    
    # Check NetworkData collection
    if "NetworkData" in db.list_collection_names():
        collection = db["NetworkData"]
        count = collection.count_documents({})
        print(f"\nNetworkData collection contains {count} documents")
        
        # Try to fetch data as the data ingestion would
        cursor = collection.find({})
        data_list = list(cursor)
        df = pd.DataFrame(data_list)
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)
        print(f"\nDataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()[:5]}...")  # Show first 5 columns
else:
    print("Bilalnetwork database not found")

# Now check what the DataIngestionConfig is using
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

# Create config objects
training_pipeline_config = TrainingPipelineConfig()
data_ingestion_config = DataIngestionConfig(training_pipeline_config)

# Print the database and collection names from config
print("\nDataIngestionConfig is using:")
print(f"Database name: {data_ingestion_config.database_name}")
print(f"Collection name: {data_ingestion_config.collection_name}")

# Check if this database and collection exist and have data
config_db_name = data_ingestion_config.database_name
config_coll_name = data_ingestion_config.collection_name

if config_db_name in client.list_database_names():
    config_db = client[config_db_name]
    if config_coll_name in config_db.list_collection_names():
        config_coll = config_db[config_coll_name]
        count = config_coll.count_documents({})
        print(f"\nConfig collection contains {count} documents")
    else:
        print(f"\nCollection {config_coll_name} not found in database {config_db_name}")
else:
    print(f"\nDatabase {config_db_name} not found")