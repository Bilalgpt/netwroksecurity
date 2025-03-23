import os
import sys
import pandas as pd
import pymongo
import certifi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_and_import_data(csv_file_path):
    """
    Check if data exists in MongoDB and import from CSV if needed
    """
    try:
        # Get MongoDB connection details from environment variables
        mongo_db_url = os.getenv("MONGO_DB_URL")
        mongo_db_database = os.getenv("MONGO_DB_DATABASE", "Bilalnetwork")
        mongo_db_collection = os.getenv("MONGO_DB_COLLECTION", "NetworkData")
        
        if not mongo_db_url:
            print("Error: MongoDB URL not found in environment variables")
            return False
            
        print(f"MongoDB URL: {mongo_db_url}")
        print(f"Database: {mongo_db_database}")
        print(f"Collection: {mongo_db_collection}")
        
        # Connect to MongoDB
        client = pymongo.MongoClient(mongo_db_url, tlsCAFile=certifi.where())
        db = client[mongo_db_database]
        collection = db[mongo_db_collection]
        
        # Check if collection exists and has data
        doc_count = collection.count_documents({})
        print(f"Found {doc_count} documents in collection")
        
        if doc_count == 0:
            print("No data found in collection. Importing from CSV file...")
            
            # Check if CSV file exists
            if not os.path.exists(csv_file_path):
                print(f"Error: CSV file not found at {csv_file_path}")
                return False
                
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            print(f"CSV file contains {len(df)} rows and {len(df.columns)} columns")
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert data into MongoDB
            result = collection.insert_many(records)
            print(f"Successfully imported {len(result.inserted_ids)} records into MongoDB")
            
            # Verify data was imported
            new_count = collection.count_documents({})
            print(f"Collection now contains {new_count} documents")
            return True
        else:
            print("Data already exists in collection. No import needed.")
            return True
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    # Get CSV file path from command line argument or use default
    csv_file_path = sys.argv[1] if len(sys.argv) > 1 else "Network_Data/phisingData.csv"
    
    # Check and import data if needed
    check_and_import_data(csv_file_path)