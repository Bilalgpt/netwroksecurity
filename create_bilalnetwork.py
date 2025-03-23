from pymongo import MongoClient
import certifi

# Connection string with your actual password
uri = "mongodb+srv://Bilal:1234@cluster0.fg42q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = MongoClient(uri, tlsCAFile=certifi.where())

# Create database by using it (MongoDB creates databases on first use)
db = client["Bilalnetwork"]

# Create a collection (this is what actually creates the database in MongoDB)
try:
    collection = db.create_collection("NetworkData")
    print("Collection 'NetworkData' created successfully!")
except Exception as e:
    print(f"Note: {e}")
    collection = db["NetworkData"]
    print("Using existing 'NetworkData' collection.")

# Insert a test document to make sure it's created
test_doc = {"test": "This is a test document for Bilalnetwork database"}
result = collection.insert_one(test_doc)

# Print confirmation
print("Database 'Bilalnetwork' and collection 'NetworkData' created successfully!")
print(f"Test document inserted with ID: {result.inserted_id}")

# List all databases to verify
print("\nAvailable databases:")
for db_name in client.list_database_names():
    print(f" - {db_name}")

# Close the connection
client.close()
print("\nConnection closed.")