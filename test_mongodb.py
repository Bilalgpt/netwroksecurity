from pymongo.mongo_client import MongoClient
import certifi

# Updated connection string with your new cluster details
uri = "mongodb+srv://Bilal:1234@cluster0.fg42q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server with SSL certificate verification
client = MongoClient(uri, tlsCAFile=certifi.where())

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
    
    # Optional: List available databases to further verify connection
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f" - {db_name}")
        
except Exception as e:
    print(f"Connection error: {e}")
finally:
    # Close the connection when done
    client.close()
    print("\nConnection closed.")