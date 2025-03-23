import os
import sys
import boto3
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class S3Sync:
    """
    Class for syncing local folders to AWS S3 bucket
    """
    
    def __init__(self):
        """
        Initialize S3Sync with credentials from environment variables
        """
        try:
            self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
            self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
            
            if not self.access_key or not self.secret_key:
                logging.warning("AWS credentials not found in environment variables")
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )
            
            logging.info("S3Sync initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing S3Sync: {str(e)}")
            raise NetworkSecurityException(e, sys)
    
    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        Sync a local folder to an S3 bucket
        
        Args:
            folder: Local folder path to sync
            aws_bucket_url: S3 URL in format s3://bucket-name/path
        """
        try:
            logging.info(f"Syncing folder {folder} to {aws_bucket_url}")
            
            # Parse bucket name and prefix from aws_bucket_url
            if not aws_bucket_url.startswith("s3://"):
                raise ValueError(f"Invalid S3 URL format: {aws_bucket_url}. Must start with s3://")
            
            s3_parts = aws_bucket_url[5:].split('/', 1)  # Remove 's3://' and split by first '/'
            bucket_name = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ""
            
            # Check if folder exists
            if not os.path.exists(folder):
                logging.warning(f"Folder does not exist: {folder}")
                return
            
            # Upload all files in the folder to S3
            for root, dirs, files in os.walk(folder):
                for file in files:
                    local_path = os.path.join(root, file)
                    
                    # Calculate relative path to maintain folder structure in S3
                    relative_path = os.path.relpath(local_path, folder)
                    s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
                    
                    logging.info(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
                    
                    # Upload file to S3
                    self.s3_client.upload_file(
                        Filename=local_path,
                        Bucket=bucket_name,
                        Key=s3_key
                    )
            
            logging.info(f"Successfully synced {folder} to {aws_bucket_url}")
            
        except Exception as e:
            logging.error(f"Error syncing folder to S3: {str(e)}")
            raise NetworkSecurityException(e, sys)