# app/services/storage.py
import os
from app.config import settings
from azure.storage.blob import BlobServiceClient

def _get_blob_service_client():
    """Helper function to create a BlobServiceClient."""
    if not settings.AZURE_STORAGE_CONNECTION_STRING:
        raise RuntimeError("Azure Storage Connection String not set in .env file.")
    return BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)

def upload_image_bytes(img_bytes: bytes, public_id: str) -> tuple[str, str]:
    """Uploads image bytes to the photo container and returns the public URL."""
    if not settings.STORAGE_ACCOUNT_NAME:
        raise RuntimeError("Storage Account Name not set in .env file.")

    blob_service_client = _get_blob_service_client()
    container_name = settings.AZURE_PHOTO_CONTAINER
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=public_id)
    blob_client.upload_blob(img_bytes, overwrite=True)
    
    url = f"https://{settings.STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{container_name}/{public_id}"
    return url, url

def upload_file(local_path: str, public_id: str, container_name: str):
    """Uploads a local file to a specified container."""
    blob_service_client = _get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=public_id)
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, timeout=300)
    print(f"Uploaded {local_path} to Azure container '{container_name}' as {public_id}")

def download_blob_to_file(public_id: str, local_path: str, container_name: str) -> bool:
    """Downloads a blob to a local file path. Returns True on success."""
    try:
        blob_service_client = _get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=public_id)
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        return True
    except Exception as e:
        # This is expected if the blob doesn't exist (e.g., first run)
        print(f"Info: Could not download {public_id} from container '{container_name}'. It may not exist yet. Error: {e}")
        return False