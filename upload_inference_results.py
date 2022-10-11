import os
import time
import asyncio
from azure.storage.blob.aio import BlobServiceClient

CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=savedmodelinference;AccountKey=0YkXMhJJoJYdDSw3iLRpqAN5vJRkYB2LHzWv3ElvDDNxYSwFkC6sPO8iuclP6jd0f4EUvZS7Lmq/+AStmh+rMQ==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "inference"
LOCAL_PATH = "saved_inference"


async def upload_image(file_name, blob_service_client):
    async with blob_service_client.get_blob_client(container=CONTAINER_NAME,
                                                   blob=file_name) as blob_client:
        exists = await blob_client.exists()
        if not exists:
            upload_file_path = os.path.join(LOCAL_PATH, file_name)
            with open(upload_file_path, "rb") as data:
                await blob_client.upload_blob(data)


async def upload_blobs():
    async with BlobServiceClient.from_connection_string(CONNECTION_STRING) as blob_service_client:
        files = [f for f in os.listdir(LOCAL_PATH)]
        await asyncio.gather(*(upload_image(file, blob_service_client) for file in files))


if __name__ == "__main__":
    start = time.time()
    asyncio.run(upload_blobs())
    end = time.time()
    print(f'File(s) Uploaded Successfully. Took { "{:.2f}".format(end-start)} seconds...')
