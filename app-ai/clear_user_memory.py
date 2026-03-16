"""
One-time script to delete ALL records from the user_memory collection.
Run from app-ai/ directory:
  python clear_user_memory.py

Safe to delete: only clears user_memory, not knowledge_base.
"""
from pymilvus import MilvusClient
from src.config.settings import settings

client = MilvusClient(
    uri=settings.milvus_uri,
    token=settings.milvus_token or None,
)

result = client.delete(
    collection_name=settings.milvus_collection_memory,
    filter="user_id != ''",  # match all records
)
print(f"Deleted: {result}")
