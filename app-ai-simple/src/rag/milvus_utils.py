from src.rag.embeddings import EmbeddingClient


def ensure_collection(client, collection_name: str) -> None:
    """Create a Milvus collection with the standard schema if it does not exist.

    Schema:
      id     : INT64 (auto-generated primary key)
      vector : FLOAT_VECTOR (1024-dim, BGE-M3)
      + dynamic fields (content, source, user_id, tags, etc.)

    Index: COSINE similarity on the vector field.
    """
    if client.has_collection(collection_name):
        return

    from pymilvus import MilvusClient, DataType

    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EmbeddingClient.DIMENSIONS)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", metric_type="COSINE")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
