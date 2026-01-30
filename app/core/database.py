from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient

from app.core.config import settings

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def get_embed_model():
    return HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL_NAME,
        device="mps",  # Использует GPU через Metal на Mac
    )
