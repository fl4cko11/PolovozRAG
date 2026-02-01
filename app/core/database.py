import requests
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.core.config import settings
from app.core.logging import logger
from app.core.ml_models import embed_model


def get_qdrant_client():
    try:
        logger.info("🔄 Начинаем соединение с qdrant")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10,
        )

        client.get_collections()
        logger.info("✅ Подключение к Qdrant успешно установлено.")
        return client
    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"❌ Ошибка протокола HTTP при подключении к Qdrant: {e}")
        raise ConnectionError(f"❌ Не удалось подключиться к Qdrant: {e}") from e
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Не удалось подключиться к Qdrant (ошибка сети): {e}")
        raise ConnectionError(f"❌ Не удалось подключиться к Qdrant: {e}") from e
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при подключении к Qdrant: {e}")
        raise


def get_qdrant_math_retriever(client: QdrantClient):
    try:
        logger.info("🔄 Начинаем создание retriever")
        retrieve_vector_store = QdrantVectorStore(client=client, collection_name="math")

        index = VectorStoreIndex.from_vector_store(vector_store=retrieve_vector_store)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=settings.QUERY_TOP_K,
            embed_model=embed_model,
        )
        logger.info("✅ Создание retriever успешно")
        return retriever
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации retriever: {e}")


client = get_qdrant_client()
retriever_math = get_qdrant_math_retriever(client)
