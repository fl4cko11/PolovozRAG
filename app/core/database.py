import requests
from llama_index.core import StorageContext, VectorStoreIndex
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


def _get_qdrant_client():
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


def _get_qdrant_ingestier_math(client: QdrantClient):
    try:
        logger.info("🔄 Начинаем создание ingestier_math")
        ingestion_vector_store = QdrantVectorStore(
            client=client,
            collection_name="math",
            distance_metric="Cosine",
        )

        ingestier_math = StorageContext.from_defaults(
            vector_store=ingestion_vector_store
        )
        logger.info("✅ Создание ingestier_math успешно")
        return ingestier_math
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации ingestier_math: {e}")


def _get_qdrant_retriever_math(client: QdrantClient):
    try:
        logger.info("🔄 Начинаем создание retriever_math")
        retrieve_vector_store = QdrantVectorStore(client=client, collection_name="math")

        index = VectorStoreIndex.from_vector_store(vector_store=retrieve_vector_store)

        retriever_math = VectorIndexRetriever(
            index=index,
            similarity_top_k=settings.QUERY_TOP_K,
            embed_model=embed_model,
        )
        logger.info("✅ Создание retriever_math успешно")
        return retriever_math
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации retriever_math: {e}")


_client = _get_qdrant_client()

_retriever_math = _get_qdrant_retriever_math(_client)

_ingestier_math = _get_qdrant_ingestier_math(_client)

retrievers = [
    _retriever_math
]  # порядок должен соответсвовать именам коллекций в конфиге

ingestiers = [_ingestier_math]
