from logging import Logger

import requests
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.core.config import Settings


def get_qdrant_client(settings: Settings, logger: Logger):
    try:

        logger.info("🔄 Начинаем соединение с qdrant")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10,
        )
        collections = client.get_collections()
        logger.info(
            f"✅ Подключение к Qdrant успешно установлено, доступные коллекции: {collections.collections}"
        )

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


class Qdrant:
    def __init__(
        self,
        settings: Settings,
        logger: Logger,
        qdrant_client: QdrantClient,
        emded_model: HuggingFaceEmbedding,
    ):
        self.settings = settings
        self.logger = logger
        self.qdrant_client = qdrant_client
        self.embed_model = emded_model

    def _get_qdrant_ingestier_math(self):
        try:
            self.logger.info("🔄 Начинаем создание ingestier_math")
            ingestion_vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name="math",
                distance_metric="Cosine",
            )

            ingestier_math = StorageContext.from_defaults(
                vector_store=ingestion_vector_store
            )
            self.logger.info("✅ Создание ingestier_math успешно")

            return ingestier_math

        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации ingestier_math: {e}")
            raise

    def _get_qdrant_retriever_math(self):
        try:
            self.logger.info("🔄 Начинаем создание retriever_math")
            retrieve_vector_store = QdrantVectorStore(
                client=self.qdrant_client, collection_name="math"
            )

            index = VectorStoreIndex.from_vector_store(
                vector_store=retrieve_vector_store, embed_model=self.embed_model
            )

            retriever_math = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.settings.QUERY_TOP_K,
                embed_model=self.embed_model,
            )
            self.logger.info("✅ Создание retriever_math успешно")

            return retriever_math

        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации retriever_math: {e}")
            raise

    def _get_qdrant_ingestiers(self):
        ingestier_math = self._get_qdrant_ingestier_math()
        return [ingestier_math]  # дописываем соотв-о коллекциям в qdrant

    def _get_qdrant_retrievers(self):
        retriever_math = self._get_qdrant_retriever_math()
        return [retriever_math]

    def get_qdrant_ingestier(self, collection_name: str):
        ingestiers = self._get_qdrant_ingestiers()

        for i in range(len(self.settings.COLLECTIONS)):
            if collection_name == self.settings.COLLECTIONS[i]:
                return ingestiers[i]

    def get_qdrant_retriever(self, collection_name: str):
        retrievers = self._get_qdrant_retrievers()

        for i in range(len(self.settings.COLLECTIONS)):
            if collection_name == self.settings.COLLECTIONS[i]:
                return retrievers[i]
