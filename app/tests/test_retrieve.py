import pytest

from app.core.config import Settings
from app.core.database import Qdrant, get_qdrant_client
from app.core.logging import get_logger
from app.core.ml_models import get_embed_model, get_reranking_model
from app.repositories.qdrant import QdrantRetrieve


def test_retrieve():
    try:
        settings = Settings()
        logger = get_logger(settings)
        embed_model = get_embed_model(settings, logger)
        rerank_model = get_reranking_model(settings, logger)
        qdrant_client = get_qdrant_client(settings, logger)
        qdrant = Qdrant(settings, logger, qdrant_client, embed_model)
        qdrant_retriever = QdrantRetrieve(
            settings, logger, qdrant.get_qdrant_retrievers(), rerank_model
        )

        query = "комплексное число"
        collection_name = "math"

        # Получаем результаты
        retrieve_nodes = qdrant_retriever.retrieve_nodes(query, collection_name)
        reranked_nodes = qdrant_retriever.retrieve_nodes_with_rerank(
            query, collection_name
        )

        # Красивая логика вывода через logger
        logger.info("🔍 Результаты поиска:")
        logger.info("=" * 80)
        logger.info(f"Запрос: '{query}'")
        logger.info(f"Коллекция: {collection_name}")
        logger.info(f"Найдено узлов (retrieve): {len(retrieve_nodes)}")
        logger.info(f"Найдено узлов (rerank): {len(reranked_nodes)}")
        logger.info("-" * 80)

        for i, node in enumerate(retrieve_nodes):
            logger.info(f"📄 [{i+1}] (до rerank)")
            logger.info(f"   Оценка: {getattr(node, 'score', 'N/A'):.4f}")
            logger.info(f"   Текст: {node.text.strip()[:300]}...")
            logger.info(f"   Метаданные: {node.metadata}")
            logger.info("")

        logger.info("🔝 Результаты после rerank:")
        logger.info("-" * 80)
        for i, node in enumerate(reranked_nodes):
            logger.info(f"📌 [{i+1}] (после rerank)")
            logger.info(f"   Оценка: {getattr(node, 'score', 'N/A'):.4f}")
            logger.info(f"   Текст: {node.text.strip()[:300]}...")
            logger.info(f"   Метаданные: {node.metadata}")
            logger.info("")

        # Проверки
        assert len(retrieve_nodes) > 0, "Метод retrieve_nodes вернул пустой результат"
        assert (
            len(reranked_nodes) > 0
        ), "Метод retrieve_nodes_with_rerank вернул пустой результат"

    except Exception as e:
        pytest.fail(f"❌ Тест упал с исключением: {type(e).__name__}: {e}")
