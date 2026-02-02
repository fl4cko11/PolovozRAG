import sys

from app.core.logging import logger
from app.repositories.qdrant import retrieve_nodes_from_qdrant


def test_retrieve():
    query = "комплексное число"
    collection_name = "math"

    try:
        nodes = retrieve_nodes_from_qdrant(query, collection_name)

        for i, node in enumerate(nodes, 1):
            logger.info(f"{i}. [Оценка: {node.score:.3f}] {node.text.strip()[:200]}...")
    except Exception as e:
        logger.error(f"❌ Ошибка при тесте retriever: {e}")
        sys.exit(1)
