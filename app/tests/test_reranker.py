import sys

from app.core.logging import logger
from app.core.ml_models import reranker_model
from app.repositories.qdrant import retrieve_nodes_from_qdrant


def test_retrieve_with_rerank():
    query = "комплексное число"
    collection_name = "math"

    try:

        initial_nodes = retrieve_nodes_from_qdrant(query, collection_name)

        logger.info("🔍 Результаты до реранкинга:")
        for i, node in enumerate(initial_nodes, 1):
            logger.info(f"  {i}. [Оценка: {node.score:.3f}]")

        logger.info("\n🔄 Применяем модель реранкинга...")
        reranked_nodes = reranker_model.postprocess_nodes(
            nodes=initial_nodes,
            query_str=query,
        )
        logger.info("\n✅ Успешно произвели реранкинг")

        logger.info("\n🔍 Результаты после реранкинга:")
        for i, node in enumerate(reranked_nodes, 1):
            logger.info(f"  {i}. [Реранк-оценка: {node.score:.3f}]")

        initial_ids = [node.node_id for node in initial_nodes]
        reranked_ids = [node.node_id for node in reranked_nodes]

        logger.info("\n📊 Сравнение позиций:")
        for new_idx, node_id in enumerate(reranked_ids, 1):
            old_idx = initial_ids.index(node_id) + 1 if node_id in initial_ids else "–"
            if str(old_idx) != str(new_idx):
                logger.info(f"  Узел {node_id} переместился: {old_idx} → {new_idx}")

    except Exception as e:
        logger.error(f"❌ Ошибка при тесте retriever с реранкингом: {e}")
        sys.exit(1)
