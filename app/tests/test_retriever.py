from app.core.logging import logger
from app.repositories.qdrant import retrieve_nodes_from_qdrant


def test_retrieve():
    query = "комплексное число"
    collection_name = "math"  # или укажи конкретную, например "math"

    nodes = retrieve_nodes_from_qdrant(query, collection_name)

    if nodes:
        logger.info(f"\n✅ Найдено {len(nodes)} релевантных фрагментов:")
        for i, node in enumerate(nodes, 1):
            logger.info(f"{i}. [Оценка: {node.score:.3f}] {node.text.strip()[:200]}...")
    else:
        logger.error("❌ Ничего не найдено.")

    assert nodes is not None  # Проверяем, что функция вернула результат
