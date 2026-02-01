from llama_index.core import VectorStoreIndex

from app.core.config import settings
from app.core.database import ingestiers, retrievers
from app.core.logging import logger
from app.core.ml_models import embed_model


def ingest_nodes_to_qdrant(nodes: list, collection_name: str):
    try:
        for i in range(len(settings.COLLECTIONS)):
            if collection_name == settings.COLLECTIONS[i]:
                ingestier = ingestiers[i]
                break

        logger.info(f"🔄 Начинаем запись в '{collection_name}' в qdrant...")
        VectorStoreIndex(
            nodes=nodes,
            storage_context=ingestier,
            embed_model=embed_model,
            show_progress=True,
        )
        logger.info(f"✅ Успешно записали {len(nodes)} в '{collection_name}' в qdrant")
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить в qdrant: {e}")


def retrieve_nodes_from_qdrant(query: str, collection_name: str):
    try:
        for i in range(len(settings.COLLECTIONS)):
            if collection_name == settings.COLLECTIONS[i]:
                retriever = retrievers[i]
                break

        logger.info(
            f"🔄 Делаем запрос на извлечение из '{collection_name}' в qdrant..."
        )
        nodes = retriever.retrieve(query)
        logger.info(f"✅ Успешно извлекли {len(nodes)} из '{collection_name}' в qdrant")
        return nodes
    except Exception as e:
        logger.error(f"❌ Не удалось загрузить в qdrant: {e}")
