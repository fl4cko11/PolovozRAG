import sys
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file import PDFReader

from app.core.config import settings
from app.core.database import ingestiers, retrievers
from app.core.logging import logger
from app.core.ml_models import embed_model
from app.utils.validators import ensure_path_exists


class IngestionPipeline:

    def load_pdf(self, file_path: Path | str) -> list[Document]:
        """Загружаем PDF по пути к файлу"""
        try:

            loc_dir = ensure_path_exists(file_path)
            logger.info(f"🔄 Загружаем PDF from: {loc_dir}")
            reader = PDFReader()
            documents = reader.load_data(file=loc_dir)
            logger.info(f"✅ Загрузили {len(documents)} document(s) from PDF.")
            return documents

        except Exception as e:
            logger.error(f"❌ Не удалось загрузить модель: {e}")
            sys.exit(1)

    def chunk_documents(self, documents: list[Document]) -> list:
        """Разбиваем документы на иерархические чанки (ноды)."""
        try:

            logger.info("🔄 Начинаем чанковать")
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])
            nodes = node_parser.get_nodes_from_documents(documents)
            logger.info(f"✅ Сгенерировано {len(nodes)} иерархических нод.")
            return nodes

        except Exception as e:
            logger.error(f"❌ Ошибка при разбиении на чанки: {e}")
            sys.exit(1)

    def ingest_nodes_to_qdrant(self, nodes: list, collection_name: str):
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
            logger.info(
                f"✅ Успешно записали {len(nodes)} в '{collection_name}' в qdrant"
            )
        except Exception as e:
            logger.error(f"❌ Не удалось загрузить в qdrant: {e}")

    def run(self, file_path: Path | str, collection_name: str):
        """
        Полный пайплайн: загрузка PDF → чанкинг → индексация в Qdrant
        :param file_path: Путь к PDF-файлу
        :param collection_name: Имя коллекции в Qdrant
        """
        documents = self.load_pdf(file_path)
        nodes = self.chunk_documents(documents)
        self.ingest_nodes_to_qdrant(nodes, collection_name)


def retrieve_nodes_from_qdrant(query: str, collection_name: str):
    try:
        for i in range(len(settings.COLLECTIONS)):
            if collection_name == settings.COLLECTIONS[i]:
                retriever = retrievers[i]
                break

        logger.info(
            f'🔄 Делаем retrieve запрос: "{query}" к "{collection_name}" в qdrant'
        )

        nodes = retriever.retrieve(query)
        logger.info(f"✅ Успешно извлекли {len(nodes)} из '{collection_name}' в qdrant")
        return nodes
    except Exception as e:
        logger.error(f"❌ Не удалось извлечь из qdrant: {e}")
