from logging import Logger
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader

from app.core.config import Settings
from app.utils.validators import ensure_path_exists


class QdrantIngestion:
    def __init__(
        self,
        settings: Settings,
        logger: Logger,
        ingestiers: list[
            StorageContext
        ],  # работаем сосписком всех ингестиеров для каждой коллекции
        emded_model: HuggingFaceEmbedding,
    ):
        self.settings = settings
        self.logger = logger
        self.ingestiers = ingestiers
        self.embed_model = emded_model

    def load_file(self, file_path: Path | str):
        """Загружаем PDF по пути к файлу"""
        try:

            loc_dir = ensure_path_exists(file_path)
            self.logger.info(f"🔄 Загружаем PDF from: {loc_dir}")
            reader = PDFReader()
            documents = reader.load_data(file=loc_dir)
            self.logger.info(f"✅ Загрузили {len(documents)} document(s) from PDF.")

            return documents

        except Exception as e:
            self.logger.error(f"❌ Не удалось загрузить модель: {e}")
            raise

    def chunk_file(self, file_path: Path | str):
        """Разбиваем документы на иерархические чанки (ноды)."""
        try:

            self.logger.info("🔄 Начинаем чанковать")
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])
            documents = self.load_file(file_path)
            nodes = node_parser.get_nodes_from_documents(documents)
            self.logger.info(f"✅ Сгенерировано {len(nodes)} иерархических нод.")

            return nodes

        except Exception as e:
            self.logger.error(f"❌ Ошибка при разбиении на чанки: {e}")
            raise

    def ingest_file_to_qdrant(self, file_path: Path | str, collection_name: str):
        try:
            nodes = self.chunk_file(file_path)

            for i in range(len(self.settings.COLLECTIONS)):
                if collection_name == self.settings.COLLECTIONS[i]:
                    ingestier = self.ingestiers[i]

            self.logger.info(f"🔄 Начинаем запись в '{collection_name}' в qdrant...")
            VectorStoreIndex(
                nodes=nodes,
                storage_context=ingestier,
                embed_model=self.embed_model,
                show_progress=True,
            )

            self.logger.info(
                f"✅ Успешно записали {len(nodes)} в '{collection_name}' в qdrant"
            )

        except Exception as e:
            self.logger.error(f"❌ Не удалось загрузить в qdrant: {e}")
            raise


class QdrantRetrieve:
    def __init__(
        self,
        settings: Settings,
        logger: Logger,
        retrievers: list[
            VectorIndexRetriever
        ],  # работаем сосписком всех ретриверов для каждой коллекции
        rerank_model: SentenceTransformerRerank,
    ):
        self.settings = settings
        self.logger = logger
        self.retrievers = retrievers
        self.rerank_model = rerank_model

    def retrieve_nodes(self, query: str, collection_name: str):
        try:
            for i in range(len(self.settings.COLLECTIONS)):
                if collection_name == self.settings.COLLECTIONS[i]:
                    retriever = self.retrievers[i]

            self.logger.info(
                f'🔄 Делаем retrieve запрос: "{query}" к "{collection_name}" в qdrant'
            )
            nodes = retriever.retrieve(query)
            self.logger.info(
                f"✅ Успешно извлекли {len(nodes)} из '{collection_name}' в qdrant"
            )

            return nodes

        except Exception as e:
            self.logger.error(f"❌ Не удалось извлечь из qdrant: {e}")
            raise

    def retrieve_nodes_with_rerank(self, query: str, collection_name: str):
        try:

            nodes = self.retrieve_nodes(query, collection_name)

            f'🔄 Делаем реранк: "{query}" к "{collection_name}" в qdrant'
            reranked_nodes = self.rerank_model.postprocess_nodes(
                nodes=nodes,
                query_str=query,
            )

            self.logger.info(
                f"✅ Успешно реранкнули {len(reranked_nodes)} из '{collection_name}' в qdrant"
            )

            return reranked_nodes

        except Exception as e:
            self.logger.error(f"❌ Не удалось извлечь из qdrant с реранком: {e}")
            raise
