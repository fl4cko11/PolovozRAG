import sys
from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.core.database import client
from app.core.ml_models import embed_model
from app.utils.validators import ensure_path_exists


class IngestionPipeline:

    def load_pdf(self, file_path: Path | str) -> list[Document]:
        """Загружаем PDF по пути к файлу"""
        try:

            loc_dir = ensure_path_exists(file_path)
            print(f"🔄 Загружаем PDF from: {loc_dir}")
            reader = PDFReader()
            documents = reader.load_data(file=loc_dir)
            print(f"✅ Загрузили {len(documents)} document(s) from PDF.")
            return documents

        except Exception as e:
            print(f"❌ Не удалось загрузить модель: {e}")
            sys.exit(1)

    def chunk_documents(self, documents: list[Document]) -> list:
        """Разбиваем документы на иерархические чанки (ноды)."""
        try:

            print("🔄 Начинаем чанковать")
            node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])
            nodes = node_parser.get_nodes_from_documents(documents)
            print(f"✅ Сгенерировано {len(nodes)} иерархических нод.")
            return nodes

        except Exception as e:
            print(f"❌ Ошибка при разбиении на чанки: {e}")
            sys.exit(1)

    def chunk2vDB(self, nodes: list, collection_name: str):
        """Записываем ноды в Qdrant"""
        try:

            print("🔄 Начинаем запись в БД...")

            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                distance_metric="Cosine",
            )

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            print(
                f"✅ Успешно wrote {len(nodes)} nodes to Qdrant collection '{collection_name}'."
            )

        except Exception as e:
            print(f"❌ Не удалось загрузить в qdrant: {e}")
            sys.exit(1)

    def run(self, file_path: Path | str, collection_name: str):
        """
        Полный пайплайн: загрузка PDF → чанкинг → индексация в Qdrant
        :param file_path: Путь к PDF-файлу
        :param collection_name: Имя коллекции в Qdrant
        """
        documents = self.load_pdf(file_path)
        nodes = self.chunk_documents(documents)
        self.chunk2vDB(nodes, collection_name)


if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "datasets" / "main_datasets" / "polovoz.pdf"
    loc_dir = ensure_path_exists(pdf_path)

    collection_name = "math"

    print(f"🔄 Loading PDF: {loc_dir.name}")
    loader = IngestionPipeline()
    loader.run(loc_dir, collection_name)
    print(f"✅ Успешно ingested into Qdrant collection '{collection_name}'")
