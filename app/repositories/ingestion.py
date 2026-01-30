from pathlib import Path

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.core.database import client, get_embed_model
from app.core.logging import logger


class File2vDB:
    def __init__(self):
        self.embed_model = get_embed_model()
        self.client = client
        self.logger = logger

    def load_pdf(self, file_path: Path | str) -> list[Document]:
        """Загружаем PDF по пути к файлу"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.logger.info(f"Loading PDF from: {file_path}")
        reader = PDFReader()
        try:
            documents = reader.load_data(file=file_path)
            self.logger.info(f"Loaded {len(documents)} document(s) from PDF.")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading PDF from {file_path}: {e}")
            raise

    def chunk_documents(self, documents: list[Document]) -> list:
        """Разбиваем документ на иерархические чанки (ноды)"""
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512])
        nodes = node_parser.get_nodes_from_documents(documents)
        self.logger.info(f"Generated {len(nodes)} hierarchical nodes.")
        return nodes

    def chunk2vDB(self, nodes: list, collection_name: str):
        """Записываем ноды в Qdrant"""
        try:
            # Требуем существование коллекции
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name not in collection_names:
                self.logger.info(
                    f"ℹ️ Collection '{collection_name}' must be created by admin"
                )
                raise
            else:
                self.logger.info(f"ℹ️ Using existing collection: {collection_name}")

            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                distance_metric="Cosine",
            )

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True,
            )

            self.logger.info(
                f"✅ Successfully wrote {len(nodes)} nodes to Qdrant collection '{collection_name}'."
            )
        except Exception as e:
            self.logger.error(f"Error writing nodes to vector DB: {e}")
            raise

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
    import sys
    from pathlib import Path

    # Путь к PDF (относительно скрипта)
    pdf_path = Path(__file__).parent / "datasets" / "petrovich_2.pdf"
    collection_name = "math"

    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        sys.exit(1)

    print(f"📄 Loading PDF: {pdf_path.name}")
    loader = File2vDB()
    try:
        loader.run(pdf_path, collection_name)
        print(f"✅ Successfully ingested into Qdrant collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        sys.exit(1)
