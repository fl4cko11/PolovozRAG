import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.core.config import Settings
from app.core.database import Qdrant, get_qdrant_client
from app.core.logging import get_logger
from app.core.ml_models import get_embed_model
from app.repositories.qdrant import QdrantIngestion

if __name__ == "__main__":
    try:

        pdf_path = Path(__file__).parent / "datasets" / "main_datasets" / "polovoz.pdf"
        collection_name = "math"

        settings = Settings()
        logger = get_logger()

        embed_model = get_embed_model(settings, logger)
        qdrant_client = get_qdrant_client(settings, logger)
        qdrant = Qdrant(settings, logger, qdrant_client, embed_model)

        print(f"🔄 Loading PDF: {pdf_path.name}")

        qdrant_ingestier = QdrantIngestion(
            settings, logger, qdrant.get_qdrant_ingestiers(), embed_model
        )
        qdrant_ingestier.ingest_file_to_qdrant(pdf_path, collection_name)

        print(f"✅ Успешно ingested into Qdrant collection '{collection_name}'")

    except Exception as e:
        print(f"❌ Не удалось загрузить в qdrant: {type(e).__name__}: {e}")
