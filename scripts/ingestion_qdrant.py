import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.repositories.qdrant import IngestionPipeline

if __name__ == "__main__":
    pdf_path = Path(__file__).parent / "datasets" / "main_datasets" / "polovoz.pdf"
    collection_name = "math"

    print(f"🔄 Loading PDF: {pdf_path.name}")
    loader = IngestionPipeline()
    loader.run(pdf_path, collection_name)
    print(f"✅ Успешно ingested into Qdrant collection '{collection_name}'")
