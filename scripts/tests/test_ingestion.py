from pathlib import Path

import pytest

from scripts.ingestion_qdrant import IngestionPipeline


@pytest.fixture
def ingestion_pipeline():
    return IngestionPipeline()


@pytest.fixture
def pdf_path():
    path = (
        Path(__file__).parent.parent / "datasets" / "test_datasets" / "polovoz_test.pdf"
    )
    if not path.exists():
        pytest.skip(f"Test PDF not found: {path}")
    return path


@pytest.fixture
def collection_name():
    return "test"


def test_upload_pdf_to_qdrant(ingestion_pipeline, pdf_path, collection_name):
    """
    Тестируем загрузку PDF в Qdrant без автоматического удаления коллекции.
    """
    print(f"\n🔄 Загружаем {pdf_path} в '{collection_name}' в qdrant...")

    # Запускаем пайплайн
    try:
        ingestion_pipeline.run(pdf_path, collection_name)
        print("✅ Успешно загрузили в qdrant")
    except Exception as e:
        pytest.fail(f"❌ Pipeline failed: {e}")
