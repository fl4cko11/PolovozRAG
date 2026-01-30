from pathlib import Path

import pytest

from app.repositories.ingestion import File2vDB


@pytest.fixture
def file2vdb():
    return File2vDB()


@pytest.fixture
def pdf_path():
    path = Path(__file__).parent / "test_datasets" / "petrovich_test.pdf"
    if not path.exists():
        pytest.skip(f"Test PDF not found: {path}")
    return path


@pytest.fixture
def collection_name():
    return "test"


def test_upload_pdf_to_qdrant(file2vdb, pdf_path, collection_name):
    """
    Тестируем загрузку PDF в Qdrant без автоматического удаления коллекции.
    """
    print(f"\n📄 Uploading {pdf_path} to Qdrant collection '{collection_name}'...")

    # Запускаем пайплайн
    try:
        file2vdb.run(pdf_path, collection_name)
        print("✅ Upload to Qdrant completed.")
    except Exception as e:
        pytest.fail(f"❌ Pipeline failed: {e}")
