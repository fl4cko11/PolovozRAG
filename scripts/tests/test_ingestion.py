import sys
from pathlib import Path

import pytest

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.repositories.qdrant import IngestionPipeline


def test_upload_pdf_to_qdrant():
    pdf_path = (
        Path(__file__).parent.parent / "datasets" / "test_datasets" / "polovoz_test.pdf"
    )
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")

    collection_name = "math"
    ingestion_pipeline = IngestionPipeline()

    ingestion_pipeline.run(pdf_path, collection_name)
