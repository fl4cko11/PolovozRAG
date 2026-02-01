import pytest

from app.core.database import _get_qdrant_client


def test_qdrant_connection():
    try:
        client = _get_qdrant_client()
        client.close()
    except Exception as e:
        pytest.fail(f"Ошибка подключения к Qdrant: {e}")
