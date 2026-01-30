import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException

from app.core.config import settings


def test_qdrant_connection():
    """Тестируем подключение к облачному Qdrant."""
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=10,  # Таймаут на случай проблем с сетью
    )

    try:
        # Пытаемся получить информацию о сервере
        info = client.get_collections()
        print("\nПодключение успешно!")
        print(f"Доступные коллекции: {[col.name for col in info.collections]}")
        assert True  # Подключение прошло успешно
    except ApiException as e:
        pytest.fail(f"Ошибка API Qdrant: {e}")
    except Exception as e:
        pytest.fail(f"Ошибка подключения к Qdrant: {e}")
