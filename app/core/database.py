import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from app.core.config import settings
from app.core.logging import logger


def get_qdrant_client():
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10,
        )

        client.get_collections()
        logger.info("✅ Подключение к Qdrant успешно установлено.")
        return client
    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"❌ Ошибка протокола HTTP при подключении к Qdrant: {e}")
        raise ConnectionError(f"❌ Не удалось подключиться к Qdrant: {e}") from e
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Не удалось подключиться к Qdrant (ошибка сети): {e}")
        raise ConnectionError(f"❌ Не удалось подключиться к Qdrant: {e}") from e
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка при подключении к Qdrant: {e}")
        raise
