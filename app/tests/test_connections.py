import pytest

from app.core.config import Settings
from app.core.database import get_qdrant_client
from app.core.llm import get_gigachat_client
from app.core.logging import get_logger
from app.core.ml_models import get_embed_model, get_reranking_model


def test_qdrant_connection():
    try:
        settings = Settings()
        logger = get_logger(settings)

        get_qdrant_client(settings, logger)

        return

    except Exception as e:
        pytest.fail(
            f"❌ Тест подключения к qdrant упал с исключением: {type(e).__name__}: {e}"
        )


def test_llm_connection():
    try:
        settings = Settings()
        logger = get_logger(settings)

        get_gigachat_client(settings, logger)

        return

    except Exception as e:
        pytest.fail(
            f"❌ Тест подключения к gigachat упал с исключением: {type(e).__name__}: {e}"
        )


def test_embed_model_connection():
    try:
        settings = Settings()
        logger = get_logger(settings)

        get_embed_model(settings, logger)

        return

    except Exception as e:
        pytest.fail(
            f"❌ Тест получения embed модели упал с исключением: {type(e).__name__}: {e}"
        )


def test_reranking_model_connection():
    try:
        settings = Settings()
        logger = get_logger(settings)

        get_reranking_model(settings, logger)

        return

    except Exception as e:
        pytest.fail(
            f"❌ Тест получения reranking модели упал с исключением: {type(e).__name__}: {e}"
        )
