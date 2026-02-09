from logging import Logger

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import Settings
from app.utils.validators import ensure_directory_exists


def get_embed_model(settings: Settings, logger: Logger):
    try:

        logger.info("🔄 Загружаем модель эмбэддингов")
        local_dir = ensure_directory_exists(settings.EMBEDDING_MODEL_DIR)

        embed_model = HuggingFaceEmbedding(
            model_name=str(local_dir),
            device="mps",  # Использует GPU через Metal на Mac
        )
        logger.info("✅ Модель эмбеддингов успешно загружена и инициализирована")

        return embed_model

    except FileNotFoundError as e:
        logger.error(f"❌ Директория не найдена: {e}")
        raise
    except NotADirectoryError as e:
        logger.error(f"❌ Путь не является директорией: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать модель эмбеддингов: {e}")
        raise


def get_reranking_model(settings: Settings, logger: Logger):
    try:

        logger.info("🔄 Загружаем модель реранкинга")
        local_dir = ensure_directory_exists(settings.RERANKING_MODEL_DIR)

        rerank_model = SentenceTransformerRerank(
            model=str(local_dir), top_n=settings.RERANK_TOP_K, device="mps"
        )
        logger.info("✅ Модель реранкинга успешно загружена и инициализирована")

        return rerank_model

    except FileNotFoundError as e:
        logger.error(f"❌ Директория не найдена: {e}")
        raise
    except NotADirectoryError as e:
        logger.error(f"❌ Путь не является директорией: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать модель реранкинга: {e}")
        raise
