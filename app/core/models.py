from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.core.config import settings
from app.core.logging import logger
from app.utils.validators import ensure_directory_exists


def get_embed_model():
    try:
        local_dir = ensure_directory_exists(settings.EMBEDDING_MODEL_DIR)

        model = HuggingFaceEmbedding(
            model_name=str(local_dir),
            device="mps",  # Использует GPU через Metal на Mac
        )
        logger.info("✅ Модель эмбеддингов успешно загружена и инициализирована.")
        return model
    except FileNotFoundError as e:
        logger.error(f"❌ Директория не найдена: {e}")
        raise
    except NotADirectoryError as e:
        logger.error(f"❌ Путь не является директорией: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Не удалось инициализировать модель эмбеддингов: {e}")
        raise
