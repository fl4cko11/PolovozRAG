import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.core.config import settings
from app.utils.validators import ensure_directory_exists


def download_emmbedding():
    try:

        print("🔄 Начинаем загрузку модели эмбэддингов с Hugging Face...")

        local_dir = ensure_directory_exists(settings.EMBEDDING_MODEL_DIR)

        snapshot_download(
            repo_id=settings.EMBEDDING_MODEL_NAME,
            local_dir=local_dir,
            token=settings.HF_TOKEN,
            local_dir_use_symlinks=False,
            revision="main",
        )
        print("✅ Загрузка завершена! Модель сохранена в:", "app/core/embeddings")

    except Exception as e:
        print(f"❌ Не удалось загрузить модель эмбэддингов: {e}")
        sys.exit(1)


def download_reranking():
    try:
        print(
            "🔄 Начинаем загрузку модели реранкинга (bge-reranker-base) с Hugging Face..."
        )

        local_dir = ensure_directory_exists(settings.RERANKING_MODEL_DIR)

        snapshot_download(
            repo_id=settings.RERANKING_MODEL_NAME,  # например, "BAAI/bge-reranker-base"
            local_dir=local_dir,
            token=settings.HF_TOKEN,
            local_dir_use_symlinks=False,
            revision="main",
        )
        print("✅ Загрузка завершена! Модель сохранена в:", local_dir)
    except Exception as e:
        print(f"❌ Не удалось загрузить модель реранкинга: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_reranking()
