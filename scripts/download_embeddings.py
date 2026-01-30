import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# Добавляем корень проекта в PYTHONPATH (одноразовый пайплайн)
root_path = Path(__file__).parent.parent  # предполагается, что скрипт в scripts/
sys.path.append(str(root_path))

from app.core.config import settings
from app.utils.validators import ensure_directory_exists

if __name__ == "__main__":
    try:

        print("Начинаем загрузку модели с Hugging Face...")

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
        print(f"❌ Не удалось загрузить модель: {e}")
        sys.exit(1)
