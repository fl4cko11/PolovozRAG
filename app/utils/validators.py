from pathlib import Path


def ensure_directory_exists(path: str | Path) -> Path:
    """
    Проверяет, что указанный путь существует и является директорией.
    Возвращает resolved Path-объект.
    """
    local_dir = Path(path).resolve()

    if not local_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {local_dir}")

    if not local_dir.is_dir():
        raise NotADirectoryError(f"Путь не является директорией: {local_dir}")

    return local_dir


def ensure_path_exists(path: str | Path) -> Path:
    """
    Проверяет, что указанный путь существует и является директорией.
    Возвращает resolved Path-объект.
    """
    local_dir = Path(path).resolve()

    if not local_dir.exists():
        raise FileNotFoundError(f"Директория не найдена: {local_dir}")

    return local_dir
