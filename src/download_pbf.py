"""
Скачивание OSM PBF (нужен только для построения дорожного графа).

Запуск из папки project:  python -m src.download_pbf
"""

import logging
from pathlib import Path
from typing import Union

from .config import PBF_DOWNLOAD_URL, PBF_PATH

logger = logging.getLogger("geoplan")

MIN_PBF_BYTES = 50 * 2 ** 20


def download_pbf(url: str = PBF_DOWNLOAD_URL,
                 path: Union[str, Path] = PBF_PATH,
                 force: bool = False) -> Path:
    """
    Скачивает PBF с докачкой прерванной загрузки.

    Args:
        url: адрес файла.
        path: куда сохранить.
        force: скачать заново, даже если файл уже есть.

    Returns:
        Путь к файлу.
    """
    import requests

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= MIN_PBF_BYTES and not force:
        logger.info("PBF уже есть: %s", path)
        return path
    if force and path.exists():
        path.unlink()
    done = path.stat().st_size if path.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
        resp.raise_for_status()
        mode = "ab" if done and resp.status_code == 206 else "wb"
        with open(path, mode) as fh:
            for chunk in resp.iter_content(chunk_size=2 ** 20):
                fh.write(chunk)
    if path.stat().st_size < MIN_PBF_BYTES:
        raise IOError(f"Файл {path} подозрительно мал — загрузка не удалась")
    logger.info("PBF сохранён: %s (%.0f МБ)", path,
                path.stat().st_size / 2 ** 20)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_pbf()
