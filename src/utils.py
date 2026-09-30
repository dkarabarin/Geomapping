"""
Вспомогательные функции: расстояния, проекция координат, сохранение CSV.
"""

import logging
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

logger = logging.getLogger("geoplan")


def calculate_distance(point1: Sequence[float],
                       point2: Sequence[float]) -> float:
    """
    Расстояние по дуге большого круга (haversine) между двумя точками.

    Args:
        point1: (latitude, longitude) в градусах.
        point2: (latitude, longitude) в градусах.

    Returns:
        Расстояние в километрах.
    """
    lat1, lon1 = np.radians(point1[0]), np.radians(point1[1])
    lat2, lon2 = np.radians(point2[0]), np.radians(point2[1])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Матрица попарных haversine-расстояний.

    Args:
        lat: массив широт, градусы.
        lon: массив долгот, градусы.

    Returns:
        Квадратная матрица расстояний в километрах.
    """
    lat_r = np.radians(np.asarray(lat, dtype=float))[:, None]
    lon_r = np.radians(np.asarray(lon, dtype=float))[:, None]
    a = (np.sin((lat_r - lat_r.T) / 2) ** 2
         + np.cos(lat_r) * np.cos(lat_r.T) * np.sin((lon_r - lon_r.T) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def to_local_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Проецирует координаты в локальную плоскость (км).

    Равнопромежуточная проекция относительно средней широты. На масштабе
    области (сотни км) погрешность < 1 %, и в отличие от градусов оси
    имеют одинаковый масштаб — это важно для евклидова KMeans.

    Args:
        lat: широты, градусы.
        lon: долготы, градусы.

    Returns:
        Массив формы (n, 2): x (восток), y (север) в километрах.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.radians(lat.mean())
    km_per_deg = np.pi * EARTH_RADIUS_KM / 180
    return np.column_stack([lon * km_per_deg * np.cos(lat0),
                            lat * km_per_deg])


def save_results(tables: Dict[str, pd.DataFrame],
                 output_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Сохраняет таблицы результата в CSV (UTF-8 с BOM — открывается в Excel).

    Args:
        tables: {имя_файла_без_расширения: DataFrame}.
        output_dir: папка для сохранения.

    Returns:
        {имя: путь к сохранённому файлу}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        paths[name] = path
        logger.info("Сохранено: %s (%d строк)", path, len(df))
    return paths


def centroid(lat: Sequence[float], lon: Sequence[float]) -> Tuple[float, float]:
    """Среднее по координатам (для компактных групп точек)."""
    return float(np.mean(lat)), float(np.mean(lon))
