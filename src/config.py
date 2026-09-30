"""
Конфигурация сервиса геопланирования.

Все параметры алгоритма собраны в dataclass ``PlannerConfig``: его можно
создать со значениями по умолчанию или переопределить нужные поля
(``PlannerConfig(working_days=20)``). Пути по умолчанию вычисляются
относительно папки проекта, поэтому проект переносится без правки кода.
Путь к дорожному графу можно задать переменной окружения ``GEO_GRAPHML``.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATA_PATH = DATA_DIR / "data.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = PROJECT_DIR / "cache"
GRAPHML_PATH = Path(os.environ.get("GEO_GRAPHML",
                                   CACHE_DIR / "volga_graph.graphml"))
ROAD_CACHE_PATH = CACHE_DIR / "road_dist_cache.json"
OSRM_CACHE_PATH = CACHE_DIR / "osrm_cache.json"
OSRM_TABLE_CACHE_PATH = CACHE_DIR / "osrm_table_cache.json"

PBF_FILENAME = "volga-fed-district-latest.osm.pbf"
PBF_PATH = DATA_DIR / PBF_FILENAME
PBF_DOWNLOAD_URL = ("https://download.geofabrik.de/russia/"
                    "volga-fed-district-latest.osm.pbf")

# Центр Нижнего Новгорода — точка старта маршрутов по умолчанию
CITY_CENTER: Tuple[float, float] = (56.3269, 44.0052)


@dataclass
class PlannerConfig:
    """
    Параметры планирования.

    Attributes:
        working_days: число рабочих дней в месяце (visit_day = 1..working_days).
        max_points_per_cluster: максимум точек в одном кластере (ТЗ: 8–12).
        min_points_per_cluster: желаемый минимум (мягкое ограничение,
            используется только в отчёте).
        max_clusters_per_manager_per_day: сколько кластеров (маршрутов)
            менеджер может пройти за один день.
        allowed_visits: допустимые значения visits_per_month.
        cluster_fill_ratio: целевая заполненность кластера; 0.9 оставляет
            запас, чтобы KMeans не «растягивал» кластеры ради заполнения.
        random_seed: seed для повторяемости кластеризации.
        distance_backend: "haversine" (по прямой, офлайн), "osrm" (по
            дорогам через OSRM, с кешем) или "road" (локальный граф OSM).
        road_detour_factor: коэффициент извилистости дорог: применяется к
            расстоянию по прямой при оценке времени в режиме haversine и
            при запасном расчёте, если дорожное расстояние не получено.
        avg_speed_kmh: средняя скорость для оценки времени в пути.
        service_time_min: время на один визит, минут.
        max_work_hours: норматив рабочего дня (для проверки, не жёсткое
            ограничение).
        depot: точка старта маршрутов (lat, lon) или None.
        default_manager: менеджер для строк, где поле manager отсутствует.
        use_two_opt: улучшать маршрут алгоритмом 2-opt.
        map_road_geometry: рисовать линии на карте по дорогам (OSRM);
            False — прямыми отрезками.
    """

    working_days: int = 22
    max_points_per_cluster: int = 12
    min_points_per_cluster: int = 8
    max_clusters_per_manager_per_day: int = 2
    allowed_visits: Tuple[int, ...] = (1, 2, 3, 4)
    cluster_fill_ratio: float = 0.9
    random_seed: int = 42
    distance_backend: str = "haversine"
    road_detour_factor: float = 1.3
    avg_speed_kmh: float = 50.0
    service_time_min: float = 15.0
    max_work_hours: float = 9.0
    depot: Optional[Tuple[float, float]] = field(default=CITY_CENTER)
    default_manager: int = 0
    use_two_opt: bool = True
    map_road_geometry: bool = True

    def __post_init__(self):
        """Проверяет согласованность параметров."""
        if self.working_days < 1:
            raise ValueError("working_days должно быть >= 1")
        if self.max_points_per_cluster < 1:
            raise ValueError("max_points_per_cluster должно быть >= 1")
        if self.max_clusters_per_manager_per_day < 1:
            raise ValueError("max_clusters_per_manager_per_day должно быть >= 1")
        if not 0 < self.cluster_fill_ratio <= 1:
            raise ValueError("cluster_fill_ratio должно быть в (0, 1]")
        if self.distance_backend not in ("haversine", "osrm", "road"):
            raise ValueError("distance_backend: 'haversine', 'osrm' или "
                             "'road'")
        too_frequent = [v for v in self.allowed_visits if v > self.working_days]
        if too_frequent:
            raise ValueError(
                f"visits_per_month {too_frequent} больше числа рабочих дней")
