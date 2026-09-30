"""
Сервис геопланирования.

Распределяет визиты к точкам по дням месяца, группирует визиты дня в
компактные кластеры (<= max_points_per_cluster) и строит маршрут обхода.
"""

from .clustering import cluster_points
from .config import PlannerConfig
from .data_loader import load_and_validate_data, summarize_points
from .distances import HaversineDistance, RoadDistance, make_distance_provider
from .pipeline import run_pipeline
from .routing import build_route
from .schedule import PlanResult, plan_month, validate_plan, visit_offsets
from .utils import calculate_distance, save_results


def visualize_map(*args, **kwargs):
    """Ленивая обёртка: folium импортируется только при построении карты."""
    from .visualize import visualize_map as _visualize_map
    return _visualize_map(*args, **kwargs)


__all__ = [
    "PlannerConfig", "PlanResult", "HaversineDistance", "RoadDistance",
    "build_route", "calculate_distance", "cluster_points",
    "load_and_validate_data", "make_distance_provider", "plan_month",
    "run_pipeline", "save_results", "summarize_points", "validate_plan",
    "visit_offsets", "visualize_map",
]

__version__ = "3.0.0"
