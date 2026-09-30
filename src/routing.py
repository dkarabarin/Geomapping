"""
Построение маршрута внутри кластера.

Стратегия: «ближайший сосед» + улучшение 2-opt.
* Если задана точка старта (depot), маршрут начинается от неё: первой
  посещается ближайшая к старту точка, дальше — ближайшая непосещённая.
* Без старта жадный маршрут строится от каждой точки кластера, берётся
  самый короткий (при <= 12 точках это дёшево и детерминированно).
* 2-opt переворачивает участки маршрута, пока это сокращает путь.
  Маршрут открытый (без возврата), старт фиксирован.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .distances import HaversineDistance


def _nearest_neighbor(dist: np.ndarray, start: int,
                      nodes: Sequence[int]) -> List[int]:
    """Жадный путь из start по узлам nodes."""
    route = [start]
    left = set(nodes) - {start}
    while left:
        last = route[-1]
        nxt = min(left, key=lambda j: (dist[last, j], j))
        route.append(nxt)
        left.remove(nxt)
    return route


def path_length(dist: np.ndarray, route: Sequence[int]) -> float:
    """Длина открытого пути по матрице расстояний."""
    return float(sum(dist[a, b] for a, b in zip(route[:-1], route[1:])))


def _two_opt(dist: np.ndarray, route: List[int]) -> List[int]:
    """
    Улучшение открытого пути 2-opt; первый узел не двигается.

    Работает и для несимметричной матрицы: длина пересчитывается целиком.
    """
    best = list(route)
    best_len = path_length(dist, best)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_len = path_length(dist, cand)
                if cand_len < best_len - 1e-9:
                    best, best_len = cand, cand_len
                    improved = True
    return best


def build_route(cluster: pd.DataFrame,
                distance=None,
                start: Optional[Tuple[float, float]] = None,
                two_opt: bool = True) -> List[str]:
    """
    Возвращает порядок обхода точек кластера.

    Args:
        cluster: DataFrame с point_id, latitude, longitude.
        distance: источник расстояний (по умолчанию haversine).
        start: (lat, lon) точки старта или None.
        two_opt: применять улучшение 2-opt.

    Returns:
        Список point_id в порядке посещения.
    """
    ids = cluster["point_id"].tolist()
    if len(ids) <= 1:
        return ids
    distance = distance or HaversineDistance()
    lat = cluster["latitude"].to_numpy(float)
    lon = cluster["longitude"].to_numpy(float)
    n = len(ids)

    if start is not None:
        dist = distance.matrix(np.r_[start[0], lat], np.r_[start[1], lon])
        route = _nearest_neighbor(dist, 0, range(n + 1))
        if two_opt:
            route = _two_opt(dist, route)
        return [ids[i - 1] for i in route[1:]]

    dist = distance.matrix(lat, lon)
    candidates = [_nearest_neighbor(dist, s, range(n)) for s in range(n)]
    route = min(candidates, key=lambda r: (path_length(dist, r), r))
    if two_opt:
        # без фиксированного старта 2-opt может перевернуть и начало
        dummy = np.zeros((n + 1, n + 1))
        dummy[1:, 1:] = dist
        route = [i - 1 for i in _two_opt(dummy, [0] + [i + 1 for i in route])[1:]]
    return [ids[i] for i in route]


def route_metrics(cluster: pd.DataFrame, order: Sequence[str],
                  distance=None,
                  start: Optional[Tuple[float, float]] = None) -> dict:
    """
    Длина маршрута и дорог от/до точки старта.

    Args:
        cluster: точки кластера.
        order: порядок point_id (из build_route).
        distance: источник расстояний.
        start: точка старта или None.

    Returns:
        {"route_km": путь между точками,
         "depot_km": старт->первая + последняя->старт (0 без старта)}.
    """
    distance = distance or HaversineDistance()
    ordered = cluster.set_index("point_id").loc[list(order)]
    lat = ordered["latitude"].to_numpy(float)
    lon = ordered["longitude"].to_numpy(float)
    if start is None:
        dist = distance.matrix(lat, lon)
        return {"route_km": path_length(dist, range(len(order))),
                "depot_km": 0.0}
    dist = distance.matrix(np.r_[start[0], lat], np.r_[start[1], lon])
    n = len(order)
    return {"route_km": path_length(dist, range(1, n + 1)),
            "depot_km": float(dist[0, 1] + dist[n, 0])}
