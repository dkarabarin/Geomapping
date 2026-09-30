"""
Кластеризация точек: KMeans с ограничением ёмкости.

Почему не обычный KMeans/DBSCAN:
* KMeans не ограничивает размер кластера — приходится дробить и склеивать
  результат эвристиками, которые ломают компактность;
* DBSCAN на плотных данных объединяет весь регион в один кластер (в
  версии 2 так и было: eps=25 км, 633 точки -> 1 кластер), а радиус
  eps ещё и передавался не в радианах.

Алгоритм (capacity-constrained KMeans):
1. Число кластеров считается автоматически:
   k = ceil(n / (max_points * fill_ratio)).
2. Центры инициализируются обычным KMeans (seed фиксирован).
3. Шаг назначения решается как задача о назначениях
   (``scipy.optimize.linear_sum_assignment``): у каждого центра ровно
   ``max_points`` «мест», точки распределяются по местам с минимальной
   суммой квадратов расстояний. Так размер кластера <= max_points
   гарантирован математически, а не проверкой после факта.
4. Центры пересчитываются; шаги 3–4 повторяются до сходимости.

Координаты перед кластеризацией проецируются в километры
(``to_local_xy``), чтобы евклидово расстояние было корректным.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .utils import to_local_xy

logger = logging.getLogger("geoplan")


def n_clusters_for(n_points: int, max_points: int,
                   fill_ratio: float = 1.0) -> int:
    """
    Автоматическое число кластеров.

    Args:
        n_points: число точек.
        max_points: максимальный размер кластера.
        fill_ratio: целевая заполненность (0, 1].

    Returns:
        Число кластеров, не меньше ceil(n / max_points).
    """
    if n_points == 0:
        return 0
    k_min = math.ceil(n_points / max_points)
    k = math.ceil(n_points / (max_points * fill_ratio))
    return min(max(k, k_min), n_points)


def cluster_points(points: pd.DataFrame,
                   max_points: int = 12,
                   n_clusters: Optional[int] = None,
                   fill_ratio: float = 1.0,
                   random_state: int = 42,
                   max_iter: int = 50) -> pd.DataFrame:
    """
    Разбивает точки на компактные кластеры размером не более max_points.

    Args:
        points: DataFrame с колонками point_id, latitude, longitude.
        max_points: максимальное число точек в кластере.
        n_clusters: число кластеров; None — вычислить автоматически.
        fill_ratio: целевая заполненность при автоматическом выборе k.
        random_state: seed (при одинаковом seed результат повторяется).
        max_iter: максимум итераций уточнения.

    Returns:
        DataFrame с колонками point_id, cluster (0..k-1). Номера
        кластеров упорядочены с запада на восток для читаемости.
    """
    n = len(points)
    if n == 0:
        return pd.DataFrame({"point_id": pd.Series(dtype=str),
                             "cluster": pd.Series(dtype=int)})
    k = n_clusters or n_clusters_for(n, max_points, fill_ratio)
    if k * max_points < n:
        raise ValueError(f"{k} кластеров по {max_points} точек не вмещают "
                         f"{n} точек")

    xy = to_local_xy(points["latitude"].values, points["longitude"].values)
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        centers = KMeans(n_clusters=k, random_state=random_state,
                         n_init=10).fit(xy).cluster_centers_
        labels = None
        for _ in range(max_iter):
            cost = cdist(xy, centers, "sqeuclidean")
            _, slots = linear_sum_assignment(np.repeat(cost, max_points,
                                                       axis=1))
            new_labels = slots // max_points
            if labels is not None and np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                members = labels == c
                if members.any():
                    centers[c] = xy[members].mean(axis=0)

    # Перенумерация: без пустых кластеров, с запада на восток
    used = np.unique(labels)
    order = sorted(used, key=lambda c: xy[labels == c, 0].mean())
    remap = {old: new for new, old in enumerate(order)}
    return pd.DataFrame({
        "point_id": points["point_id"].values,
        "cluster": [remap[c] for c in labels],
    })
