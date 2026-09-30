"""Тесты расстояний, кластеризации и маршрутов."""

import itertools

import numpy as np
import pytest

from src import build_route, calculate_distance, cluster_points
from src.distances import HaversineDistance
from src.routing import path_length


def test_calculate_distance_known_value():
    """Москва — Санкт-Петербург ≈ 634 км по прямой."""
    d = calculate_distance((55.7558, 37.6173), (59.9343, 30.3351))
    assert d == pytest.approx(634, abs=3)
    assert calculate_distance((56.3, 44.0), (56.3, 44.0)) == 0


@pytest.mark.parametrize("max_points", [8, 10, 12])
def test_cluster_size_limit_and_coverage(small_points, max_points):
    """Все точки распределены, размер кластера <= max_points."""
    labels = cluster_points(small_points, max_points=max_points)
    assert list(labels.columns) == ["point_id", "cluster"]
    assert set(labels["point_id"]) == set(small_points["point_id"])
    assert labels["cluster"].value_counts().max() <= max_points


def test_cluster_reproducible(small_points):
    """При фиксированном seed результат повторяется."""
    a = cluster_points(small_points, 10, random_state=1)
    b = cluster_points(small_points, 10, random_state=1)
    assert a.equals(b)


def test_clusters_are_compact(small_points):
    """Кластеры компактнее случайного разбиения того же размера."""
    labels = cluster_points(small_points, 10)
    pts = small_points.merge(labels, on="point_id")
    dist = HaversineDistance()

    def mean_diameter(groups):
        return np.mean([dist.matrix(g["latitude"], g["longitude"]).max()
                        for _, g in groups])

    random_labels = np.random.default_rng(0).permutation(labels["cluster"])
    shuffled = pts.assign(cluster=random_labels)
    assert mean_diameter(pts.groupby("cluster")) < \
        0.5 * mean_diameter(shuffled.groupby("cluster"))


def test_route_is_permutation_and_near_optimal(small_points):
    """Маршрут — перестановка точек; 2-opt близок к полному перебору."""
    cluster = small_points.head(7)
    order = build_route(cluster)
    assert sorted(order) == sorted(cluster["point_id"])

    dist = HaversineDistance().matrix(cluster["latitude"],
                                      cluster["longitude"])
    idx = {pid: i for i, pid in enumerate(cluster["point_id"])}
    found = path_length(dist, [idx[p] for p in order])
    optimum = min(path_length(dist, perm)
                  for perm in itertools.permutations(range(7)))
    assert found <= optimum * 1.05


def test_route_starts_near_depot(small_points):
    """Со стартом первая точка — ближайшая к старту (при 2-opt не хуже)."""
    cluster = small_points.head(8)
    depot = (56.3269, 44.0052)
    order = build_route(cluster, start=depot, two_opt=False)
    first = cluster.set_index("point_id").loc[order[0]]
    nearest = min(cluster.itertuples(), key=lambda r: calculate_distance(
        depot, (r.latitude, r.longitude)))
    assert first.name == nearest.point_id
