"""Тесты дорожного источника расстояний на синтетическом графе."""

import json

import numpy as np
import pytest

nx = pytest.importorskip("networkx")

from src.distances import RoadDistance  # noqa: E402
from src.utils import calculate_distance  # noqa: E402


def grid_graph():
    """Решётка 5×5 узлов с шагом 0.01°, рёбра в обе стороны."""
    g = nx.MultiDiGraph()
    for i in range(5):
        for j in range(5):
            g.add_node(f"{i}_{j}", y=56.30 + i * 0.01, x=44.00 + j * 0.01)
    for i in range(5):
        for j in range(5):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < 5 and j + dj < 5:
                    a, b = f"{i}_{j}", f"{i + di}_{j + dj}"
                    length = calculate_distance(
                        (g.nodes[a]["y"], g.nodes[a]["x"]),
                        (g.nodes[b]["y"], g.nodes[b]["x"])) * 1000
                    g.add_edge(a, b, length=length)
                    g.add_edge(b, a, length=length)
    return g


def test_road_distance_is_manhattan_on_grid(tmp_path):
    """По решётке путь по диагонали = сумма катетов, не гипотенуза."""
    road = RoadDistance(grid_graph(), cache_path=tmp_path / "c.json")
    lat = np.array([56.30, 56.34])
    lon = np.array([44.00, 44.04])
    m = road.matrix(lat, lon)
    straight = calculate_distance((56.30, 44.00), (56.34, 44.04))
    legs = (calculate_distance((56.30, 44.00), (56.34, 44.00))
            + calculate_distance((56.30, 44.00), (56.30, 44.04)))
    assert m[0, 1] == pytest.approx(legs, rel=0.01)
    assert m[0, 1] > straight
    assert road.stats["pairs_road"] == 2
    road.save_cache()

    again = RoadDistance(grid_graph(), cache_path=tmp_path / "c.json")
    again.matrix(lat, lon)
    assert again.stats["pairs_cache"] == 2
    assert "road" in again.describe()


def test_far_point_falls_back_and_is_counted():
    """Точка вдали от графа считается по прямой и попадает в статистику."""
    road = RoadDistance(grid_graph(), detour_factor=1.3)
    m = road.matrix(np.array([56.30, 55.00]), np.array([44.00, 44.00]))
    straight = calculate_distance((56.30, 44.00), (55.00, 44.00))
    assert m[0, 1] == pytest.approx(straight * 1.3)
    assert road.stats["pairs_fallback"] == 2


def test_reads_v2_cache_format(tmp_path):
    """Кеш версии 2 с ключами "(a, b, c, d)" читается без eval."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"(56.3, 44.0, 56.34, 44.04)": 9.99}))
    road = RoadDistance(grid_graph(), cache_path=path)
    m = road.matrix(np.array([56.30, 56.34]), np.array([44.00, 44.04]))
    assert m[0, 1] == pytest.approx(9.99)
