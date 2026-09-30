"""Тесты OSRM-источников без обращения к сети (запрос подменяется)."""

import numpy as np
import pytest

from src import PlannerConfig, plan_month
from src.distances import OsrmDistance
from src.road_geometry import OsrmRouter
from src.utils import haversine_matrix


def fake_table(lat, lon):
    """«Дороги» = прямая × 2 (чтобы отличать от запасного ×1.3)."""
    return haversine_matrix(lat, lon) * 2


def test_osrm_distance_caches_day_matrix(tmp_path, monkeypatch):
    """Один запрос на набор точек, подмножества — из кеша, кеш на диске."""
    calls = []
    dist = OsrmDistance(cache_path=tmp_path / "t.json")
    monkeypatch.setattr(dist, "_request",
                        lambda la, lo: calls.append(len(la)) or fake_table(la, lo))
    lat = np.array([56.30, 56.31, 56.33, 56.35])
    lon = np.array([44.00, 44.02, 44.01, 44.05])
    full = dist.matrix(lat, lon)
    sub = dist.matrix(lat[[3, 1]], lon[[3, 1]])
    assert calls == [4]
    assert sub[0, 1] == pytest.approx(full[3, 1])
    assert full[0, 1] == pytest.approx(fake_table(lat, lon)[0, 1])
    dist.save_cache()

    again = OsrmDistance(cache_path=tmp_path / "t.json")
    monkeypatch.setattr(again, "_request",
                        lambda la, lo: pytest.fail("кеш не использован"))
    assert again.matrix(lat, lon)[2, 3] == pytest.approx(full[2, 3])


def test_osrm_distance_fallback_is_counted(monkeypatch):
    """Без ответа сервера — прямая × detour_factor и счётчик fallback."""
    dist = OsrmDistance(detour_factor=1.3)
    monkeypatch.setattr(dist, "_request", lambda la, lo: None)
    lat, lon = np.array([56.3, 56.4]), np.array([44.0, 44.1])
    m = dist.matrix(lat, lon)
    assert m[0, 1] == pytest.approx(haversine_matrix(lat, lon)[0, 1] * 1.3)
    assert dist.stats["pairs_fallback"] == 2
    assert "прямой" in dist.describe()


def test_plan_with_osrm_backend(small_points, monkeypatch):
    """План с источником osrm проходит все обязательные проверки."""
    dist = OsrmDistance()
    monkeypatch.setattr(dist, "_request", fake_table)
    result = plan_month(small_points, PlannerConfig(distance_backend="osrm",
                                                    max_points_per_cluster=8),
                        dist)
    hard = result.checks[~result.checks["check"].str.contains("информативно")]
    assert hard["passed"].all(), hard
    assert dist.stats["pairs_fallback"] == 0
    assert result.distance_source.startswith("osrm")


def test_router_fallback_draws_straight(monkeypatch):
    """Если маршрут не получен, участки — прямые, и это отмечено."""
    router = OsrmRouter()
    monkeypatch.setattr(router, "_request", lambda coords: None)
    legs = router.legs([(56.3, 44.0), (56.4, 44.1), (56.5, 44.2)])
    assert len(legs) == 2
    assert not any(leg["road"] for leg in legs)
    assert router.stats["fallback"] == 1
