"""Тесты планирования месяца (критерии приёмки ТЗ)."""

from pathlib import Path

import pandas as pd
import pytest

from src import PlannerConfig, plan_month, run_pipeline, visit_offsets

DATA = Path(__file__).resolve().parents[1] / "data" / "data.csv"


@pytest.mark.parametrize("n,days,expected", [
    (1, 22, [0]), (2, 22, [0, 11]), (4, 22, [0, 6, 11, 17]),
    (2, 30, [0, 15]), (3, 22, [0, 7, 15])])
def test_visit_offsets(n, days, expected):
    """Визиты равномерно разнесены по месяцу."""
    assert visit_offsets(n, days) == expected


def test_plan_small(small_points):
    """Все визиты созданы, ограничения соблюдены, проверки пройдены."""
    cfg = PlannerConfig(max_points_per_cluster=8)
    result = plan_month(small_points, cfg)
    s = result.schedule
    assert result.unassigned.empty
    assert len(s) == small_points["visits_per_month"].sum()
    counts = s.groupby("point_id").size()
    required = small_points.set_index("point_id")["visits_per_month"]
    assert counts.reindex(required.index).equals(required)
    assert s.groupby("cluster_id").size().max() <= 8
    assert set(s.columns) >= {"point_id", "visit_day", "cluster_id",
                              "order_in_route"}
    hard = result.checks[~result.checks["check"].str.contains("информативно")]
    assert hard["passed"].all(), hard


def test_plan_reproducible(small_points):
    """Одинаковый seed — одинаковое расписание."""
    a = plan_month(small_points, PlannerConfig(random_seed=5)).schedule
    b = plan_month(small_points, PlannerConfig(random_seed=5)).schedule
    pd.testing.assert_frame_equal(a, b)


def test_deficit_is_reported_not_dropped(small_points):
    """Если ёмкости не хватает, визиты попадают в unassigned с причиной."""
    cfg = PlannerConfig(working_days=4, max_points_per_cluster=8,
                        max_clusters_per_manager_per_day=1)
    result = plan_month(small_points, cfg)
    assert not result.unassigned.empty
    assert result.unassigned["reason"].str.len().gt(0).all()
    total = len(result.schedule) + len(result.unassigned)
    assert total == small_points["visits_per_month"].sum()
    check = result.checks.set_index("check")
    assert not check.loc["Все визиты назначены (нет дефицита)", "passed"]


@pytest.mark.skipif(not DATA.exists(), reason="нет data/data.csv")
def test_real_data_end_to_end():
    """Реальный набор: 1125 визитов, дефицита нет."""
    out = run_pipeline(DATA, PlannerConfig(), output_dir=None)
    result = out["result"]
    assert len(result.schedule) == 1125
    assert result.unassigned.empty
    hard = result.checks[~result.checks["check"].str.contains("информативно")]
    assert hard["passed"].all()
