"""
Полный сценарий: загрузка -> планирование -> проверки -> сохранение.

Используется и из ``run.py``, и из ноутбука, поэтому логика существует
в одном экземпляре.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .config import (DATA_PATH, GRAPHML_PATH, OSRM_CACHE_PATH,
                     OSRM_TABLE_CACHE_PATH, OUTPUT_DIR, ROAD_CACHE_PATH,
                     PlannerConfig)
from .data_loader import load_and_validate_data, summarize_points
from .distances import make_distance_provider
from .schedule import PlanResult, plan_month
from .utils import save_results

logger = logging.getLogger("geoplan")


def run_pipeline(data_path: Union[str, Path, pd.DataFrame] = DATA_PATH,
                 cfg: Optional[PlannerConfig] = None,
                 output_dir: Optional[Union[str, Path]] = OUTPUT_DIR,
                 graphml_path: Union[str, Path] = GRAPHML_PATH,
                 make_map: bool = True) -> dict:
    """
    Выполняет планирование от CSV до файлов результата.

    Args:
        data_path: путь к CSV или DataFrame с точками.
        cfg: параметры (по умолчанию PlannerConfig()).
        output_dir: папка результатов; None — ничего не сохранять.
        graphml_path: дорожный граф (только для distance_backend="road").
        make_map: строить HTML-карту.

    Returns:
        Словарь: points, rejected, summary, result (PlanResult),
        distance (источник расстояний), files (пути сохранённых файлов).
    """
    cfg = cfg or PlannerConfig()
    points, rejected = load_and_validate_data(
        data_path, cfg.allowed_visits, cfg.default_manager)
    if points.empty:
        raise ValueError("Нет ни одной корректной точки")
    summary = summarize_points(points)
    capacity = (cfg.working_days * cfg.max_clusters_per_manager_per_day
                * cfg.max_points_per_cluster)
    summary["capacity"] = capacity
    summary["load_pct"] = (summary["visits_required"] / capacity * 100).round(1)
    for row in summary.itertuples():
        level = logging.WARNING if row.load_pct > 100 else logging.INFO
        logger.log(level, "Менеджер %d: требуется %d визитов, ёмкость %d "
                   "(%.0f%%)", row.manager, row.visits_required,
                   capacity, row.load_pct)

    distance = make_distance_provider(cfg.distance_backend, graphml_path,
                                      ROAD_CACHE_PATH, cfg.road_detour_factor,
                                      OSRM_TABLE_CACHE_PATH)
    result: PlanResult = plan_month(points, cfg, distance)
    if hasattr(distance, "save_cache"):
        distance.save_cache()
    logger.info("Источник расстояний: %s", result.distance_source)
    log_checks(result.checks)

    files = {}
    if output_dir is not None:
        output_dir = Path(output_dir)
        files = save_results({
            "schedule": result.schedule,
            "clusters": result.clusters,
            "daily_stats": result.daily,
            "territories": result.territories,
            "unassigned": result.unassigned,
            "rejected_rows": rejected,
            "checks": result.checks,
        }, output_dir)
        report = {
            "config": asdict(cfg),
            "distance_source": result.distance_source,
            "points_valid": len(points),
            "rows_rejected": len(rejected),
            "visits_required": int(points["visits_per_month"].sum()),
            "visits_planned": len(result.schedule),
            "visits_unassigned": len(result.unassigned),
            "clusters": len(result.clusters),
            "checks_passed": bool(result.checks["passed"].all()),
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          indent=2), encoding="utf-8")
        files["report"] = report_path
        if make_map and not result.schedule.empty:
            from .visualize import visualize_map
            router = None
            if cfg.map_road_geometry:
                from .road_geometry import OsrmRouter
                router = OsrmRouter(OSRM_CACHE_PATH)
                logger.info("Геометрия дорог для карты (OSRM)...")
            map_path = output_dir / "route_map.html"
            _, road_stats = visualize_map(points, result.schedule, cfg.depot,
                                          distance, map_path, router)
            files["map"] = map_path
            logger.info("Карта: %s", map_path)
            if router is not None:
                router.save_cache()
                logger.info(router.describe())
                files.update(save_results({"road_stats": road_stats},
                                          output_dir))

    return {"points": points, "rejected": rejected, "summary": summary,
            "result": result, "distance": distance, "files": files}


def log_checks(checks: pd.DataFrame) -> None:
    """Печатает результаты проверок плана."""
    for _, row in checks.iterrows():
        mark = "OK  " if row["passed"] else "FAIL"
        logger.info("[%s] %s %s", mark, row["check"],
                    f"({row['details']})" if row["details"] else "")
