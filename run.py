"""
Запуск сервиса геопланирования из командной строки.

Примеры (из папки project):
    python run.py
    python run.py --data data/data.csv --output outputs --days 22
    python run.py --distance road --graphml ../cache2/volga_graph.graphml
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import PlannerConfig, run_pipeline  # noqa: E402
from src.config import DATA_PATH, GRAPHML_PATH, OUTPUT_DIR  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""
    defaults = PlannerConfig()
    parser = argparse.ArgumentParser(description="Сервис геопланирования")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="входной CSV")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="папка результатов")
    parser.add_argument("--days", type=int, default=defaults.working_days,
                        help="рабочих дней в месяце")
    parser.add_argument("--max-points", type=int,
                        default=defaults.max_points_per_cluster,
                        help="максимум точек в кластере")
    parser.add_argument("--clusters-per-day", type=int,
                        default=defaults.max_clusters_per_manager_per_day,
                        help="максимум кластеров у менеджера в день")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--distance", choices=["osrm", "haversine", "road"],
                        default="osrm",
                        help="источник расстояний: osrm — по дорогам (кеш), "
                             "haversine — по прямой офлайн, road — граф GraphML")
    parser.add_argument("--graphml", default=str(GRAPHML_PATH),
                        help="дорожный граф для --distance road")
    parser.add_argument("--no-depot", action="store_true",
                        help="не использовать точку старта (центр города)")
    parser.add_argument("--no-map", action="store_true",
                        help="не строить карту")
    parser.add_argument("--straight-lines", action="store_true",
                        help="линии на карте прямыми (без запросов к OSRM)")
    parser.add_argument("--open-map", action="store_true",
                        help="открыть карту в браузере")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Точка входа. Возвращает код выхода: 0 — все проверки пройдены."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(message)s")
    cfg = PlannerConfig(
        working_days=args.days,
        max_points_per_cluster=args.max_points,
        max_clusters_per_manager_per_day=args.clusters_per_day,
        random_seed=args.seed,
        distance_backend=args.distance,
        depot=None if args.no_depot else PlannerConfig().depot,
        map_road_geometry=not args.straight_lines,
    )
    out = run_pipeline(args.data, cfg, args.output, args.graphml,
                       make_map=not args.no_map)
    result = out["result"]
    print(f"\nВизитов требуется: {int(out['points']['visits_per_month'].sum())}"
          f", назначено: {len(result.schedule)}, "
          f"дефицит: {len(result.unassigned)}")
    print(f"Отклонено строк: {len(out['rejected'])}")
    print(f"Результаты: {Path(args.output).resolve()}")
    if args.open_map and "map" in out["files"]:
        webbrowser.open(out["files"]["map"].resolve().as_uri())
    hard_checks = result.checks[~result.checks["check"].str.contains(
        "информативно")]
    return 0 if hard_checks["passed"].all() else 1


if __name__ == "__main__":
    sys.exit(main())
