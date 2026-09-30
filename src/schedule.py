"""
Распределение визитов по дням месяца и формирование кластеров дня.

Логика (для каждого менеджера отдельно):

1. **Территории.** Точки менеджера делятся на компактные группы
   размером <= max_points_per_cluster (``cluster_points``).
2. **Смещения визитов.** Точке с N визитами за D рабочих дней назначаются
   дни ``p + round(k * D / N)``, k = 0..N-1, где p — первый день её
   территории. Для D = 22: 2 визита -> p и p + 11 (ровно полмесяца),
   4 визита -> p, p+6, p+11, p+17. Интервал между визитами всегда
   >= floor(D / N).
3. **Размещение территорий.** Территория целиком «переезжает» по
   смещениям: в день p посещаются все её точки, в день p + 11 — только
   точки с 2+ визитами и т. д. Каждый такой выезд — кластер дня, он
   является подмножеством компактной территории и поэтому тоже компактен
   и не превышает лимит. Первый день p выбирается жадно так, чтобы:
   (а) у менеджера в каждый затронутый день было не больше
   ``max_clusters_per_manager_per_day`` кластеров; (б) максимальная
   оценочная загрузка дня была минимальной; (в) кластеры одного дня
   лежали рядом (меньше переезд).
4. **Без потерь.** Если территорию некуда поставить, число территорий
   уменьшается (они становятся плотнее), и попытка повторяется. Если места
   нет и так — визиты попадают в таблицу ``unassigned`` с причиной, а не
   исчезают.
5. **Маршрут дня.** Кластеры одного дня выстраиваются в цепочку
   старт -> кластер 1 -> кластер 2 -> старт; внутри кластера порядок
   строится ``build_route`` (ближайший сосед + 2-opt).
"""

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .clustering import cluster_points, n_clusters_for
from .config import PlannerConfig
from .distances import HaversineDistance
from .routing import build_route, route_metrics
from .utils import calculate_distance, centroid

logger = logging.getLogger("geoplan")


def visit_offsets(n_visits: int, working_days: int) -> List[int]:
    """
    Смещения (в днях) визитов точки относительно первого визита.

    Args:
        n_visits: визитов в месяц.
        working_days: рабочих дней в месяце.

    Returns:
        Список из n_visits возрастающих смещений, первое = 0.
    """
    return [int(math.floor(k * working_days / n_visits + 0.5))
            for k in range(n_visits)]


@dataclass
class PlanResult:
    """Результат планирования."""

    schedule: pd.DataFrame
    clusters: pd.DataFrame
    daily: pd.DataFrame
    territories: pd.DataFrame
    unassigned: pd.DataFrame
    distance_source: str = ""
    checks: Optional[pd.DataFrame] = field(default=None)


@dataclass
class _Visit:
    """Выезд территории в конкретное смещение: центр, длина обхода, точки."""

    center: Tuple[float, float]
    route_km: float
    n_points: int


@dataclass
class _Territory:
    """Территория менеджера и её выезды по смещениям."""

    tid: int
    points: pd.DataFrame
    offsets: Dict[int, pd.DataFrame]
    visits: Dict[int, _Visit]

    @property
    def last_offset(self) -> int:
        """Самое позднее смещение — ограничивает выбор первого дня."""
        return max(self.offsets)


def _day_hours(visits: List[_Visit], cfg: PlannerConfig) -> float:
    """
    Быстрая оценка длительности дня (haversine × извилистость).

    День = старт -> кластер 1 -> ... -> старт; переезды считаются между
    центрами кластеров, порядок кластеров перебирается (их <= 3).
    """
    if not visits:
        return 0.0
    best = None
    perms = (itertools.permutations(visits) if len(visits) <= 3
             else [visits])
    for perm in perms:
        stops = [v.center for v in perm]
        if cfg.depot is not None:
            stops = [cfg.depot] + stops + [cfg.depot]
        legs = sum(calculate_distance(a, b)
                   for a, b in zip(stops[:-1], stops[1:]))
        best = legs if best is None else min(best, legs)
    km = best + sum(v.route_km for v in visits)
    return (km * cfg.road_detour_factor / cfg.avg_speed_kmh
            + sum(v.n_points for v in visits) * cfg.service_time_min / 60)


def _make_territories(mdf: pd.DataFrame, labels: pd.DataFrame,
                      cfg: PlannerConfig) -> List[_Territory]:
    """Собирает территории и их выезды по смещениям."""
    mdf = mdf.merge(labels, on="point_id")
    dist = HaversineDistance()
    result = []
    for tid, tdf in mdf.groupby("cluster"):
        per_offset: Dict[int, List[int]] = {}
        for idx, n in zip(tdf.index, tdf["visits_per_month"]):
            for off in visit_offsets(int(n), cfg.working_days):
                per_offset.setdefault(off, []).append(idx)
        offsets = {o: tdf.loc[ix] for o, ix in sorted(per_offset.items())}
        visits = {}
        for o, pts in offsets.items():
            order = build_route(pts, dist, None, two_opt=False)
            visits[o] = _Visit(
                center=centroid(pts["latitude"], pts["longitude"]),
                route_km=route_metrics(pts, order, dist)["route_km"],
                n_points=len(pts))
        result.append(_Territory(tid=int(tid), points=tdf,
                                 offsets=offsets, visits=visits))
    return result


def _objective(placement: Dict[int, int], terrs: Dict[int, _Territory],
               cfg: PlannerConfig) -> float:
    """Сумма квадратов длительностей дней: штрафует пиковые дни."""
    by_day: Dict[int, List[_Visit]] = {}
    for tid, p in placement.items():
        for o, v in terrs[tid].visits.items():
            by_day.setdefault(p + o, []).append(v)
    return sum(_day_hours(v, cfg) ** 2 for v in by_day.values())


def _improve(placement: Dict[int, int], terrs: Dict[int, _Territory],
             cfg: PlannerConfig, max_passes: int = 20) -> Dict[int, int]:
    """
    Локальный поиск: перенос территории на другой свободный день и обмен
    первыми днями двух территорий. Принимаются только улучшения, лимит
    кластеров в день соблюдается всегда.
    """
    days, limit = cfg.working_days, cfg.max_clusters_per_manager_per_day
    placement = dict(placement)
    best = _objective(placement, terrs, cfg)

    def feasible(plan: Dict[int, int]) -> bool:
        load = [0] * days
        for tid, p in plan.items():
            for o in terrs[tid].visits:
                load[p + o] += 1
        return max(load) <= limit

    for _ in range(max_passes):
        improved = False
        tids = sorted(placement)
        for a in tids:
            candidates = [{a: p} for p in range(days - terrs[a].last_offset)
                          if p != placement[a]]
            candidates += [{a: placement[b], b: placement[a]} for b in tids
                           if b > a and placement[a] != placement[b]]
            for change in candidates:
                plan = {**placement, **change}
                if any(plan[t] + terrs[t].last_offset >= days
                       for t in change) or not feasible(plan):
                    continue
                score = _objective(plan, terrs, cfg)
                if score < best - 1e-9:
                    placement, best, improved = plan, score, True
        if not improved:
            break
    return placement


def _place(territories: List[_Territory], cfg: PlannerConfig
           ) -> Tuple[Dict[int, int], List[_Territory]]:
    """
    Выбирает первый день каждой территории.

    1. Жадно: сначала самые «жёсткие» территории (поздний последний визит,
       затем самые долгие по времени — обычно дальние). Для каждой
       выбирается первый день, при котором самый загруженный из
       затронутых дней минимален; при равенстве — с наименьшим приростом
       суммарного времени (соседние кластеры в один день выгоднее).
    2. Локальный поиск ``_improve`` сглаживает пиковые дни.

    Returns:
        ({tid: первый день (0-based)}, неразмещённые территории).
    """
    days = cfg.working_days
    day_visits: List[List[_Visit]] = [[] for _ in range(days)]
    day_hours = [0.0] * days
    placement, failed = {}, []
    order = sorted(territories, key=lambda t: (
        -t.last_offset,
        -sum(_day_hours([v], cfg) for v in t.visits.values()), t.tid))
    for terr in order:
        best, best_cost = None, None
        for p in range(days - terr.last_offset):
            if any(len(day_visits[p + o])
                   >= cfg.max_clusters_per_manager_per_day
                   for o in terr.visits):
                continue
            new = {p + o: _day_hours(day_visits[p + o] + [v], cfg)
                   for o, v in terr.visits.items()}
            added = sum(h - day_hours[d] for d, h in new.items())
            cost = (round(max(new.values()), 6), round(added, 6), p)
            if best_cost is None or cost < best_cost:
                best, best_cost = p, cost
        if best is None:
            failed.append(terr)
            continue
        placement[terr.tid] = best
        for o, v in terr.visits.items():
            day_visits[best + o].append(v)
            day_hours[best + o] = _day_hours(day_visits[best + o], cfg)
    placed = {t.tid: t for t in territories if t.tid in placement}
    return _improve(placement, placed, cfg), failed


def _plan_manager(manager: int, mdf: pd.DataFrame, cfg: PlannerConfig):
    """Подбирает число территорий и размещает их для одного менеджера."""
    n = len(mdf)
    k_pref = n_clusters_for(n, cfg.max_points_per_cluster,
                            cfg.cluster_fill_ratio)
    k_min = n_clusters_for(n, cfg.max_points_per_cluster, 1.0)
    best = None
    for k in range(k_pref, k_min - 1, -1):
        labels = cluster_points(mdf, cfg.max_points_per_cluster,
                                n_clusters=k, random_state=cfg.random_seed)
        terrs = _make_territories(mdf, labels, cfg)
        placement, failed = _place(terrs, cfg)
        lost = sum(len(t.points) for t in failed)
        if best is None or lost < best[3]:
            best = (terrs, placement, failed, lost)
        if not failed:
            break
        logger.info("  менеджер %s: %d территорий не помещаются, пробуем "
                    "k=%d", manager, len(failed), k - 1)
    return best[:3]


def _day_routes(clusters: List[Tuple[str, pd.DataFrame]], cfg: PlannerConfig,
                distance) -> List[Tuple[str, List[str], dict]]:
    """
    Строит маршрут дня: цепочка кластеров, каждый стартует с конца
    предыдущего. Перебираются порядки кластеров (их <= 3), берётся
    самый короткий.
    """
    if distance.name != "haversine":
        # один запрос матрицы на весь день; дальше подматрицы из кеша
        day = pd.concat([pts for _, pts in clusters])
        lat, lon = day["latitude"].to_numpy(), day["longitude"].to_numpy()
        if cfg.depot is not None:
            lat, lon = np.r_[cfg.depot[0], lat], np.r_[cfg.depot[1], lon]
        distance.matrix(lat, lon)
    best, best_km = None, None
    perms = (itertools.permutations(clusters) if len(clusters) <= 3
             else [clusters])
    for perm in perms:
        start, total, legs = cfg.depot, 0.0, []
        for cid, pts in perm:
            order = build_route(pts, distance, start, cfg.use_two_opt)
            metrics = route_metrics(pts, order, distance, start)
            first = pts.set_index("point_id").loc[order[0]]
            last = pts.set_index("point_id").loc[order[-1]]
            if start is None:
                approach = 0.0
            else:
                approach = float(distance.matrix(
                    np.array([start[0], first["latitude"]]),
                    np.array([start[1], first["longitude"]]))[0, 1])
            metrics = {"route_km": metrics["route_km"],
                       "approach_km": approach}
            legs.append((cid, order, metrics))
            total += metrics["route_km"] + approach
            start = (last["latitude"], last["longitude"])
        back = 0.0
        if cfg.depot is not None:
            back = float(distance.matrix(
                np.array([start[0], cfg.depot[0]]),
                np.array([start[1], cfg.depot[1]]))[0, 1])
        legs[-1][2]["return_km"] = back
        total += back
        if best_km is None or total < best_km - 1e-9:
            best, best_km = legs, total
    return best


def plan_month(points: pd.DataFrame, cfg: Optional[PlannerConfig] = None,
               distance=None) -> PlanResult:
    """
    Строит план посещений на месяц.

    Args:
        points: валидные точки (point_id, latitude, longitude,
            visits_per_month, manager).
        cfg: параметры планирования.
        distance: источник расстояний (по умолчанию haversine).

    Returns:
        PlanResult с таблицами schedule, clusters, daily, territories,
        unassigned.
    """
    cfg = cfg or PlannerConfig()
    distance = distance or HaversineDistance()
    speed_factor = (cfg.road_detour_factor if distance.name == "haversine"
                    else 1.0)

    sched_rows, cluster_rows, terr_rows, lost_rows = [], [], [], []
    for manager in sorted(points["manager"].unique()):
        mdf = points[points["manager"] == manager].reset_index(drop=True)
        terrs, placement, failed = _plan_manager(manager, mdf, cfg)
        logger.info("Менеджер %s: %d точек, %d визитов, %d территорий",
                    manager, len(mdf), int(mdf["visits_per_month"].sum()),
                    len(terrs))

        for terr in failed:
            for _, row in terr.points.iterrows():
                for k in range(int(row["visits_per_month"])):
                    lost_rows.append({
                        "point_id": row["point_id"], "manager": manager,
                        "visit_number": k + 1,
                        "visits_per_month": int(row["visits_per_month"]),
                        "reason": (f"нет свободного дня: у менеджера "
                                   f"{manager} все подходящие дни заняты "
                                   f"{cfg.max_clusters_per_manager_per_day} "
                                   f"кластерами")})

        by_day: Dict[int, List[Tuple[str, pd.DataFrame]]] = {}
        first_day_of: Dict[str, int] = {}
        for terr in terrs:
            if terr.tid not in placement:
                continue
            p = placement[terr.tid]
            for _, row in terr.points.iterrows():
                terr_rows.append({
                    "point_id": row["point_id"], "manager": manager,
                    "territory": f"M{manager}-T{terr.tid:02d}",
                    "first_day": p + 1})
            for off, pts in terr.offsets.items():
                day = p + off + 1
                cid = f"M{manager}-D{day:02d}-T{terr.tid:02d}"
                by_day.setdefault(day, []).append((cid, pts))
                first_day_of[cid] = p + 1

        for day in sorted(by_day):
            for seq, (cid, order, metrics) in enumerate(
                    _day_routes(by_day[day], cfg, distance), start=1):
                pts = dict(by_day[day])[cid].set_index("point_id")
                for pos, pid in enumerate(order, start=1):
                    row = pts.loc[pid]
                    n = int(row["visits_per_month"])
                    visit_no = visit_offsets(n, cfg.working_days).index(
                        day - first_day_of[cid]) + 1
                    sched_rows.append({
                        "point_id": pid, "visit_day": day, "cluster_id": cid,
                        "order_in_route": pos, "manager": manager,
                        "cluster_seq_in_day": seq, "visit_number": visit_no,
                        "visits_per_month": n,
                        "latitude": row["latitude"],
                        "longitude": row["longitude"]})
                km = (metrics["route_km"] + metrics["approach_km"]
                      + metrics.get("return_km", 0.0))
                cluster_rows.append({
                    "cluster_id": cid, "manager": manager, "visit_day": day,
                    "cluster_seq_in_day": seq,
                    "territory": cid.split("-")[0] + "-" + cid.split("-")[2],
                    "n_points": len(order),
                    "route_km": round(metrics["route_km"], 2),
                    "approach_km": round(metrics["approach_km"], 2),
                    "return_km": round(metrics.get("return_km", 0.0), 2),
                    "est_hours": round(
                        km * speed_factor / cfg.avg_speed_kmh
                        + len(order) * cfg.service_time_min / 60, 2)})

    schedule = pd.DataFrame(sched_rows)
    clusters = pd.DataFrame(cluster_rows)
    if not schedule.empty:
        schedule = schedule.sort_values(
            ["visit_day", "manager", "cluster_seq_in_day", "order_in_route"]
        ).reset_index(drop=True)
        clusters = clusters.sort_values(
            ["visit_day", "manager", "cluster_seq_in_day"]
        ).reset_index(drop=True)
    unassigned = pd.DataFrame(lost_rows, columns=[
        "point_id", "manager", "visit_number", "visits_per_month", "reason"])
    result = PlanResult(
        schedule=schedule, clusters=clusters,
        daily=_daily_stats(clusters, cfg),
        territories=pd.DataFrame(terr_rows),
        unassigned=unassigned, distance_source=distance.describe())
    result.checks = validate_plan(points, result, cfg)
    return result


def _daily_stats(clusters: pd.DataFrame, cfg: PlannerConfig) -> pd.DataFrame:
    """Загрузка по дням и менеджерам."""
    if clusters.empty:
        return pd.DataFrame()
    daily = clusters.groupby(["visit_day", "manager"]).agg(
        clusters=("cluster_id", "count"),
        visits=("n_points", "sum"),
        km=("route_km", "sum"),
        approach_km=("approach_km", "sum"),
        return_km=("return_km", "sum"),
        est_hours=("est_hours", "sum"),
    ).reset_index()
    daily["km"] = (daily["km"] + daily["approach_km"]
                   + daily["return_km"]).round(1)
    daily["est_hours"] = daily["est_hours"].round(2)
    daily["over_norm"] = daily["est_hours"] > cfg.max_work_hours
    return daily.drop(columns=["approach_km", "return_km"])


def validate_plan(points: pd.DataFrame, result: PlanResult,
                  cfg: PlannerConfig) -> pd.DataFrame:
    """
    Проверяет план по критериям приёмки ТЗ.

    Args:
        points: исходные валидные точки.
        result: результат plan_month.
        cfg: параметры.

    Returns:
        DataFrame: check, passed, details.
    """
    s = result.schedule
    checks = []

    def add(name: str, passed: bool, details: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed),
                       "details": details})

    required = points.set_index("point_id")["visits_per_month"]
    planned = (s.groupby("point_id").size() if not s.empty
               else pd.Series(dtype=int))
    lost = (result.unassigned.groupby("point_id").size()
            if not result.unassigned.empty else pd.Series(dtype=int))
    total = planned.reindex(required.index, fill_value=0) + \
        lost.reindex(required.index, fill_value=0)
    bad = total[total != required]
    add("Для каждой точки учтено ровно visits_per_month визитов "
        "(назначено + дефицит)", bad.empty,
        f"расхождений: {len(bad)}")

    add("Все визиты назначены (нет дефицита)", result.unassigned.empty,
        f"назначено {len(s)} из {int(required.sum())}, "
        f"дефицит {len(result.unassigned)}")

    missing = set(required.index) - set(planned.index) - set(lost.index)
    add("Все точки распределены по кластерам", not missing,
        f"без кластера: {len(missing)}")

    sizes = s.groupby("cluster_id").size() if not s.empty else pd.Series()
    add(f"Размер кластера <= {cfg.max_points_per_cluster}",
        (sizes <= cfg.max_points_per_cluster).all(),
        f"макс. размер {int(sizes.max()) if len(sizes) else 0}, "
        f"кластеров {len(sizes)}")

    per_day = (s.groupby(["manager", "visit_day"])["cluster_id"].nunique()
               if not s.empty else pd.Series(dtype=int))
    add(f"У менеджера <= {cfg.max_clusters_per_manager_per_day} "
        "кластеров в день",
        (per_day <= cfg.max_clusters_per_manager_per_day).all(),
        f"максимум {int(per_day.max()) if len(per_day) else 0}")

    dup = s.duplicated(["point_id", "visit_day"]).sum() if not s.empty else 0
    add("Точка не посещается дважды в один день", dup == 0,
        f"повторов: {dup}")

    gap_bad = 0
    if not s.empty:
        for pid, grp in s.groupby("point_id"):
            n = int(grp["visits_per_month"].iloc[0])
            days = np.sort(grp["visit_day"].to_numpy())
            if len(days) > 1 and np.diff(days).min() < cfg.working_days // n:
                gap_bad += 1
    add("Интервал между визитами точки >= floor(дней / visits_per_month)",
        gap_bad == 0, f"нарушений: {gap_bad}")

    in_range = s["visit_day"].between(1, cfg.working_days).all() \
        if not s.empty else True
    add(f"visit_day в диапазоне 1..{cfg.working_days}", in_range)

    order_ok = True
    for _, grp in s.groupby("cluster_id") if not s.empty else []:
        if sorted(grp["order_in_route"]) != list(range(1, len(grp) + 1)):
            order_ok = False
            break
    add("order_in_route — перестановка 1..n в каждом кластере", order_ok)

    over = int(result.daily["over_norm"].sum()) if not result.daily.empty \
        else 0
    add(f"Оценка дня <= {cfg.max_work_hours} ч (информативно)", over == 0,
        f"дней сверх нормы: {over}; макс. "
        f"{result.daily['est_hours'].max() if over or len(result.daily) else 0}"
        " ч")
    return pd.DataFrame(checks)
