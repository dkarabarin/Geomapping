"""
Визуализация плана на интерактивной карте (folium).

Правила отображения (они же выводятся легендой на карте):
* цвет точки и линии — кластер дня (у каждого кластера свой цвет);
* цифра в маркере — order_in_route из schedule.csv;
* линии проходят через точки строго в порядке order_in_route:
  старт -> кластер 1 -> кластер 2 -> старт; сплошная — обход кластера,
  пунктир — подъезд от старта, переезд между кластерами, возврат;
* если передан ``router`` (OSRM), линии идут по дорогам, а в подсказках
  показаны км и минуты по дорогам;
* каждый день — отдельный слой, слои сгруппированы по менеджерам.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
           "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
MANAGER_COLORS = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#b9770e"]


def _legend_html(lines_source: str) -> str:
    """HTML-легенда карты."""
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 6px rgba(0,0,0,.3); font: 12px sans-serif;
                max-width: 300px; line-height: 1.45">
      <b>План посещений</b><br>
      ● Цвет точки и линии — <b>кластер дня</b>.<br>
      ● Цифра в кружке — <b>order_in_route</b> из schedule.csv.<br>
      ━ Сплошная — обход кластера в порядке order_in_route.<br>
      ┅ Пунктир — подъезд от старта, переезд между кластерами,
        возврат.<br>
      Линии: <b>{lines_source}</b>.<br>
      ⌂ — старт. Слои справа: менеджер → день.<br>
      Слой «Все точки»: цвет — <b>менеджер</b>.
    </div>"""


def _marker_html(num: int, color: str) -> str:
    """Круглый маркер с номером в маршруте."""
    return (f'<div style="background:{color};color:#fff;border-radius:50%;'
            f'width:22px;height:22px;display:flex;align-items:center;'
            f'justify-content:center;font:bold 11px sans-serif;'
            f'border:2px solid #fff;box-shadow:0 0 3px rgba(0,0,0,.6)">'
            f'{num}</div>')


def _day_legs(coords: List[Tuple[float, float]], router, distance) -> list:
    """Участки дня: по OSRM, по дорожному графу или прямые."""
    if router is not None:
        return router.legs(coords)
    legs = []
    for a, b in zip(coords[:-1], coords[1:]):
        geometry = (distance.path(*a, *b) if hasattr(distance, "path")
                    else [a, b])
        legs.append({"geometry": geometry, "km": None, "min": None,
                     "road": hasattr(distance, "path")})
    return legs


def _fmt(km, minutes) -> str:
    """Подпись участка: км и минуты по дорогам, если известны."""
    if km is None:
        return "по прямой"
    return f"{km:.1f} км, {minutes:.0f} мин по дорогам"


def visualize_map(points: pd.DataFrame,
                  clusters: pd.DataFrame,
                  depot: Optional[Tuple[float, float]] = None,
                  distance=None,
                  output_path: Optional[Union[str, Path]] = None,
                  router=None):
    """
    Рисует точки, кластеры и маршруты на карте.

    Args:
        points: все валидные точки (point_id, latitude, longitude, manager).
        clusters: итоговое расписание (schedule): point_id, visit_day,
            cluster_id, order_in_route, manager, cluster_seq_in_day,
            latitude, longitude.
        depot: точка старта (lat, lon) или None.
        distance: источник расстояний; если у него есть метод ``path``
            (дорожный граф), линии рисуются по графу.
        output_path: если задан — карта сохраняется в HTML.
        router: ``OsrmRouter`` — линии по дорогам через OSRM.

    Returns:
        (folium.Map, DataFrame со статистикой по дорогам на каждый
        кластер: road_km, road_min, transfer_km, transfer_min).
    """
    import folium
    from folium.plugins import GroupedLayerControl

    center = [points["latitude"].mean(), points["longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")
    if depot is not None:
        folium.Marker(depot, tooltip="Старт маршрутов",
                      icon=folium.Icon(color="black", icon="home",
                                       prefix="fa")).add_to(fmap)

    all_layer = folium.FeatureGroup(name="Все точки", show=True)
    for row in points.itertuples():
        color = MANAGER_COLORS[int(row.manager) % len(MANAGER_COLORS)]
        folium.CircleMarker(
            [row.latitude, row.longitude], radius=4, color=color,
            fill=True, fill_opacity=0.8,
            tooltip=(f"{row.point_id} · менеджер {row.manager} · "
                     f"{row.visits_per_month} виз./мес.")).add_to(all_layer)
    all_layer.add_to(fmap)

    groups = {"Обзор": [all_layer]}
    stats, first, any_road, any_straight = [], True, False, False
    for manager, mdf in clusters.groupby("manager"):
        layers = []
        for day, ddf in mdf.groupby("visit_day"):
            ddf = ddf.sort_values(["cluster_seq_in_day", "order_in_route"])
            coords = list(zip(ddf["latitude"], ddf["longitude"]))
            seqs = ddf["cluster_seq_in_day"].tolist()
            if depot is not None:
                coords = [tuple(depot)] + coords + [tuple(depot)]
                seqs = [None] + seqs + [None]
            legs = _day_legs(coords, router, distance)
            colors = {s: PALETTE[(day * 3 + s) % len(PALETTE)]
                      for s in set(ddf["cluster_seq_in_day"])}
            ids = dict(zip(ddf["cluster_seq_in_day"], ddf["cluster_id"]))
            per_cluster = {s: {"cluster_id": ids[s], "road_km": 0.0,
                               "road_min": 0.0, "transfer_km": 0.0,
                               "transfer_min": 0.0} for s in colors}

            layer = folium.FeatureGroup(name=f"День {day}", show=first)
            first = False
            for i, leg in enumerate(legs):
                a, b = seqs[i], seqs[i + 1]
                inside = a is not None and a == b
                owner = b if b is not None else a
                color = colors[owner] if inside else "#444"
                if leg["road"]:
                    any_road = True
                else:
                    any_straight = True
                if leg["km"] is not None:
                    part = "road" if inside else "transfer"
                    per_cluster[owner][f"{part}_km"] += leg["km"]
                    per_cluster[owner][f"{part}_min"] += leg["min"]
                kind = "обход" if inside else (
                    "подъезд" if a is None else
                    "возврат" if b is None else "переезд")
                folium.PolyLine(
                    leg["geometry"], color=color,
                    weight=4 if inside else 2,
                    opacity=0.9 if inside else 0.7,
                    dash_array=None if inside else "6 6",
                    tooltip=f"{kind}: {_fmt(leg['km'], leg['min'])}",
                ).add_to(layer)

            for row in ddf.itertuples():
                s = per_cluster[row.cluster_seq_in_day]
                folium.Marker(
                    [row.latitude, row.longitude],
                    icon=folium.DivIcon(
                        html=_marker_html(row.order_in_route,
                                          colors[row.cluster_seq_in_day]),
                        icon_size=(22, 22), icon_anchor=(11, 11)),
                    tooltip=(f"{row.point_id} · {row.cluster_id} · "
                             f"№{row.order_in_route} · визит "
                             f"{row.visit_number}/{row.visits_per_month}"
                             + (f" · обход кластера {s['road_km']:.1f} км"
                                if s["road_km"] else "")),
                ).add_to(layer)
            layer.add_to(fmap)
            layers.append(layer)
            for s in per_cluster.values():
                stats.append({"manager": manager, "visit_day": day, **s})
        groups[f"Менеджер {manager}"] = layers

    source = ("по дорогам (OSRM)" if router is not None and any_road
              else "по дорожному графу" if any_road else "прямые отрезки")
    if any_road and any_straight:
        source += "; часть — прямые (маршрут не получен)"
    GroupedLayerControl(groups, exclusive_groups=False,
                        collapsed=True).add_to(fmap)
    fmap.fit_bounds([[points["latitude"].min(), points["longitude"].min()],
                     [points["latitude"].max(), points["longitude"].max()]])
    fmap.get_root().html.add_child(folium.Element(_legend_html(source)))
    if output_path:
        fmap.save(str(output_path))

    road_stats = pd.DataFrame(stats)
    if not road_stats.empty:
        num = ["road_km", "road_min", "transfer_km", "transfer_min"]
        road_stats[num] = road_stats[num].round(1)
    return fmap, road_stats
