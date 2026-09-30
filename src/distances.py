"""
Источники расстояний.

* ``HaversineDistance`` — расстояние по прямой (по умолчанию): быстро,
  без внешних данных, результат полностью воспроизводим.
* ``OsrmDistance`` — расстояния по дорогам через OSRM Table API (сеть
  нужна только при первом запуске, дальше — кеш).
* ``RoadDistance`` — кратчайший путь по дорожному графу OSM (GraphML).
  Требует networkx. Если пара точек не связана в графе, используется
  прямая × коэффициент извилистости, и это учитывается в статистике —
  в отчёте всегда видно, какой источник фактически применялся.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from .utils import haversine_matrix

logger = logging.getLogger("geoplan")


class HaversineDistance:
    """Расстояние по дуге большого круга."""

    name = "haversine"

    def __init__(self):
        self.stats = {"pairs_haversine": 0}

    def matrix(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Матрица расстояний, км.

        Args:
            lat: широты.
            lon: долготы.

        Returns:
            Квадратная матрица (n, n).
        """
        n = len(lat)
        self.stats["pairs_haversine"] += n * (n - 1)
        return haversine_matrix(lat, lon)

    def describe(self) -> str:
        """Короткое описание источника для отчёта."""
        return "haversine (по прямой)"


class RoadDistance:
    """
    Расстояние по дорожному графу.

    Граф загружается из GraphML (атрибуты узлов x/y, рёбер length в метрах).
    Ближайший узел ищется KD-деревом, кратчайшие пути — алгоритмом
    Дейкстры от каждого источника сразу до всех точек кластера.
    """

    name = "road"

    def __init__(self, graph, cache_path: Optional[Union[str, Path]] = None,
                 detour_factor: float = 1.3, max_snap_km: float = 2.0):
        """
        Args:
            graph: networkx-граф с координатами узлов x (lon), y (lat).
            cache_path: JSON-кеш дорожных расстояний.
            detour_factor: множитель к прямой, если путь не найден.
            max_snap_km: если ближайший узел дальше — точка считается вне
                графа и используется прямая.
        """
        self.graph = graph
        self.detour_factor = detour_factor
        self.max_snap_km = max_snap_km
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, float] = self._load_cache()
        self.stats = {"pairs_road": 0, "pairs_cache": 0, "pairs_fallback": 0}

        nodes = list(graph.nodes)
        lat = np.array([float(graph.nodes[n]["y"]) for n in nodes])
        lon = np.array([float(graph.nodes[n]["x"]) for n in nodes])
        self._nodes = nodes
        self._lat0 = lat.mean()
        self._tree = cKDTree(self._project(lat, lon))

    # ----------------------------------------------------------- helpers
    def _project(self, lat, lon) -> np.ndarray:
        """Проекция в км относительно широты графа (для KD-дерева)."""
        km_per_deg = 111.195
        return np.column_stack([
            np.asarray(lon, float) * km_per_deg * np.cos(np.radians(self._lat0)),
            np.asarray(lat, float) * km_per_deg])

    @staticmethod
    def _key(lat1, lon1, lat2, lon2) -> str:
        """Ключ кеша: координаты, округлённые до 6 знаков."""
        return f"{lat1:.6f},{lon1:.6f},{lat2:.6f},{lon2:.6f}"

    def _load_cache(self) -> Dict[str, float]:
        """
        Загружает кеш. Понимает и старый формат ключа "(a, b, c, d)"
        из версии 2, поэтому накопленные расстояния не теряются.
        """
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            logger.warning("Кеш %s не прочитан (%s), начат новый",
                           self.cache_path, err)
            return {}
        cache = {}
        for key, value in raw.items():
            try:
                parts = [float(p) for p in key.strip("()").split(",")]
            except ValueError:
                continue
            if len(parts) == 4:
                cache[self._key(*parts)] = float(value)
        logger.info("Кеш дорожных расстояний: %d записей", len(cache))
        return cache

    def save_cache(self) -> None:
        """Сохраняет кеш на диск."""
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")

    def _snap(self, lat, lon) -> Tuple[List, np.ndarray]:
        """Ближайшие узлы графа и расстояния до них, км."""
        dist, idx = self._tree.query(self._project(lat, lon))
        return [self._nodes[i] for i in np.atleast_1d(idx)], np.atleast_1d(dist)

    # ------------------------------------------------------------ public
    def matrix(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Матрица дорожных расстояний, км (несимметрична для one-way).

        Args:
            lat: широты.
            lon: долготы.

        Returns:
            Квадратная матрица (n, n).
        """
        import networkx as nx

        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        n = len(lat)
        straight = haversine_matrix(lat, lon)
        result = np.zeros((n, n))
        nodes, snap_km = self._snap(lat, lon)
        on_graph = snap_km <= self.max_snap_km
        cutoff_m = float(straight.max()) * 1000 * 4 + 5000

        for i in range(n):
            todo = []
            for j in range(n):
                if i == j:
                    continue
                key = self._key(lat[i], lon[i], lat[j], lon[j])
                if key in self.cache:
                    result[i, j] = self.cache[key]
                    self.stats["pairs_cache"] += 1
                else:
                    todo.append(j)
            if not todo:
                continue
            lengths = {}
            if on_graph[i]:
                lengths = nx.single_source_dijkstra_path_length(
                    self.graph, nodes[i], cutoff=cutoff_m, weight="length")
            for j in todo:
                length_m = lengths.get(nodes[j]) if on_graph[j] else None
                if length_m is None:
                    value = straight[i, j] * self.detour_factor
                    self.stats["pairs_fallback"] += 1
                else:
                    value = length_m / 1000 + snap_km[i] + snap_km[j]
                    self.stats["pairs_road"] += 1
                result[i, j] = value
                self.cache[self._key(lat[i], lon[i], lat[j], lon[j])] = value
        return result

    def path(self, lat1: float, lon1: float,
             lat2: float, lon2: float) -> List[Tuple[float, float]]:
        """
        Геометрия дорожного пути для карты.

        Returns:
            Список (lat, lon); при отсутствии пути — прямой отрезок.
        """
        import networkx as nx

        nodes, _ = self._snap([lat1, lat2], [lon1, lon2])
        try:
            route = nx.shortest_path(self.graph, nodes[0], nodes[1],
                                     weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [(lat1, lon1), (lat2, lon2)]
        coords = [(float(self.graph.nodes[v]["y"]),
                   float(self.graph.nodes[v]["x"])) for v in route]
        return [(lat1, lon1)] + coords + [(lat2, lon2)]

    def describe(self) -> str:
        """Короткое описание источника и доли запасных расчётов."""
        s = self.stats
        total = s["pairs_road"] + s["pairs_cache"] + s["pairs_fallback"]
        share = s["pairs_fallback"] / total * 100 if total else 0
        return (f"road (граф OSM): по графу {s['pairs_road']}, из кеша "
                f"{s['pairs_cache']}, по прямой×{self.detour_factor} "
                f"{s['pairs_fallback']} пар ({share:.1f}%)")


class OsrmDistance:
    """
    Расстояния по дорогам через OSRM Table API.

    Матрица запрашивается одним запросом на набор точек (например, все
    точки дня менеджера + старт) и раскладывается в кеш по парам, поэтому
    последующие запросы подмножеств (маршрут кластера, перебор порядка
    кластеров) обслуживаются из кеша без обращения к сети. Кеш хранится
    в JSON — повторный запуск полностью офлайн и воспроизводим.

    Если сервер недоступен, пара считается по прямой × detour_factor, и
    это учитывается в статистике ``pairs_fallback``.
    """

    name = "osrm"
    OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"
    MAX_COORDS = 100  # лимит публичного сервера

    def __init__(self, cache_path: Optional[Union[str, Path]] = None,
                 base_url: str = OSRM_TABLE_URL, detour_factor: float = 1.3,
                 timeout: float = 60, pause: float = 1.0, retries: int = 3):
        """
        Args:
            cache_path: JSON-кеш расстояний (км) по парам.
            base_url: адрес Table API (можно указать локальный OSRM).
            detour_factor: множитель к прямой для запасного расчёта.
            timeout: таймаут запроса, с.
            pause: пауза между запросами, с.
            retries: число попыток.
        """
        self.base_url = base_url
        self.detour_factor = detour_factor
        self.timeout = timeout
        self.pause = pause
        self.retries = retries
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(
                    self.cache_path.read_text(encoding="utf-8"))
            except ValueError:
                logger.warning("Кеш OSRM-расстояний повреждён, начат новый")
        self.stats = {"pairs_road": 0, "pairs_cache": 0,
                      "pairs_fallback": 0, "requests": 0}
        self._last_request = 0.0
        self._offline = False

    _key = staticmethod(RoadDistance._key)

    def save_cache(self) -> None:
        """Сохраняет кеш на диск."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache),
                                       encoding="utf-8")

    def _request(self, lat: np.ndarray, lon: np.ndarray):
        """Запрос матрицы расстояний, км; None — если не получена."""
        import time

        import requests

        if self._offline:
            return None
        url = self.base_url + ";".join(f"{b:.6f},{a:.6f}"
                                       for a, b in zip(lat, lon))
        for attempt in range(self.retries):
            wait = self.pause - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            self.stats["requests"] += 1
            try:
                data = requests.get(url, params={"annotations": "distance"},
                                    timeout=self.timeout).json()
            except (requests.RequestException, ValueError) as err:
                logger.debug("OSRM table: попытка %d — %s", attempt + 1, err)
                time.sleep(2 * (attempt + 1))
                continue
            if data.get("code") != "Ok":
                logger.warning("OSRM table: %s", data.get("message", data))
                return None
            dist = np.array(data["distances"], dtype=float) / 1000
            return dist  # None в ответе -> nan
        logger.warning("OSRM недоступен — дальше расстояния по прямой "
                       "× %.1f", self.detour_factor)
        self._offline = True
        return None

    def matrix(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Матрица дорожных расстояний, км (несимметрична для one-way).

        Args:
            lat: широты.
            lon: долготы.

        Returns:
            Квадратная матрица (n, n).
        """
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        n = len(lat)
        keys = [[self._key(lat[i], lon[i], lat[j], lon[j]) for j in range(n)]
                for i in range(n)]
        missing = any(keys[i][j] not in self.cache
                      for i in range(n) for j in range(n) if i != j)
        if missing and n <= self.MAX_COORDS:
            fetched = self._request(lat, lon)
            if fetched is not None:
                for i in range(n):
                    for j in range(n):
                        if i != j and np.isfinite(fetched[i, j]):
                            self.cache[keys[i][j]] = float(fetched[i, j])
                            self.stats["pairs_road"] += 1
        straight = haversine_matrix(lat, lon)
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                value = self.cache.get(keys[i][j])
                if value is None:
                    value = straight[i, j] * self.detour_factor
                    self.stats["pairs_fallback"] += 1
                elif not missing:
                    self.stats["pairs_cache"] += 1
                result[i, j] = value
        return result

    def describe(self) -> str:
        """Короткое описание источника и доли запасных расчётов."""
        s = self.stats
        total = s["pairs_road"] + s["pairs_cache"] + s["pairs_fallback"]
        share = s["pairs_fallback"] / total * 100 if total else 0
        return (f"osrm (по дорогам): запросов {s['requests']}, получено пар "
                f"{s['pairs_road']}, из кеша {s['pairs_cache']}, по "
                f"прямой×{self.detour_factor} {s['pairs_fallback']} пар "
                f"({share:.1f}%)")


def load_road_graph(graphml_path: Union[str, Path]):
    """
    Загружает дорожный граф из GraphML.

    Файл строится один раз из PBF (см. ``build_graph_from_pbf``).
    Для графа Приволжского ФО (1,5 ГБ) чтение занимает несколько минут.

    Args:
        graphml_path: путь к GraphML.

    Returns:
        networkx-граф.
    """
    import networkx as nx

    path = Path(graphml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Граф не найден: {path}. Постройте его build_graph_from_pbf() "
            f"или укажите путь через --graphml / GEO_GRAPHML")
    logger.info("Загрузка графа %s (%.0f МБ)...", path,
                path.stat().st_size / 2 ** 20)
    graph = nx.read_graphml(path, edge_key_type=int)
    for _, _, data in graph.edges(data=True):
        data["length"] = float(data.get("length", 0) or 0)
    logger.info("Граф загружен: %d узлов, %d рёбер",
                graph.number_of_nodes(), graph.number_of_edges())
    return graph


def build_graph_from_pbf(pbf_path: Union[str, Path],
                         graphml_path: Union[str, Path],
                         bbox: Optional[List[float]] = None):
    """
    Строит дорожный граф из OSM PBF через pyrosm и сохраняет в GraphML.

    Args:
        pbf_path: путь к .osm.pbf.
        graphml_path: куда сохранить GraphML.
        bbox: [west, south, east, north] — ограничение области.

    Returns:
        networkx-граф.
    """
    import networkx as nx
    from pyrosm import OSM

    osm = OSM(str(pbf_path), bounding_box=bbox)
    nodes, edges = osm.get_network(network_type="driving", nodes=True)
    graph = osm.to_graph(nodes, edges, graph_type="networkx")
    for _, data in graph.nodes(data=True):
        for key in [k for k, v in data.items()
                    if not isinstance(v, (int, float, str, bool))]:
            del data[key]
    for _, _, data in graph.edges(data=True):
        for key in [k for k, v in data.items()
                    if not isinstance(v, (int, float, str, bool))]:
            del data[key]
    nx.write_graphml(graph, graphml_path)
    return graph


def make_distance_provider(backend: str = "haversine",
                           graphml_path: Optional[Union[str, Path]] = None,
                           cache_path: Optional[Union[str, Path]] = None,
                           detour_factor: float = 1.3,
                           osrm_cache_path: Optional[Union[str, Path]] = None):
    """
    Создаёт источник расстояний.

    Если запрошен "road", но граф недоступен, возвращается haversine,
    и об этом явно пишется предупреждение (без скрытого подавления).

    Args:
        backend: "haversine", "osrm" (OSRM Table API) или "road"
            (локальный граф GraphML).
        graphml_path: путь к GraphML для режима road.
        cache_path: путь к JSON-кешу дорожных расстояний.
        detour_factor: коэффициент извилистости для запасного расчёта.
        osrm_cache_path: путь к JSON-кешу для режима osrm.

    Returns:
        HaversineDistance, OsrmDistance или RoadDistance.
    """
    if backend == "haversine":
        return HaversineDistance()
    if backend == "osrm":
        return OsrmDistance(cache_path=osrm_cache_path,
                            detour_factor=detour_factor)
    try:
        graph = load_road_graph(graphml_path)
    except (FileNotFoundError, ImportError) as err:
        logger.warning("Дорожный граф недоступен (%s). Используется "
                       "haversine.", err)
        return HaversineDistance()
    return RoadDistance(graph, cache_path=cache_path,
                        detour_factor=detour_factor)
