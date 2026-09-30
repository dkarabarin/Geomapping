"""
Геометрия маршрутов по дорогам для карты (OSRM).

Для каждого дня менеджера отправляется один запрос с цепочкой точек
старт -> точки в порядке order_in_route -> старт. Ответ разбивается на
участки (legs) между соседними точками, поэтому на карте видно, какие
участки — обход кластера, а какие — подъезд и переезд.

Ответы кешируются в JSON: повторный запуск не обращается к сети.
Если сервер недоступен, участок рисуется прямой, и это учитывается
в статистике (``stats["fallback"]``) и в легенде карты.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("geoplan")

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"


class OsrmRouter:
    """Клиент OSRM с кешем и запасным вариантом «по прямой»."""

    def __init__(self, cache_path: Optional[Union[str, Path]] = None,
                 base_url: str = OSRM_URL, timeout: float = 30,
                 pause: float = 1.0, retries: int = 3):
        """
        Args:
            cache_path: JSON-кеш ответов.
            base_url: адрес сервиса (можно указать локальный OSRM).
            timeout: таймаут запроса, с.
            pause: пауза между запросами к публичному серверу, с.
            retries: число попыток.
        """
        self.base_url = base_url
        self.timeout = timeout
        self.pause = pause
        self.retries = retries
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, List[dict]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(
                    self.cache_path.read_text(encoding="utf-8"))
            except ValueError:
                logger.warning("Кеш OSRM повреждён, начат новый")
        self.stats = {"requests": 0, "cache": 0, "fallback": 0}
        self._last_request = 0.0

    @staticmethod
    def _key(coords: Sequence[Tuple[float, float]]) -> str:
        """Ключ кеша: хеш списка координат."""
        text = "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in coords)
        return hashlib.md5(text.encode()).hexdigest()

    def save_cache(self) -> None:
        """Сохраняет кеш на диск."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache),
                                       encoding="utf-8")

    def _request(self, coords) -> Optional[List[dict]]:
        """Запрос к OSRM; None — если ответ не получен."""
        import requests

        url = self.base_url + ";".join(f"{lon:.6f},{lat:.6f}"
                                       for lat, lon in coords)
        # continue_straight=false разрешает разворот в точке визита —
        # так же, как считает матрица Table API, по которой строился план
        params = {"overview": "false", "steps": "true",
                  "geometries": "geojson", "continue_straight": "false"}
        for attempt in range(self.retries):
            wait = self.pause - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.time()
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                data = resp.json()
            except (requests.RequestException, ValueError) as err:
                logger.debug("OSRM: попытка %d — %s", attempt + 1, err)
                time.sleep(2 * (attempt + 1))
                continue
            if data.get("code") != "Ok":
                logger.debug("OSRM: %s", data.get("message", data))
                return None
            legs = []
            for leg in data["routes"][0]["legs"]:
                geometry = []
                for step in leg["steps"]:
                    for lon, lat in step["geometry"]["coordinates"]:
                        if not geometry or geometry[-1] != [lat, lon]:
                            geometry.append([lat, lon])
                legs.append({"geometry": geometry,
                             "km": leg["distance"] / 1000,
                             "min": leg["duration"] / 60})
            return legs
        return None

    def legs(self, coords: Sequence[Tuple[float, float]]) -> List[dict]:
        """
        Участки маршрута по дорогам между соседними точками.

        Args:
            coords: список (lat, lon) в порядке обхода.

        Returns:
            Список из len(coords) - 1 словарей: geometry (список [lat, lon]),
            km, min, road (True — по дорогам, False — прямая-заглушка).
        """
        coords = [(float(a), float(b)) for a, b in coords]
        key = self._key(coords)
        if key in self.cache:
            self.stats["cache"] += 1
            return [dict(leg, road=True) for leg in self.cache[key]]
        self.stats["requests"] += 1
        legs = self._request(coords)
        if legs is None or len(legs) != len(coords) - 1:
            self.stats["fallback"] += 1
            return [{"geometry": [list(a), list(b)], "km": None,
                     "min": None, "road": False}
                    for a, b in zip(coords[:-1], coords[1:])]
        self.cache[key] = legs
        return [dict(leg, road=True) for leg in legs]

    def describe(self) -> str:
        """Статистика для лога."""
        s = self.stats
        return (f"OSRM: запросов {s['requests']}, из кеша {s['cache']}, "
                f"не получено (прямые) {s['fallback']}")
