"""
Загрузка и валидация входного CSV.

Ни одна строка не исправляется молча: некорректные строки попадают в
отдельную таблицу ``rejected`` с причиной отклонения.
"""

import logging
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("geoplan")

REQUIRED_COLUMNS = ["point_id", "latitude", "longitude", "visits_per_month"]
COLUMN_ALIASES = {
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "n_visits": "visits_per_month",
}


def load_and_validate_data(
        source: Union[str, Path, pd.DataFrame],
        allowed_visits: Iterable[int] = (1, 2, 3, 4),
        default_manager: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает точки и отделяет корректные строки от некорректных.

    Поддерживаются альтернативные имена колонок: lat, lon/lng, n_visits.
    Колонка ``manager`` необязательна: если её нет, всем точкам
    назначается ``default_manager``; пустые значения в ней — тоже.

    Строка отклоняется, если:
        * отсутствует point_id, координата или visits_per_month;
        * координата не число или вне диапазона (|lat| <= 90, |lon| <= 180);
        * координаты (0, 0) — типичный признак незаполненного поля;
        * visits_per_month не целое или не входит в ``allowed_visits``;
        * point_id повторяется (сохраняется первое вхождение).

    Args:
        source: путь к CSV или готовый DataFrame.
        allowed_visits: допустимые значения visits_per_month.
        default_manager: менеджер по умолчанию.

    Returns:
        (points, rejected): корректные точки и отклонённые строки
        с колонкой ``reason``.
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        raw = pd.read_csv(path, dtype={"point_id": str})

    rename = {old: new for old, new in COLUMN_ALIASES.items()
              if old in raw.columns and new not in raw.columns}
    if rename:
        raw = raw.rename(columns=rename)
        logger.info("Переименованы колонки: %s", rename)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

    raw = raw.reset_index(drop=True)
    raw["source_row"] = raw.index + 2  # номер строки в CSV (с заголовком)
    df = raw.copy()
    reasons = pd.Series("", index=df.index, dtype=object)

    def reject(mask: pd.Series, reason: str) -> None:
        """Помечает строки причиной, если у них ещё нет причины."""
        mask = mask & (reasons == "")
        reasons[mask] = reason

    pid = df["point_id"].astype("string").str.strip()
    reject(pid.isna() | (pid == ""), "пустой point_id")
    df["point_id"] = pid

    for col in ("latitude", "longitude"):
        reject(df[col].isna(), f"пустое значение {col}")
        num = pd.to_numeric(df[col], errors="coerce")
        reject(num.isna(), f"{col} не является числом")
        df[col] = num
    reject(~df["latitude"].between(-90, 90), "latitude вне [-90, 90]")
    reject(~df["longitude"].between(-180, 180), "longitude вне [-180, 180]")
    reject((df["latitude"] == 0) & (df["longitude"] == 0),
           "координаты (0, 0)")

    visits = pd.to_numeric(df["visits_per_month"], errors="coerce")
    reject(df["visits_per_month"].isna(), "пустое значение visits_per_month")
    reject(visits.isna(), "visits_per_month не является числом")
    reject(visits.notna() & (visits % 1 != 0),
           "visits_per_month не целое число")
    allowed = sorted(set(int(v) for v in allowed_visits))
    reject(visits.notna() & ~visits.isin(allowed),
           f"visits_per_month вне допустимого списка {allowed}")

    reject(df["point_id"].duplicated(keep="first") & df["point_id"].notna(),
           "дубликат point_id")

    if "manager" not in df.columns:
        if default_manager is None:
            raise ValueError("Нет колонки manager и не задан default_manager")
        logger.warning("Колонка manager отсутствует — всем точкам назначен "
                       "менеджер %s", default_manager)
        df["manager"] = default_manager
    else:
        manager = pd.to_numeric(df["manager"], errors="coerce")
        empty = manager.isna()
        if empty.any():
            if default_manager is None:
                reject(empty, "пустой manager")
            else:
                logger.warning("%d строк без manager — назначен %s",
                               int(empty.sum()), default_manager)
                manager = manager.fillna(default_manager)
        reject(manager.notna() & (manager % 1 != 0), "manager не целое число")
        df["manager"] = manager

    bad = reasons != ""
    rejected = raw.loc[bad].copy()
    rejected["reason"] = reasons[bad]

    points = df.loc[~bad, ["point_id", "latitude", "longitude",
                           "visits_per_month", "manager"]].copy()
    points["visits_per_month"] = points["visits_per_month"].astype(float).astype(int)
    points["manager"] = points["manager"].astype(float).astype(int)
    points["latitude"] = points["latitude"].astype(float)
    points["longitude"] = points["longitude"].astype(float)
    points["point_id"] = points["point_id"].astype(str)
    points = points.reset_index(drop=True)

    logger.info("Загружено строк: %d, корректных: %d, отклонено: %d",
                len(raw), len(points), len(rejected))
    if len(rejected):
        for reason, cnt in rejected["reason"].value_counts().items():
            logger.warning("  отклонено %d: %s", cnt, reason)
    return points, rejected.reset_index(drop=True)


def summarize_points(points: pd.DataFrame) -> pd.DataFrame:
    """
    Сводка потребности по менеджерам: точки и требуемые визиты.

    Args:
        points: валидные точки.

    Returns:
        DataFrame: manager, points, visits_required, и по колонке на
        каждую частоту visits_per_month.
    """
    freq = pd.crosstab(points["manager"], points["visits_per_month"])
    freq.columns = [f"points_{c}x" for c in freq.columns]
    summary = points.groupby("manager").agg(
        points=("point_id", "count"),
        visits_required=("visits_per_month", "sum"),
    )
    return summary.join(freq).reset_index().astype(np.int64)
