"""Общие фикстуры тестов."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def small_points() -> pd.DataFrame:
    """60 точек двух менеджеров вокруг Нижнего Новгорода, частоты 1/2/4."""
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame({
        "point_id": [f"P{i}" for i in range(n)],
        "latitude": 56.3 + rng.normal(0, 0.15, n),
        "longitude": 44.0 + rng.normal(0, 0.25, n),
        "visits_per_month": rng.choice([1, 2, 4], n, p=[0.4, 0.5, 0.1]),
        "manager": rng.choice([0, 1], n),
    })
