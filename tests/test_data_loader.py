"""Тесты загрузки и валидации данных."""

import pandas as pd
import pytest

from src import load_and_validate_data


def make_raw():
    """Набор с корректными и повреждёнными строками."""
    return pd.DataFrame({
        "point_id": ["A", "B", "C", "D", "E", "F", "G", "A", None, "H"],
        "lat": [56.3, 95.0, "abc", 56.1, 56.2, 56.2, 0, 56.3, 56.0, 56.25],
        "lon": [44.0, 44.0, 44.0, 44.1, 44.2, 44.3, 0, 44.0, 44.0, None],
        "n_visits": [2, 1, 1, 1.5, 5, 1, 1, 1, 1, 1],
        "manager": [0, 0, 1, 1, 1, None, 1, 0, 0, 0],
    })


def test_rejects_with_reasons_and_does_not_clip():
    """Некорректные строки уходят в rejected с причиной; частота не режется."""
    points, rejected = load_and_validate_data(make_raw(),
                                              allowed_visits=(1, 2, 4))
    assert set(points["point_id"]) == {"A", "F"}
    reasons = dict(zip(rejected["point_id"].astype(str), rejected["reason"]))
    assert "latitude" in reasons["B"]
    assert "не является числом" in reasons["C"]
    assert "не целое" in reasons["D"]            # 1.5 -> не TypeError
    assert "допустимого списка" in reasons["E"]  # 5 -> не clip до 2
    assert "(0, 0)" in reasons["G"]
    assert "longitude" in reasons["H"]
    assert "дубликат" in rejected.loc[rejected["source_row"] == 9,
                                      "reason"].iloc[0]
    assert len(points) + len(rejected) == len(make_raw())


def test_missing_manager_gets_default():
    """Пустой manager и отсутствующая колонка -> менеджер по умолчанию."""
    points, _ = load_and_validate_data(make_raw(), (1, 2, 4),
                                       default_manager=7)
    assert points.loc[points["point_id"] == "F", "manager"].item() == 7
    raw = make_raw().drop(columns="manager")
    points, _ = load_and_validate_data(raw, (1, 2, 4), default_manager=3)
    assert (points["manager"] == 3).all()


def test_missing_required_column():
    """Без обязательной колонки — понятная ошибка."""
    with pytest.raises(ValueError, match="обязательные колонки"):
        load_and_validate_data(pd.DataFrame({"point_id": ["A"]}))
