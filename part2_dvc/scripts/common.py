"""Shared utilities for the DVC training pipeline."""

from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

TARGET_COLUMN = "price"
ID_COLUMNS = ["flat_id", "id", "building_id"]
LEAKAGE_COLUMNS = ["price_per_sqm"]


def prepare_catboost_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    forced_categorical = {
        "building_type_int",
        "has_elevator",
        "is_apartment",
        "studio",
    }
    categorical_columns = [
        column
        for column in frame.columns
        if frame[column].dtype == "object"
        or str(frame[column].dtype) in {"category", "bool", "boolean"}
        or column in forced_categorical
    ]

    prepared = frame.copy()
    for column in categorical_columns:
        values = prepared[column].astype(object)
        prepared[column] = values.where(values.notna(), "__missing__").astype(str)
    return prepared, categorical_columns


def load_params(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - clearer environment error
        raise RuntimeError("Install PyYAML to read params.yaml: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file) or {}
    return params


def load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    frame = pd.read_csv(file_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} not found in {file_path}")
    return frame


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split predictors and target, removing identifiers and target leakage."""
    data = frame.copy()
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    valid_target = y.notna() & (y > 0)
    data = data.loc[valid_target].copy()
    y = y.loc[valid_target].astype(float)
    drop_columns = [TARGET_COLUMN] + [
        name for name in ID_COLUMNS + LEAKAGE_COLUMNS if name in data.columns
    ]
    X = data.drop(columns=drop_columns, errors="ignore")
    if X.empty:
        raise ValueError("No predictor columns remain after removing target and IDs")
    return X, y


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
