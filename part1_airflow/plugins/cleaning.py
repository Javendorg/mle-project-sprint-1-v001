"""Reusable data-quality checks and cleaning functions for the project."""

from __future__ import annotations

import numpy as np
import pandas as pd

IDENTIFIER_COLUMNS = {"flat_id", "id", "building_id"}
TARGET_COLUMN = "price"


def diagnose_data(data: pd.DataFrame) -> dict:
    """Return a compact data-quality report for a notebook or a script."""
    report = {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_by_column": data.isna().sum().astype(int).to_dict(),
    }
    if "flat_id" in data.columns:
        report["duplicate_flat_ids"] = int(data["flat_id"].duplicated().sum())
    elif "id" in data.columns:
        report["duplicate_flat_ids"] = int(data["id"].duplicated().sum())
    return report


def clean_flats(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the joined flats dataset while retaining model-ready columns.

    Rows with a missing/non-positive target or non-positive total area are
    removed. Implausible floor/area combinations are rejected. Extreme
    price-per-square-metre values are filtered using the 1st/99th percentiles.
    Remaining predictor missing values are imputed with median/mode values.
    Identifiers and the derived price_per_sqm column are not imputed.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    df = data.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    id_column = "flat_id" if "flat_id" in df.columns else ("id" if "id" in df.columns else None)
    if id_column:
        df = df.drop_duplicates(subset=[id_column], keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    required = {TARGET_COLUMN, "total_area"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Required columns are missing: {sorted(missing_required)}")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df["total_area"] = pd.to_numeric(df["total_area"], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN, "total_area"])
    df = df[(df[TARGET_COLUMN] > 0) & (df["total_area"] > 0)]

    # Convert known numeric columns before physical plausibility checks.
    for column in ["floor", "floors_total", "kitchen_area", "living_area"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "floor" in df.columns and "floors_total" in df.columns:
        valid_floor = (
            df["floor"].isna()
            | df["floors_total"].isna()
            | ((df["floor"] >= 1) & (df["floors_total"] >= df["floor"]))
        )
        df = df[valid_floor]

    for area_column in ("kitchen_area", "living_area"):
        if area_column in df.columns:
            invalid_area = (df[area_column] < 0) | (df[area_column] > df["total_area"])
            df.loc[invalid_area, area_column] = np.nan

    # Derived only for quality control; never use it as a model feature.
    df["price_per_sqm"] = df[TARGET_COLUMN] / df["total_area"]
    price_per_sqm = df["price_per_sqm"].replace([np.inf, -np.inf], np.nan)
    valid_values = price_per_sqm.dropna()
    if len(valid_values) >= 20:
        low, high = valid_values.quantile([0.01, 0.99])
        df = df[df["price_per_sqm"].between(low, high)]

    # Fill missing predictor values. Identifiers and target are left untouched.
    for column in df.columns:
        if column in IDENTIFIER_COLUMNS or column in {TARGET_COLUMN, "price_per_sqm"}:
            continue
        if pd.api.types.is_bool_dtype(df[column]):
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(bool(mode.iloc[0]))
        elif pd.api.types.is_numeric_dtype(df[column]):
            median = df[column].median()
            if pd.notna(median):
                df[column] = df[column].fillna(median)
        else:
            mode = df[column].mode(dropna=True)
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    return df.reset_index(drop=True)
