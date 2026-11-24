"""Small helpers shared by the Airflow DAGs."""

import hashlib
import re

import pandas as pd


def quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return f'"{value}"'


def qualified_table(schema: str, table: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(table)}"


def staging_table_name(prefix: str, run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def dataframe_records(frame: pd.DataFrame):
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append(None)
            elif hasattr(value, "item"):
                values.append(value.item())
            else:
                values.append(value)
        yield tuple(values)
