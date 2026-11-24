"""Build a joined dataset from destination_db source tables."""

from datetime import datetime, timezone
import os
import re
import sys
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.postgres.hooks.postgres import PostgresHook

SOURCE_CONN_ID = os.getenv("AIRFLOW_SOURCE_DB_CONN_ID", "destination_db")
DESTINATION_CONN_ID = os.getenv("AIRFLOW_DESTINATION_DB_CONN_ID", "destination_db")
DB_SCHEMA = os.getenv("ML_DB_SCHEMA", "public")
TARGET_TABLE = os.getenv("FLATS_FEATURES_TABLE", "flats_features")
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
sys.path.insert(0, str(PLUGIN_DIR))
from db_utils import (  # noqa: E402
    dataframe_records,
    qualified_table,
    quote_identifier,
    staging_table_name,
)


TARGET_SQL = qualified_table(DB_SCHEMA, TARGET_TABLE)
TARGET_COLUMNS = [
    "flat_id",
    "building_id",
    "build_year",
    "building_type_int",
    "latitude",
    "longitude",
    "ceiling_height",
    "flats_count",
    "floors_total",
    "has_elevator",
    "floor",
    "kitchen_area",
    "living_area",
    "rooms",
    "is_apartment",
    "studio",
    "total_area",
    "price",
    "price_per_sqm",
]

SOURCE_COLUMN_MAP = {
    "flat_id": ("flats", "id"),
    "building_id": ("flats", "building_id"),
    "build_year": ("buildings", "build_year"),
    "building_type_int": ("buildings", "building_type_int"),
    "latitude": ("buildings", "latitude"),
    "longitude": ("buildings", "longitude"),
    "ceiling_height": ("buildings", "ceiling_height"),
    "flats_count": ("buildings", "flats_count"),
    "floors_total": ("buildings", "floors_total"),
    "has_elevator": ("buildings", "has_elevator"),
    "floor": ("flats", "floor"),
    "kitchen_area": ("flats", "kitchen_area"),
    "living_area": ("flats", "living_area"),
    "rooms": ("flats", "rooms"),
    "is_apartment": ("flats", "is_apartment"),
    "studio": ("flats", "studio"),
    "total_area": ("flats", "total_area"),
    "price": ("flats", "price"),
}

SELECT_JOINED_DATA = f"""
SELECT
    f.id AS flat_id,
    f.building_id,
    b.build_year,
    b.building_type_int,
    b.latitude,
    b.longitude,
    b.ceiling_height,
    b.flats_count,
    b.floors_total,
    b.has_elevator,
    f.floor,
    f.kitchen_area,
    f.living_area,
    f.rooms,
    f.is_apartment,
    f.studio,
    f.total_area,
    f.price,
    CASE
        WHEN f.total_area > 0
            THEN f.price::numeric / NULLIF(f.total_area::numeric, 0)
        ELSE NULL
    END AS price_per_sqm
FROM {qualified_table(DB_SCHEMA, "flats")} AS f
INNER JOIN {qualified_table(DB_SCHEMA, "buildings")} AS b
    ON b.id = f.building_id
"""


def _postgres_type(data_type, character_length, numeric_precision, numeric_scale):
    simple_types = {
        "smallint": "SMALLINT",
        "integer": "INTEGER",
        "bigint": "BIGINT",
        "real": "REAL",
        "double precision": "DOUBLE PRECISION",
        "boolean": "BOOLEAN",
        "text": "TEXT",
        "date": "DATE",
        "timestamp without time zone": "TIMESTAMP",
        "timestamp with time zone": "TIMESTAMPTZ",
        "time without time zone": "TIME",
        "time with time zone": "TIMETZ",
        "uuid": "UUID",
        "json": "JSON",
        "jsonb": "JSONB",
        "bytea": "BYTEA",
    }
    if data_type in simple_types:
        return simple_types[data_type]
    if data_type == "numeric":
        if numeric_precision is None:
            return "NUMERIC"
        return f"NUMERIC({numeric_precision},{numeric_scale or 0})"
    if data_type == "character varying":
        return "VARCHAR" if character_length is None else f"VARCHAR({character_length})"
    if data_type == "character":
        return "CHAR" if character_length is None else f"CHAR({character_length})"
    raise ValueError(f"Unsupported PostgreSQL source column type: {data_type!r}")


def _source_column_types(source_hook):
    rows = source_hook.get_records(
        """
        SELECT table_name, column_name, data_type,
               character_maximum_length, numeric_precision, numeric_scale
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = ANY(%s)
        """,
        parameters=(DB_SCHEMA, ["flats", "buildings"]),
    )
    found = {
        (table, column): _postgres_type(
            data_type, char_length, precision, scale
        )
        for table, column, data_type, char_length, precision, scale in rows
    }
    missing = sorted(set(SOURCE_COLUMN_MAP.values()) - set(found))
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")
    return found


@dag(
    dag_id="build_flats_features",
    description="Join source flats and buildings into destination_db",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule_interval=None,
    catchup=False,
    tags=["mle-project", "data-collection"],
)
def build_flats_features_dag():
    @task(task_id="create_table")
    def create_table() -> str:
        context = get_current_context()
        stage_table = staging_table_name(
            f"{TARGET_TABLE}_stage", context["run_id"]
        )
        source_hook = PostgresHook(postgres_conn_id=SOURCE_CONN_ID)
        destination_hook = PostgresHook(postgres_conn_id=DESTINATION_CONN_ID)
        source_types = _source_column_types(source_hook)
        definitions = [
            f'{quote_identifier(column)} '
            f'{source_types[SOURCE_COLUMN_MAP[column]]}'
            for column in TARGET_COLUMNS
            if column != "price_per_sqm"
        ]
        definitions.append('"price_per_sqm" NUMERIC')
        destination_hook.run(
            f"CREATE TABLE IF NOT EXISTS {TARGET_SQL} "
            f"({', '.join(definitions)})",
            autocommit=True,
        )
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        destination_hook.run(
            f"CREATE TABLE IF NOT EXISTS {stage_sql} "
            f"(LIKE {TARGET_SQL} INCLUDING ALL)",
            autocommit=True,
        )
        return stage_table

    @task(task_id="extract")
    def extract(stage_table: str):
        source_hook = PostgresHook(postgres_conn_id=SOURCE_CONN_ID)
        destination_hook = PostgresHook(postgres_conn_id=DESTINATION_CONN_ID)
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        destination_hook.run(f"TRUNCATE TABLE {stage_sql}", autocommit=True)

        source_conn = source_hook.get_conn()
        rows_copied = 0
        cursor_name = staging_table_name("source_cursor", stage_table)
        try:
            with source_conn.cursor(name=cursor_name) as cursor:
                cursor.itersize = 1000
                cursor.execute(SELECT_JOINED_DATA)
                while True:
                    batch = cursor.fetchmany(1000)
                    if not batch:
                        break
                    destination_hook.insert_rows(
                        table=stage_sql,
                        rows=batch,
                        target_fields=TARGET_COLUMNS,
                        commit_every=len(batch),
                    )
                    rows_copied += len(batch)
        finally:
            source_conn.close()

        if rows_copied == 0:
            raise ValueError("The source join returned no apartment rows")
        print(f"Copied {rows_copied:,} rows to {stage_sql}")

    @task(task_id="transform")
    def transform(stage_table: str):
        hook = PostgresHook(postgres_conn_id=DESTINATION_CONN_ID)
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        hook.run(
            f"""
            DELETE FROM {stage_sql} AS old_row
            USING {stage_sql} AS newer_row
            WHERE old_row.flat_id = newer_row.flat_id
              AND old_row.ctid < newer_row.ctid
            """,
            autocommit=True,
        )
        hook.run(f"ANALYZE {stage_sql}", autocommit=True)

    @task(task_id="load")
    def load(stage_table: str):
        hook = PostgresHook(postgres_conn_id=DESTINATION_CONN_ID)
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        columns_sql = ", ".join(f'"{column}"' for column in TARGET_COLUMNS)
        conn = hook.get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {TARGET_SQL}")
                cursor.execute(
                    f"INSERT INTO {TARGET_SQL} ({columns_sql}) "
                    f"SELECT {columns_sql} FROM {stage_sql}"
                )
                cursor.execute(f"ANALYZE {TARGET_SQL}")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @task(task_id="cleanup_stage")
    def cleanup_stage(stage_table: str):
        hook = PostgresHook(postgres_conn_id=DESTINATION_CONN_ID)
        hook.run(
            f"DROP TABLE IF EXISTS {qualified_table(DB_SCHEMA, stage_table)}",
            autocommit=True,
        )

    stage = create_table()
    extracted = extract(stage)
    transformed = transform(stage)
    loaded = load(stage)
    cleaned = cleanup_stage(stage)
    extracted >> transformed >> loaded >> cleaned


build_flats_features_dag()
