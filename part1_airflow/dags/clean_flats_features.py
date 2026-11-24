"""Clean destination_db.flats_features and save the result in flats_cleaned."""

from datetime import datetime, timezone
import os
import sys
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.postgres.hooks.postgres import PostgresHook

CONN_ID = os.getenv("AIRFLOW_DESTINATION_DB_CONN_ID", "destination_db")
DB_SCHEMA = os.getenv("ML_DB_SCHEMA", "public")
SOURCE_TABLE = os.getenv("FLATS_FEATURES_TABLE", "flats_features")
TARGET_TABLE = os.getenv("FLATS_CLEANED_TABLE", "flats_cleaned")

# Import the helper module from this repository's part1_airflow/plugins folder.
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugins"
sys.path.insert(0, str(PLUGIN_DIR))
from db_utils import (  # noqa: E402
    dataframe_records,
    qualified_table,
    staging_table_name,
)
from cleaning import clean_flats  # noqa: E402

SOURCE_SQL = qualified_table(DB_SCHEMA, SOURCE_TABLE)
TARGET_SQL = qualified_table(DB_SCHEMA, TARGET_TABLE)


@dag(
    dag_id="clean_flats_features",
    description="Remove duplicates, invalid values and missing values",
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    schedule_interval=None,
    catchup=False,
    tags=["mle-project", "data-cleaning"],
)
def clean_flats_features_dag():
    @task(task_id="create_table")
    def create_table() -> str:
        context = get_current_context()
        stage_table = staging_table_name(
            f"{TARGET_TABLE}_stage", context["run_id"]
        )
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        hook.run(
            f"CREATE TABLE IF NOT EXISTS {TARGET_SQL} "
            f"(LIKE {SOURCE_SQL} INCLUDING ALL)",
            autocommit=True,
        )
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        hook.run(
            f"CREATE TABLE IF NOT EXISTS {stage_sql} "
            f"(LIKE {SOURCE_SQL} INCLUDING ALL)",
            autocommit=True,
        )
        return stage_table

    @task(task_id="extract")
    def extract(stage_table: str):
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        frame = hook.get_pandas_df(f"SELECT * FROM {SOURCE_SQL}")
        if frame.empty:
            raise ValueError(f"Source table {SOURCE_SQL} is empty")
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        hook.run(f"TRUNCATE TABLE {stage_sql}", autocommit=True)
        hook.insert_rows(
            table=stage_sql,
            rows=dataframe_records(frame),
            target_fields=list(frame.columns),
            commit_every=1000,
        )
        print(f"Copied {len(frame):,} source rows to {stage_sql}")

    @task(task_id="transform")
    def transform(stage_table: str):
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        frame = hook.get_pandas_df(f"SELECT * FROM {stage_sql}")
        cleaned = clean_flats(frame)
        if cleaned.empty:
            raise ValueError("Cleaning removed every row from the dataset")
        hook.run(f"TRUNCATE TABLE {stage_sql}", autocommit=True)
        hook.insert_rows(
            table=stage_sql,
            rows=dataframe_records(cleaned),
            target_fields=list(cleaned.columns),
            commit_every=1000,
        )
        print(f"Cleaned dataset contains {len(cleaned):,} rows")

    @task(task_id="load")
    def load(stage_table: str):
        hook = PostgresHook(postgres_conn_id=CONN_ID)
        stage_sql = qualified_table(DB_SCHEMA, stage_table)
        conn = hook.get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"TRUNCATE TABLE {TARGET_SQL}")
                cursor.execute(
                    f"INSERT INTO {TARGET_SQL} SELECT * FROM {stage_sql}"
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
        hook = PostgresHook(postgres_conn_id=CONN_ID)
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


clean_flats_features_dag()
