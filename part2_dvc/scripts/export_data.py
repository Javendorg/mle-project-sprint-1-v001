"""Export the cleaned PostgreSQL table to a CSV file for DVC.

Required environment variables: PGHOST, PGDATABASE, PGUSER, PGPASSWORD.
Optional: PGPORT (default 5432), PGSSLMODE, PGSCHEMA (default public).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2 import sql


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="flats_cleaned", help="Table name, optionally schema.table")
    parser.add_argument("--output", default="data/raw/flats_cleaned.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if "." in args.table:
        schema, table = args.table.split(".", maxsplit=1)
    else:
        schema, table = os.getenv("PGSCHEMA", "public"), args.table

    for part in (schema, table):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise ValueError(f"Unsafe PostgreSQL identifier: {part!r}")

    required = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise EnvironmentError(
            "Missing PostgreSQL environment variables: " + ", ".join(missing)
        )

    connection = psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )
    try:
        query = sql.SQL("SELECT * FROM {}.{}").format(
            sql.Identifier(schema), sql.Identifier(table)
        )
        frame = pd.read_sql_query(query.as_string(connection), connection)
    finally:
        connection.close()

    if frame.empty:
        raise ValueError(f"The source table {schema}.{table} is empty")
    if "price" not in frame.columns:
        raise ValueError("The exported table does not contain the target column 'price'")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print(f"Exported {len(frame):,} rows and {len(frame.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
