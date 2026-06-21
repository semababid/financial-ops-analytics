"""Run the analytical SQL in sql/analytics.sql against the parquet tables.

Demonstrates the same metrics as the Python analysis, expressed in SQL via
DuckDB (which queries parquet files directly — no database server needed).

Run:  python -m src.sql_runner            # run all queries
      python -m src.sql_runner top_categories   # run one by name
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

from . import config

SQL_FILE = config.PROJECT_ROOT / "sql" / "analytics.sql"


def _parquet(path: Path) -> str:
    # DuckDB reads a parquet file as a table-valued function.
    return f"read_parquet('{path.as_posix()}')"


def load_queries() -> dict[str, str]:
    """Parse sql/analytics.sql into {name: sql}, splitting on '-- name:' headers."""
    text = SQL_FILE.read_text()
    bindings = {"order_level": _parquet(config.ORDER_LEVEL),
                "orders_master": _parquet(config.ORDERS_MASTER)}

    queries: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            if current:
                queries[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current:
        queries[current] = "\n".join(buf).strip()

    # Strip remaining comment-only lines and fill table placeholders.
    out = {}
    for name, sql in queries.items():
        sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
        out[name] = sql.strip().format(**bindings)
    return out


def run_query(name: str) -> pd.DataFrame:
    sql = load_queries()[name]
    return duckdb.sql(sql).df()


def run(names: list[str] | None = None) -> None:
    if not config.ORDER_LEVEL.exists():
        raise FileNotFoundError("Processed tables missing. Run: python -m src.data_cleaning")
    queries = load_queries()
    selected = names or list(queries)
    for name in selected:
        print("\n" + "=" * 70)
        print(f"QUERY: {name}")
        print("=" * 70)
        df = duckdb.sql(queries[name]).df()
        with pd.option_context("display.max_rows", 30, "display.width", 120):
            print(df.to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1:] or None)
