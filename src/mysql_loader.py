"""Load the processed tables into MySQL and run the MySQL version of the queries.

This is an optional layer. The analysis itself runs on DuckDB straight over the
parquet files and needs no database at all. MySQL is here because that is where
this data would realistically live in production, and because running the same
five queries on both engines shows the logic ports across dialects.

Start the server first (see docker-compose.yml):

    docker compose up -d
    python -m src.mysql_loader
    docker compose down

Connection settings come from environment variables so this also works against
a MySQL you already have, rather than only the container:

    MYSQL_HOST (127.0.0.1)  MYSQL_PORT (3307)  MYSQL_DB (olist)
    MYSQL_USER (olist)      MYSQL_PASSWORD (olist)
"""
from __future__ import annotations

import os
import re
import sys
import time

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import DECIMAL, Boolean, DateTime, Integer, String

from . import config

SQL_FILE = config.PROJECT_ROOT / "sql" / "analytics_mysql.sql"

# Only the columns the queries actually need. Loading the full item table would
# work but there is no reason to push 112k rows of unused columns over the wire.
ORDER_LEVEL_COLS = [
    "order_id", "customer_unique_id", "customer_state", "order_month",
    "items_value", "freight_value", "is_valid_revenue", "review_score",
    "delivery_delay_days",
]
ORDERS_MASTER_COLS = ["order_id", "category", "item_revenue", "is_valid_revenue"]

# Explicit types: let pandas infer and you get TEXT columns you cannot index,
# and floats where money should be fixed-point.
ORDER_LEVEL_TYPES = {
    "order_id": String(32),
    "customer_unique_id": String(32),
    "customer_state": String(4),
    "order_month": DateTime(),
    "items_value": DECIMAL(12, 2),
    "freight_value": DECIMAL(12, 2),
    "is_valid_revenue": Boolean(),
    "review_score": DECIMAL(4, 2),
    "delivery_delay_days": Integer(),
}
ORDERS_MASTER_TYPES = {
    "order_id": String(32),
    "category": String(64),
    "item_revenue": DECIMAL(12, 2),
    "is_valid_revenue": Boolean(),
}


def _url() -> str:
    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3307")
    db = os.getenv("MYSQL_DB", "olist")
    user = os.getenv("MYSQL_USER", "olist")
    pwd = os.getenv("MYSQL_PASSWORD", "olist")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset=utf8mb4"


def connect(timeout: int = 90) -> Engine:
    """Wait for the server to accept connections, then return an engine.

    A freshly started container refuses connections for a while during first
    initialisation, so this retries rather than failing on the first attempt.
    """
    engine = create_engine(_url(), pool_pre_ping=True, future=True)
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
            return engine
        except Exception as exc:  # noqa: BLE001 - any connection error is a retry
            last = exc
            time.sleep(3)
    raise RuntimeError(
        f"Could not reach MySQL at {os.getenv('MYSQL_HOST', '127.0.0.1')}:"
        f"{os.getenv('MYSQL_PORT', '3307')} after {timeout}s. "
        "Is it running? Try: docker compose up -d\n"
        f"Last error: {last}"
    )


def load_tables(engine: Engine, verbose: bool = True) -> dict[str, int]:
    """Replace the two analysis tables in MySQL with the processed data."""
    if not config.ORDER_LEVEL.exists():
        raise FileNotFoundError(
            "Processed tables missing. Run: python scripts/run_pipeline.py"
        )

    order_level = pd.read_parquet(config.ORDER_LEVEL, columns=ORDER_LEVEL_COLS)
    master = pd.read_parquet(config.ORDERS_MASTER, columns=ORDERS_MASTER_COLS)

    counts = {}
    for name, df, dtypes in (
        ("order_level", order_level, ORDER_LEVEL_TYPES),
        ("orders_master", master, ORDERS_MASTER_TYPES),
    ):
        df.to_sql(name, engine, if_exists="replace", index=False,
                  dtype=dtypes, chunksize=5000, method="multi")
        counts[name] = len(df)
        if verbose:
            print(f"  loaded {name:14s} {len(df):>8,} rows")

    # Indexes on the columns the queries group and filter by.
    with engine.begin() as c:
        for stmt in (
            "CREATE INDEX idx_ol_valid ON order_level (is_valid_revenue)",
            "CREATE INDEX idx_ol_state ON order_level (customer_state)",
            "CREATE INDEX idx_ol_cust ON order_level (customer_unique_id)",
            "CREATE INDEX idx_om_cat ON orders_master (category)",
        ):
            try:
                c.execute(text(stmt))
            except Exception:  # noqa: BLE001 - index already exists, fine
                pass
    if verbose:
        print("  indexes created")
    return counts


def load_queries() -> dict[str, str]:
    """Parse sql/analytics_mysql.sql into {name: sql}."""
    text_ = SQL_FILE.read_text()
    queries: dict[str, str] = {}
    current, buf = None, []
    for line in text_.splitlines():
        m = re.match(r"--\s*name:\s*(\w+)", line)
        if m:
            if current:
                queries[current] = "\n".join(buf).strip()
            current, buf = m.group(1), []
        elif current is not None:
            buf.append(line)
    if current:
        queries[current] = "\n".join(buf).strip()
    return {
        k: "\n".join(l for l in v.splitlines() if not l.strip().startswith("--")).strip()
        for k, v in queries.items()
    }


def run_queries(engine: Engine, names: list[str] | None = None) -> None:
    queries = load_queries()
    for name in names or list(queries):
        print("\n" + "=" * 70)
        print(f"QUERY: {name}   (MySQL)")
        print("=" * 70)
        df = pd.read_sql(text(queries[name].rstrip(";")), engine)
        with pd.option_context("display.max_rows", 30, "display.width", 120):
            print(df.to_string(index=False))


def compare_with_duckdb(engine: Engine) -> bool:
    """Run every query on both engines and check they return the same numbers.

    This is the point of having two dialects: the logic should be identical and
    only the way the data is reached should differ. Rather than assert that,
    this checks it.
    """
    import numpy as np

    from . import sql_runner

    duck_q, my_q = sql_runner.load_queries(), load_queries()
    shared = [k for k in duck_q if k in my_q]

    print("\n" + "=" * 70)
    print(f"CROSS-ENGINE CHECK - DuckDB vs MySQL ({len(shared)} queries)")
    print("=" * 70)

    all_ok = True
    for name in shared:
        d = sql_runner.run_query(name)
        m = pd.read_sql(text(my_q[name].rstrip(";")), engine)

        # monthly_revenue formats the month per dialect on purpose; align it.
        if name == "monthly_revenue":
            d = d.copy()
            d["month"] = pd.to_datetime(d["order_month"]).dt.strftime("%Y-%m")
            d = d.drop(columns=["order_month"])[m.columns]

        if list(d.columns) != list(m.columns) or len(d) != len(m):
            print(f"{name:22s} SHAPE MISMATCH")
            all_ok = False
            continue

        worst, bad = 0.0, []
        for col in d.columns:
            a, b = d[col], m[col]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                diff = np.abs(a.astype(float).values - b.astype(float).values)
                if len(diff):
                    worst = max(worst, float(np.nanmax(diff)))
                    if np.nanmax(diff) > 0.01:
                        bad.append(col)
            elif not (a.astype(str).values == b.astype(str).values).all():
                bad.append(col)

        all_ok &= not bad
        print(f"{name:22s} rows={len(d):<4} max numeric diff={worst:.4f}  "
              f"{'MATCH' if not bad else 'DIFF in ' + str(bad)}")

    print("=" * 70)
    print("All queries agree." if all_ok else "Mismatches found.")
    return all_ok


def run(names: list[str] | None = None, compare: bool = False) -> None:
    print("\n" + "=" * 70)
    print("MySQL LAYER")
    print("=" * 70)
    engine = connect()
    with engine.connect() as c:
        version = c.execute(text("SELECT VERSION()")).scalar()
    print(f"connected to MySQL {version}")
    load_tables(engine)

    if compare:
        compare_with_duckdb(engine)
        return

    run_queries(engine, names)
    print("\n" + "=" * 70)
    print("Same metrics as the DuckDB layer (sql/analytics.sql). To verify that")
    print("claim rather than take it on trust: python -m src.mysql_loader --compare")
    print("=" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    run([a for a in args if not a.startswith("--")] or None, compare="--compare" in args)
