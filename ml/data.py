"""
Transaction loading for the churn pipeline.

Two sources, one dataframe contract:

  * ``postgres`` (default) reads ``staging.sales_valid`` - the validated layer
    the Airflow DAG produces. This is the production path.
  * ``landing`` reads ``data/landing/*.csv`` and applies the same validation
    rules as ``scripts/validate.py`` in-process. This exists so training,
    tests and the monitoring job can run reproducibly without a live
    warehouse (CI, a fresh clone, or a laptop with Docker stopped). It is a
    convenience for development, not a second definition of "clean data" -
    the rules below mirror the ones documented in docs/data_dictionary.md.

Both return columns: customer_id, invoice_no, invoice_date, revenue, quantity,
stock_code, country.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_ROOT / "data" / "landing"

COLUMNS = [
    "customer_id",
    "invoice_no",
    "invoice_date",
    "revenue",
    "quantity",
    "stock_code",
    "country",
]

STAGING_QUERY = """
    SELECT customer_id, invoice_no, invoice_date, revenue,
           quantity, stock_code, country
    FROM staging.sales_valid
    WHERE customer_id IS NOT NULL
"""


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce the shared contract columns to their expected dtypes."""
    frame = frame.copy()
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    frame["revenue"] = pd.to_numeric(frame["revenue"], errors="coerce").astype(float)
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["customer_id"] = pd.to_numeric(frame["customer_id"], errors="coerce")
    frame = frame.dropna(subset=["customer_id", "invoice_date", "revenue"])
    frame["customer_id"] = frame["customer_id"].astype(int)
    frame["invoice_no"] = frame["invoice_no"].astype(str)
    return frame[COLUMNS].reset_index(drop=True)


def load_from_postgres() -> pd.DataFrame:
    """Read the validated staging table through the shared engine helper."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from scripts.db import get_engine  # noqa: PLC0415 - optional dependency path

    return _normalise(pd.read_sql(STAGING_QUERY, get_engine()))


def load_from_landing(landing_dir: Path | None = None) -> pd.DataFrame:
    """Read the monthly landing CSVs and apply the staging validation rules.

    Mirrors scripts/validate.py: cancelled orders (invoice_no starting 'C'),
    non-positive or unparseable quantity and unit_price, and missing
    customer_id are all dropped, and revenue is derived as quantity * price.
    """
    directory = landing_dir or LANDING_DIR
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no landing CSVs in {directory} - run scripts/split_landing_files.py first"
        )

    frames = []
    for path in files:
        chunk = pd.read_csv(path, dtype=str)
        quantity = pd.to_numeric(chunk["quantity"], errors="coerce")
        unit_price = pd.to_numeric(chunk["unit_price"], errors="coerce")
        customer_id = pd.to_numeric(chunk["customer_id"], errors="coerce")
        invoice_date = pd.to_datetime(chunk["invoice_date"], errors="coerce")

        keep = (
            ~chunk["invoice_no"].astype(str).str.startswith("C", na=False)
            & quantity.notna()
            & (quantity > 0)
            & unit_price.notna()
            & (unit_price > 0)
            & customer_id.notna()
            & invoice_date.notna()
            & chunk["country"].notna()
        )
        frames.append(
            pd.DataFrame(
                {
                    "customer_id": customer_id[keep],
                    "invoice_no": chunk["invoice_no"][keep],
                    "invoice_date": invoice_date[keep],
                    "revenue": (quantity[keep] * unit_price[keep]),
                    "quantity": quantity[keep],
                    "stock_code": chunk["stock_code"][keep],
                    "country": chunk["country"][keep],
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    # Same natural key as staging.sales_valid, so both sources agree on row count.
    combined = combined.drop_duplicates(subset=["invoice_no", "stock_code", "invoice_date"])
    return _normalise(combined)


def load_transactions(source: str = "postgres") -> pd.DataFrame:
    """Load transactions from the requested source.

    ``source="auto"`` prefers Postgres and falls back to the landing files,
    which is what the monitoring job and the tests use so they stay runnable
    whether or not the warehouse is up.
    """
    if source == "postgres":
        return load_from_postgres()
    if source == "landing":
        return load_from_landing()
    if source == "auto":
        try:
            return load_from_postgres()
        except Exception as exc:  # noqa: BLE001 - any connection failure should fall back
            print(f"data: postgres unavailable ({exc.__class__.__name__}), using landing CSVs")
            return load_from_landing()
    raise ValueError(f"unknown source {source!r}: expected postgres, landing or auto")
