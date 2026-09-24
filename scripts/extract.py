"""
Extract task: scans data/landing/*.csv, loads any file not already marked
'success' in meta.ingestion_log into raw.sales_raw, and logs the outcome
(row count, status, error) for every file it attempts. This is what makes
the ingestion repeatable/incremental: re-running skips already-loaded files.
"""
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from db import get_engine

LANDING_DIR = Path(__file__).resolve().parent.parent / "data" / "landing"

RAW_COLUMNS = [
    "invoice_no", "stock_code", "description", "quantity",
    "invoice_date", "unit_price", "customer_id", "country",
]


def already_loaded(engine, source_file: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM meta.ingestion_log WHERE source_file = :f AND status = 'success'"),
            {"f": source_file},
        ).fetchone()
    return row is not None


def log_ingestion(engine, source_file, row_count, status, error_message=None):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO meta.ingestion_log (source_file, row_count, status, error_message)
                VALUES (:f, :rc, :st, :err)
                ON CONFLICT (source_file, status) DO NOTHING
            """),
            {"f": source_file, "rc": row_count, "st": status, "err": error_message},
        )


def extract_new_files():
    engine = get_engine()
    if not LANDING_DIR.exists():
        raise FileNotFoundError(f"landing dir not found: {LANDING_DIR}")

    csv_files = sorted(LANDING_DIR.glob("*.csv"))
    results = {"loaded": [], "skipped": [], "failed": []}

    for path in csv_files:
        fname = path.name
        if already_loaded(engine, fname):
            results["skipped"].append(fname)
            continue

        try:
            df = pd.read_csv(path, dtype=str)  # raw layer: keep everything as text
            missing = set(RAW_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"missing expected columns: {missing}")

            df = df[RAW_COLUMNS].copy()
            df["source_file"] = fname
            df.to_sql("sales_raw", engine, schema="raw", if_exists="append", index=False)

            log_ingestion(engine, fname, len(df), "success")
            results["loaded"].append((fname, len(df)))
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the whole run
            log_ingestion(engine, fname, 0, "failed", str(exc))
            results["failed"].append((fname, str(exc)))

    print(f"extract: loaded={len(results['loaded'])} skipped={len(results['skipped'])} failed={len(results['failed'])}")
    for f, err in results["failed"]:
        print(f"  FAILED {f}: {err}")
    return results


if __name__ == "__main__":
    extract_new_files()
