"""
One-time prep script: reads the UCI Online Retail II workbook and splits it
into monthly CSV files under data/landing/, simulating the "simulated
monthly extracts" incremental source the assignment asks for.

Run once: python scripts/split_landing_files.py
"""
import pandas as pd
from pathlib import Path

SRC = Path("data/raw_download/online_retail_II.xlsx")
OUT_DIR = Path("data/landing")

COLUMN_RENAME = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(SRC)

    frames = [xl.parse(sheet) for sheet in xl.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns=COLUMN_RENAME)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["month_key"] = df["invoice_date"].dt.strftime("%Y-%m")

    months = sorted(df["month_key"].unique())
    for month in months:
        chunk = df[df["month_key"] == month].drop(columns=["month_key"])
        out_path = OUT_DIR / f"{month}.csv"
        chunk.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(chunk)} rows)")

    print(f"\n{len(months)} monthly files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
