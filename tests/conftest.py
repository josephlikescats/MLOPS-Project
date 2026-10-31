"""Shared fixtures: a synthetic transaction history with a known churn pattern.

The tests run entirely on this fixture rather than the real dataset, so they
need no database, no landing files and no trained artifact, and a failure
points at the code rather than at the data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _orders(customer_id: int, dates: list[str], revenue: float = 50.0) -> list[dict]:
    return [
        {
            "customer_id": customer_id,
            "invoice_no": f"{customer_id}-{index}",
            "invoice_date": pd.Timestamp(date),
            "revenue": revenue,
            "quantity": 5,
            "stock_code": f"SKU{index % 4}",
            "country": "United Kingdom",
        }
        for index, date in enumerate(dates)
    ]


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Two years of monthly orders for customers with distinct behaviours.

    * 1 - buys every month throughout, so never churns
    * 2 - stops after 2010-06, so churns at every later cutoff, and is
      excluded altogether once the eligibility lookback has passed
    * 3 - buys monthly to 2011-06 then returns once on 2011-12-31, which is
      the case whose label depends on the window length: churned at 30 days,
      not churned at 365
    * 4 - a single order only, which exercises the undefined
      avg_interpurchase_days path
    """
    monthly = pd.date_range("2010-01-31", "2011-12-31", freq="ME").strftime("%Y-%m-%d").tolist()
    first_half = [date for date in monthly if date < "2010-07-01"]
    lapsing = [date for date in monthly if date <= "2011-06-30"] + ["2011-12-31"]

    rows: list[dict] = []
    rows += _orders(1, monthly, revenue=120.0)
    rows += _orders(2, first_half, revenue=80.0)
    rows += _orders(3, lapsing, revenue=200.0)
    rows += _orders(4, ["2010-03-15"], revenue=35.0)
    return pd.DataFrame(rows)


@pytest.fixture
def feature_table(transactions: pd.DataFrame) -> pd.DataFrame:
    from ml.features import build_feature_table

    return build_feature_table(transactions)
