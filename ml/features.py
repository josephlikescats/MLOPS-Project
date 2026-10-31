"""
Reproducible feature engineering and churn labelling.

The churn question is "will this customer stop buying?", which only has a
meaning relative to a point in time. So the feature table is built from
*snapshots*: at each month-end cutoff every eligible customer contributes one
row whose features are computed strictly from transactions on or before the
cutoff, and whose label is read from the inactivity window that follows it.

    history (features)            inactivity window (label)
    |---------------------------|---------------------------|
                             cutoff                 cutoff + 90d

    churned = 1  <=>  the customer places no order in that window

Two consequences worth stating, because they are what make the split honest:

  * A cutoff is only labellable while the dataset still covers the whole
    inactivity window after it. The final 90 days of data can therefore be
    scored but never trained on - ``build_scoring_snapshot`` is that case.
  * Snapshots are split by cutoff date, not at random, so the test block is
    strictly later in time than the training block and no customer's future
    can leak backwards into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import (
    CHURN_CONFIG,
    FEATURES,
    INACTIVITY_DAYS,
    LABEL,
    MIN_HISTORY_DAYS,
    MIN_SNAPSHOT_HISTORY_DAYS,
    SNAPSHOT_FREQUENCY,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    ChurnConfig,
)

RECENT_WINDOW_DAYS = 90


def _resolve_frequency(frequency):
    """Normalise a month-end frequency alias to something every pandas accepts.

    The month-end alias was renamed between pandas versions: "M" is the old
    spelling (invalid from 2.2) and "ME" the new one (invalid before 2.2). This
    code has to run under both, because the Airflow containers cannot use
    pandas >= 2.2: Airflow 2.10 pins SQLAlchemy < 2.0, pandas >= 2.2 requires
    SQLAlchemy >= 2.0, and when pandas cannot import a SQLAlchemy it accepts it
    silently treats an Engine as a raw DBAPI connection and every read_sql /
    to_sql call fails with "'Engine' object has no attribute 'cursor'".

    The offset object means month-end in every version, so both aliases are
    mapped onto it. SNAPSHOT_FREQUENCY stays the string "ME" in ml/config.py so
    the value recorded in artifacts and MLflow runs is unchanged.
    """
    if frequency in ("M", "ME"):
        return pd.offsets.MonthEnd()
    return frequency


def snapshot_cutoffs(
    transactions: pd.DataFrame,
    inactivity_days: int = INACTIVITY_DAYS,
    frequency: str = SNAPSHOT_FREQUENCY,
    min_snapshot_history_days: int = MIN_SNAPSHOT_HISTORY_DAYS,
) -> list[pd.Timestamp]:
    """Month-end cutoffs that have both enough history and a full label window.

    Cutoffs are normalised to midnight so a snapshot date is a plain calendar
    date rather than inheriting the time-of-day of the first transaction.
    """
    dates = pd.to_datetime(transactions["invoice_date"])
    first, last = dates.min(), dates.max()
    earliest = (first + pd.Timedelta(min_snapshot_history_days, "D")).normalize()
    latest = (last - pd.Timedelta(inactivity_days, "D")).normalize()
    if earliest > latest:
        return []
    return list(pd.date_range(earliest, latest, freq=_resolve_frequency(frequency)))


def _aggregate_history(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-customer features from one customer's transactions up to the cutoff."""
    grouped = history.groupby("customer_id")
    features = grouped.agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        frequency=("invoice_no", "nunique"),
        monetary=("revenue", "sum"),
        total_items=("quantity", "sum"),
        distinct_products=("stock_code", "nunique"),
    )
    features["distinct_active_months"] = grouped["invoice_date"].apply(
        lambda s: s.dt.to_period("M").nunique()
    )

    features["recency_days"] = (cutoff - features["last_purchase"]).dt.days
    features["tenure_days"] = (cutoff - features["first_purchase"]).dt.days
    features["average_order_value"] = features["monetary"] / features["frequency"].clip(lower=1)
    features["avg_items_per_order"] = features["total_items"] / features["frequency"].clip(lower=1)

    # Mean gap between orders. Undefined for a single-order customer, so it is
    # left as NaN and filled by the pipeline's median imputer rather than being
    # faked with a zero (which would read as "buys every day").
    active_span = (features["last_purchase"] - features["first_purchase"]).dt.days
    repeat_orders = features["frequency"] - 1
    features["avg_interpurchase_days"] = np.where(
        repeat_orders > 0, active_span / repeat_orders.clip(lower=1), np.nan
    )

    recent = history[history["invoice_date"] > cutoff - pd.Timedelta(RECENT_WINDOW_DAYS, "D")]
    prior = history[
        (history["invoice_date"] <= cutoff - pd.Timedelta(RECENT_WINDOW_DAYS, "D"))
        & (history["invoice_date"] > cutoff - pd.Timedelta(2 * RECENT_WINDOW_DAYS, "D"))
    ]
    recent_agg = recent.groupby("customer_id").agg(
        orders_last_90d=("invoice_no", "nunique"),
        revenue_last_90d=("revenue", "sum"),
    )
    prior_revenue = prior.groupby("customer_id")["revenue"].sum().rename("revenue_prior_90d")

    features = features.join(recent_agg).join(prior_revenue)
    features[["orders_last_90d", "revenue_last_90d", "revenue_prior_90d"]] = features[
        ["orders_last_90d", "revenue_last_90d", "revenue_prior_90d"]
    ].fillna(0.0)

    # +1 smoothing keeps the ratio finite for customers with no spend in the
    # prior window: ratio > 1 means spend is accelerating, < 1 decelerating.
    features["revenue_trend_ratio"] = (features["revenue_last_90d"] + 1.0) / (
        features["revenue_prior_90d"] + 1.0
    )
    return features


def build_snapshot(
    transactions: pd.DataFrame,
    cutoff: pd.Timestamp,
    inactivity_days: int = INACTIVITY_DAYS,
    min_history_days: int = MIN_HISTORY_DAYS,
    labelled: bool = True,
) -> pd.DataFrame:
    """Build one snapshot of customer features (and its label) at ``cutoff``.

    Eligibility: the customer must have bought at least once in the
    ``min_history_days`` before the cutoff. A customer who last bought two
    years ago is not "about to churn" - they already have, and keeping them in
    would let the model score the label definition instead of behaviour.
    """
    cutoff = pd.Timestamp(cutoff)
    transactions = transactions.copy()
    transactions["invoice_date"] = pd.to_datetime(transactions["invoice_date"])

    history = transactions[transactions["invoice_date"] <= cutoff]
    if history.empty:
        return pd.DataFrame(columns=["customer_id", "snapshot_date", *FEATURES, LABEL])

    eligible = history[
        history["invoice_date"] > cutoff - pd.Timedelta(min_history_days, "D")
    ]["customer_id"].unique()
    history = history[history["customer_id"].isin(eligible)]
    if history.empty:
        return pd.DataFrame(columns=["customer_id", "snapshot_date", *FEATURES, LABEL])

    features = _aggregate_history(history, cutoff)
    features["snapshot_date"] = cutoff

    if labelled:
        window_end = cutoff + pd.Timedelta(inactivity_days, "D")
        future = transactions[
            (transactions["invoice_date"] > cutoff)
            & (transactions["invoice_date"] <= window_end)
        ]
        returning = set(future["customer_id"].unique())
        features[LABEL] = (~features.index.to_series().isin(returning)).astype(int)
        columns = ["customer_id", "snapshot_date", *FEATURES, LABEL]
    else:
        columns = ["customer_id", "snapshot_date", *FEATURES]

    return features.reset_index()[columns]


def build_feature_table(
    transactions: pd.DataFrame,
    config: ChurnConfig = CHURN_CONFIG,
) -> pd.DataFrame:
    """Stack every labellable snapshot into one chronologically ordered table."""
    cutoffs = snapshot_cutoffs(
        transactions,
        inactivity_days=config.inactivity_days,
        frequency=config.snapshot_frequency,
        min_snapshot_history_days=config.min_snapshot_history_days,
    )
    if not cutoffs:
        raise ValueError(
            "no labellable snapshots: the dataset needs at least "
            f"{config.min_snapshot_history_days} days of history plus "
            f"{config.inactivity_days} days of future per cutoff"
        )

    snapshots = [
        build_snapshot(
            transactions,
            cutoff,
            inactivity_days=config.inactivity_days,
            min_history_days=config.min_history_days,
        )
        for cutoff in cutoffs
    ]
    table = pd.concat(snapshots, ignore_index=True)
    return table.sort_values(["snapshot_date", "customer_id"]).reset_index(drop=True)


def build_scoring_snapshot(
    transactions: pd.DataFrame,
    config: ChurnConfig = CHURN_CONFIG,
) -> pd.DataFrame:
    """Features for the most recent cutoff in the data, with no label.

    This is what batch scoring and the monitoring job consume: the customers
    whose inactivity window has not finished yet, so their outcome is still
    unknown and a prediction is actually useful.
    """
    transactions = transactions.copy()
    transactions["invoice_date"] = pd.to_datetime(transactions["invoice_date"])
    cutoff = transactions["invoice_date"].max()
    return build_snapshot(
        transactions,
        cutoff,
        inactivity_days=config.inactivity_days,
        min_history_days=config.min_history_days,
        labelled=False,
    )


def chronological_split(
    features: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split snapshots into time-ordered train / validation / test blocks.

    The cut is made on the distinct snapshot dates, so a whole cutoff month
    lands in exactly one block. Validation is used for model selection and
    threshold choice; test is touched once, for the reported metrics.
    """
    cutoffs = sorted(features["snapshot_date"].unique())
    if len(cutoffs) < 3:
        raise ValueError(
            f"need at least 3 snapshot cutoffs for a 3-way chronological split, got {len(cutoffs)}"
        )

    train_end = max(1, int(len(cutoffs) * train_fraction))
    validation_end = max(train_end + 1, int(len(cutoffs) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(cutoffs) - 1)

    train_cutoffs = set(cutoffs[:train_end])
    validation_cutoffs = set(cutoffs[train_end:validation_end])
    test_cutoffs = set(cutoffs[validation_end:])

    return (
        features[features["snapshot_date"].isin(train_cutoffs)].reset_index(drop=True),
        features[features["snapshot_date"].isin(validation_cutoffs)].reset_index(drop=True),
        features[features["snapshot_date"].isin(test_cutoffs)].reset_index(drop=True),
    )


def describe_split(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict:
    """Row counts, date ranges and churn rates per block, for the run record."""

    def block(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"rows": 0}
        return {
            "rows": int(len(frame)),
            "customers": int(frame["customer_id"].nunique()),
            "snapshots": int(frame["snapshot_date"].nunique()),
            "first_cutoff": str(pd.Timestamp(frame["snapshot_date"].min()).date()),
            "last_cutoff": str(pd.Timestamp(frame["snapshot_date"].max()).date()),
            "churn_rate": float(frame[LABEL].mean()) if LABEL in frame else None,
        }

    return {"train": block(train), "validation": block(validation), "test": block(test)}
