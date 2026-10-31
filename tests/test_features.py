"""Tests for the churn label, the feature contract and the chronological split.

The label and the split are where a churn model is most easily wrong in a way
that still looks good on paper, so these are the tests that matter most: they
check that the label follows the inactivity window, that no feature is
computed from data after its cutoff, and that the split cannot leak.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ml.config import FEATURES, LABEL
from ml.features import (
    build_feature_table,
    build_scoring_snapshot,
    build_snapshot,
    chronological_split,
    describe_split,
    snapshot_cutoffs,
)


def test_label_follows_the_inactivity_window(transactions):
    """A customer who buys inside the window is not churned; one who does not, is."""
    snapshot = build_snapshot(transactions, pd.Timestamp("2010-07-31"), inactivity_days=90)
    labels = snapshot.set_index("customer_id")[LABEL]

    assert labels[1] == 0, "customer 1 keeps buying monthly, so cannot be churned"
    assert labels[2] == 1, "customer 2 stopped in June 2010, so churns"


def test_label_flips_with_the_window_length(transactions):
    """The same customer can be churned at 30 days and not at 365.

    Customer 3's last order before the cutoff is 2011-06-30 and their next is
    2011-12-31, so the label depends entirely on how long the inactivity
    window is - which is the point of defining it explicitly.
    """
    cutoff = pd.Timestamp("2011-06-30")
    short = build_snapshot(transactions, cutoff, inactivity_days=30).set_index("customer_id")
    long = build_snapshot(transactions, cutoff, inactivity_days=365).set_index("customer_id")

    assert short.loc[3, LABEL] == 1
    assert long.loc[3, LABEL] == 0


def test_features_use_no_data_after_the_cutoff(transactions):
    """Adding transactions after the cutoff must not change the features.

    This is the leakage test. Features are recomputed with a year of extra
    future activity appended; every feature value must be identical.
    """
    cutoff = pd.Timestamp("2010-12-31")
    before = build_snapshot(transactions, cutoff, labelled=False).set_index("customer_id")

    extra = transactions.copy()
    extra["invoice_date"] = extra["invoice_date"] + pd.Timedelta(400, "D")
    extra["invoice_no"] = extra["invoice_no"] + "-future"
    after = build_snapshot(
        pd.concat([transactions, extra], ignore_index=True), cutoff, labelled=False
    ).set_index("customer_id")

    pd.testing.assert_frame_equal(before[FEATURES], after.loc[before.index, FEATURES])


def test_eligibility_excludes_long_dormant_customers(transactions):
    """A customer silent for longer than the lookback is not in the population."""
    # Customer 2's last order is 2010-06; at a 2011-12 cutoff with a 365-day
    # lookback they are already gone, not "about to churn".
    snapshot = build_snapshot(
        transactions, pd.Timestamp("2011-12-31"), min_history_days=365, labelled=False
    )
    assert 2 not in set(snapshot["customer_id"])
    assert 1 in set(snapshot["customer_id"])


def test_recency_and_tenure_are_measured_from_the_cutoff(transactions):
    """Both are differences from the cutoff, not from the end of the data."""
    cutoff = pd.Timestamp("2010-12-31")
    snapshot = build_snapshot(transactions, cutoff, labelled=False).set_index("customer_id")

    # Customer 1 buys on every month-end, so the cutoff is itself an order day.
    assert snapshot.loc[1, "recency_days"] == 0
    # Their first order is 2010-01-31, i.e. 334 days before the cutoff.
    assert snapshot.loc[1, "tenure_days"] == (cutoff - pd.Timestamp("2010-01-31")).days == 334

    # Customer 4's one order was 2010-03-15, well before the cutoff.
    assert snapshot.loc[4, "recency_days"] == (cutoff - pd.Timestamp("2010-03-15")).days


def test_single_order_customer_has_undefined_interpurchase_gap(transactions):
    """Customer 4 ordered once: the mean gap is NaN, not a misleading zero."""
    snapshot = build_snapshot(
        transactions, pd.Timestamp("2010-06-30"), labelled=False
    ).set_index("customer_id")
    assert snapshot.loc[4, "frequency"] == 1
    assert pd.isna(snapshot.loc[4, "avg_interpurchase_days"])


def test_feature_table_has_the_declared_contract(feature_table):
    assert list(feature_table.columns) == ["customer_id", "snapshot_date", *FEATURES, LABEL]
    assert feature_table[LABEL].isin([0, 1]).all()
    assert not feature_table.empty


def test_cutoffs_leave_room_for_the_label_window(transactions):
    """Every cutoff must have a full inactivity window inside the data."""
    cutoffs = snapshot_cutoffs(transactions, inactivity_days=90)
    last_transaction = transactions["invoice_date"].max()
    assert cutoffs
    for cutoff in cutoffs:
        assert cutoff + pd.Timedelta(90, "D") <= last_transaction


def test_chronological_split_is_ordered_and_disjoint(feature_table):
    train, validation, test = chronological_split(feature_table)

    assert not train.empty and not validation.empty and not test.empty
    # Strictly increasing in time, with no shared cutoff between blocks.
    assert train["snapshot_date"].max() < validation["snapshot_date"].min()
    assert validation["snapshot_date"].max() < test["snapshot_date"].min()
    assert len(train) + len(validation) + len(test) == len(feature_table)

    blocks = [set(frame["snapshot_date"]) for frame in (train, validation, test)]
    assert not blocks[0] & blocks[1] and not blocks[1] & blocks[2] and not blocks[0] & blocks[2]


def test_chronological_split_rejects_too_few_cutoffs(feature_table):
    single = feature_table[feature_table["snapshot_date"] == feature_table["snapshot_date"].min()]
    with pytest.raises(ValueError, match="at least 3 snapshot cutoffs"):
        chronological_split(single)


def test_describe_split_reports_each_block(feature_table):
    summary = describe_split(*chronological_split(feature_table))
    assert set(summary) == {"train", "validation", "test"}
    for block in summary.values():
        assert block["rows"] > 0
        assert 0.0 <= block["churn_rate"] <= 1.0


def test_scoring_snapshot_is_unlabelled_and_current(transactions):
    """The scoring snapshot sits at the end of the data and carries no label."""
    scoring = build_scoring_snapshot(transactions)
    assert LABEL not in scoring.columns
    assert set(FEATURES).issubset(scoring.columns)
    assert pd.Timestamp(scoring["snapshot_date"].iloc[0]) == transactions["invoice_date"].max()


def test_build_feature_table_needs_enough_history():
    """Too short a history is an explicit error, not an empty table."""
    short = pd.DataFrame(
        [
            {
                "customer_id": 1,
                "invoice_no": "A",
                "invoice_date": pd.Timestamp("2011-01-01"),
                "revenue": 10.0,
                "quantity": 1,
                "stock_code": "X",
                "country": "United Kingdom",
            }
        ]
    )
    with pytest.raises(ValueError, match="no labellable snapshots"):
        build_feature_table(short)
