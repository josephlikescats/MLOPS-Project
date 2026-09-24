import pandas as pd

from ml.train_model import FEATURES, build_training_data


def test_training_features_create_future_purchase_label():
    transactions = pd.DataFrame([
        {"customer_id": 1, "invoice_no": "A", "invoice_date": "2024-01-01", "revenue": 10},
        {"customer_id": 1, "invoice_no": "B", "invoice_date": "2024-04-01", "revenue": 20},
        {"customer_id": 2, "invoice_no": "C", "invoice_date": "2024-01-02", "revenue": 30},
        {"customer_id": 3, "invoice_no": "D", "invoice_date": "2024-04-01", "revenue": 40},
    ])
    result = build_training_data(transactions, horizon_days=30)
    assert set(FEATURES).issubset(result.columns)
    assert result.loc[result.customer_id == 1, "repeat_purchase"].item() == 1
    assert result.loc[result.customer_id == 2, "repeat_purchase"].item() == 0