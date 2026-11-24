"""Evaluate the trained regression pipeline on the hold-out test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common import prepare_catboost_features, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/catboost.joblib")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--metrics", default="cv_results/metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    test = pd.read_csv(args.test)
    if "price" not in test.columns:
        raise ValueError("Test file must contain the 'price' target")

    y_test = pd.to_numeric(test.pop("price"), errors="coerce")
    valid = y_test.notna() & (y_test > 0)
    X_test, _ = prepare_catboost_features(test.loc[valid].copy())
    y_test = y_test.loc[valid].astype(float)
    if X_test.empty:
        raise ValueError("No valid test rows remain")

    model = joblib.load(args.model)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    non_zero = y_test != 0
    mape = float(
        np.mean(np.abs((y_test[non_zero].to_numpy() - predictions[non_zero.to_numpy()])
                       / y_test[non_zero].to_numpy())) * 100
    )

    metrics = {
        "MAE": float(mae),
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": float(r2),
        "test_rows": int(len(y_test)),
    }
    write_json(args.metrics, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
