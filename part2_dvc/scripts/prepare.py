"""Create train/test CSV files for the DVC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from common import load_csv, load_params, make_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/flats_cleaned.csv")
    parser.add_argument("--train-output", default="data/processed/train.csv")
    parser.add_argument("--test-output", default="data/processed/test.csv")
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    params = load_params(args.params).get("data", {})
    frame = load_csv(args.input)
    X, y = make_features(frame)

    test_size = float(params.get("test_size", 0.2))
    random_state = int(params.get("random_state", 42))
    if not 0 < test_size < 1:
        raise ValueError("data.test_size must be between 0 and 1")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    train = X_train.copy()
    train["price"] = y_train
    test = X_test.copy()
    test["price"] = y_test

    for output_name, data in ((args.train_output, train), (args.test_output, test)):
        output = Path(output_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output, index=False)

    print(f"Train rows: {len(train):,}; test rows: {len(test):,}")
    print(f"Predictor columns: {len(X.columns)}")


if __name__ == "__main__":
    main()
