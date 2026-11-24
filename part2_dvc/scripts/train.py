"""Train and save a CatBoost regression model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pd

from catboost import CatBoostRegressor

from common import load_params, prepare_catboost_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.csv")
    parser.add_argument("--model-output", default="models/catboost.joblib")
    parser.add_argument("--params", default="params.yaml")
    return parser.parse_args()


def build_model(categorical_columns: list[str], model_params: dict) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=int(model_params.get("iterations", 1000)),
        depth=int(model_params.get("depth", 6)),
        learning_rate=float(model_params.get("learning_rate", 0.03)),
        loss_function=model_params.get("loss_function", "RMSE"),
        random_seed=int(model_params.get("random_seed", 42)),
        thread_count=int(model_params.get("thread_count", -1)),
        cat_features=categorical_columns,
        verbose=False,
        allow_writing_files=False,
    )


def main():
    args = parse_args()
    params = load_params(args.params)
    model_params = params.get("model", {})

    train = pd.read_csv(args.train)
    if "price" not in train.columns:
        raise ValueError("Training file must contain the 'price' target")
    y_train = pd.to_numeric(train.pop("price"), errors="coerce")
    valid = y_train.notna() & (y_train > 0)
    X_train, categorical_columns = prepare_catboost_features(
        train.loc[valid].copy()
    )
    y_train = y_train.loc[valid]
    if X_train.empty:
        raise ValueError("No valid training rows remain")

    model = build_model(categorical_columns, model_params)
    model.fit(X_train, y_train)

    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")

    # MLflow logging is optional at runtime, but included when mlflow is installed.
    try:
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("apartment-price-regression")
        with mlflow.start_run(run_name="catboost"):
            mlflow.log_params(
                {
                    key: value
                    for key, value in model_params.items()
                    if value is not None and isinstance(value, (str, int, float, bool))
                }
            )
            mlflow.log_metric("train_rows", float(len(X_train)))
            mlflow.log_artifact(str(output_path), artifact_path="model")
    except ImportError:
        print("MLflow is not installed; model training completed without MLflow logging")


if __name__ == "__main__":
    main()
