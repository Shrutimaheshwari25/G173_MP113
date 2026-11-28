import os
import hashlib
import yaml
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.fuel_sales.data import clean_data, build_preprocessor


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of raw train dataset for reproducibility."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_and_split_later_data(
    later_df: pd.DataFrame,
    split_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sorts later.csv by date and splits into first 70% (retrain partition)
    and last 30% (unseen evaluation check) as instructed in PROBLEM.md Section 7.
    """
    sorted_df = later_df.sort_values("date").reset_index(drop=True)
    split_index = int(len(sorted_df) * split_ratio)
    
    later_70 = sorted_df.iloc[:split_index].copy()
    later_30 = sorted_df.iloc[split_index:].copy()
    
    return later_70, later_30


def build_v2_training_data(
    clean_train_df: pd.DataFrame,
    later_70_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines cleaned historical train.csv and the first 70% of later.csv
    to form the augmented training dataset for Model v2.
    """
    combined_df = pd.concat([clean_train_df, later_70_df], ignore_index=True)
    return combined_df


def train_v2_model(
    train_data: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    target_col: str = "litres_per_day",
    alpha: float = 1.0
) -> Pipeline:
    """
    Constructs and fits the v2 Pipeline on augmented training data.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=alpha))
    ])

    feature_cols = numeric_cols + categorical_cols
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_promotion_rule(
    mae_v1: float,
    mae_v2: float,
    threshold: float = 0.08
) -> Tuple[bool, float, str]:
    """
    Client Promotion Rule (PROBLEM.md Section 7):
    Promote v2 only if its MAE on the last 30% check partition
    is at least 8% lower than v1's MAE.
    """
    improvement = (mae_v1 - mae_v2) / mae_v1
    is_promoted = bool(improvement >= threshold)
    decision = "promoted" if is_promoted else "retained_v1"
    return is_promoted, round(float(improvement * 100), 2), decision


def execute_retraining_and_promotion(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv",
    model_path: str = "models/model.joblib",
    tracking_uri: str = "sqlite:///mlflow.db",
    registered_model_name: str = "mp113_fuel_sales_model"
) -> Dict[str, Any]:
    """
    Executes full retraining pipeline, MLflow tracking, promotion rule evaluation,
    and updates @champion alias in MLflow registry.
    """
    raw_train = pd.read_csv(train_path)
    raw_later = pd.read_csv(later_path)

    clean_train = clean_data(raw_train, is_train=True)
    clean_later = clean_data(raw_later, is_train=False)

    num_cols = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"]
    cat_cols = ["ownership", "zone"]
    feature_cols = num_cols + cat_cols
    target_col = "litres_per_day"

    # 1. 70/30 Temporal split
    later_70, later_30 = load_and_split_later_data(clean_later, split_ratio=0.7)
    train_v2 = build_v2_training_data(clean_train, later_70)

    # 2. Evaluate v1 baseline on last 30%
    v1_model = joblib.load(model_path)
    v1_preds = v1_model.predict(later_30[feature_cols])
    mae_v1 = round(float(mean_absolute_error(later_30[target_col], v1_preds)), 2)

    # 3. Train v2 model
    v2_pipeline = train_v2_model(train_v2, num_cols, cat_cols, target_col=target_col)
    v2_preds = v2_pipeline.predict(later_30[feature_cols])
    mae_v2 = round(float(mean_absolute_error(later_30[target_col], v2_preds)), 2)

    # 4. Apply 8% promotion rule
    is_promoted, imp_pct, decision = evaluate_promotion_rule(mae_v1, mae_v2, threshold=0.08)

    print("\n================ RETRAINING & PROMOTION EVALUATION ================")
    print(f"  v1 MAE on last 30% check partition : {mae_v1:.2f} litres")
    print(f"  v2 MAE on last 30% check partition : {mae_v2:.2f} litres")
    print(f"  Error Reduction (Improvement)      : {imp_pct:.2f}% (Threshold: >= 8.00%)")
    print(f"  Client Promotion Decision          : {decision.upper()}")
    print("===================================================================\n")

    # 5. Track v2 run in MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("fuel_station_daily_sales")
    client = MlflowClient()

    data_sha = compute_file_sha256(train_path)

    with mlflow.start_run(run_name="retrain_v2_augmented") as run:
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("model_type", "ridge_v2_augmented")
        mlflow.log_param("train_samples", len(train_v2))
        mlflow.log_param("retrain_split", "70_later_added")

        # Required metric containing 'mae'
        mlflow.log_metric("mae", mae_v2)
        mlflow.log_metric("mae_later_30", mae_v2)
        mlflow.log_metric("mae_v1_baseline", mae_v1)
        mlflow.log_metric("improvement_pct", imp_pct)

        # Log model with cloudpickle
        mlflow.sklearn.log_model(
            sk_model=v2_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        v2_registered = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        v2_version = v2_registered.version

    # 6. Set champion alias if promoted and save artifact
    if is_promoted:
        client.set_registered_model_alias(name=registered_model_name, alias="champion", version=v2_version)
        joblib.dump(v2_pipeline, model_path)
        print(f"Successfully promoted Model Version {v2_version} to '@champion' and updated '{model_path}'.")

    return {
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "improvement_pct": imp_pct,
        "decision": decision,
        "champion_version": int(v2_version) if is_promoted else 1
    }


if __name__ == "__main__":
    execute_retraining_and_promotion()

