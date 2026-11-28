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


if __name__ == "__main__":
    raw_train = pd.read_csv("data/raw/train.csv")
    raw_later = pd.read_csv("data/raw/later.csv")

    clean_train = clean_data(raw_train, is_train=True)
    clean_later = clean_data(raw_later, is_train=False)

    later_70, later_30 = load_and_split_later_data(clean_later, split_ratio=0.7)
    train_v2 = build_v2_training_data(clean_train, later_70)

    num_cols = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"]
    cat_cols = ["ownership", "zone"]

    v2_model = train_v2_model(train_v2, num_cols, cat_cols)
    print(f"Successfully trained Model v2 on {len(train_v2)} combined records (Train: {len(clean_train)}, Later 70%: {len(later_70)}).")
    print(f"Unseen evaluation holdout (Later 30%): {len(later_30)} records.")
