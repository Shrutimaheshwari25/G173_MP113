import os
import hashlib
import yaml
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from src.fuel_sales.data import clean_data, build_preprocessor


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_params(params_path: str = "params.yaml") -> dict:
    """Loads configuration settings using yaml.safe_load."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def train_models():
    params = load_params()

    train_path = params["data"]["raw_train_path"]
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at '{train_path}'. "
            "Please ensure train.csv is extracted into data/raw/."
        )

    # 1. Compute SHA-256 of raw train.csv
    data_sha = compute_file_sha256(train_path)
    print(f"Original train.csv SHA-256: {data_sha}")

    # 2. Configure MLflow tracking
    tracking_uri = params["tracking"]["tracking_uri"]
    experiment_name = params["tracking"]["experiment_name"]
    reg_model_name = params["tracking"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # 3. Load & clean data
    raw_df = pd.read_csv(train_path)
    cleaned_df = clean_data(raw_df, is_train=True)

    numeric_cols = params["features"]["numeric_cols"]
    categorical_cols = params["features"]["categorical_cols"]
    feature_cols = numeric_cols + categorical_cols
    target_col = params["base"]["target_col"]

    X = cleaned_df[feature_cols]
    y = cleaned_df[target_col]

    test_size = params["train"]["test_size"]
    random_state = params["base"]["random_state"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models_config = [
        ("ridge", Ridge(**params["train"]["models"]["ridge"])),
        ("random_forest", RandomForestRegressor(**params["train"]["models"]["random_forest"])),
        ("gradient_boosting", GradientBoostingRegressor(**params["train"]["models"]["gradient_boosting"])),
    ]

    runs_info = []

    for model_name, regressor in models_config:
        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            preprocessor = build_preprocessor(numeric_cols, categorical_cols)
            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor)
            ])

            pipeline.fit(X_train, y_train)

            # Predictions & evaluation
            train_preds = pipeline.predict(X_train)
            val_preds = pipeline.predict(X_val)

            train_mae = mean_absolute_error(y_train, train_preds)
            val_mae = mean_absolute_error(y_val, val_preds)

            # Log parameters
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", model_name)
            for param_key, param_val in regressor.get_params().items():
                if isinstance(param_val, (int, float, str, bool)):
                    mlflow.log_param(f"model_{param_key}", param_val)

            # Log metrics (must contain 'mae')
            mlflow.log_metric("mae_train", train_mae)
            mlflow.log_metric("mae", val_mae)
            mlflow.log_metric("mae_val", val_mae)

            # Log model with cloudpickle
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            runs_info.append({
                "model_name": model_name,
                "run_id": run.info.run_id,
                "val_mae": val_mae,
                "pipeline": pipeline
            })

            print(f"Run {model_name} (ID: {run.info.run_id}) - Validation MAE: {val_mae:.4f}")

    # 4. Select best model based on validation MAE
    best_run = min(runs_info, key=lambda r: r["val_mae"])
    print(f"\nBest Model: {best_run['model_name']} with Validation MAE: {best_run['val_mae']:.4f}")
    print(f"Best Run ID: {best_run['run_id']}")

    # 5. Register best model as version 1
    model_uri = f"runs:/{best_run['run_id']}/model"
    reg_model = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    v1_version = reg_model.version

    # Set alias champion
    client.set_registered_model_alias(name=reg_model_name, alias="champion", version=v1_version)
    print(f"Registered model '{reg_model_name}' version {v1_version} as '@champion'")

    # 6. Save model to models/model.joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_run["pipeline"], "models/model.joblib")
    print("Saved champion model to models/model.joblib")

    # 7. Update reports/findings.yaml tracking section
    findings_path = "reports/findings.yaml"
    if os.path.exists(findings_path):
        with open(findings_path, "r") as f:
            findings = yaml.safe_load(f)

        findings["tracking"]["experiment_name"] = experiment_name
        findings["tracking"]["best_run_id"] = best_run["run_id"]
        findings["tracking"]["registered_model_name"] = reg_model_name
        findings["tracking"]["champion_version"] = int(v1_version)

        with open(findings_path, "w") as f:
            yaml.dump(findings, f, sort_keys=False)
        print("Updated reports/findings.yaml tracking section.")

    return best_run


if __name__ == "__main__":
    train_models()

