import os
import joblib
import pandas as pd
from typing import Optional
from src.fuel_sales.data import clean_data


def generate_holdout_predictions(
    model_path: str = "models/model.joblib",
    holdout_path: str = "data/raw/holdout_features.csv",
    output_path: str = "predictions/holdout_predictions.csv"
) -> pd.DataFrame:
    """
    Loads champion model, predicts litres_per_day for unseen holdout records,
    and writes predictions/holdout_predictions.csv with exactly two columns:
    record_id,prediction as mandated by PROBLEM.md Section 8.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Champion model not found at '{model_path}'. Please train the model first.")

    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"Holdout features file not found at '{holdout_path}'.")

    # Load holdout data and model
    raw_holdout = pd.read_csv(holdout_path)
    model = joblib.load(model_path)

    # Clean holdout features without row dropping
    clean_holdout = clean_data(raw_holdout, is_train=False)

    feature_cols = [
        "highway_distance_km",
        "nearby_pumps",
        "price_rs",
        "trucks_passing",
        "temp_c",
        "staff",
        "ownership",
        "zone",
    ]

    # Batch prediction
    raw_predictions = model.predict(clean_holdout[feature_cols])

    # Construct final submission DataFrame
    predictions_df = pd.DataFrame({
        "record_id": raw_holdout["record_id"],
        "prediction": [round(float(p), 2) for p in raw_predictions]
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV without index
    predictions_df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(predictions_df)} holdout predictions -> '{output_path}'.")
    print(f"Columns: {predictions_df.columns.tolist()}")

    return predictions_df


if __name__ == "__main__":
    generate_holdout_predictions()
