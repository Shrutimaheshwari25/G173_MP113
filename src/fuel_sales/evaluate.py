import os
import joblib
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error

from src.fuel_sales.data import clean_data


def evaluate_by_subgroup(
    model: any,
    df: pd.DataFrame,
    group_col: str = "zone",
    target_col: str = "litres_per_day",
    feature_cols: List[str] = None
) -> Tuple[Dict[str, float], str]:
    """
    Evaluates model MAE separately across each subgroup category
    and identifies the worst-served group as required by PROBLEM.md Section 7.
    """
    if feature_cols is None:
        feature_cols = [
            "highway_distance_km",
            "nearby_pumps",
            "price_rs",
            "trucks_passing",
            "temp_c",
            "staff",
            "ownership",
            "zone"
        ]

    subgroup_maes = {}
    unique_groups = df[group_col].unique()

    for group_val in sorted(unique_groups):
        sub_df = df[df[group_col] == group_val]
        if len(sub_df) > 0:
            preds = model.predict(sub_df[feature_cols])
            mae = mean_absolute_error(sub_df[target_col], preds)
            subgroup_maes[str(group_val)] = round(float(mae), 2)

    worst_group = max(subgroup_maes, key=subgroup_maes.get) if subgroup_maes else ""
    return subgroup_maes, worst_group


def run_subgroup_evaluation(
    model_path: str = "models/model.joblib",
    later_path: str = "data/raw/later.csv"
) -> Tuple[Dict[str, float], str]:
    """
    Executes v1 model evaluation on the later.csv dataset across all zones.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    later_raw = pd.read_csv(later_path)
    clean_later = clean_data(later_raw, is_train=False)

    group_maes, worst_group = evaluate_by_subgroup(model, clean_later, group_col="zone")

    print("\n================ SUBGROUP PERFORMANCE (ZONE-WISE MAE) ================")
    for zone_name, mae_score in group_maes.items():
        tag = "WORST SERVED" if zone_name == worst_group else "NORMAL"
        print(f"  Zone {zone_name:<14}: MAE = {mae_score:<10.2f} [{tag}]")
    print(f"\nWorst Served Group: {worst_group} (MAE = {group_maes.get(worst_group, 0):.2f})")
    print("=======================================================================\n")

    return group_maes, worst_group


if __name__ == "__main__":
    run_subgroup_evaluation()
