import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from src.fuel_sales.data import clean_data


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between reference (expected)
    and current (actual) distributions using 10 bins as taught in Lab P13.

    PSI interpretation:
    - PSI < 0.1: No significant change / stable
    - 0.1 <= PSI < 0.25: Moderate change / slight drift
    - PSI >= 0.25: Significant change / severe drift
    """
    # Define quantile bin edges based on expected distribution
    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles * 100)
    bin_edges = np.unique(bin_edges)
    
    # Extend outer boundaries to cover all potential actual values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Apply epsilon to prevent division by zero or log(0)
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi_value, 4))


def compute_numeric_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: List[str]
) -> Tuple[Dict[str, float], str]:
    """
    Calculates PSI for all numeric features and identifies the most drifted feature.
    """
    psi_results = {}
    for col in numeric_cols:
        if col in reference_df.columns and col in current_df.columns:
            psi = calculate_psi(
                reference_df[col].dropna().values,
                current_df[col].dropna().values,
                num_bins=10
            )
            psi_results[col] = psi

    most_drifted = max(psi_results, key=psi_results.get) if psi_results else ""
    return psi_results, most_drifted


def run_drift_analysis(
    train_path: str = "data/raw/train.csv",
    later_path: str = "data/raw/later.csv"
) -> Tuple[Dict[str, float], str]:
    """
    Loads raw train and later datasets, applies cleaning, and executes drift analysis.
    """
    train_raw = pd.read_csv(train_path)
    later_raw = pd.read_csv(later_path)

    clean_train = clean_data(train_raw, is_train=True)
    clean_later = clean_data(later_raw, is_train=False)

    numeric_cols = [
        "highway_distance_km",
        "nearby_pumps",
        "price_rs",
        "trucks_passing",
        "temp_c",
        "staff",
    ]

    psi_results, most_drifted = compute_numeric_drift(clean_train, clean_later, numeric_cols)

    print("\n================ POPULATION STABILITY INDEX (PSI) ================")
    for feature, psi in psi_results.items():
        status = "CRITICAL DRIFT (>=0.25)" if psi >= 0.25 else "STABLE (<0.1)"
        print(f"  {feature:<22}: PSI = {psi:<8.4f} [{status}]")
    print(f"\nMost Drifted Feature: {most_drifted} (PSI = {psi_results.get(most_drifted, 0):.4f})")
    print("==================================================================\n")

    return psi_results, most_drifted


if __name__ == "__main__":
    run_drift_analysis()
