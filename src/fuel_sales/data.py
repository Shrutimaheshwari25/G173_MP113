import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VALID_OWNERSHIPS = ["public_sector", "private", "cooperative"]


def clean_ownership(val):
    """Normalize spelling variants in ownership column."""
    if pd.isna(val):
        return "public_sector"
    s = str(val).strip().lower()
    if "pub" in s:
        return "public_sector"
    elif "pri" in s:
        return "private"
    elif "coo" in s:
        return "cooperative"
    return s if s in VALID_OWNERSHIPS else "public_sector"


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans raw fuel sales dataframe by resolving the issues identified in Stage 1:
    1. Duplicate rows
    2. Unit mixup in highway_distance_km for Industrial zone (meters -> km)
    3. Stuck sensor in temp_c between 2026-02-08 and 2026-02-22
    4. Spelling variants in ownership
    """
    cleaned = df.copy()

    # 1. Duplicate rows
    if is_train:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Unit mixup: Industrial highway distance is in meters
    if "zone" in cleaned.columns and "highway_distance_km" in cleaned.columns:
        mask_industrial = cleaned["zone"] == "Industrial"
        # Values > 50 in highway_distance_km are clearly in meters
        cleaned.loc[mask_industrial & (cleaned["highway_distance_km"] > 50), "highway_distance_km"] /= 1000.0

    # 3. Stuck sensor in temp_c (constant 37.1 from 2026-02-08 to 2026-02-22)
    if "date" in cleaned.columns and "temp_c" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"])
        stuck_mask = (cleaned["date"] >= "2026-02-08") & (cleaned["date"] <= "2026-02-22")
        if stuck_mask.any():
            # Interpolate or fill with monthly mean temperature
            cleaned.loc[stuck_mask, "temp_c"] = np.nan
            cleaned = cleaned.sort_values("date")
            cleaned["temp_c"] = cleaned["temp_c"].interpolate(method="linear").bfill().ffill()

    # 4. Ownership spelling variants
    if "ownership" in cleaned.columns:
        cleaned["ownership"] = cleaned["ownership"].apply(clean_ownership)

    return cleaned


def build_preprocessor(numeric_cols, categorical_cols):
    """Builds scikit-learn ColumnTransformer for preprocessing."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor

