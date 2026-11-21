import pandas as pd
from src.fuel_sales.data import clean_data, clean_ownership, build_preprocessor


def test_clean_ownership():
    assert clean_ownership("pub.") == "public_sector"
    assert clean_ownership("PUBLIC_SECTOR") == "public_sector"
    assert clean_ownership("pri.") == "private"
    assert clean_ownership("COOPERATIVE") == "cooperative"
    assert clean_ownership("coo.") == "cooperative"


def test_clean_data_unit_mixup_and_duplicates():
    # Synthetic test data without requiring external CSV
    raw_sample = pd.DataFrame({
        "record_id": ["R01", "R01", "R02"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "highway_distance_km": [7500.0, 7500.0, 5.0],
        "nearby_pumps": [5, 5, 8],
        "price_rs": [95.0, 95.0, 96.0],
        "trucks_passing": [500, 500, 600],
        "temp_c": [25.0, 25.0, 26.0],
        "staff": [10, 10, 12],
        "ownership": ["pub.", "pub.", "pri."],
        "zone": ["Industrial", "Industrial", "City"],
        "litres_per_day": [18000, 18000, 22000]
    })

    cleaned = clean_data(raw_sample, is_train=True)

    # Duplicates removed: 2 rows remaining
    assert len(cleaned) == 2
    # Industrial highway distance divided by 1000
    assert cleaned.loc[cleaned["zone"] == "Industrial", "highway_distance_km"].iloc[0] == 7.5
    # Ownership cleaned
    assert cleaned["ownership"].tolist() == ["public_sector", "private"]


def test_build_preprocessor():
    num_cols = ["highway_distance_km", "nearby_pumps"]
    cat_cols = ["ownership", "zone"]
    preprocessor = build_preprocessor(num_cols, cat_cols)

    df = pd.DataFrame({
        "highway_distance_km": [5.0, 10.0],
        "nearby_pumps": [4, 8],
        "ownership": ["public_sector", "private"],
        "zone": ["Highway", "City"]
    })

    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] >= 4

