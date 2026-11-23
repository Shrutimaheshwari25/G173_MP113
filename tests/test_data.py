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

def test_clean_ownership_edge_cases():
    """Verify that None or unexpected values fallback safely."""
    assert clean_ownership(None) == "public_sector"
    assert clean_ownership("random_unknown_type") == "public_sector"
    assert clean_ownership("PRIVATE_PUMP") == "private"


def test_clean_data_stuck_sensor_interpolation():
    """Verify stuck sensor interpolation between 2026-02-08 and 2026-02-22."""
    df_stuck = pd.DataFrame({
        "record_id": ["S1", "S2", "S3"],
        "date": ["2026-02-07", "2026-02-15", "2026-02-23"],
        "temp_c": [20.0, 37.1, 30.0],
        "zone": ["City", "City", "City"],
        "highway_distance_km": [5.0, 5.0, 5.0]
    })
    cleaned = clean_data(df_stuck, is_train=True)
    # The stuck middle reading (37.1) should interpolate linearly to 25.0
    assert cleaned.loc[cleaned["date"] == "2026-02-15", "temp_c"].iloc[0] == 25.0


def test_clean_data_normal_zone_distance_unchanged():
    """Verify highway distance for Highway/City is not divided."""
    df = pd.DataFrame({
        "highway_distance_km": [15.0, 8.5],
        "zone": ["Highway", "City"]
    })
    cleaned = clean_data(df, is_train=False)
    assert cleaned["highway_distance_km"].tolist() == [15.0, 8.5]