import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from src.fuel_sales.data import build_preprocessor


def test_model_pipeline_train_predict():
    numeric_cols = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing", "temp_c", "staff"]
    categorical_cols = ["ownership", "zone"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])

    X_train = pd.DataFrame({
        "highway_distance_km": [5.0, 10.0, 3.0, 8.0],
        "nearby_pumps": [4, 8, 12, 6],
        "price_rs": [95.0, 96.0, 94.0, 95.5],
        "trucks_passing": [500, 600, 700, 550],
        "temp_c": [30.0, 32.0, 28.0, 31.0],
        "staff": [10, 12, 8, 11],
        "ownership": ["public_sector", "private", "cooperative", "public_sector"],
        "zone": ["Highway", "City", "Industrial", "City"]
    })
    y_train = [18000, 22000, 15000, 20000]

    pipeline.fit(X_train, y_train)

    X_test = pd.DataFrame({
        "highway_distance_km": [6.0],
        "nearby_pumps": [5],
        "price_rs": [95.2],
        "trucks_passing": [520],
        "temp_c": [29.5],
        "staff": [10],
        "ownership": ["private"],
        "zone": ["Highway"]
    })

    preds = pipeline.predict(X_test)
    assert len(preds) == 1
    assert preds[0] > 0

