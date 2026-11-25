from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_valid():
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 10,
        "price_rs": 95.5,
        "trucks_passing": 800,
        "temp_c": 32.0,
        "staff": 12,
        "ownership": "public_sector",
        "zone": "Highway"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "litres_per_day" in data
    assert isinstance(data["litres_per_day"], float)


def test_predict_rule_nearby_pumps_exceeded():
    # Rule 1: nearby_pumps above 20 must be rejected with 422
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 25,
        "price_rs": 95.5,
        "trucks_passing": 800,
        "temp_c": 32.0,
        "staff": 12,
        "ownership": "public_sector",
        "zone": "Highway"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rule_trucks_passing_negative():
    # Rule 2: trucks_passing below 0 must be rejected with 422
    payload = {
        "highway_distance_km": 5.2,
        "nearby_pumps": 10,
        "price_rs": 95.5,
        "trucks_passing": -10,
        "temp_c": 32.0,
        "staff": 12,
        "ownership": "public_sector",
        "zone": "Highway"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_boundary_rules_accepted():
    """Verify that boundary values (nearby_pumps=20 and trucks_passing=0) are valid."""
    payload = {
        "highway_distance_km": 0.0,
        "nearby_pumps": 20,  # Exact upper bound for Rule 1
        "price_rs": 96.0,
        "trucks_passing": 0,  # Exact lower bound for Rule 2
        "temp_c": 30.0,
        "staff": 5,
        "ownership": "private",
        "zone": "City"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "litres_per_day" in data
    assert data["litres_per_day"] > 0


def test_predict_missing_required_fields():
    """Verify that omitting required schema fields triggers 422 Unprocessable Entity."""
    incomplete_payload = {
        "nearby_pumps": 5,
        "price_rs": 95.0,
        "temp_c": 28.0
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_invalid_data_type():
    """Verify that passing incorrect types (e.g., text for float) triggers 422."""
    invalid_payload = {
        "highway_distance_km": "very_far",
        "nearby_pumps": 5,
        "price_rs": 95.0,
        "trucks_passing": 400,
        "temp_c": 30.0,
        "staff": 8,
        "ownership": "cooperative",
        "zone": "Industrial"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_different_zones_and_ownership():
    """Verify prediction across different valid zones and ownership structures."""
    for zone in ["Highway", "City", "Industrial"]:
        for ownership in ["public_sector", "private", "cooperative"]:
            payload = {
                "highway_distance_km": 4.5,
                "nearby_pumps": 6,
                "price_rs": 94.8,
                "trucks_passing": 350,
                "temp_c": 31.5,
                "staff": 10,
                "ownership": ownership,
                "zone": zone
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert isinstance(response.json()["litres_per_day"], float)


