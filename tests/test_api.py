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

