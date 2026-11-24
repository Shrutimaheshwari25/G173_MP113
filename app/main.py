import os
import json
import logging
import sys
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Setup structured JSON logging to stdout (as per Handbook Lab P10 requirements)
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            log_entry.update(record.props)
        return json.dumps(log_entry)

logger = logging.getLogger("fuel_sales_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
if not logger.handlers:
    logger.addHandler(handler)

app = FastAPI(
    title="MP113 - Fuel Station Daily Sales Prediction API",
    description="Production prediction API for Bhopal Fuel Retailers' Federation with Pydantic validation rules and structured JSON logging.",
    version="1.0.0"
)

MODEL_PATH = os.path.join("models", "model.joblib")
model = None


def get_model():
    """Lazy loader for champion model artifact."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded champion model artifact successfully", extra={"props": {"model_path": MODEL_PATH}})
        else:
            logger.warning("Champion model artifact not found. Service running in fallback mode.", extra={"props": {"model_path": MODEL_PATH}})
    return model


class PredictionRequest(BaseModel):
    """
    Request payload for daily fuel sales prediction.
    Strictly implements client business rules (PROBLEM.md Section 5):
    1. nearby_pumps above 20 is impossible and must be rejected (le=20).
    2. trucks_passing below 0 is a typing mistake and must be rejected (ge=0).
    """
    highway_distance_km: float = Field(..., ge=0.0, description="Distance to highway in km")
    nearby_pumps: int = Field(..., ge=0, le=20, description="Rule 1: Nearby pumps within 3 km (Max 20)")
    price_rs: float = Field(..., gt=0.0, description="Petrol price in rupees per litre")
    trucks_passing: int = Field(..., ge=0, description="Rule 2: Trucks passing per day (Min 0)")
    temp_c: float = Field(..., description="Maximum temperature in deg C")
    staff: int = Field(..., ge=0, description="Attendants on duty")
    ownership: str = Field(..., description="Ownership category (public_sector, private, cooperative)")
    zone: str = Field(..., description="Location zone (Highway, City, Industrial)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "highway_distance_km": 5.2,
                "nearby_pumps": 10,
                "price_rs": 95.5,
                "trucks_passing": 800,
                "temp_c": 32.0,
                "staff": 12,
                "ownership": "public_sector",
                "zone": "Highway"
            }
        }
    )


class PredictionResponse(BaseModel):
    litres_per_day: float = Field(..., description="Predicted fuel sold in litres per day")


@app.get("/health")
def health():
    """Health check endpoint required by deployment readiness/liveness probes."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    """
    Predict daily fuel sales in litres for a fuel station.
    Validates rules in schema (422 for violations) and logs prediction in JSON.
    """
    start_time = time.time()
    loaded_model = get_model()

    input_data = pd.DataFrame([{
        "highway_distance_km": payload.highway_distance_km,
        "nearby_pumps": payload.nearby_pumps,
        "price_rs": payload.price_rs,
        "trucks_passing": payload.trucks_passing,
        "temp_c": payload.temp_c,
        "staff": payload.staff,
        "ownership": payload.ownership,
        "zone": payload.zone,
    }])

    if loaded_model is not None:
        prediction = float(loaded_model.predict(input_data)[0])
    else:
        # Fallback baseline prediction if model artifact is absent
        prediction = 15000.0

    duration_ms = round((time.time() - start_time) * 1000, 2)
    predicted_val = round(prediction, 2)

    # Structured JSON stdout logging without PII
    logger.info(
        "Prediction completed",
        extra={
            "props": {
                "prediction_litres": predicted_val,
                "zone": payload.zone,
                "duration_ms": duration_ms
            }
        }
    )

    return PredictionResponse(litres_per_day=predicted_val)


