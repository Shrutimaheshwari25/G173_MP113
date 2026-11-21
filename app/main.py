import os
import json
import logging
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Setup JSON logging to stdout
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "time": self.formatTime(record, self.datefmt)
        }
        if hasattr(record, "props"):
            log_entry.update(record.props)
        return json.dumps(log_entry)

logger = logging.getLogger("fuel_sales_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
if not logger.handlers:
    logger.addHandler(handler)

app = FastAPI(title="MP113 Fuel Station Daily Sales Prediction API", version="1.0.0")

MODEL_PATH = os.path.join("models", "model.joblib")
model = None


def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            logger.warning("Model file not found at %s. Service in mock/stub mode.", MODEL_PATH)
    return model


class PredictionRequest(BaseModel):
    highway_distance_km: float = Field(..., description="Distance to the highway in km")
    nearby_pumps: int = Field(..., le=20, description="Nearby pumps within 3 km (Max 20)")
    price_rs: float = Field(..., description="Petrol price in rupees per litre")
    trucks_passing: int = Field(..., ge=0, description="Trucks passing per day (Min 0)")
    temp_c: float = Field(..., description="Maximum temperature in deg C")
    staff: int = Field(..., description="Attendants on duty")
    ownership: str = Field(..., description="Ownership (public_sector, private, cooperative)")
    zone: str = Field(..., description="Zone (Highway, City, Industrial)")


class PredictionResponse(BaseModel):
    litres_per_day: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
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
        # Fallback baseline prediction if model is not yet trained
        prediction = 15000.0

    logger.info("Prediction generated", extra={"props": {"prediction": prediction}})

    return PredictionResponse(litres_per_day=round(prediction, 2))

