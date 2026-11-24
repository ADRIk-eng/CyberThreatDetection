"""Optional API for integration with the future multi-input frontend."""
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict_url import load_predictor, predict

app = FastAPI(title="Cyber Threat Detector API", version="0.1.0")

class UrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)

@lru_cache(maxsize=1)
def model_bundle():
    return load_predictor()

@app.get("/health")
def health():
    return {"status": "ok", "model": "url_random_forest"}

@app.post("/predict/url")
def predict_url(request: UrlRequest):
    try:
        return predict(request.url, model_bundle())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
