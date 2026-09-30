"""FastAPI inference app. Set MODEL_PATH to override artifacts/model.keras."""

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .inference import load_model, predict_bytes

app = FastAPI(title="Fashion-MNIST CNN", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(Path(os.getenv("MODEL_PATH", "artifacts/model.keras")))
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _model


@app.get("/health")
def health():
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.keras"))
    return {"status": "ready" if path.is_file() else "model_missing"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type != "image/png":
        raise HTTPException(status_code=415, detail="Upload a PNG image")
    contents = await file.read(2 * 1024 * 1024 + 1)
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 2 MB")
    try:
        predictions = predict_bytes(get_model(), contents)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"predictions": predictions}
