"""Meaningful API/inference contract checks without running model training."""

from io import BytesIO

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from fashion_ai import api
from fashion_ai.data import decode_image
from fashion_ai.inference import predict_bytes


def png_bytes():
    buffer = BytesIO()
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeModel:
    def __call__(self, image, training=False):
        assert image.shape == (1, 28, 28, 1)
        return np.arange(10, dtype=np.float32)[None, :]


def test_preprocessing_and_ordered_probabilities():
    image = decode_image(png_bytes())
    assert image.shape == (1, 28, 28, 1)
    assert image.dtype == np.float32
    top3 = predict_bytes(FakeModel(), png_bytes())
    assert [p["class_id"] for p in top3] == [9, 8, 7]
    assert all(0 < p["probability"] < 1 for p in top3)


def test_api_contract(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    client = TestClient(api.app)
    assert client.post("/predict", files={"file": ("sample.png", png_bytes(), "image/png")}).status_code == 200
    assert client.post("/predict", files={"file": ("bad.jpg", b"x", "image/jpeg")}).status_code == 415
    assert client.post("/predict", files={"file": ("bad.png", b"bad", "image/png")}).status_code == 422
