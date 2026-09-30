"""Prediction helpers shared by CLI and API."""

from pathlib import Path
import numpy as np
import tensorflow as tf
from . import LABELS
from .data import decode_image


def load_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run the training script first.")
    return tf.keras.models.load_model(path, compile=False)


def predict_bytes(model: tf.keras.Model, contents: bytes, top_k: int = 3) -> list[dict]:
    image = decode_image(contents)
    logits = np.asarray(model(image, training=False))[0]
    if logits.shape != (len(LABELS),):
        raise ValueError("The model must return 10 logits")
    probabilities = tf.nn.softmax(logits).numpy()
    ranked = np.argsort(-probabilities)[:top_k]
    return [
        {"class_id": int(i), "label": LABELS[i], "probability": float(probabilities[i])}
        for i in ranked
    ]
