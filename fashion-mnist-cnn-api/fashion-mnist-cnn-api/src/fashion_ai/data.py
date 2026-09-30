"""Dataset loading and image preprocessing."""

import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_splits(seed: int = 42):
    """Return stratified train/validation splits and untouched official test set."""
    (x_dev, y_dev), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    idx_train, idx_val = train_test_split(
        np.arange(len(y_dev)), test_size=0.1, stratify=y_dev, random_state=seed
    )
    def prepare(x):
        return x.astype("float32")[..., None] / 255.0
    return (
        (prepare(x_dev[idx_train]), y_dev[idx_train]),
        (prepare(x_dev[idx_val]), y_dev[idx_val]),
        (prepare(x_test), y_test),
    )


def decode_image(contents: bytes) -> np.ndarray:
    """Decode a PNG image for inference; resize to the dataset's 28x28 format."""
    try:
        image = Image.open(BytesIO(contents))
        if image.format != "PNG":
            raise ValueError("Only PNG images are accepted")
        image = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid PNG image") from exc
    return np.asarray(image, dtype="float32")[None, ..., None] / 255.0
