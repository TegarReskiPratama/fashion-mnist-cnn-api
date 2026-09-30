"""Small convolutional classifier. Outputs logits, not probabilities."""

import tensorflow as tf


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1), name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    logits = tf.keras.layers.Dense(10, name="logits")(x)
    model = tf.keras.Model(inputs, logits, name="fashion_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model
