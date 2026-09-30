"""Train, evaluate, and save all reproducible run artifacts."""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import tensorflow as tf

from . import LABELS
from .data import load_splits
from .model import build_model


def save_confusion_matrix(matrix: np.ndarray, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=np.arange(10), yticks=np.arange(10),
           xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted", ylabel="Actual", title="Test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_history(history: tf.keras.callbacks.History, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set(xlabel="Epoch", title=metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(epochs: int, out_dir: Path, seed: int = 42, subset: int | None = None) -> dict:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_splits(seed)
    if subset:
        if subset < 10 or subset > len(y_train):
            raise ValueError("subset must be between 10 and the training split size")
        selected, _ = train_test_split(
            np.arange(len(y_train)), train_size=subset, stratify=y_train, random_state=seed
        )
        x_train, y_train = x_train[selected], y_train[selected]

    # A genuine linear baseline on the same training records. Test labels are
    # only used after all fitting and model selection has finished.
    baseline = SGDClassifier(loss="log_loss", random_state=seed, max_iter=200, tol=1e-3)
    baseline.fit(x_train.reshape(len(x_train), -1), y_train)
    baseline_val_accuracy = accuracy_score(
        y_val, baseline.predict(x_val.reshape(len(x_val), -1))
    )

    model = build_model()
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=128, callbacks=callbacks, verbose=2,
    )
    baseline_test_pred = baseline.predict(x_test.reshape(len(x_test), -1))
    cnn_test_pred = np.argmax(model.predict(x_test, batch_size=256, verbose=0), axis=1)
    report = classification_report(y_test, cnn_test_pred, target_names=LABELS,
                                   output_dict=True, zero_division=0)
    metrics = {
        "run_type": "smoke_test" if subset else "full",
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "baseline_validation_accuracy": float(baseline_val_accuracy),
        "baseline_test_accuracy": float(accuracy_score(y_test, baseline_test_pred)),
        "cnn_test_accuracy": float(accuracy_score(y_test, cnn_test_pred)),
        "cnn_test_macro_f1": float(f1_score(y_test, cnn_test_pred, average="macro")),
        "classification_report": report,
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
        "tensorflow_version": tf.__version__,
    }
    model.save(out_dir / "model.keras")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    save_confusion_matrix(confusion_matrix(y_test, cnn_test_pred, labels=range(10)),
                          out_dir / "confusion_matrix.png")
    save_history(history, out_dir / "learning_curves.png")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None,
                        help="Training subset for a quick smoke test (not final results)")
    args = parser.parse_args()
    results = run(args.epochs, args.out_dir, args.seed, args.subset)
    print(json.dumps({k: v for k, v in results.items() if k not in
                      {"classification_report", "history"}}, indent=2))


if __name__ == "__main__":
    main()
