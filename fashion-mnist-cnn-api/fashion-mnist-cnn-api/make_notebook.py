"""Generate the step-by-step analysis notebook (run with nbclient afterward)."""

from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md("""# Fashion-MNIST: CNN + baseline + inference

**Author:** Tegar Reski Pratama (Informatika, Universitas Muhammadiyah Malang). This notebook analyzes the trained model from `python -m fashion_ai.train`. Run the cells sequentially after training. Test data is used for final reporting, never for training or early stopping.
""")
md("## 1. Imports and reproducibility")
code("""from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from fashion_ai import LABELS
from fashion_ai.data import load_splits
from fashion_ai.inference import load_model, predict_bytes

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
ARTIFACTS = ROOT / 'artifacts'
print('Artifact directory:', ARTIFACTS)
""")
md("## 2. Load the dataset and check label distribution")
code("""(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_splits(seed=42)
print('Train:', x_train.shape, 'Validation:', x_val.shape, 'Test:', x_test.shape)
print('Train class counts:', np.bincount(y_train).tolist())
print('Validation class counts:', np.bincount(y_val).tolist())
print('Test class counts:', np.bincount(y_test).tolist())
assert x_train.min() >= 0 and x_train.max() <= 1
assert x_test.shape == (10000, 28, 28, 1)
""")
md("## 3. Inspect sample images")
code("""fig, axes = plt.subplots(2, 5, figsize=(11, 5))
for label, ax in enumerate(axes.flat):
    index = int(np.flatnonzero(y_train == label)[0])
    ax.imshow(x_train[index, :, :, 0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(LABELS[label]); ax.axis('off')
plt.tight_layout(); plt.show()
""")
md("## 4. Read the genuine training record")
code("""metrics = json.loads((ARTIFACTS / 'metrics.json').read_text())
print('Run type:', metrics['run_type'], 'Epochs:', metrics['epochs_run'])
for key in ['baseline_validation_accuracy', 'baseline_test_accuracy', 'cnn_test_accuracy', 'cnn_test_macro_f1']:
    print(f'{key}: {metrics[key]:.4f}')
assert metrics['run_type'] == 'full'
""")
md("## 5. Compare learning curves")
code("""history = metrics['history']
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ['loss', 'accuracy']):
    ax.plot(history[metric], label='Train')
    ax.plot(history['val_' + metric], label='Validation')
    ax.set(xlabel='Epoch', title=metric.capitalize()); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
""")
md("## 6. Load saved model and evaluate held-out test data")
code("""model = load_model(ARTIFACTS / 'model.keras')
logits = np.asarray(model.predict(x_test, batch_size=256, verbose=0))
pred = logits.argmax(axis=1)
print(classification_report(y_test, pred, target_names=LABELS, digits=4))
print('Accuracy from model:', round(float(np.mean(pred == y_test)), 4))
assert abs(float(np.mean(pred == y_test)) - metrics['cnn_test_accuracy']) < 1e-8
""")
md("## 7. Which classes get confused?")
code("""matrix = confusion_matrix(y_test, pred, labels=range(10))
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
plt.tight_layout(); plt.show()
tmp = matrix.copy(); np.fill_diagonal(tmp, 0)
flat = np.argsort(tmp.ravel())[-5:][::-1]
for pos in flat:
    actual, predicted = np.unravel_index(pos, tmp.shape)
    print(f'{LABELS[actual]} -> {LABELS[predicted]}: {tmp[actual, predicted]}')
""")
md("## 8. Inspect difficult mistakes")
code("""prob = np.exp(logits - logits.max(axis=1, keepdims=True))
prob /= prob.sum(axis=1, keepdims=True)
wrong = np.flatnonzero(pred != y_test)
most_confident = wrong[np.argsort(prob[wrong, pred[wrong]])[-10:][::-1]]
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in zip(most_confident, axes.flat):
    ax.imshow(x_test[i, :, :, 0], cmap='gray')
    ax.set_title(f'True: {LABELS[y_test[i]]} | Pred: {LABELS[pred[i]]} ({prob[i, pred[i]]:.0%})', fontsize=9)
    ax.axis('off')
plt.tight_layout(); plt.show()
""")
md("## 9. Verify the saved model through the same PNG inference path used by the API")
code("""from io import BytesIO
from PIL import Image
buffer = BytesIO()
Image.fromarray((x_test[0, :, :, 0] * 255).astype('uint8')).save(buffer, format='PNG')
result = predict_bytes(model, buffer.getvalue())
print('True class:', LABELS[y_test[0]])
print('Top-3:', result)
assert result[0]['class_id'] == int(pred[0])
""")
md("""## Interpretation

Fashion-MNIST is a 28×28 grayscale benchmark, so the API expects similarly prepared images. Compare test accuracy and macro F1 with the linear baseline; inspect the confusion matrix and confident mistakes before claiming strengths. The model is a demonstration, not a production-ready fashion product recognizer. See `README.md` for setup, data citation, limits, and API usage.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
})
path = root / 'notebooks' / 'fashion_mnist_analysis.ipynb'
nbf.write(notebook, path)
print(path)
