import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from src.config import TEST_DIR, RESULTS_DIR, MODELS_DIR, IMG_SIZE


def ensure_dirs():
    (RESULTS_DIR / "confusion_matrix").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "sample_predictions").mkdir(parents=True, exist_ok=True)


def load_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
    )
    return test_ds


def save_confusion_matrix(cm, class_names, out_path: Path):
    plt.figure()
    plt.imshow(cm, cmap="YlGnBu")
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=20)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_sample_predictions(model, test_ds, class_names, out_dir: Path, max_images=12, threshold=0.5):
    images, labels = next(iter(test_ds.take(1)))
    preds = model.predict(images, verbose=0).reshape(-1)

    pred_labels = (preds >= threshold).astype(int)

    # Save a grid image
    n = min(len(images), max_images)
    plt.figure(figsize=(12, 8))
    for i in range(n):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_name = class_names[int(labels[i].numpy())]
        pred_name = class_names[int(pred_labels[i])]
        conf = preds[i] if pred_labels[i] == 1 else (1 - preds[i])  
        plt.title(f"T:{true_name}\nP:{pred_name} ({conf:.2f})", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "sample_predictions_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ensure_dirs()

    model_path = MODELS_DIR / "smart_drug_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    test_ds = load_test_dataset()
    class_names = test_ds.class_names 
    print("Test class names:", class_names)

    # Evaluate (loss + accuracy)
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTEST Accuracy: {test_acc:.4f}")
    print(f"TEST Loss: {test_loss:.4f}")

    y_true = []
    y_pred = []

    for batch_images, batch_labels in test_ds:
        probs = model.predict(batch_images, verbose=0).reshape(-1)
        preds = (probs >= 0.5).astype(int)
        y_true.extend(batch_labels.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )

    # Save confusion matrix
    cm_path = RESULTS_DIR / "confusion_matrix" / "confusion_matrix.png"
    save_confusion_matrix(cm, class_names, cm_path)

    # Save report
    report_path = RESULTS_DIR / "confusion_matrix" / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    # Save a JSON summary for your report
    summary = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    (RESULTS_DIR / "confusion_matrix" / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Save sample predictions grid
    save_sample_predictions(
        model, test_ds, class_names,
        RESULTS_DIR / "sample_predictions",
        max_images=12
    )

    print("\nSaved:")
    print(f"- Confusion Matrix: {cm_path}")
    print(f"- Classification Report: {report_path}")
    print(f"- Sample Predictions: {RESULTS_DIR / 'sample_predictions' / 'sample_predictions_grid.png'}")
    print("\nDONE Evaluation completed.")


if __name__ == "__main__":
    main()

