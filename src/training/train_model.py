import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import (
    RAW_DIR,
    PROCESSED_DIR,
    VAL_DIR,
    TEST_DIR,
    TRAIN_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    MODELS_DIR,
    IMG_SIZE
)

def ensure_output_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "accuracy_loss_plots").mkdir(parents=True, exist_ok=True)


def build_model(img_size=(224, 224)):
    # Base model: MobileNetV2 pretrained on ImageNet (Transfer Learning)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x) 
    model = tf.keras.Model(inputs, outputs)
    return model

def plot_history(history: dict, out_dir: Path):
    # Accuracy plot
    plt.figure()
    plt.plot(history.get("accuracy", []))
    plt.plot(history.get("val_accuracy", []))
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"])
    plt.savefig(out_dir / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(history.get("loss", []))
    plt.plot(history.get("val_loss", []))
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"])
    plt.savefig(out_dir / "loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    
def main():
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_output_dirs()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False,
    )
    
    class_names = train_ds.class_names 
    print("Class names (folder order):", class_names)
    print("Note: For binary label_mode, the second class is label 1.")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # --------- Model ---------
    model = build_model(img_size=IMG_SIZE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
     # --------- Training ---------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=callbacks,
    )
    
    history = history_obj.history
    
        
    # --------- Fine-tuning ---------
    base_model = model.get_layer("mobilenetv2_1.00_224")
    print("Base model name:", base_model.name)
    base_model.trainable = True

    N = 30
    for layer in base_model.layers[:-N]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    fine_history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
    )

    for k, v in fine_history_obj.history.items():
        history[k] = history.get(k, []) + v


    # --------- Save Model ---------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "smart_drug_model.keras"
    model.save(model_path)


    # --------- Save History + Plots ---------
    stamp = datetime.utcnow().isoformat() + "Z"
    out_dir = RESULTS_DIR / "accuracy_loss_plots"
    plot_history(history, out_dir)

    history_out = RESULTS_DIR / "training_history.json"
    payload = {
        "created_at": stamp,
        "img_size": list(IMG_SIZE),
        "class_names": class_names,
        "epochs_ran": len(history.get("loss", [])),
        "history": history,
        "saved_model": str(model_path),
    }
    history_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nDONE Training finished.")
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {out_dir}")
    print(f"History saved to: {history_out}")


if __name__ == "__main__":
    main()

