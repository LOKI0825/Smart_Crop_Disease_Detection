"""Train and evaluate a MobileNetV2 plant-leaf classifier.

Expected dataset format: one directory per class, with images directly inside it.
Example: data/PlantVillage/Tomato___healthy/*.JPG
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = (224, 224)
SEED = 42
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("plant-disease-trainer")


def count_images_in_folder(folder: Path) -> int:
    return sum(1 for item in folder.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS)


def find_class_root(root: Path) -> Path:
    """Find the directory whose direct child directories are image classes.

    Supports common nested downloads such as data/PlantVillage/PlantVillage/...
    without recursively mixing class directories and individual images.
    """
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Download/extract PlantVillage and pass its path with --data-dir."
        )

    current = root
    for _ in range(4):
        class_dirs = [
            d for d in current.iterdir()
            if d.is_dir() and not d.name.startswith(".") and count_images_in_folder(d) > 0
        ]
        if len(class_dirs) >= 2:
            return current

        child_dirs = [d for d in current.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(child_dirs) != 1:
            break
        current = child_dirs[0]

    # Fallback search for a nested directory containing at least two labeled folders.
    candidates: list[tuple[int, Path]] = []
    for candidate in root.rglob("*"):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        try:
            class_dirs = [
                d for d in candidate.iterdir()
                if d.is_dir() and not d.name.startswith(".") and count_images_in_folder(d) > 0
            ]
        except OSError:
            continue
        if len(class_dirs) >= 2:
            candidates.append((len(class_dirs), candidate))
    if candidates:
        candidates.sort(key=lambda pair: (pair[0], str(pair[1])), reverse=True)
        return candidates[0][1]

    raise ValueError(
        f"Could not find class folders under {root}. Expected one folder per class, such as "
        "PlantVillage/Tomato___healthy/*.JPG. See README.md for the correct structure."
    )


def collect_dataset(data_dir: Path) -> tuple[list[str], list[int], list[str], Path]:
    class_root = find_class_root(data_dir)
    class_dirs = sorted(
        d for d in class_root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and count_images_in_folder(d) > 0
    )
    class_names: list[str] = []
    all_paths: list[str] = []
    all_labels: list[int] = []

    for class_index, class_dir in enumerate(class_dirs):
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            continue
        class_names.append(class_dir.name)
        all_paths.extend(str(p.resolve()) for p in images)
        all_labels.extend([class_index] * len(images))

    # If an empty class directory was skipped, rebuild the label IDs safely.
    actual_names = [d.name for d in class_dirs if count_images_in_folder(d) > 0]
    if class_names != actual_names:
        raise RuntimeError("Internal label mapping error. Please check the dataset folders.")

    if len(class_names) < 2:
        raise ValueError(f"Found only {len(class_names)} class(es) in {class_root}; at least two are required.")

    counts = pd.Series(all_labels).value_counts()
    if int(counts.min()) < 4:
        raise ValueError("Each class needs at least four images to create train/validation/test splits.")

    LOGGER.info("Dataset directory: %s", class_root)
    LOGGER.info("Images: %d | classes: %d", len(all_paths), len(class_names))
    return all_paths, all_labels, class_names, class_root


def make_tf_dataset(paths: list[str], labels: list[int], batch_size: int, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        content = tf.io.read_file(path)
        image = tf.io.decode_image(content, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32)  # Values remain in [0, 255]; model rescales them.
        return image, tf.cast(label, tf.int32)

    dataset = dataset.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(paths), 12000), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenet_rescaling")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="disease_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, backbone


def save_evaluation(model: tf.keras.Model, test_ds: tf.data.Dataset, test_labels: list[int], class_names: list[str], output_dir: Path) -> None:
    LOGGER.info("Evaluating the best checkpoint on the held-out test split...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    report = classification_report(
        test_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv", index=True)

    matrix = confusion_matrix(test_labels, predicted_labels, labels=list(range(len(class_names))))
    figure_size = max(10, min(22, len(class_names) * 0.45))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.85))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Plant Disease Classifier — Test Confusion Matrix",
        ylabel="True class",
        xlabel="Predicted class",
    )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_images": int(len(test_labels)),
        "num_classes": len(class_names),
        "class_names": class_names,
        "split": "70% train / 15% validation / 15% test (stratified random split)",
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    LOGGER.info("Test accuracy: %.4f | test loss: %.4f", test_accuracy, test_loss)
    LOGGER.info("Evaluation files saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PlantVillage plant disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "PlantVillage", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Head-training epochs (default: 10)")
    parser.add_argument("--fine-tune-epochs", type=int, default=3, help="Fine-tuning epochs (default: 3; set 0 to skip)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size; use 16 or 8 on lower-memory systems")
    args = parser.parse_args()

    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive; fine-tune-epochs cannot be negative")

    tf.keras.utils.set_random_seed(SEED)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    output_dir = BASE_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    image_paths, labels, class_names, class_root = collect_dataset(args.data_dir)
    indices = np.arange(len(image_paths))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=np.asarray(labels)
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1764705882,  # 15% of the full dataset, after holding out 15% test.
        random_state=SEED,
        stratify=np.asarray(labels)[train_val_idx],
    )

    paths_array = np.asarray(image_paths)
    labels_array = np.asarray(labels, dtype=np.int32)
    train_paths, train_labels = paths_array[train_idx].tolist(), labels_array[train_idx].tolist()
    val_paths, val_labels = paths_array[val_idx].tolist(), labels_array[val_idx].tolist()
    test_paths, test_labels = paths_array[test_idx].tolist(), labels_array[test_idx].tolist()

    LOGGER.info("Split sizes — train: %d | validation: %d | test: %d", len(train_paths), len(val_paths), len(test_paths))
    train_ds = make_tf_dataset(train_paths, train_labels, args.batch_size, training=True)
    val_ds = make_tf_dataset(val_paths, val_labels, args.batch_size, training=False)
    test_ds = make_tf_dataset(test_paths, test_labels, args.batch_size, training=False)

    with (output_dir / "classes.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    model_path = output_dir / "plant_disease_model.keras"
    head_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_path), monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    )
    head_callbacks = [
        head_checkpoint,
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
    ]

    model, backbone = build_model(num_classes=len(class_names))
    LOGGER.info("Stage 1/2 — training the classification head...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=head_callbacks, verbose=1)

    # Keep the best head-only model as a candidate so fine-tuning cannot replace it
    # with a lower-validation-accuracy model.
    best_head_model = tf.keras.models.load_model(model_path)

    if args.fine_tune_epochs > 0:
        LOGGER.info("Stage 2/2 — fine-tuning the last MobileNetV2 layers...")
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tuned_candidate_path = output_dir / "fine_tuned_candidate.keras"
        fine_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(fine_tuned_candidate_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
        fine_callbacks = [
            fine_checkpoint,
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
        ]
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=fine_callbacks,
            verbose=1,
        )

        if fine_tuned_candidate_path.exists():
            fine_model = tf.keras.models.load_model(fine_tuned_candidate_path)
            _, head_val_accuracy = best_head_model.evaluate(val_ds, verbose=0)
            _, fine_val_accuracy = fine_model.evaluate(val_ds, verbose=0)
            LOGGER.info("Validation accuracy — frozen head: %.4f | fine-tuned: %.4f", head_val_accuracy, fine_val_accuracy)
            if fine_val_accuracy >= head_val_accuracy:
                fine_model.save(model_path)
                LOGGER.info("Selected fine-tuned checkpoint.")
            else:
                LOGGER.info("Keeping the higher-scoring head-only checkpoint.")
            fine_tuned_candidate_path.unlink(missing_ok=True)

    # Reload the highest-validation-accuracy checkpoint before final test evaluation.
    best_model = tf.keras.models.load_model(model_path)
    save_evaluation(best_model, test_ds, test_labels, class_names, BASE_DIR / "reports")
    LOGGER.info("Training completed. Model: %s", model_path)
    LOGGER.info("Detected class root: %s", class_root)
    LOGGER.info("Next step: run `python app.py` and open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
