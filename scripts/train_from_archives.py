from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from endometrial_app.data_prep import (
    assign_grouped_splits,
    assign_similarity_groups,
    build_similarity_group_summary,
    compute_difference_hash,
    summarize_cross_split_similarity,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an endometrial infection classifier directly from infected and notinfected zip archives."
    )
    parser.add_argument("--infected-zip", type=Path, required=True)
    parser.add_argument("--uninfected-zip", type=Path, required=True)
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("data/processed/training_workspace"),
        help="Directory where extracted files and split folders will be created.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("models/endometrial_classifier.keras"),
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("artifacts/class_names.json"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("artifacts/training_summary.json"),
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("artifacts/training_history.csv"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=4,
        help="Maximum dHash Hamming distance treated as a near-duplicate for grouped splitting.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def extract_archive(archive_path: Path, label: str, destination_root: Path) -> list[Path]:
    class_dir = destination_root / label
    if class_dir.exists():
        shutil.rmtree(class_dir)
    class_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            file_name = Path(member.filename).name
            if not file_name:
                continue
            target_path = class_dir / file_name
            target_path.write_bytes(archive.read(member))
            extracted_files.append(target_path)
    return extracted_files


def inspect_image(image_path: Path, label: str) -> dict[str, object]:
    record: dict[str, object] = {
        "label": label,
        "file_name": image_path.name,
        "source_path": str(image_path),
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }

    try:
        with Image.open(image_path) as image:
            image.load()
            record["width"] = image.width
            record["height"] = image.height
            record["dhash"] = compute_difference_hash(image)
        record["is_valid"] = True
    except Exception as exc:
        record["width"] = None
        record["height"] = None
        record["dhash"] = None
        record["is_valid"] = False
        record["error"] = str(exc)

    return record


def build_manifest(extracted_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for label in ["infected", "uninfected"]:
        for image_path in sorted((extracted_dir / label).glob("*")):
            if image_path.is_file():
                records.append(inspect_image(image_path, label))
    return pd.DataFrame(records)


def deduplicate_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = manifest_df.loc[manifest_df["is_valid"]].copy()

    cross_class_duplicates = (
        valid_df.groupby("sha256")["label"]
        .nunique()
        .reset_index(name="num_labels")
        .query("num_labels > 1")
    )
    if not cross_class_duplicates.empty:
        raise ValueError("Cross-class duplicate images detected. Resolve them before training.")

    clean_df = (
        valid_df.drop_duplicates(subset=["label", "sha256"], keep="first")
        .sort_values(["label", "file_name"])
        .reset_index(drop=True)
    )
    return clean_df


def materialize_splits(
    grouped_df: pd.DataFrame,
    split_dir: Path,
    seed: int,
) -> dict[str, pd.DataFrame]:
    if split_dir.exists():
        shutil.rmtree(split_dir)

    split_assignments = assign_grouped_splits(grouped_df, seed=seed)
    split_frames = {
        split_name: split_assignments.loc[split_assignments["split"] == split_name].reset_index(drop=True)
        for split_name in ["train", "validation", "test"]
    }
    for split_name, frame in split_frames.items():
        frame = frame.copy()
        target_paths: dict[int, str] = {}
        for label in ["infected", "uninfected"]:
            class_dir = split_dir / split_name / label
            class_dir.mkdir(parents=True, exist_ok=True)
            subset = frame.loc[frame["label"] == label]
            for row in subset.itertuples():
                target_name = f"{row.sha256[:12]}_{row.file_name}"
                target_path = class_dir / target_name
                shutil.copy2(Path(row.source_path), target_path)
                target_paths[int(row.Index)] = str(target_path)
        frame["target_path"] = frame.index.map(target_paths.get)
        split_frames[split_name] = frame

    return split_frames


def make_dataset(directory: Path, image_size: int, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["infected", "uninfected"],
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(image_size: int) -> tuple[keras.Model, keras.Model, bool]:
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    pretrained_weights_loaded = True
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception:
        pretrained_weights_loaded = False
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=None,
        )

    backbone.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model, backbone, pretrained_weights_loaded


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    workspace_dir = args.workspace_dir.resolve()
    extracted_dir = workspace_dir / "extracted"
    split_dir = workspace_dir / "splits"
    audit_dir = args.summary_path.resolve().parent / "audit"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    extract_archive(args.infected_zip.resolve(), "infected", extracted_dir)
    extract_archive(args.uninfected_zip.resolve(), "uninfected", extracted_dir)

    manifest_df = build_manifest(extracted_dir)
    raw_manifest_path = audit_dir / "raw_manifest.csv"
    manifest_df.to_csv(raw_manifest_path, index=False)
    clean_df = deduplicate_manifest(manifest_df)
    clean_manifest_path = audit_dir / "clean_manifest.csv"
    clean_df.to_csv(clean_manifest_path, index=False)
    grouped_df = assign_similarity_groups(
        clean_df,
        threshold=args.near_duplicate_threshold,
    )
    grouped_manifest_path = audit_dir / "grouped_manifest.csv"
    grouped_df.to_csv(grouped_manifest_path, index=False)
    split_frames = materialize_splits(grouped_df, split_dir, args.seed)
    split_audit_df = pd.concat(split_frames.values(), ignore_index=True)
    split_manifest_path = audit_dir / "split_manifest.csv"
    split_audit_df.to_csv(split_manifest_path, index=False)
    similarity_summary = build_similarity_group_summary(grouped_df)
    cross_split_similarity = summarize_cross_split_similarity(
        split_audit_df,
        threshold=args.near_duplicate_threshold,
    )

    train_ds = make_dataset(split_dir / "train", args.image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(split_dir / "validation", args.image_size, args.batch_size, False, args.seed)
    test_ds = make_dataset(split_dir / "test", args.image_size, args.batch_size, False, args.seed)

    model, backbone, pretrained_weights_loaded = build_model(args.image_size)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    backbone.trainable = True
    fine_tune_at = max(1, len(backbone.layers) - 30)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=len(history_head.history["loss"]),
        epochs=len(history_head.history["loss"]) + args.fine_tune_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model.resolve())

    args.labels_path.parent.mkdir(parents=True, exist_ok=True)
    args.labels_path.write_text(json.dumps(["infected", "uninfected"], indent=2), encoding="utf-8")

    history_df = pd.concat(
        [
            pd.DataFrame(history_head.history),
            pd.DataFrame(history_fine.history),
        ],
        ignore_index=True,
    )
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(args.history_path.resolve(), index=False)

    summary = {
        "pretrained_weights_loaded": pretrained_weights_loaded,
        "raw_counts": manifest_df.loc[manifest_df["is_valid"], "label"].value_counts().sort_index().to_dict(),
        "data_quality": {
            "original_image_count": int(manifest_df.shape[0]),
            "valid_image_count": int(manifest_df["is_valid"].sum()),
            "invalid_images_removed": int((~manifest_df["is_valid"]).sum()),
            "exact_duplicates_removed": int(manifest_df["is_valid"].sum() - clean_df.shape[0]),
            "near_duplicate_hash": "difference-hash-8x8",
            "near_duplicate_threshold": int(args.near_duplicate_threshold),
            "split_strategy": "group-aware split over perceptual similarity clusters",
            "similarity_groups": similarity_summary,
            "cross_split_similarity_audit": cross_split_similarity,
        },
        "clean_counts": clean_df["label"].value_counts().sort_index().to_dict(),
        "split_counts": {
            split_name: frame["label"].value_counts().sort_index().to_dict()
            for split_name, frame in split_frames.items()
        },
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
        "evaluation_scope": (
            "Internal held-out evaluation after exact-duplicate removal and grouped perceptual-similarity splitting. "
            "External validation, repeated grouped resampling, and study-level partitioning are still recommended."
        ),
        "audit_artifacts": {
            "raw_manifest_path": str(raw_manifest_path),
            "clean_manifest_path": str(clean_manifest_path),
            "grouped_manifest_path": str(grouped_manifest_path),
            "split_manifest_path": str(split_manifest_path),
        },
        "model_path": str(args.output_model.resolve()),
        "labels_path": str(args.labels_path.resolve()),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
