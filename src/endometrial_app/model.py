from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LoadedModel:
    backend: str
    model: Any
    class_names: list[str]


def _ensure_model_exists(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found at {model_path}. Export a trained model before starting the app."
        )


def load_model(model_path: Path, class_names: list[str]) -> LoadedModel:
    _ensure_model_exists(model_path)

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    return LoadedModel(backend="tensorflow", model=model, class_names=class_names)


def preprocess_image(image: Any, image_size: tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def _normalize_probabilities(raw_output: np.ndarray, class_names: list[str]) -> np.ndarray:
    output = np.asarray(raw_output, dtype=np.float32).squeeze()

    if output.ndim == 0:
        positive_score = float(output)
        if positive_score < 0.0 or positive_score > 1.0:
            positive_score = 1.0 / (1.0 + np.exp(-positive_score))
        return np.array([1.0 - positive_score, positive_score], dtype=np.float32)

    if output.ndim == 1 and output.shape[0] == len(class_names):
        if np.all(output >= 0.0) and np.isclose(output.sum(), 1.0, atol=1e-3):
            return output.astype(np.float32)
        shifted = output - np.max(output)
        exp_output = np.exp(shifted)
        return (exp_output / exp_output.sum()).astype(np.float32)

    raise ValueError(
        "Unsupported model output shape. Expected a scalar sigmoid output or a vector with one score per class."
    )


def predict_probabilities(loaded_model: LoadedModel, image_batch: np.ndarray) -> dict[str, float]:
    raw_predictions = loaded_model.model.predict(image_batch, verbose=0)
    probabilities = _normalize_probabilities(raw_predictions, loaded_model.class_names)
    return {
        class_name: float(probability)
        for class_name, probability in zip(loaded_model.class_names, probabilities.tolist())
    }
