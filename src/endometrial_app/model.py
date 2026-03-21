from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


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


def _resolve_target_score(predictions: Any, predicted_index: int) -> Any:
    output_shape = tuple(predictions.shape)
    if len(output_shape) == 2 and output_shape[-1] == 1:
        return predictions[:, 0] if predicted_index == 1 else 1.0 - predictions[:, 0]
    return predictions[:, predicted_index]


def _to_uint8(array: np.ndarray) -> np.ndarray:
    clipped = np.asarray(array, dtype=np.float32)
    clipped = clipped - clipped.min()
    maximum = clipped.max()
    if maximum > 0:
        clipped = clipped / maximum
    return np.uint8(np.clip(clipped * 255.0, 0, 255))


def _activation_region_label(centroid_x: float, centroid_y: float) -> str:
    horizontal = "left" if centroid_x < 0.33 else "center" if centroid_x < 0.66 else "right"
    vertical = "upper" if centroid_y < 0.33 else "middle" if centroid_y < 0.66 else "lower"
    return f"{vertical} {horizontal}"


def build_attention_explanation(
    loaded_model: LoadedModel,
    image: Image.Image,
    image_batch: np.ndarray,
    predicted_index: int,
    probabilities: dict[str, float],
) -> dict[str, Any]:
    import tensorflow as tf

    model = loaded_model.model
    image_tensor = tf.convert_to_tensor(image_batch)
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor, training=False)
        target_score = _resolve_target_score(predictions, predicted_index)

    gradients = tape.gradient(target_score, image_tensor)[0]
    saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
    if float(tf.reduce_max(saliency)) > 0:
        saliency = saliency / tf.reduce_max(saliency)
    heatmap_array = saliency.numpy()

    model_input = Image.fromarray(np.uint8(np.clip(image_batch[0], 0, 255)))
    heatmap_u8 = Image.fromarray(_to_uint8(heatmap_array)).convert("L").resize(
        model_input.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_color = ImageOps.colorize(
        heatmap_u8,
        black="#dfe7ea",
        mid="#0e4d73",
        white="#1cb595",
    )

    overlay = model_input.convert("RGBA")
    overlay_color = heatmap_color.convert("RGBA")
    overlay_alpha = heatmap_u8.point(lambda pixel: int(pixel * 0.72))
    overlay_color.putalpha(overlay_alpha)
    overlay = Image.alpha_composite(overlay, overlay_color).convert("RGB")

    active_threshold = float(np.quantile(heatmap_array, 0.85))
    active_mask = heatmap_array >= active_threshold
    if not np.any(active_mask):
        active_mask = heatmap_array >= float(heatmap_array.max())

    active_indices = np.argwhere(active_mask)
    centroid_y = float(active_indices[:, 0].mean() / heatmap_array.shape[0])
    centroid_x = float(active_indices[:, 1].mean() / heatmap_array.shape[1])
    focus_region = _activation_region_label(centroid_x, centroid_y)
    focus_coverage = float(active_mask.mean())

    ordered_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    winning_label, winning_score = ordered_probabilities[0]
    runner_up_label, runner_up_score = ordered_probabilities[1]
    margin = float(winning_score - runner_up_score)

    return {
        "model_input_image": model_input,
        "attention_overlay_image": overlay,
        "attention_heatmap_image": heatmap_color,
        "focus_region": focus_region,
        "focus_coverage": focus_coverage,
        "winning_label": winning_label,
        "runner_up_label": runner_up_label,
        "margin": margin,
        "attention_layer": "input-gradient saliency",
    }
