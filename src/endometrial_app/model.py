from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


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


def _normalize_heatmap(array: np.ndarray) -> np.ndarray:
    normalized = np.asarray(array, dtype=np.float32)
    normalized = normalized - normalized.min()
    maximum = float(normalized.max())
    if maximum > 0:
        normalized = normalized / maximum
    return normalized


def _otsu_threshold(array: np.ndarray, bins: int = 256) -> float:
    flattened = np.clip(np.asarray(array, dtype=np.float32).ravel(), 0.0, 1.0)
    if flattened.size == 0:
        return 1.0

    histogram, bin_edges = np.histogram(flattened, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    if np.count_nonzero(histogram) <= 1:
        return float(flattened.max())

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * bin_centers)
    total_weight = cumulative_weight[-1]
    total_mean = cumulative_mean[-1]

    background_weight = cumulative_weight
    foreground_weight = total_weight - cumulative_weight
    numerator = (total_mean * background_weight - cumulative_mean) ** 2
    denominator = background_weight * foreground_weight
    between_class_variance = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(between_class_variance))
    return float(bin_centers[best_index])


def _smooth_heatmap(array: np.ndarray, blur_radius: float = 6.0) -> np.ndarray:
    heatmap_u8 = Image.fromarray(_to_uint8(array)).convert("L")
    smoothed = heatmap_u8.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return _normalize_heatmap(np.asarray(smoothed, dtype=np.float32))


def _activation_region_label(centroid_x: float, centroid_y: float) -> str:
    horizontal = "left" if centroid_x < 0.33 else "center" if centroid_x < 0.66 else "right"
    vertical = "upper" if centroid_y < 0.33 else "middle" if centroid_y < 0.66 else "lower"
    return f"{vertical} {horizontal}"


def _focus_pattern_label(coverage: float) -> str:
    if coverage <= 0.08:
        return "compact"
    if coverage <= 0.18:
        return "moderately concentrated"
    if coverage <= 0.30:
        return "broad"
    return "diffuse"


def _build_attention_mask(heatmap_array: np.ndarray) -> tuple[np.ndarray, float]:
    normalized_heatmap = _smooth_heatmap(heatmap_array)
    otsu_threshold = _otsu_threshold(normalized_heatmap)
    adaptive_floor = float(normalized_heatmap.mean() + 0.20 * normalized_heatmap.std())
    blended_threshold = 0.55 * otsu_threshold + 0.45 * adaptive_floor
    active_threshold = float(np.clip(max(blended_threshold, 0.18), 0.18, 0.75))
    active_mask = normalized_heatmap >= active_threshold

    if not np.any(active_mask):
        fallback_threshold = float(np.clip(normalized_heatmap.max() * 0.85, 0.0, 1.0))
        active_mask = normalized_heatmap >= fallback_threshold
        active_threshold = fallback_threshold

    return active_mask, active_threshold


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
    heatmap_array = _normalize_heatmap(saliency.numpy())

    model_input = Image.fromarray(np.uint8(np.clip(image_batch[0], 0, 255)))
    heatmap_u8 = Image.fromarray(_to_uint8(heatmap_array)).convert("L").resize(
        model_input.size,
        Image.Resampling.BILINEAR,
    )
    heatmap_u8 = ImageOps.autocontrast(heatmap_u8)
    heatmap_color = ImageOps.colorize(
        heatmap_u8,
        black="#0b1620",
        mid="#0e6fb2",
        white="#19d3a5",
    )

    overlay = model_input.convert("RGBA")
    overlay_color = heatmap_color.convert("RGBA")
    overlay_alpha = heatmap_u8.point(lambda pixel: int(pixel * 0.72))
    overlay_color.putalpha(overlay_alpha)
    overlay = Image.alpha_composite(overlay, overlay_color).convert("RGB")

    active_mask, active_threshold = _build_attention_mask(heatmap_array)
    active_indices = np.argwhere(active_mask)
    active_weights = heatmap_array[active_mask]
    if float(active_weights.sum()) > 0:
        centroid_y = float(np.average(active_indices[:, 0], weights=active_weights) / heatmap_array.shape[0])
        centroid_x = float(np.average(active_indices[:, 1], weights=active_weights) / heatmap_array.shape[1])
    else:
        centroid_y = float(active_indices[:, 0].mean() / heatmap_array.shape[0])
        centroid_x = float(active_indices[:, 1].mean() / heatmap_array.shape[1])
    focus_region = _activation_region_label(centroid_x, centroid_y)
    focus_coverage = float(active_mask.mean())
    focus_pattern = _focus_pattern_label(focus_coverage)

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
        "focus_pattern": focus_pattern,
        "high_attention_threshold": active_threshold,
        "winning_label": winning_label,
        "runner_up_label": runner_up_label,
        "margin": margin,
        "attention_layer": "input-gradient saliency",
    }
