from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Protocol

from PIL import Image

from endometrial_app.config import Settings, get_settings
from endometrial_app.model import (
    LoadedModel,
    build_attention_explanation,
    load_model,
    predict_probabilities,
    preprocess_image,
)


class PredictorProtocol(Protocol):
    def predict(self, image: Image.Image) -> dict[str, float]:
        ...


@dataclass
class PredictionService:
    settings: Settings
    eager_load: bool = False

    def __post_init__(self) -> None:
        if self.eager_load:
            _ = self.model_bundle

    @classmethod
    def from_settings(cls) -> "PredictionService":
        return cls(settings=get_settings(), eager_load=False)

    @cached_property
    def model_bundle(self) -> LoadedModel:
        return load_model(self.settings.model_path, self.settings.class_names)

    def is_ready(self) -> bool:
        try:
            _ = self.model_bundle
            return True
        except Exception:
            return False

    def health(self) -> dict[str, object]:
        model_loaded = self.is_ready()
        return {
            "status": "ok" if model_loaded else "model_not_ready",
            "model_loaded": model_loaded,
            "model_path": str(self.settings.model_path),
            "class_names": self.settings.class_names,
        }

    def predict(self, image: Image.Image) -> dict[str, object]:
        image_batch = preprocess_image(image, self.settings.image_size)
        probabilities = predict_probabilities(self.model_bundle, image_batch)

        predicted_label = max(probabilities, key=probabilities.get)
        predicted_index = self.settings.class_names.index(predicted_label)
        confidence = float(probabilities[predicted_label])

        return {
            "predicted_label": predicted_label,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def explain_prediction(self, image: Image.Image, prediction: dict[str, object]) -> dict[str, Any]:
        image_batch = preprocess_image(image, self.settings.image_size)
        try:
            return build_attention_explanation(
                loaded_model=self.model_bundle,
                image=image,
                image_batch=image_batch,
                predicted_index=int(prediction["predicted_index"]),
                probabilities={str(key): float(value) for key, value in prediction["probabilities"].items()},
            )
        except Exception as exc:
            return {
                "model_input_image": Image.fromarray(image_batch[0].astype("uint8")),
                "attention_overlay_image": None,
                "attention_heatmap_image": None,
                "focus_region": "unavailable",
                "focus_coverage": 0.0,
                "winning_label": str(prediction["predicted_label"]),
                "runner_up_label": "unavailable",
                "margin": 0.0,
                "attention_layer": "unavailable",
                "error": str(exc),
            }
