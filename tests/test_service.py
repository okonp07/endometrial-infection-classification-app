from __future__ import annotations

from pathlib import Path

from PIL import Image

from endometrial_app.config import Settings
from endometrial_app.service import PredictionService


class FakeService(PredictionService):
    def __init__(self) -> None:
        settings = Settings(
            project_name="Test",
            project_root=Path("."),
            model_path=Path("models/fake.keras"),
            class_names_path=Path("artifacts/class_names.json"),
            image_width=224,
            image_height=224,
            threshold=0.5,
            host="127.0.0.1",
            port=7860,
        )
        super().__init__(settings=settings)

    def is_ready(self) -> bool:
        return True

    def predict(self, image: Image.Image) -> dict[str, object]:
        return {
            "predicted_label": "infected",
            "predicted_index": 0,
            "confidence": 0.92,
            "probabilities": {"infected": 0.92, "uninfected": 0.08},
        }


def test_service_health_shape() -> None:
    service = FakeService()
    payload = service.health()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["class_names"] == ["infected", "uninfected"]
