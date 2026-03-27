from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from endometrial_app.api import create_api_app
from endometrial_app.config import Settings
from endometrial_app.service import PredictionService


class FakeService(PredictionService):
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        settings = Settings(
            project_name="Test",
            project_root=project_root,
            model_path=project_root / "models" / "fake.keras",
            class_names_path=project_root / "artifacts" / "class_names.json",
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
            "confidence": 0.87,
            "probabilities": {"infected": 0.87, "uninfected": 0.13},
        }


def make_image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(180, 32, 32))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(create_api_app(FakeService()))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint() -> None:
    client = TestClient(create_api_app(FakeService()))
    response = client.post(
        "/api/predict",
        files={"file": ("sample.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["predicted_label"] == "infected"


def test_download_demo_pack_endpoint() -> None:
    client = TestClient(create_api_app(FakeService()))
    response = client.get("/downloads/demo-pack")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert "endometrial-demo-test-images.zip" in response.headers["content-disposition"]
