from __future__ import annotations

from pathlib import Path
import zipfile

import gradio as gr

from endometrial_app.config import Settings
from endometrial_app.service import PredictionService
from endometrial_app.ui import _build_demo_bundle, build_ui


def make_service() -> PredictionService:
    project_root = Path(__file__).resolve().parents[1]
    settings = Settings(
        project_name="Test",
        project_root=project_root,
        model_path=project_root / "models" / "endometrial_classifier.keras",
        class_names_path=project_root / "artifacts" / "class_names.json",
        image_width=224,
        image_height=224,
        threshold=0.5,
        host="127.0.0.1",
        port=7860,
    )
    return PredictionService(settings=settings)


def test_build_ui_returns_blocks() -> None:
    ui = build_ui(make_service())
    assert isinstance(ui, gr.Blocks)


def test_demo_sample_bundle_contains_twenty_images() -> None:
    samples_dir = Path(__file__).resolve().parents[1] / "assets" / "demo_samples"
    sample_images = sorted(samples_dir.glob("*.jpg"))
    assert len(sample_images) == 20


def test_download_bundle_contains_samples_and_manifest() -> None:
    project_root = Path(__file__).resolve().parents[1]
    bundle_path = Path(_build_demo_bundle(project_root))

    try:
        assert bundle_path.exists()
        with zipfile.ZipFile(bundle_path) as archive:
            names = archive.namelist()
            assert "README.txt" in names
            image_members = [name for name in names if name.startswith("demo_samples/")]
            assert len(image_members) == 20
    finally:
        if bundle_path.exists():
            bundle_path.unlink()
