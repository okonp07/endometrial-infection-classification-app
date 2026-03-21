from __future__ import annotations

from pathlib import Path
import zipfile

import gradio as gr
import pytest
from PIL import Image

from endometrial_app.config import Settings
from endometrial_app.service import PredictionService
from endometrial_app.ui import (
    _build_class_distribution_frame,
    _build_demo_bundle,
    _build_demo_profile_frame,
    _build_split_distribution_frame,
    _load_training_history,
    _load_training_summary,
    _safe_chart_limit,
    build_ui,
)


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


def test_eda_frames_match_expected_project_counts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    summary = _load_training_summary(project_root)
    history = _load_training_history(project_root)
    class_frame = _build_class_distribution_frame(summary)
    split_frame = _build_split_distribution_frame(summary)
    demo_profile_frame = _build_demo_profile_frame(project_root)

    assert int(class_frame["count"].sum()) == 1560
    assert int(split_frame["count"].sum()) == 1560
    assert len(history) == 4
    assert "epoch" in history.columns
    assert len(demo_profile_frame) == 20


def test_safe_chart_limit_starts_from_zero() -> None:
    frame = _build_class_distribution_frame(
        {
            "clean_counts": {
                "infected": 779,
                "uninfected": 781,
            }
        }
    )

    chart_limit = _safe_chart_limit(frame, "count", minimum=10.0)

    assert chart_limit[0] == 0.0
    assert chart_limit[1] > 781.0


def test_service_generates_explanation_artifacts() -> None:
    pytest.importorskip("tensorflow")
    project_root = Path(__file__).resolve().parents[1]
    service = make_service()
    sample_path = project_root / "assets" / "demo_samples" / "infected_01.jpg"

    with Image.open(sample_path).convert("RGB") as image:
        prediction = service.predict(image)
        explanation = service.explain_prediction(image, prediction)

    assert explanation["model_input_image"] is not None
    assert explanation["attention_overlay_image"] is not None
    assert explanation["attention_layer"] != "unavailable"
