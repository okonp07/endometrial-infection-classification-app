from __future__ import annotations

from pathlib import Path
import zipfile

import gradio as gr
import pytest
from PIL import Image

from endometrial_app.config import Settings
from endometrial_app.service import PredictionService
from endometrial_app.ui import (
    AUTHOR_PROFILES,
    CUSTOM_CSS,
    _build_class_distribution_frame,
    _build_demo_bundle,
    _build_demo_profile_frame,
    _build_split_distribution_frame,
    _future_dev_markdown,
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


def test_css_pins_light_color_scheme_for_theme_collision_edge_case() -> None:
    assert "color-scheme: only light !important;" in CUSTOM_CSS
    assert ".explanation-shell strong" in CUSTOM_CSS
    assert "-webkit-text-fill-color: currentColor !important;" in CUSTOM_CSS


def test_demo_sample_bundle_contains_twenty_images() -> None:
    samples_dir = Path(__file__).resolve().parents[1] / "assets" / "demo_samples"
    sample_images = sorted(samples_dir.glob("*.jpg"))
    assert len(sample_images) == 20


def test_future_development_assets_exist() -> None:
    project_root = Path(__file__).resolve().parents[1]
    roadmap_path = project_root / "future development.md"

    assert roadmap_path.exists()
    roadmap_text = roadmap_path.read_text(encoding="utf-8")
    assert "Future Development" in roadmap_text
    assert "Joseph Edet" in roadmap_text
    assert "future development.md" in _future_dev_markdown()


def test_author_profiles_cover_three_authors() -> None:
    author_names = {profile["name"] for profile in AUTHOR_PROFILES}
    prince_profile = next(profile for profile in AUTHOR_PROFILES if profile["name"] == "Okon Prince")
    joseph_profile = next(profile for profile in AUTHOR_PROFILES if profile["name"] == "Joseph Edet")

    assert author_names == {
        "Okon Prince",
        "Dr. Obi Cajetan",
        "Joseph Edet",
    }
    assert prince_profile["image_asset"] == "author/okon-prince.png"
    assert "MIVA Open University" in prince_profile["role"]
    assert "production-ready intelligence" in prince_profile["bio"]
    assert joseph_profile["image_asset"] == "author/joseph-edet.png"
    assert "WorldQuant University" in joseph_profile["role"]
    assert "Financial Engineering" in joseph_profile["bio"]


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
            assert all("infected_" not in name and "uninfected_" not in name for name in image_members)
            assert all(name.split("/")[-1].startswith("scan_") for name in image_members)
            manifest_text = archive.read("README.txt").decode("utf-8")
            assert "neutral scan filenames" in manifest_text
            assert "10 infected examples" not in manifest_text
            assert "10 uninfected examples" not in manifest_text
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
    assert set(split_frame["split_class"]) == {
        "Train - Infected",
        "Train - Uninfected",
        "Validation - Infected",
        "Validation - Uninfected",
        "Test - Infected",
        "Test - Uninfected",
    }
    assert summary["raw_counts"] == {"infected": 781, "uninfected": 791}
    assert summary["data_quality"]["near_duplicate_threshold"] >= 4
    assert "audit_artifacts" in summary
    assert len(history) >= 1
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
    assert explanation["focus_pattern"] in {
        "compact",
        "moderately concentrated",
        "broad",
        "diffuse",
    }
    assert 0.0 <= float(explanation["high_attention_threshold"]) <= 1.0
