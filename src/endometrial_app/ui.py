from __future__ import annotations

import html
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

from endometrial_app.service import PredictionService


FUTURE_DEVELOPMENT_URL = (
    "https://github.com/okonp07/endometrial-infection-classification-app/blob/main/"
    "future%20development.md"
)


CUSTOM_CSS = """
:root {
    --brand-blue: #0e4d73;
    --brand-blue-deep: #092d46;
    --brand-green: #178b76;
    --brand-green-soft: #dff4ee;
    --brand-ash: #eef3f4;
    --brand-ink: #12242d;
    --brand-slate: #4d6069;
    --brand-white: #ffffff;
}

html,
body,
.gradio-container {
    color-scheme: only light !important;
    forced-color-adjust: none;
    background:
        radial-gradient(circle at top right, rgba(23, 139, 118, 0.18), transparent 32%),
        radial-gradient(circle at top left, rgba(14, 77, 115, 0.12), transparent 24%),
        linear-gradient(180deg, #f8fbfb 0%, #edf2f3 100%);
    color: var(--brand-ink);
    font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
    font-size: 16px;
    line-height: 1.6;
}

.gradio-container,
.gradio-container * {
    scrollbar-color: rgba(14, 77, 115, 0.4) rgba(238, 243, 244, 0.9);
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: inherit;
}

.gradio-container .prose strong,
.gradio-container .prose b,
.gradio-container .prose code,
.helper-copy strong,
.sample-copy strong,
.about-copy strong,
.author-copy strong,
.download-card strong,
.download-highlight strong,
.prediction-shell strong,
.explanation-shell strong,
.footer-note strong,
.eda-note strong {
    color: var(--brand-blue-deep) !important;
    -webkit-text-fill-color: var(--brand-blue-deep) !important;
}

.gradio-container .prose code,
.helper-copy code,
.sample-copy code,
.about-copy code,
.author-copy code,
.download-card code,
.prediction-shell code,
.explanation-shell code,
.footer-note code,
.eda-note code {
    background: rgba(14, 77, 115, 0.08);
    border-radius: 0.45rem;
    padding: 0.08rem 0.32rem;
}

.helper-copy,
.sample-copy,
.about-copy,
.author-copy,
.prediction-shell,
.explanation-shell,
.download-card,
.download-highlight,
.footer-note,
.eda-note,
.panel-card .json-container,
.result-card .json-container,
.explanation-panel .json-container {
    color-scheme: only light !important;
}

.helper-copy p,
.helper-copy li,
.sample-copy p,
.sample-copy li,
.about-copy p,
.about-copy li,
.author-copy p,
.author-copy li,
.prediction-shell p,
.prediction-shell li,
.explanation-shell p,
.explanation-shell li,
.download-card p,
.download-card li,
.download-highlight p,
.footer-note p,
.eda-note p,
.helper-copy span,
.sample-copy span,
.about-copy span,
.author-copy span,
.prediction-shell span,
.explanation-shell span,
.download-card span,
.footer-note span,
.eda-note span {
    -webkit-text-fill-color: currentColor !important;
}

.panel-card .json-container,
.result-card .json-container,
.explanation-panel .json-container {
    background: linear-gradient(180deg, #f7fafb 0%, #eef3f4 100%) !important;
    border: 1px solid rgba(9, 45, 70, 0.08) !important;
    color: var(--brand-ink) !important;
}

.panel-card .json-container *,
.result-card .json-container *,
.explanation-panel .json-container * {
    color: var(--brand-ink) !important;
    -webkit-text-fill-color: currentColor !important;
}

.gradio-container code,
.gradio-container pre {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere;
    word-break: break-word;
}

[role="tablist"] {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 22px;
    padding: 0.4rem;
    box-shadow: 0 12px 32px rgba(18, 36, 45, 0.08);
    gap: 0.35rem;
    overflow-x: auto;
    scrollbar-width: none;
    -ms-overflow-style: none;
}

[role="tablist"]::-webkit-scrollbar {
    display: none;
}

button[role="tab"] {
    border-radius: 16px !important;
    color: var(--brand-blue-deep) !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
    min-height: 44px;
    padding: 0.7rem 1rem !important;
    white-space: nowrap;
    flex: 0 0 auto;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green)) !important;
    color: var(--brand-white) !important;
    box-shadow: 0 12px 28px rgba(14, 77, 115, 0.25);
}

.hero-shell,
.author-row {
    gap: 1rem;
    align-items: stretch;
}

.workspace-row {
    gap: 1rem;
    align-items: stretch;
}

.hero-copy,
.hero-banner-wrap,
.panel-card,
.author-copy-card,
.author-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 28px;
    box-shadow: 0 20px 56px rgba(18, 36, 45, 0.08);
    padding: 1.4rem !important;
}

.top-card {
    height: 100%;
}

.hero-copy {
    background: linear-gradient(135deg, rgba(9, 45, 70, 0.98) 0%, rgba(14, 77, 115, 0.95) 55%, rgba(23, 139, 118, 0.92) 100%);
    color: var(--brand-white);
}

.hero-copy .hero-eyebrow {
    display: inline-block;
    margin-bottom: 0.85rem;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.hero-copy h1 {
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 3rem !important;
    line-height: 1.02 !important;
    margin-bottom: 1rem !important;
}

.hero-copy p {
    color: rgba(255, 255, 255, 0.92) !important;
    font-size: 1.03rem;
    line-height: 1.75;
}

.hero-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin-top: 1.4rem;
}

.hero-stat {
    padding: 1rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.16);
    backdrop-filter: blur(10px);
}

.hero-stat-value {
    display: block;
    color: var(--brand-white);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
}

.hero-stat-label {
    display: block;
    margin-top: 0.35rem;
    color: rgba(255, 255, 255, 0.84);
    font-size: 0.92rem;
}

.hero-banner-wrap {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.72) 0%, rgba(223, 244, 238, 0.72) 100%);
    display: flex;
    align-items: stretch;
    justify-content: stretch;
    padding: 0.95rem !important;
    min-height: 100%;
}

.hero-banner {
    width: 100%;
    min-height: 100%;
    display: flex;
    align-items: stretch;
}

.hero-banner > div,
.hero-banner .image-container,
.hero-banner [data-testid="image"] {
    width: 100% !important;
    height: 100% !important;
}

.hero-banner img {
    width: 100%;
    height: 100% !important;
    border-radius: 22px;
    object-fit: cover;
    box-shadow: 0 16px 34px rgba(18, 36, 45, 0.16);
    display: block;
}

.section-kicker {
    display: inline-block;
    margin-bottom: 0.6rem;
    color: var(--brand-green);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.panel-card h2 {
    margin-top: 0 !important;
    line-height: 1.12;
}

.helper-copy p,
.sample-copy p,
.about-copy p,
.about-copy li,
.author-copy p,
.helper-copy li,
.sample-copy li,
.author-copy li {
    color: var(--brand-slate) !important;
    line-height: 1.75;
}

.button-row {
    justify-content: flex-start;
    gap: 0.8rem;
}

.button-row button {
    border-radius: 16px !important;
    font-weight: 700 !important;
}

.button-row button.primary {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green)) !important;
    border: none !important;
    box-shadow: 0 14px 28px rgba(14, 77, 115, 0.22);
}

.prediction-shell {
    padding: 1.2rem;
    border-radius: 22px;
    background: linear-gradient(180deg, #f7fafb 0%, #edf5f3 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
}

.prediction-shell.placeholder {
    background: linear-gradient(180deg, #f8fbfb 0%, #f1f5f6 100%);
}

.prediction-kicker {
    color: var(--brand-green);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.prediction-title {
    margin-top: 0.5rem;
    color: var(--brand-blue-deep);
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
}

.prediction-chip {
    display: inline-flex;
    margin-top: 0.9rem;
    padding: 0.48rem 0.88rem;
    border-radius: 999px;
    font-weight: 700;
}

.prediction-chip.infected {
    background: rgba(14, 77, 115, 0.12);
    color: var(--brand-blue-deep);
}

.prediction-chip.uninfected {
    background: rgba(23, 139, 118, 0.12);
    color: #0e6c5b;
}

.prediction-shell p {
    margin-top: 0.85rem;
    color: var(--brand-slate);
    line-height: 1.7;
}

.disclaimer p {
    margin-top: 1rem;
    color: var(--brand-slate) !important;
    text-align: center;
    font-size: 0.95rem;
}

.sample-note {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(14, 77, 115, 0.08), rgba(23, 139, 118, 0.12));
    border: 1px solid rgba(23, 139, 118, 0.12);
}

.sample-note p {
    margin: 0;
    color: var(--brand-blue-deep);
    line-height: 1.7;
}

.download-card {
    padding: 1.15rem 1.2rem;
    border-radius: 22px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.94) 0%, rgba(223, 244, 238, 0.7) 100%);
    border: 1px solid rgba(9, 45, 70, 0.08);
    box-shadow: 0 18px 42px rgba(18, 36, 45, 0.08);
}

.download-card p,
.download-card li {
    color: var(--brand-slate);
    line-height: 1.75;
}

.download-highlight {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: rgba(14, 77, 115, 0.08);
    border: 1px solid rgba(14, 77, 115, 0.1);
}

.download-highlight p {
    margin: 0;
    color: var(--brand-blue-deep);
    line-height: 1.7;
}

.explanation-shell {
    margin-top: 0;
    padding: 0;
    border: 0;
    background: transparent;
}

.explanation-shell p,
.explanation-shell li {
    color: var(--brand-slate);
    line-height: 1.7;
}

.explanation-shell .explanation-title {
    color: var(--brand-blue-deep);
    -webkit-text-fill-color: var(--brand-blue-deep) !important;
    font-family: "Space Grotesk", "Manrope", sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
}

.visual-row {
    margin-top: 1rem;
    gap: 0.85rem;
    align-items: flex-start;
}

.explanation-panel {
    margin-top: 0.15rem;
}

.explanation-panel .json-container,
.result-card .json-container {
    border-radius: 20px;
}

.eda-note {
    padding: 1rem 1.1rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(14, 77, 115, 0.08), rgba(23, 139, 118, 0.1));
    border: 1px solid rgba(14, 77, 115, 0.08);
}

.eda-note p {
    margin: 0;
    color: var(--brand-blue-deep);
    line-height: 1.7;
}

.author-row {
    align-items: stretch;
}

.author-card {
    height: 100%;
}

.author-photo {
    width: min(220px, 100%);
    margin: 0 auto 1rem;
}

.author-photo img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 28px;
    border: 4px solid rgba(23, 139, 118, 0.14);
    box-shadow: 0 18px 46px rgba(18, 36, 45, 0.16);
}

.author-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    width: min(220px, 100%);
    aspect-ratio: 1 / 1;
    margin: 0 auto 1rem;
    border-radius: 28px;
    border: 2px dashed rgba(14, 77, 115, 0.22);
    background: linear-gradient(135deg, rgba(14, 77, 115, 0.08), rgba(23, 139, 118, 0.12));
    color: var(--brand-blue-deep);
    text-align: center;
    font-weight: 700;
    padding: 1rem;
}

.author-placeholder span {
    display: block;
    margin-top: 0.45rem;
    color: var(--brand-slate);
    font-size: 0.88rem;
    font-weight: 600;
}

.author-card h3 {
    margin-top: 0 !important;
    margin-bottom: 0.35rem !important;
    color: var(--brand-blue-deep);
    -webkit-text-fill-color: var(--brand-blue-deep) !important;
    font-family: "Space Grotesk", "Manrope", sans-serif;
}

.author-card p {
    color: var(--brand-slate);
    -webkit-text-fill-color: var(--brand-slate) !important;
    line-height: 1.75;
}

.author-card .author-role {
    color: var(--brand-green);
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.footer-note {
    margin-top: 1.4rem;
    padding: 1.15rem 1.4rem 1.8rem;
    text-align: center;
    color: var(--brand-slate);
}

.footer-note p {
    margin: 0.2rem 0;
    line-height: 1.7;
}

@media (max-width: 960px) {
    .hero-shell,
    .workspace-row,
    .author-row,
    .visual-row {
        gap: 0.85rem;
    }

    .hero-copy,
    .hero-banner-wrap,
    .panel-card,
    .author-card,
    .download-card {
        padding: 1rem !important;
        border-radius: 22px;
    }

    .hero-copy h1 {
        font-size: 2.15rem !important;
        line-height: 1.06 !important;
    }

    .hero-copy p {
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .hero-copy .hero-eyebrow {
        font-size: 0.74rem;
        padding: 0.38rem 0.7rem;
    }

    .hero-stat-grid {
        grid-template-columns: 1fr;
    }

    .hero-stat {
        padding: 0.9rem;
    }

    .hero-stat-value {
        font-size: 1.45rem;
    }

    .hero-stat-label,
    .prediction-shell p,
    .helper-copy p,
    .sample-copy p,
    .about-copy p,
    .about-copy li,
    .author-copy p,
    .download-card p,
    .download-card li,
    .explanation-shell p,
    .footer-note p {
        font-size: 0.97rem;
        line-height: 1.65;
    }

    .section-kicker,
    .prediction-kicker {
        font-size: 0.74rem;
    }

    .panel-card h2,
    .about-copy h2,
    .sample-copy h2 {
        font-size: 1.75rem !important;
        line-height: 1.14 !important;
    }

    .prediction-title {
        font-size: 1.22rem;
    }

    .button-row {
        gap: 0.7rem;
    }

    .button-row button {
        width: 100%;
    }

    .upload-card [data-testid="image"],
    .upload-card .image-container {
        min-height: 300px !important;
    }

    .visual-row [data-testid="image"],
    .visual-row .image-container {
        min-height: 230px !important;
    }

    .footer-note {
        padding: 1rem 0.9rem 1.35rem;
    }
}

@media (max-width: 640px) {
    .gradio-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }

    [role="tablist"] {
        padding: 0.3rem;
        border-radius: 18px;
    }

    button[role="tab"] {
        font-size: 0.86rem !important;
        padding: 0.66rem 0.82rem !important;
    }

    .hero-copy,
    .hero-banner-wrap,
    .panel-card,
    .author-card,
    .download-card {
        padding: 0.92rem !important;
        border-radius: 20px;
    }

    .hero-copy h1 {
        font-size: 1.85rem !important;
    }

    .hero-copy p {
        font-size: 0.94rem;
    }

    .panel-card h2,
    .about-copy h2,
    .sample-copy h2 {
        font-size: 1.5rem !important;
    }

    .author-photo,
    .author-placeholder {
        width: min(180px, 100%);
    }

    .prediction-shell {
        padding: 1rem;
        border-radius: 18px;
    }

    .panel-card .json-container,
    .result-card .json-container,
    .explanation-panel .json-container {
        font-size: 0.84rem;
    }
}

@media (max-width: 480px) {
    .hero-copy .hero-eyebrow {
        width: 100%;
        text-align: center;
    }

    .hero-copy h1 {
        font-size: 1.62rem !important;
        line-height: 1.08 !important;
    }

    .hero-copy p,
    .prediction-shell p,
    .helper-copy p,
    .sample-copy p,
    .about-copy p,
    .about-copy li,
    .author-copy p,
    .download-card p,
    .download-card li,
    .explanation-shell p,
    .footer-note p {
        font-size: 0.92rem;
    }

    .panel-card h2,
    .about-copy h2,
    .sample-copy h2 {
        font-size: 1.34rem !important;
    }

    .prediction-chip {
        width: 100%;
        justify-content: center;
    }
}
"""


FOOTER_HTML = """
<div class="footer-note">
    <p><strong>&copy; 2026</strong></p>
    <p>
        This project is based on research work by Okon Prince of Miva Open University, Dr. Obi Cajetan of the University of Calabar Teaching Hospital, and Joseph Edet of WorldQuant University.
        It is covered by the MIT License, and the authors should be acknowledged if the product or methods are referenced in future research.
    </p>
    <p>Enquiries: okonp07@gmail.com</p>
</div>
"""


def _load_training_summary(project_root: Path) -> dict[str, Any]:
    summary_path = project_root / "artifacts" / "training_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _collect_demo_samples(project_root: Path) -> list[str]:
    samples_dir = project_root / "assets" / "demo_samples"
    infected = sorted(samples_dir.glob("infected_*.jpg"))
    uninfected = sorted(samples_dir.glob("uninfected_*.jpg"))

    ordered_paths: list[str] = []
    for infected_path, uninfected_path in zip(infected, uninfected):
        ordered_paths.extend([str(infected_path), str(uninfected_path)])
    return ordered_paths


def _hero_stats_html(summary: dict[str, Any]) -> str:
    clean_counts = summary.get("clean_counts", {})
    split_counts = summary.get("split_counts", {})
    total_clean = sum(int(count) for count in clean_counts.values())
    test_total = sum(int(count) for count in split_counts.get("test", {}).values())

    cards = [
        ("2", "Target classes"),
        (f"{total_clean:,}" if total_clean else "N/A", "Deduplicated scans"),
        (f"{test_total:,}" if test_total else "N/A", "Held-out test images"),
    ]
    stats_markup = "".join(
        f"""
        <div class="hero-stat">
            <span class="hero-stat-value">{value}</span>
            <span class="hero-stat-label">{label}</span>
        </div>
        """
        for value, label in cards
    )
    return f'<div class="hero-stat-grid">{stats_markup}</div>'


def _demo_bundle_filename() -> str:
    return "endometrial-demo-test-images.zip"


def _build_demo_bundle(project_root: Path) -> str:
    sample_paths = _collect_demo_samples(project_root)
    with tempfile.NamedTemporaryFile(
        prefix="endometrial-demo-samples-",
        suffix=".zip",
        delete=False,
    ) as temporary_file:
        bundle_path = Path(temporary_file.name)

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        manifest_lines = [
            "Endometrial Infection Classification App",
            "",
            "Bundle contents:",
            "- 20 demo images from the held-out test split",
            "- 10 infected examples",
            "- 10 uninfected examples",
            "",
            "Intended use:",
            "These images are provided so users can test the deployed application without sourcing their own scans.",
        ]
        archive.writestr("README.txt", "\n".join(manifest_lines))
        for sample_path in sample_paths:
            sample_file = Path(sample_path)
            archive.write(sample_file, arcname=f"demo_samples/{sample_file.name}")

    return str(bundle_path)


def _load_training_history(project_root: Path) -> pd.DataFrame:
    history_path = project_root / "artifacts" / "training_history.csv"
    if not history_path.exists():
        return pd.DataFrame()

    history = pd.read_csv(history_path)
    history.insert(0, "epoch", range(1, len(history) + 1))
    return history


def _build_class_distribution_frame(summary: dict[str, Any]) -> pd.DataFrame:
    clean_counts = summary.get("clean_counts", {})
    return pd.DataFrame(
        [
            {"class": class_name.title(), "count": int(count), "series": "Dataset"}
            for class_name, count in clean_counts.items()
        ]
    )


def _build_split_distribution_frame(summary: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, class_counts in summary.get("split_counts", {}).items():
        for class_name, count in class_counts.items():
            rows.append(
                {
                    "split": split_name.title(),
                    "class": class_name.title(),
                    "count": int(count),
                    "split_class": f"{split_name.title()} - {class_name.title()}",
                }
            )
    return pd.DataFrame(rows)


def _build_test_metrics_frame(summary: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": metric.replace("_", " ").title(), "value": float(value)}
            for metric, value in summary.get("test_metrics", {}).items()
        ]
    )


def _build_curve_frame(history: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["epoch", "series", "value"])

    curve_frame = history[["epoch", *columns.keys()]].melt(
        id_vars="epoch",
        var_name="series",
        value_name="value",
    )
    curve_frame["series"] = curve_frame["series"].map(columns)
    return curve_frame


def _build_demo_profile_frame(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_path in _collect_demo_samples(project_root):
        sample_file = Path(sample_path)
        with Image.open(sample_file).convert("L") as image:
            image_array = np.asarray(image, dtype=np.float32)
            rows.append(
                {
                    "file": sample_file.name,
                    "class_label": "Infected" if sample_file.name.startswith("infected") else "Uninfected",
                    "width": int(image.width),
                    "height": int(image.height),
                    "mean_intensity": round(float(image_array.mean()), 2),
                    "std_intensity": round(float(image_array.std()), 2),
                }
            )
    return pd.DataFrame(rows)


def _safe_chart_limit(frame: pd.DataFrame, column: str, minimum: float = 1.0) -> list[float]:
    if frame.empty or column not in frame:
        return [0.0, minimum]

    maximum = float(frame[column].max())
    padded_maximum = max(minimum, maximum * 1.1)
    return [0.0, padded_maximum]


def _research_safeguards_markdown(summary: dict[str, Any]) -> str:
    data_quality = summary.get("data_quality", {})
    similarity_groups = data_quality.get("similarity_groups", {})
    exact_duplicates_removed = data_quality.get("exact_duplicates_removed", "N/A")
    threshold = data_quality.get("near_duplicate_threshold", "N/A")
    multi_image_group_count = similarity_groups.get("multi_image_group_count", "N/A")
    images_in_multi_image_groups = similarity_groups.get("images_in_multi_image_groups", "N/A")
    audit_artifacts = summary.get("audit_artifacts", {})
    audit_note = ""
    if audit_artifacts:
        audit_note = (
            "\nThe raw, cleaned, grouped, and split manifests for this run are also exported as audit artifacts so "
            "the evaluation protocol can be inspected outside the UI."
        )

    return f"""
<span class="section-kicker">Evaluation Protocol</span>
## Leakage-control safeguards

This training run removed **{exact_duplicates_removed} exact duplicates** and then applied a **group-aware perceptual-similarity split** so visually similar scans would remain in the same partition rather than leak across train, validation, and test.

The similarity audit used a **difference-hash threshold of {threshold}** and detected **{multi_image_group_count} multi-image similarity groups** covering **{images_in_multi_image_groups} scans**. This produces a materially stricter evaluation protocol than a naive random image-level split and reduces one important source of optimistic bias in the reported internal metrics.{audit_note}
"""


def _eda_overview_html(summary: dict[str, Any]) -> str:
    clean_total = sum(int(count) for count in summary.get("clean_counts", {}).values())
    exact_duplicates_removed = summary.get("data_quality", {}).get("exact_duplicates_removed", "N/A")
    threshold = summary.get("data_quality", {}).get("near_duplicate_threshold", "N/A")
    return f"""
    <div class="eda-note">
        <p>
            This EDA Lab documents the curated cohort, partitioning protocol, optimization traces, and internal held-out evaluation behind the deployed classifier.
            The current production run is based on <strong>{clean_total:,}</strong> cleaned images after removal of <strong>{exact_duplicates_removed}</strong> exact duplicates, with grouped similarity controls applied at a dHash threshold of <strong>{threshold}</strong>.
        </p>
    </div>
    """


def _class_distribution_markdown(summary: dict[str, Any]) -> str:
    raw_counts = summary.get("raw_counts", {})
    clean_counts = summary.get("clean_counts", {})
    data_quality = summary.get("data_quality", {})
    raw_infected = raw_counts.get("infected", "N/A")
    raw_uninfected = raw_counts.get("uninfected", "N/A")
    clean_infected = clean_counts.get("infected", "N/A")
    clean_uninfected = clean_counts.get("uninfected", "N/A")
    exact_duplicates_removed = data_quality.get("exact_duplicates_removed", "N/A")

    return f"""
<span class="section-kicker">Dataset Profile</span>
## Curated class counts

The raw archive contributed **{raw_infected} infected** and **{raw_uninfected} uninfected** valid images. After removing **{exact_duplicates_removed} exact duplicates**, the curated production dataset contains **{clean_infected} infected** and **{clean_uninfected} uninfected** scans.

The resulting class distribution remains close to balanced even though the cleaned counts are not perfectly identical. The chart below uses a zero-based y-axis so small differences are represented proportionally rather than visually amplified.
"""


def _split_strategy_markdown(summary: dict[str, Any]) -> str:
    split_counts = summary.get("split_counts", {})
    train_total = sum(int(count) for count in split_counts.get("train", {}).values())
    validation_total = sum(int(count) for count in split_counts.get("validation", {}).values())
    test_total = sum(int(count) for count in split_counts.get("test", {}).values())
    threshold = summary.get("data_quality", {}).get("near_duplicate_threshold", "N/A")
    train_infected = split_counts.get("train", {}).get("infected", "N/A")
    train_uninfected = split_counts.get("train", {}).get("uninfected", "N/A")
    validation_infected = split_counts.get("validation", {}).get("infected", "N/A")
    validation_uninfected = split_counts.get("validation", {}).get("uninfected", "N/A")
    test_infected = split_counts.get("test", {}).get("infected", "N/A")
    test_uninfected = split_counts.get("test", {}).get("uninfected", "N/A")

    return f"""
<span class="section-kicker">Partition Protocol</span>
## Curated split composition

The cleaned corpus was partitioned into **{train_total} training**, **{validation_total} validation**, and **{test_total} test** images. Class balance was preserved across the three partitions, and visually related scans were assigned at the similarity-group level so they would not be dispersed across train and evaluation subsets.

Per-class counts are: **train = {train_infected} infected / {train_uninfected} uninfected**, **validation = {validation_infected} / {validation_uninfected}**, and **test = {test_infected} / {test_uninfected}**.

The grouped partitioning logic was applied with a perceptual-similarity threshold of **{threshold}**, which makes the split protocol more defensible for research reporting than a simple image-level random split.
"""


def _accuracy_curves_markdown(history: pd.DataFrame) -> str:
    epoch_count = int(len(history))
    return f"""
<span class="section-kicker">Optimization Trace</span>
## Accuracy across epochs

These curves summarize how training and validation accuracy evolved across **{epoch_count} recorded epochs**. They are included to help readers assess whether predictive performance improved in a consistent manner across optimization rather than diverging sharply between the development and validation partitions.
"""


def _loss_curves_markdown(history: pd.DataFrame) -> str:
    epoch_count = int(len(history))
    return f"""
<span class="section-kicker">Optimization Trace</span>
## Loss across epochs

Loss is shown alongside accuracy because it is sensitive to probability quality, not only final class decisions. Across **{epoch_count} recorded epochs**, this panel helps indicate whether optimization converged cleanly or showed signs of instability during training.
"""


def _held_out_evaluation_markdown(summary: dict[str, Any]) -> str:
    data_quality = summary.get("data_quality", {})
    threshold = data_quality.get("near_duplicate_threshold", "N/A")
    metrics = summary.get("test_metrics", {})
    perfect_internal_result = (
        metrics
        and all(float(metrics.get(metric, 0.0)) >= 0.999 for metric in ["accuracy", "auc", "precision", "recall"])
    )
    caution = ""
    if perfect_internal_result:
        caution = (
            "\nThis internal split still produces near-perfect scores even after the stricter grouped similarity controls. "
            "That is encouraging, but it should be interpreted as an internal estimate rather than proof of broad external generalization."
        )

    return f"""
<span class="section-kicker">Internal Evaluation</span>
## Held-out test metrics

These metrics summarize the current production model on the reserved test partition after exact-duplicate removal and a similarity-group split designed to reduce train/test leakage. The grouped audit used a perceptual-similarity threshold of **{threshold}** and should therefore be read as a stricter internal evaluation than a basic random split.{caution}
"""


def _interpretation_note_markdown(summary: dict[str, Any]) -> str:
    data_quality = summary.get("data_quality", {})
    threshold = data_quality.get("near_duplicate_threshold", "N/A")
    evaluation_scope = summary.get(
        "evaluation_scope",
        "This should be treated as an internal held-out evaluation rather than external validation.",
    )
    return f"""
<span class="section-kicker">Interpretation</span>
## Reading the results carefully

The current evaluation uses a grouped split with a perceptual-similarity threshold of **{threshold}**, which is materially stronger than a naive random image split. Even so, the most defensible next steps are repeated grouped resampling, external validation, and, where possible, study-level or patient-level partitioning.

{evaluation_scope}
"""


def _demo_profile_markdown() -> str:
    return """
<span class="section-kicker">Descriptive Statistics</span>
## Public demo-bundle profile

This table reports lightweight descriptive statistics for the downloadable public demo bundle, including image dimensions and grayscale intensity summaries. It is intended as a compact characterization aid for app users and reviewers, not as a substitute for full dataset-level radiological feature analysis.
"""


def _project_about_markdown(summary: dict[str, Any]) -> str:
    raw_counts = summary.get("raw_counts", {})
    clean_counts = summary.get("clean_counts", {})
    split_counts = summary.get("split_counts", {})
    data_quality = summary.get("data_quality", {})
    raw_infected = raw_counts.get("infected", "N/A")
    raw_uninfected = raw_counts.get("uninfected", "N/A")
    infected_count = clean_counts.get("infected", "N/A")
    uninfected_count = clean_counts.get("uninfected", "N/A")
    test_infected = split_counts.get("test", {}).get("infected", "N/A")
    test_uninfected = split_counts.get("test", {}).get("uninfected", "N/A")
    exact_duplicates_removed = data_quality.get("exact_duplicates_removed", "N/A")
    split_strategy = data_quality.get("split_strategy", "a structured train, validation, and test workflow")
    threshold = data_quality.get("near_duplicate_threshold", "N/A")
    evaluation_scope = summary.get(
        "evaluation_scope",
        "The current results should be treated as internal held-out evaluation rather than external validation.",
    )

    return f"""
## About the Project

This application is an end-to-end image classification solution built to distinguish between **infected** and **uninfected** endometrial scan images. The goal is to make the screening workflow practical, reproducible, and accessible through a clean browser interface backed by the same inference service exposed through the API.

### What problem the solution addresses

Endometrial infection assessment from imaging data can be difficult to operationalize in a way that is fast, repeatable, and easy for researchers or clinicians to interact with. This project turns the underlying research work into a deployable application so that a user can upload a scan, receive a class prediction, inspect class probabilities, and use the tool as part of a research-support workflow.

### Why this research is important

Research in this area matters because reproductive and gynecological health conditions can be under-supported by scalable digital tools, especially in settings where specialist capacity, standardized screening support, or advanced infrastructure may be limited. Building AI systems around endometrial image classification helps move valuable medical research toward tools that can support consistency, speed, and broader access to analytical support.

This matters not only for model development, but also for translational impact: it demonstrates how clinical research can be transformed into an interactive system that researchers, innovation teams, and future medical-AI collaborators can actually use, evaluate, and improve.

### How the solution works

1. A user uploads an endometrial scan image or downloads the demo test-image bundle for structured app testing.
2. The app preprocesses the image into the format expected by the TensorFlow model.
3. The trained classifier scores the image against the two target classes: `infected` and `uninfected`.
4. The interface returns the predicted class, the model confidence, a class-probability breakdown, and inference metadata.
5. The same model powers both the Gradio interface and the FastAPI backend, which keeps browser predictions and API predictions consistent.

### Data and evaluation summary

The deployed model was trained from curated archive data after duplicate handling and split generation. The raw archive contributed **{raw_infected} infected** and **{raw_uninfected} uninfected** valid images. After curation, the current production bundle was prepared from **{infected_count} infected** images and **{uninfected_count} uninfected** images, with a held-out test set of **{test_infected} infected** and **{test_uninfected} uninfected** images.

To improve research rigor, the pipeline removed **{exact_duplicates_removed} exact duplicate files** and used **{split_strategy}** rather than a plain random image-level split. It also grouped near-duplicate scans using a perceptual-similarity threshold of **{threshold}** before assigning train, validation, and test partitions. This helps reduce the risk that near-identical scans inflate the reported performance.

{evaluation_scope}

### What the output means

The predicted label is the class the model considers most likely for the uploaded scan. The confidence score shows how strongly the model favors that decision, while the class-probability panel reveals the distribution across both classes. This is useful because it allows the user to see not only the final prediction, but also how decisive or uncertain the model is.

### Why the solution is useful to humanity

The broader human value of this work is that it helps make specialized medical-image research more usable, testable, and deployable. A system like this can contribute to faster triage support, more reproducible research pipelines, better experimentation, and wider access to AI-assisted tools that extend expert capacity rather than replace it.

When solutions like this are responsibly developed, they can reduce manual bottlenecks, encourage earlier analytical review, and create a foundation for future diagnostic-support systems that are more accessible across institutions and resource settings. In that sense, the project is not just a classifier; it is part of a larger movement toward practical, human-centered medical AI.

### Responsible use

This application is best understood as a **research and AI-assisted classification tool**. It is designed to support structured analysis, experimentation, and decision support. It should not be treated as a standalone clinical diagnosis without expert interpretation and appropriate medical context.
"""


def _future_dev_markdown() -> str:
    return f"""
## Future Development

This page highlights the main directions for future enrichment of the project and links to the full roadmap in the repository.

### Full roadmap

The detailed development roadmap is available here:

[Open `future development.md`]({FUTURE_DEVELOPMENT_URL})

### Priority development tracks

1. **Stronger evaluation design**  
   Move toward patient-level or study-level partitioning, repeated grouped resampling, and external validation so the evidence standard becomes stronger than internal held-out evaluation alone.

2. **Clinically grounded model improvement**  
   Explore region-of-interest localization, segmentation support, stronger interpretability validation, and uncertainty calibration so the model becomes more anatomically meaningful and more trustworthy.

3. **Richer product capability**  
   Add clinician-in-the-loop review workflows, structured audit exports, robust deployment controls, and potentially multimodal metadata support so the system becomes more useful in real research settings.

4. **Expanded research scope**  
   Benchmark stronger architectures, test robustness under distribution shift, and explore whether future datasets can support multi-class or severity-aware classification.

### Recommended near-term sequence

- recover or curate metadata that support study-level grouping
- run an external validation experiment on an independent dataset
- add calibration analysis and uncertainty-aware thresholding
- strengthen the current explanation pipeline with additional interpretability methods
- formalize experiment tracking so future model comparisons remain reproducible

### Development philosophy

Future work should enrich the project without weakening its transparency. Improvements should raise the quality of the evidence, the clarity of the outputs, and the usefulness of the system to researchers and clinicians.
"""


AUTHOR_MARKDOWN = """
## About the Authors

This project currently credits three authors. The cards below show the intended author layout, with a live profile image for Okon Prince and placeholders for the remaining authors until their final images are provided.
"""


AUTHOR_PROFILES = [
    {
        "name": "Okon Prince",
        "role": "Senior Data Scientist at MIVA Open University | AI Engineer & Data Scientist",
        "image_asset": "author/okon-prince.png",
        "bio": (
            "I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.\n\n"
            "My core stack includes Python, Streamlit, BigQuery, Supabase, Hugging Face, PySpark, SQL, "
            "Machine Learning, LLMs, and Transformers.\n\n"
            "My work spans risk scoring systems, A/B testing, Traditional and AI-powered dashboards, RAG "
            "pipelines, predictive analytics, LLM-based solutions and AI research.\n\n"
            "Currently, I work as a Senior Data Scientist in the department of Research and Development at "
            "MIVA Open University, where I carry out AI / ML Research and build intelligent systems that "
            "drive analytics, decision support and scalable AI innovation.\n\n"
            "I believe: models are trained, systems are engineered and impact is delivered."
        ),
    },
    {
        "name": "Dr. Obi Cajetan",
        "role": "University of Calabar Teaching Hospital",
        "placeholder_label": "Image placeholder",
        "bio": (
            "Placeholder bio: add a short profile describing Dr. Obi Cajetan's clinical, medical, or "
            "research contribution to the project, including subject-matter leadership, data context, and "
            "the domain relevance of the endometrial infection study."
        ),
    },
    {
        "name": "Joseph Edet",
        "role": "WorldQuant University",
        "placeholder_label": "Image placeholder",
        "bio": (
            "Placeholder bio: add a short profile describing Joseph Edet's role in the research support, "
            "analytical contribution, technical collaboration, or future extension of the endometrial "
            "classification project."
        ),
    },
]


def _author_placeholder_html(profile: dict[str, str]) -> str:
    name = html.escape(profile["name"])
    placeholder_label = html.escape(profile["placeholder_label"])
    return f"""
    <div class="author-placeholder">
        <div>
            {placeholder_label}
            <span>{name}</span>
        </div>
    </div>
    """


def _author_card_markdown(profile: dict[str, str]) -> str:
    return f"""
### {profile["name"]}

<div class="author-role">{profile["role"]}</div>

{profile["bio"]}
"""


def _author_image_path(profile: dict[str, str], assets_dir: Path) -> Path | None:
    image_asset = profile.get("image_asset")
    if not image_asset:
        return None

    image_path = assets_dir / image_asset
    if image_path.exists():
        return image_path
    return None


def _prediction_placeholder_html() -> str:
    return """
    <div class="prediction-shell placeholder">
        <div class="prediction-kicker">Awaiting Inference</div>
        <div class="prediction-title">Upload a scan to generate a prediction</div>
        <p>
            The result card will show the predicted class, confidence score, and a short interpretation once the model runs.
        </p>
    </div>
    """


def _prediction_card_html(result: dict[str, Any]) -> str:
    predicted_label = str(result["predicted_label"]).strip().lower()
    confidence = float(result["confidence"])
    label_text = predicted_label.replace("_", " ").title()
    chip_class = "infected" if predicted_label == "infected" else "uninfected"
    interpretation = (
        "The model sees stronger evidence for the infected class in this scan."
        if predicted_label == "infected"
        else "The model sees stronger evidence for the uninfected class in this scan."
    )
    return f"""
    <div class="prediction-shell">
        <div class="prediction-kicker">Model Decision</div>
        <div class="prediction-title">{html.escape(label_text)}</div>
        <span class="prediction-chip {chip_class}">{html.escape(label_text)}</span>
        <p><strong>Confidence:</strong> {confidence:.2%}</p>
        <p>{html.escape(interpretation)}</p>
    </div>
    """


def _explanation_placeholder_html() -> str:
    return """
    <div class="explanation-shell">
        <div class="explanation-title">Why the model predicted this</div>
        <p>
            After inference, this section will explain the processing steps used by the model and show an attention-based visual summary for the current scan.
        </p>
    </div>
    """


def _explanation_card_html(result: dict[str, Any], explanation: dict[str, Any], image_size: tuple[int, int]) -> str:
    predicted_label = str(result["predicted_label"]).replace("_", " ").title()
    focus_region = str(explanation.get("focus_region", "unavailable")).title()
    focus_coverage = float(explanation.get("focus_coverage", 0.0))
    focus_pattern = str(explanation.get("focus_pattern", "unavailable")).replace("_", " ")
    high_attention_threshold = float(explanation.get("high_attention_threshold", 0.0))
    margin = float(explanation.get("margin", 0.0))
    runner_up_label = str(explanation.get("runner_up_label", "unavailable")).replace("_", " ").title()
    attention_layer = str(explanation.get("attention_layer", "unavailable"))
    error_message = explanation.get("error")

    detail_copy = (
        f"""
        <p>
            The model first resized the uploaded scan to <strong>{image_size[0]} x {image_size[1]}</strong> pixels, then passed it through the trained TensorFlow feature extractor.
            For this run, the dominant high-attention mass was centered in the <strong>{html.escape(focus_region)}</strong> region of the scan and the resulting saliency mask was <strong>{html.escape(focus_pattern)}</strong>, spanning about <strong>{focus_coverage:.1%}</strong> of the analyzed frame.
        </p>
        <p>
            The winning class, <strong>{html.escape(predicted_label)}</strong>, finished ahead of <strong>{html.escape(runner_up_label)}</strong> by a probability margin of <strong>{margin:.2%}</strong>.
            The attention heatmap below is an input-saliency view that highlights the areas that most influenced the network's decision, derived from <code>{html.escape(attention_layer)}</code>. The high-attention mask is computed with an adaptive threshold calibrated to the current saliency map rather than a fixed percentage of pixels.
        </p>
        <p>
            Brighter colored zones indicate stronger influence on the model output. For this image, the adaptive threshold was <strong>{high_attention_threshold:.2f}</strong>. This is an interpretability aid for the current inference run, not a clinical segmentation or a standalone diagnosis.
        </p>
        """
    )

    if error_message:
        detail_copy = (
            f"""
            <p>
                The prediction is available, but the attention view could not be generated for this run.
                The fallback model-input preview is still shown below.
            </p>
            <p><strong>Technical detail:</strong> {html.escape(str(error_message))}</p>
            """
        )

    return f"""
    <div class="explanation-shell">
        <div class="explanation-title">Why the model predicted this</div>
        {detail_copy}
    </div>
    """


def build_ui(service: PredictionService) -> gr.Blocks:
    project_root = service.settings.project_root
    assets_dir = project_root / "assets"
    banner_path = assets_dir / "banner" / "endometrium_banner.png"
    training_summary = _load_training_summary(project_root)
    training_history = _load_training_history(project_root)
    class_distribution_frame = _build_class_distribution_frame(training_summary)
    split_distribution_frame = _build_split_distribution_frame(training_summary)
    test_metrics_frame = _build_test_metrics_frame(training_summary)
    accuracy_curve_frame = _build_curve_frame(
        training_history,
        {"accuracy": "Training Accuracy", "val_accuracy": "Validation Accuracy"},
    )
    loss_curve_frame = _build_curve_frame(
        training_history,
        {"loss": "Training Loss", "val_loss": "Validation Loss"},
    )
    demo_profile_frame = _build_demo_profile_frame(project_root)
    class_chart_limit = _safe_chart_limit(class_distribution_frame, "count", minimum=10.0)
    split_chart_limit = _safe_chart_limit(split_distribution_frame, "count", minimum=10.0)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
        neutral_hue="slate",
        font=[
            gr.themes.GoogleFont("Manrope"),
            "Avenir Next",
            "Segoe UI",
            "sans-serif",
        ],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    )

    def classify(
        image: Image.Image,
    ) -> tuple[str, dict[str, float], str, Image.Image | None, Image.Image | None, dict[str, Any]]:
        if image is None:
            raise gr.Error("Please upload an image before running inference.")

        if not service.is_ready():
            raise gr.Error("The model is not loaded yet. Export a trained model into the models directory first.")

        result = service.predict(image)
        explanation = service.explain_prediction(image, result)
        metadata = {
            "predicted_index": result["predicted_index"],
            "class_order": service.settings.class_names,
            "model_path": str(service.settings.model_path),
            "input_size": list(service.settings.image_size),
            "attention_layer": explanation.get("attention_layer", "unavailable"),
            "focus_region": explanation.get("focus_region", "unavailable"),
            "focus_pattern": explanation.get("focus_pattern", "unavailable"),
            "focus_coverage": round(float(explanation.get("focus_coverage", 0.0)), 4),
            "high_attention_threshold": round(float(explanation.get("high_attention_threshold", 0.0)), 4),
        }
        return (
            _prediction_card_html(result),
            result["probabilities"],
            _explanation_card_html(result, explanation, service.settings.image_size),
            explanation.get("model_input_image"),
            explanation.get("attention_heatmap_image"),
            metadata,
        )

    def download_demo_bundle() -> str:
        return _build_demo_bundle(project_root)

    def clear_classification_view() -> tuple[
        None,
        str,
        dict[str, float],
        str,
        None,
        None,
        dict[str, Any],
    ]:
        return (
            None,
            _prediction_placeholder_html(),
            {},
            _explanation_placeholder_html(),
            None,
            None,
            {},
        )

    with gr.Blocks(
        title="Endometrial Infection Classification App",
        theme=theme,
        css=CUSTOM_CSS,
    ) as demo:
        with gr.Tabs():
            with gr.Tab("Classify"):
                with gr.Row(elem_classes="hero-shell"):
                    with gr.Column(scale=6, elem_classes="hero-copy"):
                        gr.Markdown(
                            """
                            <span class="hero-eyebrow">AI-Assisted Endometrial Screening</span>
                            # Endometrial Infection Classification App

                            This application helps classify endometrial scan images into **infected** and **uninfected** classes through a production-ready TensorFlow inference pipeline. It combines a modern web interface, a reusable FastAPI backend, and a curated medical-image workflow so the model can move beyond the notebook into a real deployment setting.
                            """,
                            elem_classes="hero-markdown",
                        )
                        gr.HTML(_hero_stats_html(training_summary))
                    with gr.Column(scale=5, elem_classes="hero-banner-wrap"):
                        gr.Image(
                            value=str(banner_path),
                            show_label=False,
                            interactive=False,
                            container=False,
                            show_download_button=False,
                            show_fullscreen_button=False,
                            elem_classes="hero-banner",
                        )

                with gr.Row(elem_classes="workspace-row"):
                    with gr.Column(scale=5, elem_classes="panel-card top-card upload-card"):
                        gr.Markdown(
                            """
                            <span class="section-kicker">Step 1</span>
                            ## Upload an image

                            Add an endometrial scan and send it through the classifier. If you do not have a scan available, open the **Download** tab and use the bundled test image pack.
                            """,
                            elem_classes="helper-copy",
                        )
                        image_input = gr.Image(
                            type="pil",
                            label="Endometrial scan",
                            image_mode="RGB",
                            sources=["upload"],
                            height=420,
                            show_download_button=False,
                            show_fullscreen_button=False,
                            show_share_button=False,
                        )

                    with gr.Column(scale=5, elem_classes="panel-card top-card result-card"):
                        gr.Markdown(
                            """
                            <span class="section-kicker">Step 2</span>
                            ## Review the result

                            The app returns the predicted class, the model confidence, the class-probability distribution, and the inference metadata used for the prediction request.
                            """,
                            elem_classes="helper-copy",
                        )
                        summary_output = gr.HTML(value=_prediction_placeholder_html())
                        probability_output = gr.Label(label="Class probabilities", num_top_classes=2)

                with gr.Row(elem_classes="button-row"):
                    submit_button = gr.Button("Run classification", variant="primary")
                    clear_button = gr.Button("Clear", variant="secondary")

                with gr.Column(elem_classes="panel-card explanation-panel"):
                    explanation_output = gr.HTML(value=_explanation_placeholder_html())
                    with gr.Row(elem_classes="visual-row"):
                        model_input_output = gr.Image(
                            label="Model input used for inference",
                            interactive=False,
                            type="pil",
                            height=300,
                            show_download_button=False,
                            show_fullscreen_button=False,
                        )
                        attention_heatmap_output = gr.Image(
                            label="Model attention heatmap",
                            interactive=False,
                            type="pil",
                            height=300,
                            show_download_button=False,
                            show_fullscreen_button=False,
                        )
                    metadata_output = gr.JSON(label="Inference metadata")

                submit_button.click(
                    fn=classify,
                    inputs=image_input,
                    outputs=[
                        summary_output,
                        probability_output,
                        explanation_output,
                        model_input_output,
                        attention_heatmap_output,
                        metadata_output,
                    ],
                )
                clear_button.click(
                    fn=clear_classification_view,
                    outputs=[
                        image_input,
                        summary_output,
                        probability_output,
                        explanation_output,
                        model_input_output,
                        attention_heatmap_output,
                        metadata_output,
                    ],
                )

                gr.Markdown(
                    """
                    This tool supports research, experimentation, and AI-assisted screening workflows. Final clinical interpretation should remain with qualified medical experts.
                    """,
                    elem_classes="disclaimer",
                )

            with gr.Tab("Download"):
                gr.Markdown(
                    """
                    ## Download Test Images

                    This tab lets users download the demo test-image bundle used for public app exploration. The package is designed for quick testing, demos, and interface validation when real endometrial scans are not immediately available.
                    """,
                    elem_classes="sample-copy",
                )
                gr.HTML(
                    """
                    <div class="download-highlight">
                        <p>
                            The download bundle contains 20 curated test images drawn from the held-out test split: 10 infected examples and 10 uninfected examples.
                        </p>
                    </div>
                    """
                )
                with gr.Row():
                    with gr.Column(scale=6, elem_classes="download-card"):
                        gr.Markdown(
                            """
                            ### What is included

                            - A balanced set of 20 demo scan images
                            - Files organized for quick local testing
                            - A small `README.txt` inside the archive describing the bundle

                            ### Suggested use

                            - Download the bundle
                            - Upload any image into the **Classify** tab
                            - Compare outputs across infected and uninfected examples
                            """,
                        )
                    with gr.Column(scale=4, elem_classes="download-card"):
                        gr.Markdown(
                            f"""
                            ### Ready to download

                            File name: `{_demo_bundle_filename()}`
                            """,
                        )
                        gr.DownloadButton(
                            label="Download test image bundle",
                            value=download_demo_bundle,
                            variant="primary",
                            size="lg",
                        )

            with gr.Tab("EDA Lab"):
                gr.Markdown(
                    """
                    ## EDA Lab

                    This section presents a compact research-facing view of the data curation workflow, split design, optimization traces, and internal evaluation underpinning the deployed classifier.
                    """,
                    elem_classes="sample-copy",
                )
                gr.HTML(_eda_overview_html(training_summary))
                with gr.Row():
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown(
                            _class_distribution_markdown(training_summary),
                            elem_classes="helper-copy",
                        )
                        gr.BarPlot(
                            value=class_distribution_frame,
                            x="class",
                            y="count",
                            color="class",
                            color_map={"Infected": "#0e4d73", "Uninfected": "#178b76"},
                            y_title="Images",
                            x_title="Class",
                            y_lim=class_chart_limit,
                            show_fullscreen_button=False,
                            show_export_button=False,
                        )
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown(
                            _split_strategy_markdown(training_summary),
                            elem_classes="helper-copy",
                        )
                        gr.BarPlot(
                            value=split_distribution_frame,
                            x="split_class",
                            y="count",
                            color="class",
                            color_map={"Infected": "#0e4d73", "Uninfected": "#178b76"},
                            y_title="Images",
                            x_title="Partition / Class",
                            y_lim=split_chart_limit,
                            sort=[
                                "Train - Infected",
                                "Train - Uninfected",
                                "Validation - Infected",
                                "Validation - Uninfected",
                                "Test - Infected",
                                "Test - Uninfected",
                            ],
                            x_label_angle=-20,
                            show_fullscreen_button=False,
                            show_export_button=False,
                        )
                with gr.Row():
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown(
                            _accuracy_curves_markdown(training_history),
                            elem_classes="helper-copy",
                        )
                        gr.LinePlot(
                            value=accuracy_curve_frame,
                            x="epoch",
                            y="value",
                            color="series",
                            color_map={
                                "Training Accuracy": "#0e4d73",
                                "Validation Accuracy": "#178b76",
                            },
                            y_title="Accuracy",
                            x_title="Epoch",
                            show_fullscreen_button=False,
                            show_export_button=False,
                        )
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown(
                            _loss_curves_markdown(training_history),
                            elem_classes="helper-copy",
                        )
                        gr.LinePlot(
                            value=loss_curve_frame,
                            x="epoch",
                            y="value",
                            color="series",
                            color_map={
                                "Training Loss": "#0e4d73",
                                "Validation Loss": "#178b76",
                            },
                            y_title="Loss",
                            x_title="Epoch",
                            show_fullscreen_button=False,
                            show_export_button=False,
                        )
                with gr.Row():
                    with gr.Column(scale=4, elem_classes="panel-card"):
                        gr.Markdown(
                            _research_safeguards_markdown(training_summary),
                            elem_classes="helper-copy",
                        )
                    with gr.Column(scale=6, elem_classes="panel-card"):
                        gr.Markdown(
                            _held_out_evaluation_markdown(training_summary),
                            elem_classes="helper-copy",
                        )
                        gr.BarPlot(
                            value=test_metrics_frame,
                            x="metric",
                            y="value",
                            color_map={"value": "#178b76"},
                            y_title="Score",
                            x_title="Metric",
                            show_fullscreen_button=False,
                            show_export_button=False,
                        )
                with gr.Row():
                    with gr.Column(scale=4, elem_classes="panel-card"):
                        gr.Markdown(
                            _demo_profile_markdown(),
                            elem_classes="helper-copy",
                        )
                        gr.Dataframe(
                            value=demo_profile_frame,
                            interactive=False,
                            wrap=True,
                            row_count=len(demo_profile_frame),
                            col_count=len(demo_profile_frame.columns),
                        )
                    with gr.Column(scale=6, elem_classes="panel-card"):
                        gr.Markdown(
                            _interpretation_note_markdown(training_summary),
                            elem_classes="helper-copy",
                        )
                        gr.Markdown(
                            """
                            - These performance estimates are internal to the current curated archive and should not be interpreted as population-level performance.
                            - Group-aware splitting reduces one major source of optimistic bias, but study-level and external validation remain the stronger tests of generalization.
                            - The app now exposes its curation and audit safeguards so readers can review the evaluation protocol with greater transparency.
                            """,
                            elem_classes="helper-copy",
                        )
            with gr.Tab("About"):
                gr.Markdown(
                    _project_about_markdown(training_summary),
                    elem_classes="about-copy",
                )
                gr.Markdown(AUTHOR_MARKDOWN, elem_classes="author-copy")
                with gr.Row(elem_classes="author-row"):
                    for profile in AUTHOR_PROFILES:
                        with gr.Column(scale=1, elem_classes="author-card"):
                            author_image_path = _author_image_path(profile, assets_dir)
                            if author_image_path is not None:
                                gr.Image(
                                    value=str(author_image_path),
                                    show_label=False,
                                    interactive=False,
                                    container=False,
                                    show_download_button=False,
                                    show_fullscreen_button=False,
                                    height=220,
                                    elem_classes="author-photo",
                                )
                            else:
                                gr.HTML(_author_placeholder_html(profile))
                            gr.Markdown(_author_card_markdown(profile), elem_classes="author-copy")
            with gr.Tab("Future Dev"):
                gr.Markdown(
                    _future_dev_markdown(),
                    elem_classes="sample-copy",
                )
                with gr.Row():
                    with gr.Column(scale=6, elem_classes="panel-card"):
                        gr.Markdown(
                            """
                            <span class="section-kicker">Roadmap Highlights</span>
                            ## Where the project can grow next

                            The roadmap emphasizes stronger evaluation, richer clinical grounding, more robust deployment practice, and a more useful human-review workflow. The aim is not only to improve model performance, but also to improve evidence quality and research usability.
                            """,
                            elem_classes="helper-copy",
                        )
                        gr.Markdown(
                            """
                            - strengthen study-level and external validation
                            - improve calibration, uncertainty handling, and explainability
                            - explore ROI-focused and multimodal modeling
                            - support clinician feedback loops and richer audit trails
                            - benchmark stronger architectures under the same evaluation protocol
                            """,
                            elem_classes="helper-copy",
                        )
                    with gr.Column(scale=4, elem_classes="panel-card"):
                        gr.Markdown(
                            f"""
                            <span class="section-kicker">Repository Guide</span>
                            ## Open the full roadmap

                            The complete future-development document lives in the repository as `future development.md`.

                            [View the full roadmap on GitHub]({FUTURE_DEVELOPMENT_URL})
                            """,
                            elem_classes="helper-copy",
                        )

        gr.HTML(FOOTER_HTML)

    return demo
