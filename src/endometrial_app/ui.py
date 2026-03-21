from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from endometrial_app.service import PredictionService


CUSTOM_CSS = """
:root {
    --brand-blue: #0e4d73;
    --brand-blue-deep: #092d46;
    --brand-green: #178b76;
    --brand-green-soft: #dff4ee;
    --brand-ash: #eef3f4;
    --brand-ink: #12242d;
    --brand-slate: #60717a;
    --brand-white: #ffffff;
}

.gradio-container {
    background:
        radial-gradient(circle at top right, rgba(23, 139, 118, 0.18), transparent 32%),
        radial-gradient(circle at top left, rgba(14, 77, 115, 0.12), transparent 24%),
        linear-gradient(180deg, #f8fbfb 0%, #edf2f3 100%);
    color: var(--brand-ink);
    font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: inherit;
}

[role="tablist"] {
    background: rgba(255, 255, 255, 0.78);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 22px;
    padding: 0.4rem;
    box-shadow: 0 12px 32px rgba(18, 36, 45, 0.08);
}

button[role="tab"] {
    border-radius: 16px !important;
    color: var(--brand-blue-deep) !important;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--brand-blue-deep), var(--brand-green)) !important;
    color: var(--brand-white) !important;
    box-shadow: 0 12px 28px rgba(14, 77, 115, 0.25);
}

.hero-shell,
.workspace-row,
.author-row {
    gap: 1rem;
    align-items: stretch;
}

.hero-copy,
.hero-banner-wrap,
.panel-card,
.author-copy-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(9, 45, 70, 0.08);
    border-radius: 28px;
    box-shadow: 0 20px 56px rgba(18, 36, 45, 0.08);
    padding: 1.4rem !important;
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
}

.hero-banner img {
    width: 100%;
    min-height: 100%;
    border-radius: 22px;
    object-fit: cover;
    box-shadow: 0 16px 34px rgba(18, 36, 45, 0.16);
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
}

.helper-copy p,
.sample-copy p,
.about-copy p,
.about-copy li,
.author-copy p {
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

.author-row {
    align-items: center;
}

.author-image-column {
    display: flex;
    justify-content: center;
    align-items: center;
}

.author-photo img {
    width: 100%;
    max-width: 270px;
    margin: 0 auto;
    border-radius: 999px;
    border: 6px solid rgba(23, 139, 118, 0.12);
    box-shadow: 0 18px 46px rgba(18, 36, 45, 0.16);
}

.author-copy-card h2 {
    margin-top: 0 !important;
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
    .hero-copy h1 {
        font-size: 2.35rem !important;
    }

    .hero-stat-grid {
        grid-template-columns: 1fr;
    }
}
"""


FOOTER_HTML = """
<div class="footer-note">
    <p><strong>&copy; Okon Prince, 2026</strong></p>
    <p>
        This project is based on research work by Dr. Obi Cajetan of the University of Calabar Teaching Hospital and Prince Okon.
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


def _project_about_markdown(summary: dict[str, Any]) -> str:
    clean_counts = summary.get("clean_counts", {})
    split_counts = summary.get("split_counts", {})
    infected_count = clean_counts.get("infected", "N/A")
    uninfected_count = clean_counts.get("uninfected", "N/A")
    test_infected = split_counts.get("test", {}).get("infected", "N/A")
    test_uninfected = split_counts.get("test", {}).get("uninfected", "N/A")

    return f"""
## About the Project

This application is an end-to-end image classification solution built to distinguish between **infected** and **uninfected** endometrial scan images. The goal is to make the screening workflow practical, reproducible, and accessible through a clean browser interface backed by the same inference service exposed through the API.

### What problem the solution addresses

Endometrial infection assessment from imaging data can be difficult to operationalize in a way that is fast, repeatable, and easy for researchers or clinicians to interact with. This project turns the underlying research work into a deployable application so that a user can upload a scan, receive a class prediction, inspect class probabilities, and use the tool as part of a research-support workflow.

### How the solution works

1. A user uploads an endometrial scan image or selects one of the built-in demo samples.
2. The app preprocesses the image into the format expected by the TensorFlow model.
3. The trained classifier scores the image against the two target classes: `infected` and `uninfected`.
4. The interface returns the predicted class, the model confidence, a class-probability breakdown, and inference metadata.
5. The same model powers both the Gradio interface and the FastAPI backend, which keeps browser predictions and API predictions consistent.

### Data and evaluation summary

The deployed model was trained from curated archive data after duplicate handling and split generation. The current production bundle was prepared from **{infected_count} infected** images and **{uninfected_count} uninfected** images, with a held-out test set of **{test_infected} infected** and **{test_uninfected} uninfected** images. This makes the deployment grounded in a proper train, validation, and test workflow rather than a notebook-only demonstration.

### What the output means

The predicted label is the class the model considers most likely for the uploaded scan. The confidence score shows how strongly the model favors that decision, while the class-probability panel reveals the distribution across both classes. This is useful because it allows the user to see not only the final prediction, but also how decisive or uncertain the model is.

### Responsible use

This application is best understood as a **research and AI-assisted classification tool**. It is designed to support structured analysis, experimentation, and decision support. It should not be treated as a standalone clinical diagnosis without expert interpretation and appropriate medical context.
"""


AUTHOR_MARKDOWN = """
## About the Author

**Okon Prince**  
Senior Data Scientist at MIVA Open University | AI Engineer & Data Scientist

I design and deploy end-to-end data systems that turn raw data into production-ready intelligence.

My core stack includes Python, Streamlit, BigQuery, Supabase, Hugging Face, PySpark, SQL, Machine Learning, LLMs, and Transformers.

My work spans risk scoring systems, A/B testing, traditional and AI-powered dashboards, RAG pipelines, predictive analytics, LLM-based solutions, and AI research.

Currently, I work as a Senior Data Scientist in the Department of Research and Development at MIVA Open University, where I carry out AI and machine learning research and build intelligent systems that drive analytics, decision support, and scalable AI innovation.

I believe: models are trained, systems are engineered, and impact is delivered.
"""


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


def build_ui(service: PredictionService) -> gr.Blocks:
    project_root = service.settings.project_root
    assets_dir = project_root / "assets"
    banner_path = assets_dir / "banner" / "endometrium_banner.png"
    author_path = assets_dir / "author" / "okon-prince.png"
    training_summary = _load_training_summary(project_root)
    demo_samples = _collect_demo_samples(project_root)

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

    def classify(image: Image.Image) -> tuple[str, dict[str, float], dict[str, Any]]:
        if image is None:
            raise gr.Error("Please upload an image before running inference.")

        if not service.is_ready():
            raise gr.Error("The model is not loaded yet. Export a trained model into the models directory first.")

        result = service.predict(image)
        metadata = {
            "predicted_index": result["predicted_index"],
            "class_order": service.settings.class_names,
            "model_path": str(service.settings.model_path),
            "input_size": list(service.settings.image_size),
        }
        return _prediction_card_html(result), result["probabilities"], metadata

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
                            elem_classes="hero-banner",
                        )

                with gr.Row(elem_classes="workspace-row"):
                    with gr.Column(scale=5, elem_classes="panel-card"):
                        gr.Markdown(
                            """
                            <span class="section-kicker">Step 1</span>
                            ## Upload an image

                            Add an endometrial scan and send it through the classifier. If you do not have a scan available, open the **Sample Images** tab and click any of the 20 curated demo scans.
                            """,
                            elem_classes="helper-copy",
                        )
                        image_input = gr.Image(
                            type="pil",
                            label="Endometrial scan",
                            image_mode="RGB",
                        )

                    with gr.Column(scale=5, elem_classes="panel-card"):
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
                        metadata_output = gr.JSON(label="Inference metadata")

                with gr.Row(elem_classes="button-row"):
                    submit_button = gr.Button("Run classification", variant="primary")
                    gr.ClearButton(
                        [image_input, summary_output, probability_output, metadata_output],
                        value="Clear",
                    )

                submit_button.click(
                    fn=classify,
                    inputs=image_input,
                    outputs=[summary_output, probability_output, metadata_output],
                )

                gr.Markdown(
                    """
                    This tool supports research, experimentation, and AI-assisted screening workflows. Final clinical interpretation should remain with qualified medical experts.
                    """,
                    elem_classes="disclaimer",
                )

            with gr.Tab("Sample Images"):
                gr.Markdown(
                    """
                    ## Sample Images

                    These 20 demo scans were extracted from the held-out test split so visitors can try the app without needing their own endometrial image. Click any thumbnail below and it will populate the uploader in the **Classify** tab.
                    """,
                    elem_classes="sample-copy",
                )
                gr.HTML(
                    """
                    <div class="sample-note">
                        <p>
                            The sample set is balanced across the two target classes and is intended purely for demonstration and interface testing.
                        </p>
                    </div>
                    """
                )
                gr.Examples(
                    examples=demo_samples,
                    inputs=image_input,
                    examples_per_page=20,
                    label="Demo scan gallery",
                )

            with gr.Tab("About"):
                gr.Markdown(
                    _project_about_markdown(training_summary),
                    elem_classes="about-copy",
                )
                with gr.Row(elem_classes="author-row"):
                    with gr.Column(scale=2, elem_classes="author-image-column"):
                        gr.Image(
                            value=str(author_path),
                            show_label=False,
                            interactive=False,
                            container=False,
                            show_download_button=False,
                            elem_classes="author-photo",
                        )
                    with gr.Column(scale=5, elem_classes="author-copy-card"):
                        gr.Markdown(AUTHOR_MARKDOWN, elem_classes="author-copy")

        gr.HTML(FOOTER_HTML)

    return demo
