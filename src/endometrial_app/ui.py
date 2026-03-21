from __future__ import annotations

from typing import Any

import gradio as gr
from PIL import Image

from endometrial_app.service import PredictionService


def build_ui(service: PredictionService) -> gr.Blocks:
    def classify(image: Image.Image) -> tuple[str, dict[str, float], dict[str, Any]]:
        if image is None:
            raise gr.Error("Please upload an image before running inference.")

        if not service.is_ready():
            raise gr.Error("The model is not loaded yet. Export a trained model into the models directory first.")

        result = service.predict(image)
        summary = (
            f"Prediction: {result['predicted_label']}\n"
            f"Confidence: {result['confidence']:.4f}"
        )
        metadata = {
            "predicted_index": result["predicted_index"],
            "class_order": service.settings.class_names,
            "model_path": str(service.settings.model_path),
        }
        return summary, result["probabilities"], metadata

    with gr.Blocks(title="Endometrial Infection Classifier") as demo:
        gr.Markdown(
            """
            # Endometrial Infection Classifier

            Upload an endometrial image to classify it as either `infected` or `uninfected`.

            This UI runs on the same prediction service as the FastAPI backend, so the browser demo and the API stay consistent.
            """
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload image")

            with gr.Column():
                summary_output = gr.Textbox(label="Prediction summary")
                probability_output = gr.Label(label="Class probabilities", num_top_classes=2)
                metadata_output = gr.JSON(label="Inference metadata")

        submit_button = gr.Button("Run classification", variant="primary")
        submit_button.click(
            fn=classify,
            inputs=image_input,
            outputs=[summary_output, probability_output, metadata_output],
        )

        gr.Markdown(
            """
            ## API endpoints

            - `GET /health`
            - `GET /api/metadata`
            - `POST /api/predict`
            - `GET /docs`
            """
        )

    return demo
