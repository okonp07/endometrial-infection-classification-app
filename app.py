from __future__ import annotations

import gradio as gr

from endometrial_app.api import create_api_app
from endometrial_app.service import PredictionService
from endometrial_app.ui import build_ui


service = PredictionService.from_settings()
api_app = create_api_app(service)
ui = build_ui(service)
app = gr.mount_gradio_app(api_app, ui, path="/")
