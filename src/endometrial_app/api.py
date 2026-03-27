from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.background import BackgroundTask

from endometrial_app.demo_bundle import build_demo_bundle, demo_bundle_filename
from endometrial_app.schemas import ErrorResponse, HealthResponse, PredictionResponse
from endometrial_app.service import PredictionService


def create_api_app(service: PredictionService) -> FastAPI:
    app = FastAPI(
        title="Endometrial Infection Classifier API",
        version="1.0.0",
        description="FastAPI backend for endometrial infection image classification.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict[str, object]:
        return service.health()

    @app.post(
        "/api/predict",
        response_model=PredictionResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(file: UploadFile = File(...)) -> dict[str, object]:
        if not service.is_ready():
            raise HTTPException(status_code=503, detail="Model is not ready. Export a trained model first.")

        try:
            payload = await file.read()
            image = Image.open(BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from None

        return service.predict(image)

    @app.get("/api/metadata")
    def metadata() -> JSONResponse:
        return JSONResponse(
            {
                "project_name": service.settings.project_name,
                "class_names": service.settings.class_names,
                "image_size": {
                    "width": service.settings.image_width,
                    "height": service.settings.image_height,
                },
                "model_path": str(service.settings.model_path),
            }
        )

    @app.get("/downloads/demo-pack")
    def download_demo_pack() -> FileResponse:
        bundle_path = Path(build_demo_bundle(service.settings.project_root))
        return FileResponse(
            bundle_path,
            media_type="application/zip",
            filename=demo_bundle_filename(),
            background=BackgroundTask(bundle_path.unlink, missing_ok=True),
        )

    return app
