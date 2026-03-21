from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_label: str
    predicted_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    class_names: list[str]


class ErrorResponse(BaseModel):
    detail: str


class GradioOutput(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    metadata: dict[str, Any]
