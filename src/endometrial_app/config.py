from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_CLASS_NAMES = ["infected", "uninfected"]


@dataclass(frozen=True)
class Settings:
    project_name: str
    project_root: Path
    model_path: Path
    class_names_path: Path
    image_width: int
    image_height: int
    threshold: float
    host: str
    port: int

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    @property
    def class_names(self) -> list[str]:
        if self.class_names_path.exists():
            return json.loads(self.class_names_path.read_text(encoding="utf-8"))
        return DEFAULT_CLASS_NAMES.copy()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    return Settings(
        project_name=os.getenv("PROJECT_NAME", "Endometrial Infection Classifier"),
        project_root=project_root,
        model_path=Path(
            os.getenv(
                "MODEL_PATH",
                str(project_root / "models" / "endometrial_classifier.keras"),
            )
        ),
        class_names_path=Path(
            os.getenv(
                "CLASS_NAMES_PATH",
                str(project_root / "artifacts" / "class_names.json"),
            )
        ),
        image_width=int(os.getenv("IMAGE_WIDTH", "224")),
        image_height=int(os.getenv("IMAGE_HEIGHT", "224")),
        threshold=float(os.getenv("MODEL_THRESHOLD", "0.5")),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("APP_PORT", "7860"))),
    )
