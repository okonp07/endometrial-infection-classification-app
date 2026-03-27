from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_FIELDS = [
    "submitted_at_utc",
    "name",
    "role",
    "recommendation",
    "rating",
    "thoughts",
    "suggestions",
]


def feedback_log_path(project_root: Path) -> Path:
    return project_root / "artifacts" / "feedback" / "feedback_submissions.csv"


def save_feedback(
    project_root: Path,
    *,
    name: str,
    role: str,
    recommendation: str,
    rating: int,
    thoughts: str,
    suggestions: str,
) -> Path:
    log_path = feedback_log_path(project_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    payload = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": name.strip(),
        "role": role.strip(),
        "recommendation": recommendation.strip(),
        "rating": int(rating),
        "thoughts": thoughts.strip(),
        "suggestions": suggestions.strip(),
    }

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(payload)

    return log_path
