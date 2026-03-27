from __future__ import annotations

import csv

from endometrial_app.feedback import feedback_log_path, save_feedback


def test_save_feedback_creates_csv_log(tmp_path) -> None:
    log_path = save_feedback(
        tmp_path,
        name="Research Reviewer",
        role="Clinical collaborator",
        recommendation="Recommend",
        rating=5,
        thoughts="The explanation flow is clear.",
        suggestions="Add external validation notes.",
    )

    assert log_path == feedback_log_path(tmp_path)
    assert log_path.exists()

    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["name"] == "Research Reviewer"
    assert rows[0]["role"] == "Clinical collaborator"
    assert rows[0]["recommendation"] == "Recommend"
    assert rows[0]["rating"] == "5"
    assert rows[0]["thoughts"] == "The explanation flow is clear."
    assert rows[0]["suggestions"] == "Add external validation notes."
