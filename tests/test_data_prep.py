from __future__ import annotations

import pandas as pd

from endometrial_app.data_prep import (
    assign_grouped_splits,
    assign_similarity_groups,
    hamming_distance,
    summarize_cross_split_similarity,
)


def test_assign_similarity_groups_clusters_near_duplicates_within_label() -> None:
    manifest = pd.DataFrame(
        [
            {"label": "infected", "file_name": "a.jpg", "dhash": 0b0000},
            {"label": "infected", "file_name": "b.jpg", "dhash": 0b0001},
            {"label": "infected", "file_name": "c.jpg", "dhash": 0b1111},
            {"label": "uninfected", "file_name": "d.jpg", "dhash": 0b0000},
            {"label": "uninfected", "file_name": "e.jpg", "dhash": 0b0011},
        ]
    )

    grouped = assign_similarity_groups(manifest, threshold=1)

    infected_groups = grouped.loc[grouped["label"] == "infected", "similarity_group"].tolist()
    uninfected_groups = grouped.loc[grouped["label"] == "uninfected", "similarity_group"].tolist()

    assert infected_groups[0] == infected_groups[1]
    assert infected_groups[0] != infected_groups[2]
    assert uninfected_groups[0] != infected_groups[0]


def test_assign_grouped_splits_keeps_similarity_groups_together() -> None:
    rows: list[dict[str, object]] = []
    for label, prefix in [("infected", "i"), ("uninfected", "u")]:
        for group_index in range(6):
            for image_index in range(2):
                rows.append(
                    {
                        "label": label,
                        "file_name": f"{prefix}_{group_index}_{image_index}.jpg",
                        "similarity_group": f"{label}-group-{group_index}",
                        "dhash": group_index * 10 + image_index,
                    }
                )

    manifest = pd.DataFrame(rows)
    assigned = assign_grouped_splits(manifest, seed=42)

    split_by_group = assigned.groupby("similarity_group")["split"].nunique()

    assert split_by_group.eq(1).all()
    assert set(assigned["split"].unique()) == {"train", "validation", "test"}


def test_similarity_audit_counts_close_cross_split_items() -> None:
    manifest = pd.DataFrame(
        [
            {"label": "infected", "split": "train", "dhash": 0b0000},
            {"label": "infected", "split": "validation", "dhash": 0b0001},
            {"label": "infected", "split": "test", "dhash": 0b1111},
            {"label": "uninfected", "split": "train", "dhash": 0b1000},
            {"label": "uninfected", "split": "validation", "dhash": 0b1001},
            {"label": "uninfected", "split": "test", "dhash": 0b0000},
        ]
    )

    audit = summarize_cross_split_similarity(manifest, threshold=1)

    assert audit["items_with_nearest_train_distance_le_threshold"]["infected_validation"] == 1
    assert audit["items_with_nearest_train_distance_le_threshold"]["infected_test"] == 0
    assert audit["nearest_distance_by_split"]["uninfected_validation"]["min"] == hamming_distance(0b1001, 0b1000)
