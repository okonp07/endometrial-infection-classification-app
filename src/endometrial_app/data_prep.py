from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def compute_difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    grayscale = image.convert("L")
    resized = grayscale.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float32)
    differences = pixels[:, 1:] > pixels[:, :-1]

    hash_value = 0
    for bit in differences.flatten():
        hash_value = (hash_value << 1) | int(bool(bit))
    return int(hash_value)


def compute_image_hash(image_path: Path, hash_size: int = 8) -> int:
    with Image.open(image_path) as image:
        image.load()
        return compute_difference_hash(image, hash_size=hash_size)


def hamming_distance(left_hash: int, right_hash: int) -> int:
    return int(left_hash ^ right_hash).bit_count()


@dataclass
class _UnionFind:
    size: int

    def __post_init__(self) -> None:
        self.parent = list(range(self.size))
        self.rank = [0] * self.size

    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def assign_similarity_groups(
    manifest_df: pd.DataFrame,
    *,
    hash_column: str = "dhash",
    label_column: str = "label",
    threshold: int = 2,
) -> pd.DataFrame:
    if manifest_df.empty:
        result = manifest_df.copy()
        result["similarity_group"] = pd.Series(dtype="object")
        return result

    grouped_frames: list[pd.DataFrame] = []
    for label, label_df in manifest_df.groupby(label_column, sort=True):
        frame = label_df.copy().reset_index(drop=True)
        hash_values = frame[hash_column].astype(object).tolist()
        union_find = _UnionFind(len(frame))

        for left_index, left_hash in enumerate(hash_values):
            for right_index in range(left_index + 1, len(hash_values)):
                if hamming_distance(int(left_hash), int(hash_values[right_index])) <= threshold:
                    union_find.union(left_index, right_index)

        roots = [union_find.find(index) for index in range(len(frame))]
        root_map: dict[int, str] = {}
        next_group_index = 0
        group_ids: list[str] = []

        for root in roots:
            if root not in root_map:
                root_map[root] = f"{label}-group-{next_group_index:04d}"
                next_group_index += 1
            group_ids.append(root_map[root])

        frame["similarity_group"] = group_ids
        grouped_frames.append(frame)

    return pd.concat(grouped_frames, ignore_index=True)


def build_similarity_group_summary(
    manifest_df: pd.DataFrame,
    *,
    group_column: str = "similarity_group",
    label_column: str = "label",
) -> dict[str, Any]:
    if manifest_df.empty or group_column not in manifest_df:
        return {
            "group_count": 0,
            "multi_image_group_count": 0,
            "images_in_multi_image_groups": 0,
            "largest_group_size": 0,
            "per_class_group_count": {},
        }

    grouped_sizes = manifest_df.groupby(group_column).size()
    multi_image_groups = grouped_sizes[grouped_sizes > 1]
    per_class_group_count = (
        manifest_df.groupby(label_column)[group_column].nunique().sort_index().to_dict()
    )

    return {
        "group_count": int(grouped_sizes.shape[0]),
        "multi_image_group_count": int(multi_image_groups.shape[0]),
        "images_in_multi_image_groups": int(multi_image_groups.sum()) if not multi_image_groups.empty else 0,
        "largest_group_size": int(grouped_sizes.max()) if not grouped_sizes.empty else 0,
        "per_class_group_count": {key: int(value) for key, value in per_class_group_count.items()},
    }


def assign_grouped_splits(
    manifest_df: pd.DataFrame,
    *,
    group_column: str = "similarity_group",
    label_column: str = "label",
    seed: int = 42,
    split_ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    if split_ratios is None:
        split_ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}

    expected_splits = ("train", "validation", "test")
    if tuple(split_ratios.keys()) != expected_splits:
        split_ratios = {split: split_ratios[split] for split in expected_splits}

    randomizer = random.Random(seed)
    split_frames: list[pd.DataFrame] = []

    for label, label_df in manifest_df.groupby(label_column, sort=True):
        label_frame = label_df.copy()
        group_sizes = (
            label_frame.groupby(group_column)
            .size()
            .reset_index(name="image_count")
            .to_dict("records")
        )
        randomizer.shuffle(group_sizes)
        group_sizes.sort(key=lambda row: row["image_count"], reverse=True)

        total_images = int(label_frame.shape[0])
        split_targets = {
            split_name: total_images * ratio for split_name, ratio in split_ratios.items()
        }
        split_counts = {split_name: 0 for split_name in split_ratios}
        group_assignments: dict[str, str] = {}

        for group_row in group_sizes:
            group_name = str(group_row[group_column])
            image_count = int(group_row["image_count"])

            ranked_splits = sorted(
                split_ratios,
                key=lambda split_name: (
                    max(0.0, split_counts[split_name] + image_count - split_targets[split_name]),
                    abs(split_targets[split_name] - (split_counts[split_name] + image_count)),
                    split_counts[split_name],
                    split_name,
                ),
            )
            chosen_split = ranked_splits[0]
            group_assignments[group_name] = chosen_split
            split_counts[chosen_split] += image_count

        label_frame["split"] = label_frame[group_column].map(group_assignments)
        split_frames.append(label_frame)

    return pd.concat(split_frames, ignore_index=True)


def summarize_cross_split_similarity(
    manifest_df: pd.DataFrame,
    *,
    hash_column: str = "dhash",
    label_column: str = "label",
    split_column: str = "split",
    threshold: int = 2,
) -> dict[str, Any]:
    if manifest_df.empty or split_column not in manifest_df:
        return {
            "nearest_distance_by_split": {},
            "items_with_nearest_train_distance_le_threshold": {},
            "threshold": threshold,
        }

    summary: dict[str, Any] = {
        "threshold": int(threshold),
        "nearest_distance_by_split": {},
        "items_with_nearest_train_distance_le_threshold": {},
    }

    for label, label_df in manifest_df.groupby(label_column, sort=True):
        train_hashes = label_df.loc[label_df[split_column] == "train", hash_column].astype(object).tolist()
        if not train_hashes:
            continue

        for split_name in ["validation", "test"]:
            split_hashes = label_df.loc[label_df[split_column] == split_name, hash_column].astype(object).tolist()
            if not split_hashes:
                continue

            nearest_distances = [
                min(hamming_distance(int(candidate_hash), int(train_hash)) for train_hash in train_hashes)
                for candidate_hash in split_hashes
            ]
            summary["nearest_distance_by_split"][f"{label}_{split_name}"] = {
                "min": int(min(nearest_distances)),
                "median": float(pd.Series(nearest_distances).median()),
                "max": int(max(nearest_distances)),
            }
            summary["items_with_nearest_train_distance_le_threshold"][f"{label}_{split_name}"] = int(
                sum(distance <= threshold for distance in nearest_distances)
            )

    return summary
