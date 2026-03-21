from __future__ import annotations

import numpy as np

from endometrial_app.model import _build_attention_mask, _focus_pattern_label


def test_attention_mask_coverage_varies_with_heatmap_structure() -> None:
    compact_heatmap = np.zeros((64, 64), dtype=np.float32)
    compact_heatmap[30:34, 30:34] = 1.0

    diffuse_heatmap = np.zeros((64, 64), dtype=np.float32)
    diffuse_heatmap[14:50, 14:50] = 0.65
    diffuse_heatmap[24:40, 24:40] = 1.0

    compact_mask, compact_threshold = _build_attention_mask(compact_heatmap)
    diffuse_mask, diffuse_threshold = _build_attention_mask(diffuse_heatmap)

    assert compact_mask.mean() < diffuse_mask.mean()
    assert 0.0 <= compact_threshold <= 1.0
    assert 0.0 <= diffuse_threshold <= 1.0


def test_focus_pattern_label_tracks_coverage_bands() -> None:
    assert _focus_pattern_label(0.04) == "compact"
    assert _focus_pattern_label(0.12) == "moderately concentrated"
    assert _focus_pattern_label(0.24) == "broad"
    assert _focus_pattern_label(0.42) == "diffuse"
