"""
v15 per-direction pleasure (docs/v15_plan.md, v15_experiment_log.md Sec.2):
P[k] = sum_c value_c * (#pixels of colour c in camera bin k) / 128; gray
cubes match no colour mask, so they contribute 0. With values in [-1, 1]
and the four landmarks in disjoint 16-column quadrants, P is in [-1, 1].
Colour masks and geometry are v11core.pleasure's.
"""

import numpy as np
import torch

from v11core.pleasure import BIN_PIXELS, COLS_PER_BIN, N_BINS, column_counts
from .world import COLORS

RANK_VALUES = (1.0, 0.5, 0.0, -1.0)  # 1st .. 4th


def rank_values(rank, values=RANK_VALUES):
    """rank: tuple of colours, most liked first -> {colour: value}."""
    return {c: values[i] for i, c in enumerate(rank)}


def pleasure_np(v, values):
    """(..., 16, 64, 3) -> (..., 8) float32."""
    d = sum(np.float32(values[c]) * column_counts(v, c).astype(np.float32) for c in COLORS)
    d = d.reshape(d.shape[:-1] + (N_BINS, COLS_PER_BIN)).sum(-1)
    return (d / BIN_PIXELS).astype(np.float32)


def pleasure_torch(v_raw, values):
    """v_raw: (B, 3, 16, 64) float in [0,1]; values: {colour: value} -> (B, 8) float32.
    Same quantity as pleasure_np (checked against it in analyze/v15/check_s2_model_v15.py)."""
    from v11core.pleasure import _color_mask_torch
    d = torch.zeros(v_raw.shape[0], v_raw.shape[-1], device=v_raw.device)
    for c in COLORS:
        if values[c] != 0:
            d = d + float(values[c]) * _color_mask_torch(v_raw, c).sum(1).float()
    d = d.reshape(d.size(0), N_BINS, COLS_PER_BIN).sum(-1)
    return d / BIN_PIXELS
