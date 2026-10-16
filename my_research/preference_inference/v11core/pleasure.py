"""
v11 innate per-direction pleasure signal and colour geometry
(docs/v11_experiment_log.md Sec.3.2, 4.3, 4.4, 5.0).

The panorama (16 x 64) is world-fixed: column c faces the arena angle
(c + 0.5) * 5.625 deg - 180 deg (atan2(dy, dx), y = world z), measured on
the root `grid` (mean residual 2.4 deg). Bin k = columns 8k..8k+7 = one of
the 8 cameras, centred at -157.5 deg + 45 k. Each landmark fills exactly one
16-column quadrant, so the liked and disliked colours never share a bin.

P[k] = (#liked-colour pixels - #disliked-colour pixels in bin k) / 128,
in [-1, 1] by construction (scale constant c = 1).

Colour names follow the project convention (array channel 0 = "Red"; see
sim_v10/layout.py). Both numpy (collection / analysis, HWC) and torch
(model forward, NCHW) versions are given; they must agree exactly
(gate 6.1 item 3).
"""

import numpy as np
import torch

N_COLS = 64
N_BINS = 8
COLS_PER_BIN = N_COLS // N_BINS
HEIGHT = 16
BIN_PIXELS = HEIGHT * COLS_PER_BIN  # 128

COL_ANGLES = np.deg2rad((np.arange(N_COLS) + 0.5) * 360.0 / N_COLS - 180.0)
BIN_ANGLES = np.deg2rad(-157.5 + 45.0 * np.arange(N_BINS))
BIN_UNIT = np.stack([np.cos(BIN_ANGLES), np.sin(BIN_ANGLES)], 1)  # (8, 2)

COLORS = ('Red', 'Green', 'Blue', 'Cyan')


def color_mask(v, color):
    """v: (..., H, W, 3) float in [0,1] or uint8 -> bool (..., H, W)."""
    v = np.asarray(v)
    if v.dtype == np.uint8:
        v = v.astype(np.float32) / 255.0
    c0, c1, c2 = v[..., 0], v[..., 1], v[..., 2]
    if color == 'Red':
        return (c0 > .9) & (c1 < .1) & (c2 < .1)
    if color == 'Green':
        return (c1 > .9) & (c0 < .1) & (c2 < .1)
    if color == 'Blue':
        return (c2 > .9) & (c0 < .1) & (c1 < .1)
    if color == 'Cyan':
        return (c1 > .9) & (c2 > .9) & (c0 < .1)
    raise ValueError(color)


def column_counts(v, color):
    """(..., H, W, 3) -> (..., W) pixel counts of `color` per column."""
    return color_mask(v, color).sum(-2)


def pleasure_np(v, liked='Red', disliked='Green'):
    """(..., 16, 64, 3) -> (..., 8) float32."""
    d = column_counts(v, liked).astype(np.float32) - column_counts(v, disliked)
    d = d.reshape(d.shape[:-1] + (N_BINS, COLS_PER_BIN)).sum(-1)
    return (d / BIN_PIXELS).astype(np.float32)


def _color_mask_torch(v, color):
    """v: (B, 3, H, W) float in [0,1]."""
    c0, c1, c2 = v[:, 0], v[:, 1], v[:, 2]
    if color == 'Red':
        return (c0 > .9) & (c1 < .1) & (c2 < .1)
    if color == 'Green':
        return (c1 > .9) & (c0 < .1) & (c2 < .1)
    if color == 'Blue':
        return (c2 > .9) & (c0 < .1) & (c1 < .1)
    if color == 'Cyan':
        return (c1 > .9) & (c2 > .9) & (c0 < .1)
    raise ValueError(color)


def pleasure_torch(v_raw, liked='Red', disliked='Green'):
    """v_raw: (B, 3, 16, 64) float in [0,1] -> (B, 8) float32."""
    d = (_color_mask_torch(v_raw, liked).sum(1)
         - _color_mask_torch(v_raw, disliked).sum(1)).float()  # (B, W)
    d = d.reshape(d.size(0), N_BINS, COLS_PER_BIN).sum(-1)
    return d / BIN_PIXELS


def color_direction(v, color):
    """Circular centroid direction of `color` pixels (weights = pixels per
    column). v: (16, 64, 3). Returns a unit 2-vector or None if the colour
    is not visible (or the centroid is degenerate)."""
    w = column_counts(v, color).astype(np.float64)
    if w.sum() == 0:
        return None
    z = (w * np.exp(1j * COL_ANGLES)).sum()
    if abs(z) < 1e-9 * w.sum():
        return None
    return np.array([z.real, z.imag]) / abs(z)
