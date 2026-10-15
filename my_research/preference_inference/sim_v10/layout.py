"""
v10 landmark layouts -- the single place where layouts are defined
(docs/v10_instructions.md Sec.6.0, docs/v10_experiment_log.md Sec.4).

Only red and green swap places; blue/cyan, cube geometry, lighting, colours
and the reward definition are unchanged. The swap is a material swap in a
copy of the field mesh (resource/model/field_swap_rg.obj: the `usemtl`
lines of Cube.001 and Cube.002 exchanged).

Naming caution: Camera.read_pixel() reads GL_BGR, so the project's "Red"
(array channel 0) is the `blue` *material* cube (Cube.001) in field.mtl,
"Blue" is the `red` material (Cube), "Cyan" is the `yellow` material, and
"Green" is the `green` material (Cube.002).

Reference points are the project's (+-9, +-9) convention (used by v4-v7
analyses); the cubes themselves are centred at (+-13, +-13), outside the
+-10 arena -- kept as CUBE_CENTERS for the reference-only numbers.
"""

import numpy as np

ORIGINAL = 'original'
SWAP_RG = 'swap_rg'
LAYOUT_NAMES = (ORIGINAL, SWAP_RG)

FIELD_OBJ = {
    ORIGINAL: 'field.obj',
    SWAP_RG: 'field_swap_rg.obj',
}

# arena (x, y) = world (x, z)
LANDMARKS = {
    ORIGINAL: {
        'Red': np.array([-9.0, 9.0]),
        'Green': np.array([-9.0, -9.0]),
        'Blue': np.array([9.0, -9.0]),
        'Cyan': np.array([9.0, 9.0]),
    },
    SWAP_RG: {
        'Red': np.array([-9.0, -9.0]),
        'Green': np.array([-9.0, 9.0]),
        'Blue': np.array([9.0, -9.0]),
        'Cyan': np.array([9.0, 9.0]),
    },
}

CUBE_CENTERS = {
    ORIGINAL: {
        'Red': np.array([-13.0, 13.0]),
        'Green': np.array([-13.0, -13.0]),
        'Blue': np.array([13.0, -13.0]),
        'Cyan': np.array([13.0, 13.0]),
    },
    SWAP_RG: {
        'Red': np.array([-13.0, -13.0]),
        'Green': np.array([-13.0, 13.0]),
        'Blue': np.array([13.0, -13.0]),
        'Cyan': np.array([13.0, 13.0]),
    },
}

# integer code stored per episode in v10 datasets (`layout` field)
LAYOUT_CODE = {ORIGINAL: 0, SWAP_RG: 1}
CODE_LAYOUT = {v: k for k, v in LAYOUT_CODE.items()}


def other_layout(name):
    return SWAP_RG if name == ORIGINAL else ORIGINAL


def color_fractions(vision):
    """vision: (H, W, 3) float in [0,1] or uint8. Same thresholds as the
    reward in train_rl_v3.py (channel > 0.9, others < 0.1); cyan = ch1 and
    ch2 > 0.9, ch0 < 0.1."""
    v = np.asarray(vision)
    if v.dtype == np.uint8:
        v = v.astype(np.float32) / 255.0
    c0, c1, c2 = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    n = v.shape[0] * v.shape[1]
    return {
        'Red': float(((c0 > .9) & (c1 < .1) & (c2 < .1)).sum()) / n,
        'Green': float(((c1 > .9) & (c0 < .1) & (c2 < .1)).sum()) / n,
        'Blue': float(((c2 > .9) & (c0 < .1) & (c1 < .1)).sum()) / n,
        'Cyan': float(((c1 > .9) & (c2 > .9) & (c0 < .1)).sum()) / n,
    }
