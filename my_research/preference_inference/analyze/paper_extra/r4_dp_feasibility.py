"""
Paper round 4, work 5 (estimate only; docs/paper_extra_log.md Sec.30.6): on which share of episodes could G3 (and so
d_P) be computed on the B4 test data (v13_s1_mixed80 test) compared with the common test? A colour's slope needs the
points p_20 + d * unit(ref_X - p_20), d = 0..5, inside the arena (|q| <= 10; render_cfdist_v11 + localise_v11.dstar,
DMIN_VALID = 5); g = slope(Red) - mean of the others needs Red. Positions and layouts only (no rendering, no model).
"""
import os
import sys

import h5py
import numpy as np

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa

for nm in ('v11_s1_random', 'v13_s1_mixed80'):
    g = h5py.File(os.path.join(_PI, 'data', 'data', nm, 'data.h5'), 'r')['test']
    p, lay = g['self_position'][:, 20].astype(np.float64), g['layout'][()]
    ok = np.zeros((len(p), 4), bool)
    for n in range(len(p)):
        name = L.CODE_LAYOUT[int(lay[n])]
        for k, X in enumerate(('Red', 'Green', 'Blue', 'Cyan')):
            r = L.LANDMARKS[name][X] - p[n]
            u = r / max(np.linalg.norm(r), 1e-9)
            ok[n, k] = all(np.all(np.abs(p[n] + d * u) <= 10) for d in range(6))
    g_ok = ok[:, 0] & ok[:, 1:].any(1)
    print(nm, 'Red usable', round(float(ok[:, 0].mean()), 3), 'per colour', ok.mean(0).round(3).tolist(), 'g computable', round(float(g_ok.mean()), 3))
