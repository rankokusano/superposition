"""
v15 Sec.37 (descriptive, after E2): share of the other's h2 points (1-10 steps after eating) that the h1 readout calls blue,
per model and label set, and the share of h2 points that fall inside the range of the self h1 points on PC1-2 (E1 axes).
    python analyze/v15/e2_blue_share_v15.py   # -> data/result/v15/e123/e2_blue_share.json
"""
import json
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.e123_v15 import CONDS, D, SEEDS, load, name, readout  # noqa

R = {}
for c, stem, _ in [(None, None, None)] + list(CONDS):
    for s in ((0,) if c is None else SEEDS):
        n = 'ref0_v15i' if c is None else name(stem, s)
        z = load(n)
        R[n] = {}
        for tag in ('goalcap', 'rankXY'):
            _, clf, _ = readout(z, tag)
            h2 = z['h2'][:, 1:11].reshape(-1, z['h2'].shape[-1])
            pb = clf.predict(h2)
            p = PCA(2).fit(z[f'{tag}_h1']); P1 = p.transform(z[f'{tag}_h1']); P2 = p.transform(h2)
            lo, hi = np.percentile(P1, 1, 0), np.percentile(P1, 99, 0)
            R[n][tag] = {'share_called_blue': float(pb.mean()),
                         'share_inside_h1_range_pc12': float(np.all((P2 >= lo) & (P2 <= hi), 1).mean()),
                         'norm_h2_over_h1': float(np.linalg.norm(h2, axis=1).mean() / np.linalg.norm(z[f'{tag}_h1'], axis=1).mean())}
        print(n, {t: {k: round(v, 3) for k, v in R[n][t].items()} for t in R[n]})
json.dump(R, open(os.path.join(D, 'e2_blue_share.json'), 'w'), indent=1)
