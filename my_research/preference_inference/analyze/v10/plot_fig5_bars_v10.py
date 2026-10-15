"""v10 stage A: Fig.5 error bars (A0 original grid vs A1 swapped grid) from vpt_metrics_v10.py JSONs."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

D = os.path.join(OUT_ROOT, 'stageA', 'fig5')
KEYS = ['self_self', 'other_other', 'self_other', 'other_self']
MODELS = [('fig5_v4_exp1l1000', 'R3-A_400 Encoder (frozen, original paper)'),
          ('fig5_v5_base', 'v5_base_l1 Encoder (retrained)')]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True, sharey=True)
x = np.arange(len(KEYS))
for ax, (m, title) in zip(axes, MODELS):
    for i, (c, col) in enumerate((('A0', '#4C78A8'), ('A1', '#F58518'))):
        r = json.load(open(os.path.join(D, f'{m}_ep100_{c}.json')))
        v = [r[k]['mean'] for k in KEYS]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.35, color=col,
               label=f'{c} ({"original grid" if c == "A0" else "swapped grid"}), margin {r["margin_other_self_minus_other_other"]:.3f}')
        for j, val in enumerate(v):
            ax.text(x[j] + (i - 0.5) * 0.38, val + 0.003, f'{val:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['self true vs\nself rec', 'other true vs\nother rec', 'self true vs\nother rec', 'other true vs\nself rec'])
    ax.set_title(f'{title}, decoder {m} ep100', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', color='0.9')
axes[0].set_ylabel('mean |error| (grid, margin 1)')
fig.suptitle('v10 stage A: Fig.5 perspective-taking errors; margin = other_self - other_other')
fig.savefig(os.path.join(D, 'fig5_bars.png'), dpi=110)
print('saved', os.path.join(D, 'fig5_bars.png'))
