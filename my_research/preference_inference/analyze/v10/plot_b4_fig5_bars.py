"""v10 B4: Fig.5 errors per layout (original grid / swapped grid), B3-frozen (retrained decoder, shared by 3 seeds;
plus the old fig5_v4_exp1l1000 decoder as reference) and B3-ft s0-2."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT  # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa
B4 = os.path.join(OUT_ROOT, 'b4', 'fig5'); SA = os.path.join(OUT_ROOT, 'stageA', 'fig5')
KEYS = ['self_self', 'other_other', 'self_other', 'other_self']
ROWS = [('B3-frozen (retrained dec.)', lambda g: os.path.join(B4, f'v10_fig5_frozen_ep100_{g}.json')),
        ('B3-frozen enc. + old dec. (ref.)', lambda g: os.path.join(SA, 'fig5_v4_exp1l1000_ep100_' + ('A0' if g == 'grid' else 'A1') + '.json')),
        ('B3-ft s0', lambda g: os.path.join(B4, f'v10_fig5_ft_s0_ep100_{g}.json')),
        ('B3-ft s1', lambda g: os.path.join(B4, f'v10_fig5_ft_s1_ep100_{g}.json')),
        ('B3-ft s2', lambda g: os.path.join(B4, f'v10_fig5_ft_s2_ep100_{g}.json'))]
cols = ['#4C78A8', '#9ecae9', '#E45756', '#F28E8C', '#FBB4AE']
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True, sharey=True)
for ax, (g, ttl) in zip(axes, (('grid', 'original layout (grid)'), ('v10_swap_grid', 'swapped layout (v10_swap_grid)'))):
    x = np.arange(len(KEYS)); w = 0.16
    for i, (lab, p) in enumerate(ROWS):
        r = json.load(open(p(g)))
        v = [r[k]['mean'] for k in KEYS]
        ax.bar(x + (i - 2) * w, v, w * 0.92, color=cols[i], label=f"{lab}  margin {r['margin_other_self_minus_other_other']:+.3f}")
    ax.set_xticks(x); ax.set_xticklabels(['self true vs\nself rec', 'other true vs\nother rec', 'self true vs\nother rec', 'other true vs\nself rec'])
    ax.set_title(ttl, fontsize=10); ax.legend(fontsize=7, loc='upper left'); ax.grid(axis='y', color='0.93')
axes[0].set_ylabel('mean |error| (margin 1, decoder ep100)')
fig.suptitle('v10 B4: Fig.5 perspective-taking errors (margin = other_self - other_other; > 0 = A-2 view recovered better than A-1 view)')
fig.savefig(os.path.join(OUT_ROOT, 'b4', 'b4_fig5_bars.png'), dpi=110)
print('saved')
