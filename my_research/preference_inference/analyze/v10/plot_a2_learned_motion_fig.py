"""
v10 summary figure (by-product): on the re-evaluation data (a) the other agent A-2 stands still,
but B3-ft (trained where A-2 always walks to green) predicts it as if it had walked toward green.
B3-ft s0 ep400, v10_swapmix_eval_a2stay test episode 0 (A-1's own view). Source: the saved
arrays diag_vision/ft_s0_a2stay_ep0.npz (input frame t, truth = frame t+1, prediction of t+1).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT  # noqa
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa
D = os.path.join(OUT_ROOT, 'b4_reeval', 'diag_vision')
d = np.load(os.path.join(D, 'ft_s0_a2stay_ep0.npz'))
T = [0, 5, 10, 20, 50]
rows = [('input (t)', 'inp'), ('truth (t+1)', 'truth'), ('pred. (t+1)', 'pred')]
fig, ax = plt.subplots(len(rows), len(T), figsize=(3.3 * len(T), 3.6), constrained_layout=True)
for c, t in enumerate(T):
    for r, (lab, k) in enumerate(rows):
        ax[r, c].imshow(np.clip(d[k][t], 0, 1), aspect='auto', interpolation='nearest')
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        if c == 0:
            ax[r, c].set_ylabel(lab, fontsize=10)
    ax[0, c].set_title(f't = {t}', fontsize=10)
fig.suptitle('B3-ft (seed 0, ep400) on data where the other agent A-2 stands still: the black A-2 stays in place in the truth (swapped layout: green on the right),\n'
             'but the prediction moves it toward green (as A-2 always did in training) - A-2\'s regularity learned by observation only',
             fontsize=10)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(D, f'fig_a2_learned_motion.{ext}'), dpi=200)
print('saved')
