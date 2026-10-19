"""
v12 trial comparison figure (instructions Sec.5.1): self/other code alignment cos(ov(A-1 view),
sv(A-2 view)) and the Fig.5 margin (other_self - other_other, original / swap grid) for the v12 trial
(from scratch), (P-a), (P-b), (P-c) and v4 (the paper encoder, reproduced). ep150 / Fig.5 Decoder ep100.
Writes data/result/v12/trial/figs/trial_alignment_fig5.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa

T = os.path.join(RES, 'v12', 'trial')


def main():
    J = lambda f: json.load(open(os.path.join(T, f)))  # noqa
    rows = []
    for lab, ev, f5 in (('v12 scratch\nno exp.', 'eval_random_ep150.json', 'fig5/fig5_v12_trial_fig5_random_s0_ep100.json'),
                        ('v12 scratch\nexp.', 'eval_mixed_ep150.json', 'fig5/fig5_v12_trial_fig5_mixed_s0_ep100.json'),
                        ('(P-a) paper model\non v11 data', 'pa_eval_random_ep150.json', 'fig5_pa_random/fig5_v12_trial_fig5_pa_random_s0_ep100.json'),
                        ('(P-b) v4 enc.\nfrozen', 'pb_eval_mixed_ep150.json', 'fig5_pb_mixed/fig5_v12_trial_fig5_pb_mixed_s0_ep100.json'),
                        ('(P-c) v4 enc.\n+ P held', 'pc_eval_mixed_ep150.json', 'fig5_pc_mixed/fig5_v12_trial_fig5_pc_mixed_s0_ep100.json')):
        e, f = J(ev), J(f5)
        rows.append((lab, e['alignment']['cos_ov(A1)_sv(A2)'], f['original']['margin_other_self_minus_other_other'],
                     f['swap_rg']['margin_other_self_minus_other_other']))
    rows.append(('v4 (paper\nencoder)', 0.7993, 0.0911, 0.0519))
    lab = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(rows))
    axes[0].bar(x, [r[1] for r in rows], color=['0.6', 'tab:red', 'tab:orange', 'tab:blue', 'tab:purple', 'k'])
    axes[0].set_title('self/other code alignment  cos(ov(A-1 view), sv(A-2 view))', fontsize=9)
    axes[1].bar(x - 0.2, [r[2] for r in rows], 0.4, label='original grid')
    axes[1].bar(x + 0.2, [r[3] for r in rows], 0.4, label='swapped grid')
    axes[1].set_title('Fig.5 margin (other_self - other_other)', fontsize=9); axes[1].legend(fontsize=8)
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(lab, fontsize=7); a.axhline(0, color='k', lw=0.8)
    fig.suptitle('Why the encoders are frozen in v12 (trial ep150; (P-b) and (P-c) share the frozen v4 encoder, so their values coincide)', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.join(T, 'figs'), exist_ok=True)
    fig.savefig(os.path.join(T, 'figs', 'trial_alignment_fig5.png'), dpi=130)
    print([(r[0].replace('\n', ' '), round(r[1], 3), round(r[2], 4), round(r[3], 4)) for r in rows])


if __name__ == '__main__':
    main()
