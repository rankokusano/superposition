"""
v11 summary figures (docs/v11_summary.md Sec.7), from saved outputs (plus one model run for the
M1(d) example frame):
  fig_m1d_colours.png  : stage-1 M1(d) S_X per colour (mse late-5, per seed), random vs mixed;
  fig_m1d_example.png  : one frame: actual next view, counterfactual "one step toward red" view,
                         prediction (mixed s0, mse ep200, canonical masking);
  fig_fig5_transform.png: Fig.5 margin (self_other - other_other) without / with the linear map
                         ov(A-1 view) -> sv(A-2 view), and the self/other code alignment (cos) before / after.
Writes under data/result/v11/summary_figs/.
"""

import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, RES, load_model  # noqa
import util  # noqa

OUTD = os.path.join(RES, 'v11', 'summary_figs')
MAIN = os.path.join(RES, 'v11', 'main')
COLORS = ('Red', 'Green', 'Blue', 'Cyan')
EPS = [160, 170, 180, 190, 200]


def fig_m1d_colours():
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.13
    for si, s in enumerate((0, 1, 2)):
        for ci, (c, col) in enumerate((('random', 'tab:gray'), ('mixed', 'tab:red'))):
            vals = np.mean([[json.load(open(os.path.join(MAIN, 'm1', f'm1_v11_s1_X_mse_s{s}_ep{e}_pmaskoff.json')))[c]['M1d']['S'][X]['mean']
                             for X in COLORS] for e in EPS], 0)
            ax.bar(np.arange(4) + (si * 2 + ci - 2.5) * w, vals, w, color=col, alpha=0.5 + 0.25 * si,
                   label=f'{"no experience" if c == "random" else "experience"} s{s}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(COLORS)
    ax.set_ylabel('S_X = L1(pred, actual next) - L1(pred, cf_X)')
    ax.set_title('Stage 1 M1(d): pull of the prediction toward "one step toward X" (mse late-5)', fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, 'fig_m1d_colours.png'), dpi=150); plt.close(fig)


@torch.no_grad()
def fig_m1d_example(ep=0):
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    cf = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_views.h5'), 'r')
    sv = g['self_vision'][ep:ep + 1]
    pos = g['self_position'][ep]
    model, p_mask = load_model('v11_s1_mixed_mse', 0, 200, dev)
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv))).to(dev)
    util.seed_all(0)
    model.init_state(1)
    preds = []
    for t in range(100):
        pm = 0 if t == 0 else p_mask
        p = model({'self_vision': x[:, t]}, pm, pm)['self_vision']
        preds.append(((p + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy())
    valid = cf['valid'][ep][:, 0]
    cfr = cf['cf_vision'][ep][:, 0].astype(np.float32) / 255
    nxt = sv[0, 1:101].astype(np.float32) / 255
    m = pos[1:101] - pos[:100]
    # frames where A-2 (black) is small in both views, so the pull is not an occlusion effect
    black = lambda v: (v.max(-1) < 0.1).mean()  # noqa
    ok = [valid[t] and black(nxt[t]) < 0.01 and black(cfr[t]) < 0.01 for t in range(100)]
    gain = np.array([np.abs(preds[t] - nxt[t]).mean() - np.abs(preds[t] - cfr[t]).mean() if ok[t] else -1 for t in range(100)])
    t = int(np.argmax(gain))
    fig, axes = plt.subplots(3, 1, figsize=(6, 4.2))
    for a, (img, ttl) in zip(axes, ((nxt[t], f'actual next view v_(t+1)  (episode {ep}, t={t}: largest pull among frames where A-2 covers < 1% of both views; illustrative only)'),
                                    (cfr[t], 'counterfactual: one step toward red'),
                                    (preds[t], f'prediction (experience s0, mse ep200): L1 to actual {np.abs(preds[t] - nxt[t]).mean():.3f}, to counterfactual {np.abs(preds[t] - cfr[t]).mean():.3f}'))):
        a.imshow(img, aspect='auto'); a.set_title(ttl, fontsize=7); a.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, 'fig_m1d_example.png'), dpi=150); plt.close(fig)
    return t, float(np.degrees(np.arctan2(m[t, 1], m[t, 0])))


def fig_fig5_transform():
    e = json.load(open(os.path.join(MAIN, 'fig5_reason', 'fig5_extra.json')))['check2']
    names = list(e)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [e[n]['fig5_unmapped_original']['self_other_minus_other_other'] for n in names], 0.4, label='without map', color='tab:gray')
    axes[0].bar(x + 0.2, [e[n]['fig5_mapped_original']['self_other_minus_other_other'] for n in names], 0.4, label='with linear map', color='tab:blue')
    axes[0].set_title('Fig.5 margin: self_other - other_other (original grid)', fontsize=9)
    axes[1].bar(x - 0.2, [e[n]['cos_before'] for n in names], 0.4, label='before map', color='tab:gray')
    axes[1].bar(x + 0.2, [e[n]['cos_after'] for n in names], 0.4, label='after map', color='tab:blue')
    axes[1].set_title('cos(ov(A-1 view), sv(A-2 view)): self/other code alignment', fontsize=9)
    for a in axes:
        a.axhline(0, color='k', lw=0.8); a.set_xticks(x)
        a.set_xticklabels([n.replace('v11_', '').replace('v4_paper', 'v4 (paper)') for n in names], rotation=30, fontsize=8)
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUTD, 'fig_fig5_transform.png'), dpi=150); plt.close(fig)


def main():
    os.makedirs(OUTD, exist_ok=True)
    fig_m1d_colours()
    print('example frame / actual step angle', fig_m1d_example())
    fig_fig5_transform()
    print('saved', OUTD)


if __name__ == '__main__':
    main()
