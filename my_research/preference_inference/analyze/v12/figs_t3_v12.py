"""
Thesis figure: T3 (process-2 input replaced) across the stage-2 attempts (docs/v12_summary.md Sec.12), CPU, saved
outputs only. Prediction error with process-2 = generated Q2 / zero / D6 (A-2's true pleasure P_A2) / position proxy,
on the registered (visible) test, each G model on its own setting's set (green-liker model on set 1, preference model
on set 2), G late-5 (ep160-200), both conditions:
  (a) other-side feature prediction error (MSE), (b) self-vision error (L1 per pixel);
  bars = mean over pairs x conditions x checkpoints, dots = per pair x condition (late-5 mean);
  (c) stage 2c on the HIDDEN test (Sec.46 C4): self-vision error with generated / zero / D6, split by A-2 hidden /
      visible frames (set 1 models).
Writes data/result/v12/report/figs/t3_attempts.png.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(_PI, 'data', 'result', 'v12')
EPS = (160, 170, 180, 190, 200)
ATT = [('formal', 'stage2', (0, 1, 2, 3)), ('2b', 'stage2b', (0, 1, 2, 3)), ('shape A', 'stage2a', (0,)),
       ('2c (visible test)', 'stage2c_vis', (0, 1, 2))]
KINDS = [('zero', '#bdbdbd'), ('gen', '#d62728'), ('d6', '#1f77b4'), ('proxy', '#9467bd')]
LAB = {'zero': 'zero', 'gen': 'generated Q2', 'd6': 'D6 (true P_A2)', 'proxy': 'position proxy'}


def collect(out, seeds, metric):
    vals = {k: [] for k, _ in KINDS}
    for s in seeds:
        for st in ('green', 'pref'):
            for c in ('random', 'mixed'):
                per = {k: [] for k, _ in KINDS}
                for e in EPS:
                    f = os.path.join(V, out, 't1t3', f'{st}_{c}_s{s}_ep{e}.json')
                    if not os.path.exists(f):
                        continue
                    j = json.load(open(f))[st]
                    for k, _ in KINDS:
                        per[k].append(j[k][metric])
                for k in per:
                    if per[k]:
                        vals[k].append(float(np.mean(per[k])))
    return vals


def main():
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={'width_ratios': [1.25, 1.25, 0.8]})
    for ax, metric, ttl in ((axes[0], 'other_fp_mse', '(a) other-side feature prediction error (MSE)'),
                            (axes[1], 'self_L1', '(b) self-vision prediction error (L1 per pixel)')):
        xt = []
        for i, (name, out, seeds) in enumerate(ATT):
            vals = collect(out, seeds, metric)
            for j, (k, col) in enumerate(KINDS):
                x = i * 5 + j
                v = vals[k]
                ax.bar(x, np.mean(v), 0.8, color=col, alpha=0.75, label=LAB[k] if i == 0 else None)
                ax.scatter(np.full(len(v), x) + np.linspace(-0.25, 0.25, len(v)), v, s=8, color='k', zorder=3)
            xt.append((i * 5 + 1.5, f'{name}\n({len(seeds)} pair{"s" if len(seeds) > 1 else ""})'))
        ax.set_xticks([x for x, _ in xt]); ax.set_xticklabels([t for _, t in xt], fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        lo = min(min(collect(o, s, metric)['gen']) for _, o, s in ATT)
        ax.set_ylim(lo * 0.8, ax.get_ylim()[1] * 1.12)
    # (c) stage 2c hidden test, by frames
    ax = axes[2]
    vals = {(k, fr): [] for k in ('zero', 'gen', 'd6') for fr in ('hidden', 'visible')}
    for s in (0, 1, 2):
        for c in ('random', 'mixed'):
            for k in ('zero', 'gen', 'd6'):
                for fr in ('hidden', 'visible'):
                    vals[(k, fr)].append(np.mean([json.load(open(os.path.join(V, 'stage2c', 'eval', f'green_{c}_s{s}_ep{e}.json')))
                                                  ['T3_self_L1_green_by_frames'][f'{k}_{fr}'] for e in EPS]))
    for i, fr in enumerate(('hidden', 'visible')):
        for j, (k, col) in enumerate([('zero', '#bdbdbd'), ('gen', '#d62728'), ('d6', '#1f77b4')]):
            x = i * 4 + j
            v = vals[(k, fr)]
            ax.bar(x, np.mean(v), 0.8, color=col, alpha=0.75, label=LAB[k] if i == 0 else None)
            ax.scatter(np.full(len(v), x) + np.linspace(-0.25, 0.25, len(v)), v, s=8, color='k', zorder=3)
    ax.set_xticks([1, 5]); ax.set_xticklabels(['A-2 hidden frames', 'A-2 visible frames'], fontsize=8)
    ax.set_title('(c) stage 2c on the hidden test:\nself-vision error by frames (set-1 models, 3 pairs)', fontsize=10)
    lo = min(min(v) for v in vals.values())
    ax.set_ylim(lo * 0.8, ax.get_ylim()[1] * 1.1)
    ax.legend(fontsize=7, loc='upper right')
    fig.suptitle('T3: prediction error with process-2 = zero / generated Q2 / D6 (A-2\'s true pleasure) / position proxy '
                 '(G late-5; bars = mean, dots = pair x condition)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(V, 'report', 'figs', 't3_attempts.png')
    fig.savefig(out, dpi=120)
    print('saved', out)
    for name, o, s in ATT:
        print(name, {m: {k: round(float(np.mean(v)), 3) for k, v in collect(o, s, m).items()} for m in ('other_fp_mse', 'self_L1')})
    print('2c hidden', {f'{k}_{fr}': round(float(np.mean(v)), 4) for (k, fr), v in vals.items()})


if __name__ == '__main__':
    main()
